"""
HotFlip adversarial attack implementation for text models.

HotFlip calculates directional derivatives w.r.t token embeddings to identify
influential token positions and select discrete token substitutions that maximize
adversarial loss (or minimize loss toward a target class).

Reference:
    Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text Classification" (ACL 2017)
"""

from typing import Any, List, Optional, Set

import torch
import torch.nn as nn

from app.attack_engine.attack_registry import register_attack
from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attacks.text._text_utils import (
    compute_embedding_gradients,
    forward_pass_tokens,
    get_embedding_layer,
    get_raw_model,
    get_special_token_ids,
    get_tokenizer,
    prepare_text_inputs,
    reconstruct_output,
    validate_labels,
)
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackConfigurationError
from app.attack_engine.models import AttackMetadata
from app.attack_engine.utils.tensor_utils import get_device


class HotFlip(BaseAttack):
    """
    HotFlip white-box gradient-guided discrete token substitution attack.
    """

    metadata = AttackMetadata(
        name="hotflip",
        domain="text",
        category="gradient",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L0"],
        computational_cost="medium",
        supported_model_types=["pytorch"],
        parameters={
            "token_level": True,
            "requires_gradient": True,
            "embedding_based": True,
            "input_agnostic": False,
        },
    )

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        """
        Initialize HotFlip attack.

        Args:
            model: PyTorch model or BaseModelAdapter.
            tokenizer: Optional tokenizer instance.
        """
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)
        self.tokenizer = tokenizer or get_tokenizer(self.model)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate HotFlip configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        max_modifications = params.get("max_modifications", params.get("max_flips", 5))
        if not isinstance(max_modifications, int) or max_modifications <= 0:
            raise AttackConfigurationError(
                f"max_modifications must be a positive integer, got {max_modifications}"
            )

        candidate_vocab_size = params.get(
            "candidate_vocab_size", params.get("top_k_candidates", 50)
        )
        if not isinstance(candidate_vocab_size, int) or candidate_vocab_size <= 0:
            raise AttackConfigurationError(
                f"candidate_vocab_size must be a positive integer, got {candidate_vocab_size}"
            )

        if "epsilon" in params and params["epsilon"] < 0:
            raise AttackConfigurationError(
                f"Epsilon cannot be negative, got {params['epsilon']}"
            )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate HotFlip adversarial examples.

        Args:
            inputs: Raw text string, list of strings, token IDs tensor, or dict.
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig.

        Returns:
            Adversarial examples formatted according to inputs.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        targeted = params.get("targeted", False)
        target_label = params.get("target_label", None)
        max_flips = params.get("max_modifications", params.get("max_flips", 5))
        candidate_vocab_size = params.get(
            "candidate_vocab_size", params.get("top_k_candidates", 50)
        )
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()

        tokenizer = params.get("tokenizer", self.tokenizer)
        device = get_device(self.raw_model)
        embedding_layer = get_embedding_layer(self.raw_model, params)
        embedding_matrix = embedding_layer.weight  # shape (vocab_size, embed_dim)
        vocab_size = embedding_matrix.size(0)

        # Standardize inputs
        input_ids, attention_mask, input_type, _ = prepare_text_inputs(
            inputs=inputs,
            tokenizer=tokenizer,
            device=device,
        )

        batch_size, seq_len = input_ids.shape
        labels_tensor = validate_labels(
            labels=labels,
            batch_size=batch_size,
            device=device,
            targeted=targeted,
            target_label=target_label,
        )

        special_token_ids = get_special_token_ids(tokenizer, vocab_size=vocab_size)

        # Prepare copy of token IDs for adversarial modification
        adv_input_ids = input_ids.clone()

        # Execute sample-by-sample for fine-grained token-level control
        for b in range(batch_size):
            sample_ids = adv_input_ids[b : b + 1].clone()
            sample_mask = attention_mask[b : b + 1].clone()
            sample_label = labels_tensor[b : b + 1]

            # Check initial prediction
            with torch.no_grad():
                init_logits = forward_pass_tokens(
                    self.raw_model, sample_ids, sample_mask
                )
                init_pred = init_logits.argmax(dim=-1).item()

            if targeted and init_pred == sample_label.item():
                # Already predicts target class
                adv_input_ids[b] = sample_ids[0]
                continue
            elif not targeted and init_pred != sample_label.item():
                # Already misclassified
                adv_input_ids[b] = sample_ids[0]
                continue

            # Track modified positions to respect budget without re-flipping same index if desired
            modified_positions: Set[int] = set()

            for flip_step in range(max_flips):
                # Compute gradient w.r.t embeddings
                grad, _, _ = compute_embedding_gradients(
                    model=self.raw_model,
                    embedding_layer=embedding_layer,
                    input_ids=sample_ids,
                    attention_mask=sample_mask,
                    labels=sample_label,
                    loss_fn=loss_fn,
                    targeted=targeted,
                )
                # grad shape: (1, seq_len, embed_dim)
                grad_b = grad[0]  # (seq_len, embed_dim)

                # Find valid non-special token positions
                valid_positions: List[int] = []
                for pos in range(seq_len):
                    tok_id = sample_ids[0, pos].item()
                    # Skip special tokens, padding, and masked positions
                    if tok_id in special_token_ids or sample_mask[0, pos].item() == 0:
                        continue
                    valid_positions.append(pos)

                if not valid_positions:
                    break

                # Score candidate replacements
                # First-order Taylor approximation:
                # Untargeted: Delta L = (v_j - e_i)^T g_i -> maximize
                # Targeted:   Delta L_target = (e_i - v_j)^T g_target -> maximize
                best_score = float("-inf")
                best_pos: Optional[int] = None
                best_token: Optional[int] = None

                # Subsample or evaluate candidate vocabulary
                # If candidate_vocab_size < vocab_size, select top candidate tokens by gradient dot product
                cand_pool_size = min(candidate_vocab_size, vocab_size)

                for pos in valid_positions:
                    current_tok = sample_ids[0, pos].item()
                    curr_emb = embedding_matrix[current_tok]  # (embed_dim,)
                    g_i = grad_b[pos]  # (embed_dim,)

                    if targeted:
                        # Direction to decrease target loss: move in -g_i direction
                        eff_g = -g_i
                    else:
                        # Direction to increase true class loss: move in +g_i direction
                        eff_g = g_i

                    # (v_j - e_i) dot eff_g = v_j dot eff_g - e_i dot eff_g
                    proj_scores = torch.matmul(embedding_matrix, eff_g) - torch.dot(
                        curr_emb, eff_g
                    )

                    # Mask special tokens and current token
                    for sid in special_token_ids:
                        if 0 <= sid < vocab_size:
                            proj_scores[sid] = float("-inf")
                    proj_scores[current_tok] = float("-inf")

                    top_vals, top_indices = torch.topk(proj_scores, k=cand_pool_size)

                    if top_vals[0].item() > best_score:
                        best_score = top_vals[0].item()
                        best_pos = pos
                        best_token = top_indices[0].item()

                if best_pos is None or best_token is None:
                    break

                # Apply token flip
                sample_ids[0, best_pos] = best_token
                modified_positions.add(best_pos)

                # Re-evaluate model prediction
                with torch.no_grad():
                    curr_logits = forward_pass_tokens(
                        self.raw_model, sample_ids, sample_mask
                    )
                    curr_pred = curr_logits.argmax(dim=-1).item()

                if targeted and curr_pred == sample_label.item():
                    # Attack succeeded
                    break
                elif not targeted and curr_pred != sample_label.item():
                    # Attack succeeded
                    break

            adv_input_ids[b] = sample_ids[0]

        return reconstruct_output(
            adv_input_ids=adv_input_ids,
            input_type=input_type,
            tokenizer=tokenizer,
            original_input=inputs,
        )


# Self-register
register_attack("hotflip", HotFlip)
