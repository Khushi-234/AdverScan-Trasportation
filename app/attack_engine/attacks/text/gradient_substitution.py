"""
Gradient-Guided Token Substitution attack implementation for text models.

Calculates token saliency using embedding gradients, ranks positions by importance,
generates valid substitution candidates using embedding-space cosine similarity,
and evaluates candidates to find the most effective replacement.

Reference:
    Alzantot et al., "Generating Natural Language Adversarial Examples" (EMNLP 2018)
    Jin et al., "Is BERT Really Robust?" (AAAI 2020)
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from app.attack_engine.attack_registry import register_attack
from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attacks.text._text_utils import (
    compute_embedding_gradients,
    compute_token_saliency,
    forward_pass_tokens,
    get_candidate_tokens_by_similarity,
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


class GradientSubstitution(BaseAttack):
    """
    White-box saliency-guided token substitution attack.
    """

    metadata = AttackMetadata(
        name="gradient_substitution",
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
        Initialize GradientSubstitution attack.

        Args:
            model: Target PyTorch model or BaseModelAdapter.
            tokenizer: Optional tokenizer instance.
        """
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)
        self.tokenizer = tokenizer or get_tokenizer(self.model)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate GradientSubstitution configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        max_modifications = params.get("max_modifications", 5)
        if not isinstance(max_modifications, int) or max_modifications <= 0:
            raise AttackConfigurationError(
                f"max_modifications must be a positive integer, got {max_modifications}"
            )

        num_candidates = params.get("num_candidates", 10)
        if not isinstance(num_candidates, int) or num_candidates <= 0:
            raise AttackConfigurationError(
                f"num_candidates must be a positive integer, got {num_candidates}"
            )

        sim_thresh = params.get("similarity_threshold", 0.4)
        if not isinstance(sim_thresh, (int, float)) or not (0.0 <= sim_thresh <= 1.0):
            raise AttackConfigurationError(
                f"similarity_threshold must be between 0.0 and 1.0, got {sim_thresh}"
            )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate adversarial examples using gradient-guided substitution.

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

        max_modifications = int(params.get("max_modifications", 5))
        num_candidates = int(params.get("num_candidates", 10))
        similarity_threshold = float(params.get("similarity_threshold", 0.4))
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()

        tokenizer = params.get("tokenizer", self.tokenizer)
        device = get_device(self.raw_model)
        embedding_layer = get_embedding_layer(self.raw_model, params)
        embedding_matrix = embedding_layer.weight
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
        adv_input_ids = input_ids.clone()

        was_training = self.raw_model.training
        self.raw_model.eval()

        for b in range(batch_size):
            sample_ids = adv_input_ids[b : b + 1].clone()
            sample_mask = attention_mask[b : b + 1].clone()
            sample_label = labels_tensor[b : b + 1]

            # Step 1: Initial prediction check
            with torch.no_grad():
                init_logits = forward_pass_tokens(
                    self.raw_model, sample_ids, sample_mask
                )
                init_pred = init_logits.argmax(dim=-1).item()

            if targeted and init_pred == sample_label.item():
                adv_input_ids[b] = sample_ids[0]
                continue
            elif not targeted and init_pred != sample_label.item():
                adv_input_ids[b] = sample_ids[0]
                continue

            num_modified = 0

            for mod_step in range(max_modifications):
                # Step 2: Compute embedding gradients for current token sequence
                grad, _, _ = compute_embedding_gradients(
                    model=self.raw_model,
                    embedding_layer=embedding_layer,
                    input_ids=sample_ids,
                    attention_mask=sample_mask,
                    labels=sample_label,
                    loss_fn=loss_fn,
                    targeted=targeted,
                )

                saliency = compute_token_saliency(grad, method="norm")[0]  # (seq_len,)

                # Mask out special tokens and padding
                for pos in range(seq_len):
                    tok_id = sample_ids[0, pos].item()
                    if tok_id in special_token_ids or sample_mask[0, pos].item() == 0:
                        saliency[pos] = float("-inf")

                # Rank positions by saliency
                sorted_positions = torch.argsort(saliency, descending=True).tolist()
                eligible_positions = [
                    pos for pos in sorted_positions if saliency[pos] != float("-inf")
                ]

                if not eligible_positions:
                    break

                # Pick the most salient position
                target_pos = eligible_positions[0]
                current_tok = sample_ids[0, target_pos].item()
                current_emb = embedding_matrix[current_tok]

                # Step 3: Candidate generation decoupled from gradient ranking
                candidates = get_candidate_tokens_by_similarity(
                    token_embedding=current_emb,
                    embedding_weight=embedding_matrix,
                    top_k=num_candidates,
                    special_ids=special_token_ids,
                    original_token_id=current_tok,
                    min_similarity=similarity_threshold,
                )

                if not candidates:
                    # Fallback to nearest without strict threshold if none met threshold
                    candidates = get_candidate_tokens_by_similarity(
                        token_embedding=current_emb,
                        embedding_weight=embedding_matrix,
                        top_k=num_candidates,
                        special_ids=special_token_ids,
                        original_token_id=current_tok,
                    )

                if not candidates:
                    continue

                # Step 4: Evaluate candidates on model to find best substitution
                best_cand: Optional[int] = None
                best_score = float("-inf") if not targeted else float("inf")

                with torch.no_grad():
                    for cand_id in candidates:
                        test_ids = sample_ids.clone()
                        test_ids[0, target_pos] = cand_id

                        test_logits = forward_pass_tokens(
                            self.raw_model, test_ids, sample_mask
                        )
                        cand_loss = loss_fn(test_logits, sample_label).item()
                        cand_pred = test_logits.argmax(dim=-1).item()

                        # Immediate success check
                        if targeted and cand_pred == sample_label.item():
                            best_cand = cand_id
                            break
                        elif not targeted and cand_pred != sample_label.item():
                            best_cand = cand_id
                            break

                        if targeted:
                            if cand_loss < best_score:
                                best_score = cand_loss
                                best_cand = cand_id
                        else:
                            if cand_loss > best_score:
                                best_score = cand_loss
                                best_cand = cand_id

                if best_cand is not None and best_cand != current_tok:
                    sample_ids[0, target_pos] = best_cand
                    num_modified += 1

                    # Check if attack succeeded
                    with torch.no_grad():
                        curr_logits = forward_pass_tokens(
                            self.raw_model, sample_ids, sample_mask
                        )
                        curr_pred = curr_logits.argmax(dim=-1).item()

                    if targeted and curr_pred == sample_label.item():
                        break
                    elif not targeted and curr_pred != sample_label.item():
                        break
                else:
                    # No candidate improved loss at this position
                    break

            adv_input_ids[b] = sample_ids[0]

        self.raw_model.train(was_training)

        return reconstruct_output(
            adv_input_ids=adv_input_ids,
            input_type=input_type,
            tokenizer=tokenizer,
            original_input=inputs,
        )


# Self-register
register_attack("gradient_substitution", GradientSubstitution)
