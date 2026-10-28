"""
Embedding-Space Projected Gradient Descent (PGD) attack for text models.

Performs continuous iterative PGD optimization in the model's embedding space,
followed by a discrete nearest-neighbor projection to map perturbed continuous
embeddings back to valid discrete vocabulary tokens.

Reference:
    Miyato et al., "Adversarial Training Methods for Semi-Supervised Text Classification" (ICLR 2017)
    Zhu et al., "FreeLB: Enhanced Adversarial Training for Natural Language Understanding" (ICLR 2020)
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from app.attack_engine.attack_registry import register_attack
from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attacks.text._text_utils import (
    forward_pass_embeddings,
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
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.attack_engine.models import AttackMetadata
from app.attack_engine.utils.tensor_utils import get_device


class EmbeddingPGD(BaseAttack):
    """
    Embedding-space Projected Gradient Descent attack with discrete vocabulary projection.
    """

    metadata = AttackMetadata(
        name="embedding_pgd",
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
        supported_norms=["Linf", "L2"],
        computational_cost="medium",
        supported_model_types=["pytorch"],
        parameters={
            "embedding_based": True,
            "token_level": True,
            "requires_gradient": True,
            "input_agnostic": False,
        },
    )

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        """
        Initialize EmbeddingPGD attack.

        Args:
            model: Target PyTorch model or BaseModelAdapter.
            tokenizer: Optional tokenizer instance.
        """
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)
        self.tokenizer = tokenizer or get_tokenizer(self.model)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate EmbeddingPGD configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        epsilon = params.get("epsilon", config.epsilon)
        if epsilon < 0:
            raise AttackConfigurationError(f"Epsilon cannot be negative, got {epsilon}")

        alpha = params.get("alpha", params.get("step_size", 0.1))
        if alpha <= 0:
            raise AttackConfigurationError(
                f"Step size (alpha) must be positive, got {alpha}"
            )

        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iterations", 10))
        )
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise AttackConfigurationError(
                f"Iterations (num_steps) must be a positive integer, got {num_steps}"
            )

        norm = params.get("norm", "Linf")
        if norm.upper() not in ("LINF", "INF", "L2", "2"):
            raise AttackConfigurationError(
                f"Unsupported norm '{norm}'. Supported norms: ['Linf', 'L2']"
            )

        k_nearest = params.get("k_nearest", params.get("num_candidates", 1))
        if not isinstance(k_nearest, int) or k_nearest <= 0:
            raise AttackConfigurationError(
                f"k_nearest must be a positive integer, got {k_nearest}"
            )

        if "max_modifications" in params and params["max_modifications"] is not None:
            max_mod = params["max_modifications"]
            if not isinstance(max_mod, int) or max_mod <= 0:
                raise AttackConfigurationError(
                    f"max_modifications must be a positive integer, got {max_mod}"
                )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate adversarial examples using embedding PGD and k-NN projection.

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

        epsilon = float(
            params.get("epsilon", config.epsilon if config.epsilon > 0 else 0.5)
        )
        alpha = float(
            params.get(
                "alpha", params.get("step_size", epsilon / 4.0 if epsilon > 0 else 0.1)
            )
        )
        num_steps = int(
            params.get("num_steps", params.get("steps", params.get("iterations", 10)))
        )
        norm = str(params.get("norm", "Linf")).upper()
        k_nearest = int(params.get("k_nearest", params.get("num_candidates", 1)))
        max_modifications = params.get("max_modifications", None)
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()

        tokenizer = params.get("tokenizer", self.tokenizer)
        device = get_device(self.raw_model)
        embedding_layer = get_embedding_layer(self.raw_model, params)
        embedding_matrix = embedding_layer.weight  # shape: (vocab_size, embed_dim)
        vocab_size, embed_dim = embedding_matrix.shape

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

        # Build mask of modifiable positions: shape (batch_size, seq_len)
        # 1 for modifiable, 0 for protected (special tokens, padding)
        modifiable_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for b in range(batch_size):
            for s in range(seq_len):
                tok_id = input_ids[b, s].item()
                if tok_id in special_token_ids or attention_mask[b, s].item() == 0:
                    modifiable_mask[b, s] = False

        was_training = self.raw_model.training
        self.raw_model.eval()

        # Phase 1: Continuous Optimization in Embedding Space
        # Extract clean embeddings: shape (batch_size, seq_len, embed_dim)
        clean_embeddings = embedding_layer(input_ids).detach()
        # Initialize continuous perturbation delta
        delta = torch.zeros_like(clean_embeddings, requires_grad=True)

        for step in range(num_steps):
            perturbed_embeddings = clean_embeddings + delta
            logits = forward_pass_embeddings(
                self.raw_model,
                embedding_layer,
                input_ids,
                perturbed_embeddings,
                attention_mask,
            )

            if targeted:
                # Minimize loss toward target class
                loss = -loss_fn(logits, labels_tensor)
            else:
                # Maximize loss away from true class
                loss = loss_fn(logits, labels_tensor)

            grad = torch.autograd.grad(
                loss,
                delta,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                raise AttackExecutionError(
                    "PGD gradient computation failed in embedding space."
                )

            with torch.no_grad():
                # Apply gradient step according to norm constraint
                if norm in ("LINF", "INF"):
                    grad_sign = grad.sign()
                    delta = delta + alpha * grad_sign
                    delta = torch.clamp(delta, -epsilon, epsilon)
                elif norm in ("L2", "2"):
                    grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-12
                    delta = delta + alpha * (grad / grad_norm)
                    # Project onto L2 ball per token
                    delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
                    factor = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)
                    delta = delta * factor

                # Zero out perturbations at protected/special token positions
                mask_expanded = modifiable_mask.unsqueeze(-1).expand_as(delta)
                delta = delta * mask_expanded.float()

            delta.requires_grad_(True)

        self.raw_model.train(was_training)

        # Phase 2: Discrete Nearest-Neighbor Vocabulary Projection
        # Optimized continuous embeddings:
        with torch.no_grad():
            final_embeddings = (clean_embeddings + delta).detach()
            adv_input_ids = input_ids.clone()

            # Precompute squared norms of vocabulary embeddings for fast batched L2 distance
            vocab_weight = embedding_matrix.detach()  # (vocab_size, embed_dim)
            vocab_norm_sq = (vocab_weight**2).sum(dim=-1)  # (vocab_size,)

            for b in range(batch_size):
                # Calculate displacement magnitude per position to prioritize modifications
                displacements = torch.norm(delta[b], p=2, dim=-1)  # (seq_len,)

                # Candidate positions to project
                eligible_positions = [
                    pos for pos in range(seq_len) if modifiable_mask[b, pos].item()
                ]

                # If max_modifications is set, only modify the top-N positions with largest displacement
                if (
                    max_modifications is not None
                    and len(eligible_positions) > max_modifications
                ):
                    eligible_positions = sorted(
                        eligible_positions,
                        key=lambda p: float(displacements[p].item()),
                        reverse=True,
                    )[:max_modifications]

                for pos in eligible_positions:
                    target_vector = final_embeddings[b, pos]  # (embed_dim,)
                    target_norm_sq = (target_vector**2).sum()

                    # L2 distance squared = ||x - v||^2 = ||x||^2 + ||v||^2 - 2 * (x dot v)
                    dists_sq = (
                        target_norm_sq
                        + vocab_norm_sq
                        - 2.0 * torch.matmul(vocab_weight, target_vector)
                    )

                    # Exclude special tokens from nearest neighbor selection
                    for sid in special_token_ids:
                        if 0 <= sid < vocab_size:
                            dists_sq[sid] = float("inf")

                    if k_nearest == 1:
                        best_tok_id = torch.argmin(dists_sq).item()
                        adv_input_ids[b, pos] = best_tok_id
                    else:
                        # Evaluate top-k nearest candidate tokens
                        top_vals, top_cand_ids = torch.topk(
                            dists_sq, k=min(k_nearest, vocab_size), largest=False
                        )
                        best_tok_id = adv_input_ids[b, pos].item()
                        best_loss = float("-inf") if not targeted else float("inf")

                        for cand_id_tensor in top_cand_ids:
                            cand_id = cand_id_tensor.item()
                            if cand_id in special_token_ids:
                                continue

                            test_ids = adv_input_ids[b : b + 1].clone()
                            test_ids[0, pos] = cand_id
                            test_logits = forward_pass_tokens(
                                self.raw_model, test_ids, attention_mask[b : b + 1]
                            )
                            test_loss = loss_fn(
                                test_logits, labels_tensor[b : b + 1]
                            ).item()

                            if targeted:
                                if test_loss < best_loss:
                                    best_loss = test_loss
                                    best_tok_id = cand_id
                            else:
                                if test_loss > best_loss:
                                    best_loss = test_loss
                                    best_tok_id = cand_id

                        adv_input_ids[b, pos] = best_tok_id

        return reconstruct_output(
            adv_input_ids=adv_input_ids,
            input_type=input_type,
            tokenizer=tokenizer,
            original_input=inputs,
        )


# Self-register
register_attack("embedding_pgd", EmbeddingPGD)
