"""
Universal Adversarial Trigger attack implementation for text models.

Optimizes an input-agnostic sequence of trigger tokens across multiple input samples
using model gradients, which when attached (prepended or appended) to inputs consistently
forces the model toward a target class (or causes misclassification).

Reference:
    Wallace et al., "Universal Adversarial Triggers for Attacking and Analyzing NLP" (EMNLP 2019)
"""

from typing import Any, List, Optional, Set, Tuple

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


class UniversalTrigger(BaseAttack):
    """
    Universal Adversarial Trigger attack.

    Input-agnostic trigger optimization across a collection of text inputs.
    """

    metadata = AttackMetadata(
        name="universal_trigger",
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
        computational_cost="high",
        supported_model_types=["pytorch"],
        parameters={
            "input_agnostic": True,
            "token_level": True,
            "requires_gradient": True,
            "embedding_based": True,
        },
    )

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        """
        Initialize UniversalTrigger attack.

        Args:
            model: Target PyTorch model or BaseModelAdapter.
            tokenizer: Optional tokenizer instance.
        """
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)
        self.tokenizer = tokenizer or get_tokenizer(self.model)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate UniversalTrigger configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        trigger_length = params.get("trigger_length", 3)
        if not isinstance(trigger_length, int) or trigger_length <= 0:
            raise AttackConfigurationError(
                f"trigger_length must be a positive integer, got {trigger_length}"
            )

        num_iterations = params.get("num_iterations", params.get("num_steps", 5))
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise AttackConfigurationError(
                f"num_iterations must be a positive integer, got {num_iterations}"
            )

        num_candidates = params.get(
            "candidate_vocab_size", params.get("num_candidates", 20)
        )
        if not isinstance(num_candidates, int) or num_candidates <= 0:
            raise AttackConfigurationError(
                f"candidate_vocab_size must be a positive integer, got {num_candidates}"
            )

        placement = params.get("placement", "prefix")
        if placement not in ("prefix", "suffix"):
            raise AttackConfigurationError(
                f"Unsupported trigger placement '{placement}'. Supported: ['prefix', 'suffix']"
            )

    def _attach_trigger(
        self,
        input_ids: torch.Tensor,
        trigger_ids: List[int],
        placement: str,
        special_token_ids: Set[int],
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attach trigger tokens to a batch of token sequences.

        For prefix: inserts trigger immediately after leading special tokens (like [CLS]).
        For suffix: inserts trigger immediately before trailing special tokens (like [SEP] / [PAD]).
        """
        batch_size, seq_len = input_ids.shape
        trigger_len = len(trigger_ids)
        new_seq_len = seq_len + trigger_len
        device = input_ids.device

        new_input_ids = torch.full(
            (batch_size, new_seq_len), pad_token_id, dtype=torch.long, device=device
        )
        new_attention_mask = torch.zeros(
            (batch_size, new_seq_len), dtype=torch.long, device=device
        )

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            # Identify content boundaries (skipping leading/trailing special tokens and padding)
            leading_special: List[int] = []
            content: List[int] = []
            trailing_special: List[int] = []

            idx = 0
            while idx < len(tokens) and tokens[idx] in special_token_ids:
                leading_special.append(tokens[idx])
                idx += 1

            while idx < len(tokens) and tokens[idx] != pad_token_id:
                if tokens[idx] in special_token_ids:
                    trailing_special.append(tokens[idx])
                else:
                    content.append(tokens[idx])
                idx += 1

            if placement == "prefix":
                assembled = (
                    leading_special + list(trigger_ids) + content + trailing_special
                )
            else:  # suffix
                assembled = (
                    leading_special + content + list(trigger_ids) + trailing_special
                )

            assembled_tensor = torch.tensor(assembled, dtype=torch.long, device=device)
            cur_len = min(len(assembled), new_seq_len)
            new_input_ids[b, :cur_len] = assembled_tensor[:cur_len]
            new_attention_mask[b, :cur_len] = 1

        return new_input_ids, new_attention_mask

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Optimize universal trigger tokens across inputs and return triggered adversarial examples.

        Args:
            inputs: Raw text string, list of strings, token IDs tensor, or dict.
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig.

        Returns:
            Adversarial examples with optimized trigger attached.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        trigger_length = int(params.get("trigger_length", 3))
        num_iterations = int(params.get("num_iterations", params.get("num_steps", 5)))
        num_candidates = int(
            params.get("candidate_vocab_size", params.get("num_candidates", 20))
        )
        placement = str(params.get("placement", "prefix"))
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
        pad_token_id = (
            getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
        )

        # Step 1: Initialize trigger tokens (e.g. common token or configured)
        init_trigger = params.get("initial_trigger", None)
        if isinstance(init_trigger, list) and len(init_trigger) == trigger_length:
            trigger_ids = [int(t) for t in init_trigger]
        elif isinstance(init_trigger, str) and tokenizer is not None:
            encoded_trig = tokenizer.encode(init_trigger, add_special_tokens=False)
            trigger_ids = (
                encoded_trig * ((trigger_length // max(len(encoded_trig), 1)) + 1)
            )[:trigger_length]
        else:
            # Default to a safe common non-special token ID (e.g. 10 or 100, or first non-special token)
            default_id = 1
            while default_id in special_token_ids and default_id < vocab_size - 1:
                default_id += 1
            trigger_ids = [default_id] * trigger_length

        was_training = self.raw_model.training
        self.raw_model.eval()

        # Step 2: Iterative Universal Trigger Optimization
        for iteration in range(num_iterations):
            for trig_idx in range(trigger_length):
                # Attach current trigger tokens to batch
                trig_input_ids, trig_mask = self._attach_trigger(
                    input_ids, trigger_ids, placement, special_token_ids, pad_token_id
                )

                # Compute gradients of batch loss w.r.t embeddings
                embeddings = (
                    embedding_layer(trig_input_ids)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                logits = forward_pass_embeddings(
                    self.raw_model,
                    embedding_layer,
                    trig_input_ids,
                    embeddings,
                    trig_mask,
                )

                if targeted:
                    # Minimize loss toward target class
                    loss = loss_fn(logits, labels_tensor)
                else:
                    # Maximize loss away from true class
                    loss = -loss_fn(logits, labels_tensor)

                loss.backward()

                if embeddings.grad is None:
                    raise AttackExecutionError(
                        "Failed to compute gradients for universal trigger tokens."
                    )

                # Average gradient over batch at the trigger position
                # Determine trigger position in sequences
                if placement == "prefix":
                    # Trigger starts right after leading special token (typically index 1)
                    target_pos = 1 + trig_idx
                else:
                    # Suffix trigger position
                    target_pos = max(
                        1, trig_input_ids.size(1) - 2 - trigger_length + trig_idx
                    )

                target_pos = min(target_pos, embeddings.grad.size(1) - 1)
                avg_grad = embeddings.grad[:, target_pos, :].mean(dim=0)  # (embed_dim,)

                # Linear approximation score for replacement candidates:
                # Targeted: we want to minimize loss -> score = (v_j - e_trig) dot (-avg_grad)
                # Untargeted: we want to maximize loss -> score = (v_j - e_trig) dot (avg_grad)
                current_trig_tok = trigger_ids[trig_idx]
                curr_trig_emb = embedding_matrix[current_trig_tok].detach()

                if targeted:
                    eff_grad = -avg_grad
                else:
                    eff_grad = avg_grad

                proj_scores = torch.matmul(
                    embedding_matrix.detach(), eff_grad
                ) - torch.dot(curr_trig_emb, eff_grad)

                # Mask special tokens
                for sid in special_token_ids:
                    if 0 <= sid < vocab_size:
                        proj_scores[sid] = float("-inf")
                for sid in (pad_token_id,):
                    if 0 <= sid < vocab_size:
                        proj_scores[sid] = float("-inf")

                # Top-K replacement candidates
                _, top_cands = torch.topk(
                    proj_scores, k=min(num_candidates, vocab_size)
                )

                # Step 3: Evaluate candidates across the batch to find best candidate
                best_cand_id = current_trig_tok
                best_eval_loss = float("inf") if targeted else float("-inf")

                with torch.no_grad():
                    for cand_tensor in top_cands:
                        cand_id = cand_tensor.item()
                        if cand_id in special_token_ids:
                            continue

                        test_trigger_ids = list(trigger_ids)
                        test_trigger_ids[trig_idx] = cand_id

                        test_input_ids, test_mask = self._attach_trigger(
                            input_ids,
                            test_trigger_ids,
                            placement,
                            special_token_ids,
                            pad_token_id,
                        )

                        test_logits = forward_pass_tokens(
                            self.raw_model, test_input_ids, test_mask
                        )
                        cand_loss = loss_fn(test_logits, labels_tensor).item()

                        if targeted:
                            if cand_loss < best_eval_loss:
                                best_eval_loss = cand_loss
                                best_cand_id = cand_id
                        else:
                            if cand_loss > best_eval_loss:
                                best_eval_loss = cand_loss
                                best_cand_id = cand_id

                trigger_ids[trig_idx] = best_cand_id

        self.raw_model.train(was_training)

        # Decode trigger string if tokenizer is available
        trigger_text = ""
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                trigger_text = tokenizer.decode(trigger_ids, skip_special_tokens=True)
            except Exception:
                pass

        # Record trigger details into attack parameters for inspectability
        params["trigger_tokens"] = list(trigger_ids)
        params["trigger_text"] = trigger_text
        params["input_agnostic"] = True

        # Generate final triggered batch
        final_input_ids, final_mask = self._attach_trigger(
            input_ids, trigger_ids, placement, special_token_ids, pad_token_id
        )

        return reconstruct_output(
            adv_input_ids=final_input_ids,
            input_type=input_type,
            tokenizer=tokenizer,
            original_input=inputs,
        )


# Self-register
register_attack("universal_trigger", UniversalTrigger)
