"""
White-Box TextBugger adversarial attack implementation.

Combines white-box gradient saliency analysis to locate the most influential
tokens/words with character-level and token-level perturbations ("bugs").

Reference:
    Li et al., "TextBugger: Generating Adversarial Text Against Real-world Applications" (NDSS 2019)
"""

from typing import Any, Dict, List, Optional, Set

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


class TextBuggerWB(BaseAttack):
    """
    White-Box TextBugger attack combining gradient saliency and character/token-level bugs.
    """

    metadata = AttackMetadata(
        name="textbugger_wb",
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
            "character_level": True,
            "requires_gradient": True,
            "input_agnostic": False,
        },
    )

    # Visual and phonetic character substitution table (Sub-C)
    SIMILAR_CHAR_MAP: Dict[str, List[str]] = {
        "a": ["@", "4", "а", "à"],
        "b": ["8", "6", "d"],
        "c": ["(", "<", "с", "ç"],
        "d": ["b", "cl"],
        "e": ["3", "е", "€", "è", "é"],
        "g": ["9", "q"],
        "h": ["#", "n"],
        "i": ["1", "!", "|", "l", "ì"],
        "l": ["1", "|", "i"],
        "m": ["rn", "nn"],
        "n": ["m", "r"],
        "o": ["0", "о", "ö", "ò"],
        "p": ["q", "р"],
        "s": ["$", "5", "z"],
        "t": ["7", "+"],
        "u": ["v", "µ"],
        "v": ["u", "w"],
        "w": ["vv", "uu"],
        "x": ["*", "%"],
        "y": ["j", "v"],
        "z": ["2", "s"],
    }

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        """
        Initialize TextBuggerWB attack.

        Args:
            model: Target PyTorch model or BaseModelAdapter.
            tokenizer: Optional tokenizer instance.
        """
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)
        self.tokenizer = tokenizer or get_tokenizer(self.model)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate TextBuggerWB configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        max_modifications = params.get("max_modifications", 5)
        if not isinstance(max_modifications, int) or max_modifications <= 0:
            raise AttackConfigurationError(
                f"max_modifications must be a positive integer, got {max_modifications}"
            )

        bug_types = params.get(
            "bug_types", ["insert", "delete", "swap", "sub_c", "sub_w"]
        )
        if not isinstance(bug_types, (list, tuple)) or len(bug_types) == 0:
            raise AttackConfigurationError(
                "bug_types must be a non-empty list of bug types."
            )

        valid_bugs = {"insert", "delete", "swap", "sub_c", "sub_w"}
        for bug in bug_types:
            if bug.lower() not in valid_bugs:
                raise AttackConfigurationError(
                    f"Unknown bug type '{bug}'. Valid options: {sorted(list(valid_bugs))}"
                )

    def _generate_bugs_for_word(
        self,
        word: str,
        bug_types: List[str],
    ) -> List[str]:
        """
        Generate candidate character-level bugged words for a single text token/word.

        Bugs supported:
            - insert: insert whitespace or character into word
            - delete: remove random character from word
            - swap: swap two adjacent characters
            - sub_c: replace character with visually similar glyph / symbol
        """
        candidates: Set[str] = set()
        clean_word = word.strip()
        if not clean_word:
            return []

        word_len = len(clean_word)

        # Bug 1: Insert (insert a character or space)
        if "insert" in bug_types and word_len >= 2:
            # Insert at middle position
            mid = word_len // 2
            candidates.add(clean_word[:mid] + "_" + clean_word[mid:])
            candidates.add(clean_word[:mid] + clean_word[mid - 1] + clean_word[mid:])

        # Bug 2: Delete (delete a random interior character if word is long enough)
        if "delete" in bug_types and word_len >= 3:
            for idx in range(1, word_len - 1):
                candidates.add(clean_word[:idx] + clean_word[idx + 1 :])

        # Bug 3: Swap (swap adjacent characters)
        if "swap" in bug_types and word_len >= 2:
            for idx in range(word_len - 1):
                swapped = list(clean_word)
                swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
                candidates.add("".join(swapped))

        # Bug 4: Sub-C (character substitution with visually similar glyph)
        if "sub_c" in bug_types:
            for idx, ch in enumerate(clean_word):
                lower_ch = ch.lower()
                if lower_ch in self.SIMILAR_CHAR_MAP:
                    for rep in self.SIMILAR_CHAR_MAP[lower_ch]:
                        new_word = clean_word[:idx] + rep + clean_word[idx + 1 :]
                        candidates.add(new_word)

        # Remove unmodified word from candidates
        candidates.discard(clean_word)
        return list(candidates)

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate TextBugger adversarial examples.

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

        max_modifications = params.get("max_modifications", 5)
        bug_types = [
            b.lower()
            for b in params.get(
                "bug_types", ["insert", "delete", "swap", "sub_c", "sub_w"]
            )
        ]
        targeted = params.get("targeted", False)
        target_label = params.get("target_label", None)
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()

        tokenizer = params.get("tokenizer", self.tokenizer)
        device = get_device(self.raw_model)
        embedding_layer = get_embedding_layer(self.raw_model, params)
        embedding_matrix = embedding_layer.weight
        vocab_size = embedding_matrix.size(0)

        # Standardize inputs
        input_ids, attention_mask, input_type, original_texts = prepare_text_inputs(
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

            # Step 2: Compute white-box gradients to rank token positions by saliency
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

            # Zero out special and padding tokens
            for pos in range(seq_len):
                tok_id = sample_ids[0, pos].item()
                if tok_id in special_token_ids or sample_mask[0, pos].item() == 0:
                    saliency[pos] = float("-inf")

            # Rank positions in descending order of saliency
            sorted_positions = torch.argsort(saliency, descending=True).tolist()
            ranked_valid_positions = [
                pos for pos in sorted_positions if saliency[pos] != float("-inf")
            ]

            num_modified = 0

            # Step 3: Iterate through top salient positions and apply candidate bugs
            for pos in ranked_valid_positions:
                if num_modified >= max_modifications:
                    break

                current_tok_id = sample_ids[0, pos].item()
                candidate_tokens: List[int] = []

                # A. Sub-W (Token-level substitution using embedding similarity)
                if "sub_w" in bug_types:
                    tok_emb = embedding_matrix[current_tok_id]
                    sim_cands = get_candidate_tokens_by_similarity(
                        token_embedding=tok_emb,
                        embedding_weight=embedding_matrix,
                        top_k=5,
                        special_ids=special_token_ids,
                        original_token_id=current_tok_id,
                    )
                    candidate_tokens.extend(sim_cands)

                # B. Character-level bugs (Insert, Delete, Swap, Sub-C) via tokenizer if available
                has_char_bugs = any(
                    b in bug_types for b in ("insert", "delete", "swap", "sub_c")
                )
                if (
                    has_char_bugs
                    and tokenizer is not None
                    and hasattr(tokenizer, "decode")
                ):
                    try:
                        word = tokenizer.decode([current_tok_id]).strip()
                        if (
                            word
                            and not word.startswith("##")
                            and not word.startswith("[")
                        ):
                            bugged_words = self._generate_bugs_for_word(word, bug_types)
                            for bw in bugged_words:
                                if hasattr(tokenizer, "encode"):
                                    cand_ids = tokenizer.encode(
                                        bw, add_special_tokens=False
                                    )
                                    if (
                                        len(cand_ids) == 1
                                        and cand_ids[0] not in special_token_ids
                                    ):
                                        candidate_tokens.append(cand_ids[0])
                    except Exception:
                        pass

                if not candidate_tokens:
                    continue

                # Step 4: Evaluate candidate bugs with white-box forward passes
                best_candidate: Optional[int] = None
                best_metric = float("-inf") if not targeted else float("inf")

                for cand_id in candidate_tokens:
                    test_ids = sample_ids.clone()
                    test_ids[0, pos] = cand_id

                    with torch.no_grad():
                        test_logits = forward_pass_tokens(
                            self.raw_model, test_ids, sample_mask
                        )
                        test_loss = loss_fn(test_logits, sample_label).item()
                        test_pred = test_logits.argmax(dim=-1).item()

                    # Check immediate success
                    if targeted and test_pred == sample_label.item():
                        best_candidate = cand_id
                        break
                    elif not targeted and test_pred != sample_label.item():
                        best_candidate = cand_id
                        break

                    # Tracking best candidate by loss
                    if targeted:
                        if test_loss < best_metric:
                            best_metric = test_loss
                            best_candidate = cand_id
                    else:
                        if test_loss > best_metric:
                            best_metric = test_loss
                            best_candidate = cand_id

                if best_candidate is not None and best_candidate != current_tok_id:
                    sample_ids[0, pos] = best_candidate
                    num_modified += 1

                    # Re-evaluate prediction to check early stopping
                    with torch.no_grad():
                        new_logits = forward_pass_tokens(
                            self.raw_model, sample_ids, sample_mask
                        )
                        new_pred = new_logits.argmax(dim=-1).item()

                    if targeted and new_pred == sample_label.item():
                        break
                    elif not targeted and new_pred != sample_label.item():
                        break

            adv_input_ids[b] = sample_ids[0]

        self.raw_model.train(was_training)

        return reconstruct_output(
            adv_input_ids=adv_input_ids,
            input_type=input_type,
            tokenizer=tokenizer,
            original_input=inputs,
        )


# Self-register with primary name and alias
register_attack("textbugger_wb", TextBuggerWB)
register_attack("textbugger", TextBuggerWB)
