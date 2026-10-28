"""
Text utility functions and shared helpers for white-box text adversarial attacks.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
)
from app.ingestion.adapters.base_adapter import BaseModelAdapter


def get_raw_model(model: Any) -> Any:
    """Extract raw framework model if model is wrapped in a BaseModelAdapter."""
    if isinstance(model, BaseModelAdapter):
        return model.get_model()
    return model


def get_embedding_layer(
    model: Any, config_params: Optional[Dict[str, Any]] = None
) -> nn.Embedding:
    """
    Locate and return the word embedding layer (nn.Embedding) of a target model.
    Supports HuggingFace transformer models (BERT, RoBERTa, DistilBERT, etc.)
    and generic PyTorch modules.

    Args:
        model: Target model or adapter.
        config_params: Optional dictionary of attack parameters.

    Returns:
        The nn.Embedding layer.

    Raises:
        UnsupportedModelError: If no embedding layer can be identified.
    """
    params = config_params or {}

    # 1. Explicit embedding layer in configuration
    if "embedding_layer" in params and isinstance(
        params["embedding_layer"], nn.Embedding
    ):
        return params["embedding_layer"]

    raw_model = get_raw_model(model)

    # 2. Check for explicit attribute on model
    if hasattr(raw_model, "embedding_layer") and isinstance(
        raw_model.embedding_layer, nn.Embedding
    ):
        return raw_model.embedding_layer

    # 3. Hugging Face standard get_input_embeddings()
    if hasattr(raw_model, "get_input_embeddings") and callable(
        raw_model.get_input_embeddings
    ):
        emb = raw_model.get_input_embeddings()
        if isinstance(emb, nn.Embedding):
            return emb

    # 4. Standard attribute paths (BERT, RoBERTa, DistilBERT, GPT2, Electra)
    attr_paths = [
        ("bert", "embeddings", "word_embeddings"),
        ("roberta", "embeddings", "word_embeddings"),
        ("distilbert", "embeddings", "word_embeddings"),
        ("electra", "embeddings", "word_embeddings"),
        ("deberta", "embeddings", "word_embeddings"),
        ("embeddings", "word_embeddings"),
        ("transformer", "wte"),
        ("embeddings",),
        ("embedding",),
        ("word_embeddings",),
        ("wte",),
    ]
    for path in attr_paths:
        curr = raw_model
        found = True
        for attr in path:
            if hasattr(curr, attr):
                curr = getattr(curr, attr)
            else:
                found = False
                break
        if found and isinstance(curr, nn.Embedding):
            return curr

    # 5. Search all submodules for the first nn.Embedding
    for module in raw_model.modules():
        if isinstance(module, nn.Embedding):
            return module

    raise UnsupportedModelError(
        f"Could not locate an nn.Embedding layer in model {type(raw_model).__name__}. "
        "Provide 'embedding_layer' in attack config params or ensure model has get_input_embeddings()."
    )


def get_tokenizer(
    model: Any, config_params: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Retrieve the tokenizer instance from attack config parameters or model attribute.
    """
    params = config_params or {}
    if "tokenizer" in params and params["tokenizer"] is not None:
        return params["tokenizer"]

    raw_model = get_raw_model(model)
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        return model.tokenizer
    if hasattr(raw_model, "tokenizer") and raw_model.tokenizer is not None:
        return raw_model.tokenizer

    return None


def get_special_token_ids(
    tokenizer: Optional[Any], vocab_size: Optional[int] = None
) -> Set[int]:
    """
    Extract set of special token IDs (e.g. [CLS], [SEP], [PAD], [UNK], [MASK], <s>, </s>, <pad>)
    that must be protected from modification or replacement.
    """
    special_ids: Set[int] = set()

    if tokenizer is not None:
        # Check all_special_ids
        if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids:
            special_ids.update(int(tid) for tid in tokenizer.all_special_ids)

        # Check individual special token attributes
        attrs = [
            "cls_token_id",
            "sep_token_id",
            "pad_token_id",
            "unk_token_id",
            "mask_token_id",
            "bos_token_id",
            "eos_token_id",
        ]
        for attr in attrs:
            val = getattr(tokenizer, attr, None)
            if val is not None:
                special_ids.add(int(val))

    # By convention, if pad_token_id was not detected, token 0 is often PAD/special
    if not special_ids and vocab_size is not None and vocab_size > 0:
        special_ids.add(0)

    return special_ids


def prepare_text_inputs(
    inputs: Any,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, str, Optional[List[str]]]:
    """
    Standardize text inputs into (input_ids, attention_mask, input_type, original_texts).

    Supported input formats:
        - str: Single raw string
        - List[str]: Batch of raw strings
        - torch.Tensor: 1D or 2D tensor of token IDs
        - dict: Dictionary containing 'input_ids' and optional 'attention_mask'

    Returns:
        Tuple of:
            input_ids: 2D LongTensor of shape (batch_size, sequence_length)
            attention_mask: 2D LongTensor of shape (batch_size, sequence_length)
            input_type: One of 'str', 'list_str', 'tensor', 'dict'
            original_texts: List of original text strings if inputs were text, else None

    Raises:
        AttackExecutionError: If inputs are empty or invalid.
        AttackConfigurationError: If raw strings are provided without a tokenizer.
    """
    if inputs is None:
        raise AttackExecutionError("Input text or tensor cannot be None.")

    target_device = device or torch.device("cpu")

    # 1. Single string input
    if isinstance(inputs, str):
        if not inputs.strip():
            raise AttackExecutionError(
                "Input text string cannot be empty or whitespace only."
            )
        if tokenizer is None:
            raise AttackConfigurationError(
                "A tokenizer must be provided (via config.params['tokenizer'] or model.tokenizer) "
                "when inputs are raw text strings."
            )
        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            target_device
        )
        return input_ids, attention_mask, "str", [inputs]

    # 2. List or tuple of strings or ints
    if isinstance(inputs, (list, tuple)):
        if len(inputs) == 0:
            raise AttackExecutionError("Input list cannot be empty.")

        if all(isinstance(x, str) for x in inputs):
            if any(not str(x).strip() for x in inputs):
                raise AttackExecutionError(
                    "Input list contains empty or whitespace-only strings."
                )
            if tokenizer is None:
                raise AttackConfigurationError(
                    "A tokenizer must be provided when inputs are raw text strings."
                )
            encoded = tokenizer(
                list(inputs),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(target_device)
            attention_mask = encoded.get(
                "attention_mask", torch.ones_like(input_ids)
            ).to(target_device)
            return input_ids, attention_mask, "list_str", list(inputs)

        elif all(isinstance(x, (int, np.integer)) for x in inputs):
            tensor_1d = torch.tensor(inputs, dtype=torch.long, device=target_device)
            input_ids = tensor_1d.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask, "tensor", None

        else:
            raise AttackExecutionError(
                f"Unsupported list item types in inputs: {[type(x) for x in inputs[:3]]}"
            )

    # 3. PyTorch Tensor
    if isinstance(inputs, torch.Tensor):
        if inputs.numel() == 0:
            raise AttackExecutionError("Input tensor cannot be empty.")
        if not torch.is_floating_point(inputs):
            input_ids = inputs.to(dtype=torch.long, device=target_device)
        else:
            input_ids = inputs.to(dtype=torch.long, device=target_device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() != 2:
            raise AttackExecutionError(
                f"Expected 1D or 2D token IDs tensor, got shape {tuple(inputs.shape)}"
            )

        pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
        attention_mask = (input_ids != pad_id).long().to(target_device)
        # Ensure at least non-empty mask
        if attention_mask.sum() == 0:
            attention_mask = torch.ones_like(input_ids)

        original_texts = None
        if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
            try:
                original_texts = tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
            except Exception:
                pass

        return input_ids, attention_mask, "tensor", original_texts

    # 4. Dictionary containing 'input_ids' (including BatchEncoding/Mapping)
    if isinstance(inputs, (dict, Mapping)):
        if "input_ids" not in inputs:
            raise AttackExecutionError(
                "Inputs dictionary must contain 'input_ids' key."
            )
        raw_ids = inputs["input_ids"]
        if not isinstance(raw_ids, torch.Tensor):
            raw_ids = torch.tensor(raw_ids, dtype=torch.long)
        else:
            raw_ids = raw_ids.to(dtype=torch.long)

        if raw_ids.dim() == 1:
            raw_ids = raw_ids.unsqueeze(0)

        input_ids = raw_ids.to(target_device)

        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            mask = inputs["attention_mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.long)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            attention_mask = mask.to(dtype=torch.long, device=target_device)
        else:
            pad_id = (
                getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
            )
            attention_mask = (input_ids != pad_id).long().to(target_device)

        original_texts = None
        if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
            try:
                original_texts = tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
            except Exception:
                pass

        return input_ids, attention_mask, "dict", original_texts

    raise AttackExecutionError(
        f"Unsupported input type for text attack: {type(inputs)}. "
        "Expected str, List[str], torch.Tensor, or dict."
    )


def reconstruct_output(
    adv_input_ids: torch.Tensor,
    input_type: str,
    tokenizer: Optional[Any] = None,
    original_input: Optional[Any] = None,
) -> Any:
    """
    Reconstruct adversarial output data matching the original input type.

    Args:
        adv_input_ids: 2D LongTensor of perturbed token IDs.
        input_type: 'str', 'list_str', 'tensor', or 'dict'.
        tokenizer: Optional tokenizer for text decoding.
        original_input: Original input to preserve format (e.g. 1D tensor shape).

    Returns:
        Adversarial result in the appropriate format.
    """
    if input_type == "str":
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
        return adv_input_ids

    if input_type == "list_str":
        if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
            return tokenizer.batch_decode(adv_input_ids, skip_special_tokens=True)
        elif tokenizer is not None and hasattr(tokenizer, "decode"):
            return [
                tokenizer.decode(seq, skip_special_tokens=True) for seq in adv_input_ids
            ]
        return adv_input_ids

    if input_type == "tensor":
        if isinstance(original_input, torch.Tensor) and original_input.dim() == 1:
            return adv_input_ids.squeeze(0)
        return adv_input_ids

    if input_type == "dict":
        result = dict(original_input) if isinstance(original_input, (dict, Mapping)) else {}
        result["input_ids"] = adv_input_ids
        if "attention_mask" in result and hasattr(result["attention_mask"], "shape"):
            # Ensure attention mask matches new sequence length if changed
            pad_id = (
                getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
            )
            result["attention_mask"] = (adv_input_ids != pad_id).long()
        return result

    return adv_input_ids


def forward_pass_embeddings(
    model: Any,
    embedding_layer: nn.Embedding,
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass through the model using continuous input embeddings.
    First attempts HuggingFace standard 'inputs_embeds' keyword argument.
    If unsupported, registers a forward hook on embedding_layer to inject embeddings.

    Returns:
        Logits tensor of shape (batch_size, num_classes).
    """
    raw_model = get_raw_model(model)

    # 1. Try Hugging Face inputs_embeds keyword argument
    try:
        if attention_mask is not None:
            outputs = raw_model(inputs_embeds=embeddings, attention_mask=attention_mask)
        else:
            outputs = raw_model(inputs_embeds=embeddings)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits
    except (TypeError, ValueError):
        pass

    # 2. Hook injection into embedding_layer for custom PyTorch architectures
    hook_handle = embedding_layer.register_forward_hook(
        lambda module, in_args, out: embeddings
    )
    try:
        if attention_mask is not None:
            try:
                outputs = raw_model(input_ids, attention_mask=attention_mask)
            except TypeError:
                outputs = raw_model(input_ids)
        else:
            outputs = raw_model(input_ids)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits
    finally:
        hook_handle.remove()


def forward_pass_tokens(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass through the model using discrete token IDs.

    Returns:
        Logits tensor of shape (batch_size, num_classes).
    """
    raw_model = get_raw_model(model)
    if attention_mask is not None:
        try:
            outputs = raw_model(input_ids, attention_mask=attention_mask)
        except TypeError:
            outputs = raw_model(input_ids)
    else:
        outputs = raw_model(input_ids)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits


def compute_embedding_gradients(
    model: Any,
    embedding_layer: nn.Embedding,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: torch.Tensor,
    loss_fn: Optional[Any] = None,
    targeted: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute gradient of classification loss w.r.t input embeddings.

    For untargeted:
        Loss = CE(logits, labels).
        Gradient points in direction that increases loss (away from true class).
    For targeted:
        Loss = CE(logits, labels_target).
        Gradient points in direction that decreases loss toward target class.

    Returns:
        Tuple of (gradients, logits, loss_value).
    """
    raw_model = get_raw_model(model)
    was_training = raw_model.training
    raw_model.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    embeddings = embedding_layer(input_ids).clone().detach().requires_grad_(True)
    logits = forward_pass_embeddings(
        raw_model, embedding_layer, input_ids, embeddings, attention_mask
    )

    if targeted:
        # We want to MINIMIZE loss toward target class, so loss is cross entropy with target
        loss = loss_fn(logits, labels)
    else:
        # We want to MAXIMIZE loss away from true class
        loss = loss_fn(logits, labels)

    grad = torch.autograd.grad(
        loss,
        embeddings,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]

    raw_model.train(was_training)

    if grad is None:
        raise AttackExecutionError(
            "Gradients w.r.t token embeddings could not be computed."
        )

    return grad, logits.detach(), float(loss.item())


def compute_token_saliency(
    embedding_gradients: torch.Tensor,
    embeddings: Optional[torch.Tensor] = None,
    method: str = "norm",
) -> torch.Tensor:
    """
    Compute scalar saliency score per token position.

    Args:
        embedding_gradients: Tensor of shape (batch_size, seq_len, embed_dim)
        embeddings: Optional embedding tensor of shape (batch_size, seq_len, embed_dim)
        method: 'norm' (L2 norm) or 'dot' (absolute dot product |g * e|)

    Returns:
        Saliency scores tensor of shape (batch_size, seq_len).
    """
    if method == "dot" and embeddings is not None:
        return (embedding_gradients * embeddings).sum(dim=-1).abs()
    return torch.norm(embedding_gradients, p=2, dim=-1)


def validate_labels(
    labels: Any,
    batch_size: int,
    device: torch.device,
    targeted: bool = False,
    target_label: Optional[Union[int, List[int], torch.Tensor]] = None,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Validate and format labels tensor.

    Args:
        labels: True labels (or target labels if targeted=True).
        batch_size: Expected batch size.
        device: Target torch device.
        targeted: Whether the attack is targeted.
        target_label: Explicit target label(s) if targeted.
        num_classes: Optional number of classes for bounds check.

    Returns:
        1D LongTensor of shape (batch_size,).

    Raises:
        AttackExecutionError: If labels are missing or invalid.
        AttackConfigurationError: If targeted mode is requested without valid target labels.
    """
    if targeted:
        target = target_label if target_label is not None else labels
        if target is None:
            raise AttackConfigurationError(
                "Targeted attack requires 'target_label' in config.params or target labels as labels argument."
            )
        effective_labels = target
    else:
        if labels is None:
            raise AttackExecutionError("Labels must be provided for attack generation.")
        effective_labels = labels

    if isinstance(effective_labels, (int, np.integer)):
        labels_tensor = torch.full(
            (batch_size,), int(effective_labels), dtype=torch.long, device=device
        )
    elif isinstance(effective_labels, (list, tuple)):
        if len(effective_labels) == 1 and batch_size > 1:
            labels_tensor = torch.full(
                (batch_size,), int(effective_labels[0]), dtype=torch.long, device=device
            )
        elif len(effective_labels) != batch_size:
            raise AttackExecutionError(
                f"Labels length ({len(effective_labels)}) does not match batch size ({batch_size})."
            )
        else:
            labels_tensor = torch.tensor(
                [int(x) for x in effective_labels], dtype=torch.long, device=device
            )
    elif isinstance(effective_labels, torch.Tensor):
        labels_tensor = effective_labels.to(dtype=torch.long, device=device)
        if labels_tensor.dim() == 0:
            labels_tensor = labels_tensor.repeat(batch_size)
        elif labels_tensor.size(0) != batch_size:
            raise AttackExecutionError(
                f"Labels batch size ({labels_tensor.size(0)}) does not match input batch size ({batch_size})."
            )
    else:
        raise AttackExecutionError(
            f"Unsupported labels type: {type(effective_labels)}. Expected int, list, or torch.Tensor."
        )

    if num_classes is not None:
        if (labels_tensor < 0).any() or (labels_tensor >= num_classes).any():
            raise AttackConfigurationError(
                f"Label values must be within [0, {num_classes - 1}], got {labels_tensor.tolist()}"
            )

    return labels_tensor


def get_candidate_tokens_by_similarity(
    token_embedding: torch.Tensor,
    embedding_weight: torch.Tensor,
    top_k: int = 50,
    special_ids: Optional[Set[int]] = None,
    original_token_id: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> List[int]:
    """
    Find top-k tokens in vocabulary closest in cosine similarity to token_embedding,
    excluding special tokens and the original token.

    Args:
        token_embedding: 1D tensor of shape (embed_dim,)
        embedding_weight: Vocabulary embedding matrix (vocab_size, embed_dim)
        top_k: Number of candidates to return.
        special_ids: Set of token IDs to exclude.
        original_token_id: The original token ID to exclude.
        min_similarity: Minimum cosine similarity threshold (e.g. 0.4).

    Returns:
        List of candidate token IDs.
    """
    special = special_ids or set()

    # Normalize vectors for cosine similarity
    token_norm = F.normalize(token_embedding.unsqueeze(0), dim=-1)
    vocab_norm = F.normalize(embedding_weight.detach(), dim=-1)

    sims = torch.matmul(vocab_norm, token_norm.t()).squeeze(-1)  # (vocab_size,)

    # Mask out special tokens and original token
    exclude = set(special)
    if original_token_id is not None:
        exclude.add(int(original_token_id))

    for eid in exclude:
        if 0 <= eid < sims.size(0):
            sims[eid] = float("-inf")

    k = min(top_k * 2, sims.size(0))
    top_vals, top_indices = torch.topk(sims, k=k)

    candidates: List[int] = []
    for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
        if val == float("-inf"):
            continue
        if min_similarity is not None and val < min_similarity:
            continue
        candidates.append(int(idx))
        if len(candidates) >= top_k:
            break

    return candidates
