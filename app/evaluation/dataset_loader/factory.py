"""
Factory function to construct domain-specific dataset loaders for evaluation.
"""

from typing import Any, Optional

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader
from app.evaluation.dataset_loader.vision_loader import HFVisionDatasetLoader
from app.evaluation.dataset_loader.text_loader import HFTextDatasetLoader
from app.evaluation.dataset_loader.timeseries_loader import TimeSeriesDatasetLoader
from app.evaluation.dataset_loader.tabular_loader import TabularDatasetLoader
from app.evaluation.dataset_loader.generic_loader import GenericDatasetLoader


def resolve_dataset_file(
    identifier: Optional[str],
    candidate_extensions: Any,
    model_identifier: Optional[str] = None,
) -> Optional[str]:
    """
    Dynamically locate a dataset file on disk without hardcoding any model or dataset names.
    Searches for the identifier directly, then in data/ and datasets/ directories,
    and optionally checks the model's stem name as fallback.
    """
    from pathlib import Path
    import re

    if not identifier and not model_identifier:
        return None

    search_dirs = [Path("data"), Path("datasets"), Path(".")]
    candidates_to_try = []

    if identifier:
        p = Path(identifier)
        if p.is_file():
            return str(p)
        clean_id = p.stem.lower().replace("-", "_")
        candidates_to_try.extend([
            clean_id,
            f"{clean_id}_test",
            f"{clean_id}_data",
            clean_id.replace("_", "-"),
            f"{clean_id.replace('_', '-')}_test",
        ])

    if model_identifier:
        m_p = Path(model_identifier)
        clean_m = m_p.stem.lower().replace("-", "_")
        stripped_suffix = re.sub(r"([_-]+(?:classifier|model|mlp|net|pt|pth))+$", "", clean_m)
        stripped_prefix = re.sub(r"^((?:tabular|time_series|time|series|ts|model|mlp)[_-]+)+", "", clean_m)
        stripped_both = re.sub(r"^((?:tabular|time_series|time|series|ts|model|mlp)[_-]+)+", "", stripped_suffix)
        tokens = [t for t in clean_m.split("_") if t not in ("model", "classifier", "net", "mlp")]
        model_candidates = {clean_m, stripped_suffix, stripped_prefix, stripped_both, "_".join(tokens)}

        for cand in list(model_candidates):
            if cand:
                candidates_to_try.extend([
                    cand,
                    f"{cand}_test",
                    f"{cand}_data",
                    cand.replace("_", "-"),
                    f"{cand.replace('_', '-')}-test",
                ])

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for ext in candidate_extensions:
            for name in candidates_to_try:
                candidate = s_dir / f"{name}{ext}"
                if candidate.is_file():
                    return str(candidate)

    return None


def get_dataset_loader(
    dataset_name: str,
    data_domain: str = "image",
    processor_name: Optional[str] = None,
    split: str = "test",
    batch_size: int = 32,
    **kwargs: Any,
) -> BaseDatasetLoader:
    """
    Factory function to construct domain-specific dataset loaders for evaluation.
    Supports data_domain: 'image'/'vision', 'text'/'nlp', 'time_series'/'timeseries'/'sequence',
    'tabular'/'csv'/'table', 'generic'/'custom'.
    """
    domain_clean = str(data_domain).strip().lower().replace("-", "_")

    if domain_clean in ("image", "vision"):
        return HFVisionDatasetLoader(
            dataset_name=dataset_name,
            processor_name=processor_name,
            split=split,
            batch_size=batch_size,
            **kwargs,
        )
    elif domain_clean in ("text", "nlp"):
        return HFTextDatasetLoader(
            dataset_name=dataset_name,
            tokenizer_name=processor_name,
            split=split,
            batch_size=batch_size,
            **kwargs,
        )
    elif domain_clean in ("time_series", "timeseries", "sequence"):
        inputs = kwargs.get("inputs")
        targets = kwargs.get("targets")

        if inputs is None or targets is None:
            # Dynamically resolve file from data arg or dataset_name without hardcoded names
            candidate_id = kwargs.get("data") or dataset_name
            model_id = kwargs.get("model_name") or kwargs.get("model_id")
            resolved_file = resolve_dataset_file(
                candidate_id, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id
            )
            if resolved_file:
                import torch
                loaded = torch.load(resolved_file, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    inputs = loaded.get("inputs", loaded.get("X"))
                    targets = loaded.get("targets", loaded.get("y"))
                elif isinstance(loaded, tuple) and len(loaded) == 2:
                    inputs, targets = loaded

        if inputs is None or targets is None:
            raise ValueError(
                "TimeSeriesDatasetLoader requires both 'inputs' and 'targets' keyword arguments."
            )
        return TimeSeriesDatasetLoader(
            inputs=inputs,
            targets=targets,
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    elif domain_clean in ("tabular", "csv", "table"):
        data_source = kwargs.get("data")
        if data_source is None:
            # Dynamically resolve dataset file without hardcoded names
            model_id = kwargs.get("model_name") or kwargs.get("model_id")
            data_source = resolve_dataset_file(
                dataset_name, [".csv", ".tsv"], model_identifier=model_id
            )

        if data_source is None:
            raise ValueError(
                "TabularDatasetLoader requires 'data' keyword argument (file path or (X, y) tuple)."
            )
        return TabularDatasetLoader(
            data=data_source,
            target_col=kwargs.get("target_col", "target"),
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    elif domain_clean in ("generic", "custom"):
        if "inputs" not in kwargs or "targets" not in kwargs:
            raise ValueError(
                "GenericDatasetLoader requires both 'inputs' and 'targets' keyword arguments."
            )
        return GenericDatasetLoader(
            inputs=kwargs["inputs"],
            targets=kwargs["targets"],
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unsupported data domain: '{data_domain}'. "
            f"Expected one of: 'image', 'vision', 'text', 'nlp', 'time_series', 'timeseries', 'sequence', 'tabular', 'csv', 'table', 'generic', 'custom'."
        )
