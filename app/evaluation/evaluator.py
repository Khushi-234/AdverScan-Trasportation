"""
Baseline evaluator engine for executing clean baseline evaluations in AdverScan.
"""

from ast import List
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.evaluation.dataset_loader import (
    BaseDatasetLoader,
    GTSRBDatasetLoader,
    HFVisionDatasetLoader,
    get_dataset_loader,
)
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.results import EvaluationResult


class BaselineEvaluator:
    """
    Generalized evaluator engine that consumes standardized M1 model adapters,
    runs clean baseline evaluation over dataset splits, and records metrics.
    Works dynamically for any model domain and class count.
    """

    def __init__(
        self,
        adapter: BaseModelAdapter,
        dataset_loader: BaseDatasetLoader,
        num_classes: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize BaselineEvaluator.

        Args:
            adapter: Standardized model adapter from Module 1 (BaseModelAdapter).
            dataset_loader: Dataset loader instance (BaseDatasetLoader).
            num_classes: Optional target classes count. If None, auto-inferred during evaluation.
            model_name: Optional custom model identifier name.
        """
        if not isinstance(adapter, BaseModelAdapter):
            raise TypeError(f"Expected adapter instance of BaseModelAdapter, got {type(adapter)}")

        self.adapter = adapter
        self.dataset_loader = dataset_loader
        self.num_classes = num_classes
        self.model_name = model_name or getattr(adapter, "model_name", "Target_Model")

    def evaluate(
        self,
        output_dir: Optional[Union[str, Path]] = "results/baseline",
        log_mlflow: bool = False,
    ) -> EvaluationResult:
        """
        Run baseline evaluation over full dataset split.

        Args:
            output_dir: Optional directory to save evaluation result JSON artifact.
            log_mlflow: Whether to log metrics to MLflow.

        Returns:
            EvaluationResult dataclass containing full metrics.
        """
        self.adapter.eval()
        
        all_targets: list[int] = []
        all_preds: list[int] = []
        all_probs_list: list[np.ndarray] = []

        # Batch inference loop
        for batch_pixels, batch_targets, _ in self.dataset_loader.iterate_batches():
            # Perform inference using M1 adapter
            outputs = self.adapter.predict(batch_pixels)
            
            # Support HuggingFace model outputs (ImageClassifierOutput)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            # Convert to PyTorch Tensor if outputs are NumPy array
            if isinstance(outputs, np.ndarray):
                logits_tensor = torch.from_numpy(outputs)
            else:
                logits_tensor = outputs

            # Dynamically infer num_classes from output dimension if not explicitly provided
            if self.num_classes is None:
                self.num_classes = int(logits_tensor.shape[-1])

            # Slice logits to active target classes if output contains unused buffer classes
            if logits_tensor.shape[-1] > self.num_classes:
                logits_tensor = logits_tensor[:, : self.num_classes]

            # Compute probabilities & predictions
            probs_tensor = torch.softmax(logits_tensor, dim=-1)
            preds_tensor = torch.argmax(probs_tensor, dim=-1)

            all_targets.extend(batch_targets.cpu().numpy().tolist())
            all_preds.extend(preds_tensor.cpu().numpy().tolist())
            all_probs_list.append(probs_tensor.cpu().numpy())

        y_true = np.array(all_targets, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_probs = np.concatenate(all_probs_list, axis=0)

        # Fallback dynamic calculation of num_classes if dataset labels exceed num_classes
        if self.num_classes is None or (len(y_true) > 0 and int(np.max(y_true)) >= self.num_classes):
            self.num_classes = max(int(np.max(y_true)) + 1 if len(y_true) > 0 else 1, y_probs.shape[-1])

        # Compute classification, confidence, entropy, and confusion matrix metrics
        metrics = MetricsCalculator.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            num_classes=self.num_classes,
        )

        device_str = str(getattr(self.adapter, "device", "cpu"))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract model metadata: parameter count, architecture, and class names
        total_params: Optional[int] = None
        trainable_params: Optional[int] = None
        model_arch: Optional[str] = None
        class_names: Optional[List[str]] = None

        try:
            raw_model = self.adapter.get_model() if hasattr(self.adapter, "get_model") else None
            if raw_model is not None:
                model_arch = raw_model.__class__.__name__
                if hasattr(raw_model, "parameters"):
                    params_list = [p for p in raw_model.parameters() if hasattr(p, "numel")]
                    if params_list:
                        total_params = int(sum(p.numel() for p in params_list))
                        trainable_params = int(sum(p.numel() for p in params_list if getattr(p, "requires_grad", False)))

                if hasattr(raw_model, "config") and hasattr(raw_model.config, "id2label"):
                    id2label = getattr(raw_model.config, "id2label", None)
                    if isinstance(id2label, dict) and id2label:
                        try:
                            sorted_keys = sorted(id2label.keys(), key=lambda k: int(k))
                            class_names = [str(id2label[k]) for k in sorted_keys]
                        except Exception:
                            class_names = [str(v) for v in id2label.values()]
        except Exception:
            pass

        if not class_names and hasattr(self.dataset_loader, "_dataset"):
            ds = getattr(self.dataset_loader, "_dataset", None)
            if hasattr(ds, "features") and ds.features:
                for col in ["label", "labels", "ClassId", "target", "class_id", "category", "fine_label"]:
                    if col in ds.features and hasattr(ds.features[col], "names"):
                        class_names = [str(n) for n in ds.features[col].names]
                        break

        result = EvaluationResult(
            dataset_name=self.dataset_loader.dataset_name,
            model_name=self.model_name,
            num_samples=len(y_true),
            num_classes=self.num_classes,
            accuracy=metrics["accuracy"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            f1_macro=metrics["f1_macro"],
            precision_weighted=metrics["precision_weighted"],
            recall_weighted=metrics["recall_weighted"],
            f1_weighted=metrics["f1_weighted"],
            average_confidence=metrics["average_confidence"],
            average_entropy=metrics["average_entropy"],
            per_class_metrics=metrics["per_class_metrics"],
            confusion_matrix=metrics["confusion_matrix"],
            batch_size=getattr(self.dataset_loader, "batch_size", 32),
            device=device_str,
            timestamp=timestamp,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_architecture=model_arch,
            class_names=class_names,
        )

        # Persist results JSON if output_dir is provided
        if output_dir is not None:
            out_path = Path(output_dir) / f"baseline_{self.model_name.replace('/', '_').lower()}.json"
            result.save_json(out_path)

        if log_mlflow:
            result.log_to_mlflow()

        return result


def evaluate_baseline(
    adapter: BaseModelAdapter,
    dataset_name: Optional[str] = None,
    processor_name: Optional[str] = None,
    data_domain: str = "image",
    split: str = "test",
    batch_size: int = 32,
    num_classes: Optional[int] = None,
    model_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = "results/baseline",
    log_mlflow: bool = False,
    **kwargs: Any,
) -> EvaluationResult:
    """
    Generalized convenience function for performing baseline evaluation on any model and dataset.

    Args:
        adapter: Module 1 model adapter.
        dataset_name: Optional dataset identifier (default 'bazyl/GTSRB').
        processor_name: Optional processor model identifier.
        data_domain: Target data domain ('image', 'text', 'time_series', 'tabular', etc.).
        split: Dataset split ('test', 'train', 'validation').
        batch_size: Evaluation batch size.
        num_classes: Optional number of target classes (auto-inferred if None).
        model_name: Optional model identifier.
        output_dir: Output directory path to save JSON results.
        log_mlflow: Whether to log metrics to MLflow.
        **kwargs: Additional keyword arguments forwarded to get_dataset_loader.

    Returns:
        EvaluationResult object.
    """
    resolved_dataset = dataset_name or "bazyl/GTSRB"
    resolved_model_name = model_name or getattr(adapter, "model_name", "Target_Model")

    loader = get_dataset_loader(
        dataset_name=resolved_dataset,
        data_domain=data_domain,
        processor_name=processor_name,
        split=split,
        batch_size=batch_size,
        **kwargs,
    )
    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,
        num_classes=num_classes,
        model_name=resolved_model_name,
    )
    return evaluator.evaluate(output_dir=output_dir, log_mlflow=log_mlflow)

