"""
Baseline evaluator engine for executing clean baseline evaluations in AdverScan.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.evaluation.dataset_loader import BaseDatasetLoader, GTSRBDatasetLoader
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.results import EvaluationResult


class BaselineEvaluator:
    """
    Evaluator engine that consumes standardized M1 model adapters,
    runs clean baseline evaluation over dataset splits, and records metrics.
    """

    def __init__(
        self,
        adapter: BaseModelAdapter,
        dataset_loader: BaseDatasetLoader,
        num_classes: int = 43,
        model_name: str = "GTSRB_Model",
    ):
        """
        Initialize BaselineEvaluator.

        Args:
            adapter: Standardized model adapter from Module 1 (BaseModelAdapter).
            dataset_loader: Dataset loader instance (BaseDatasetLoader).
            num_classes: Active target classes count (default 43 for GTSRB).
            model_name: Model identifier name.
        """
        if not isinstance(adapter, BaseModelAdapter):
            raise TypeError(f"Expected adapter instance of BaseModelAdapter, got {type(adapter)}")

        self.adapter = adapter
        self.dataset_loader = dataset_loader
        self.num_classes = num_classes
        self.model_name = model_name

    def evaluate(
        self,
        output_dir: Optional[Union[str, Path]] = "results/baseline",
        log_mlflow: bool = False,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Run baseline evaluation over full dataset split.

        Args:
            output_dir: Optional directory to save evaluation result JSON artifact.
            log_mlflow: Whether to log metrics to MLflow.
            show_progress: Whether to display a progress bar for batch evaluation.

        Returns:
            EvaluationResult dataclass containing full metrics.
        """
        self.adapter.eval()
        
        all_targets: list[int] = []
        all_preds: list[int] = []
        all_probs_list: list[np.ndarray] = []

        # Determine total batches for progress display
        total_batches = None
        if hasattr(self.dataset_loader, "__len__"):
            try:
                total_batches = len(self.dataset_loader)
            except Exception:
                total_batches = None

        batch_iter = self.dataset_loader.iterate_batches()
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iter = tqdm(
                    batch_iter,
                    total=total_batches,
                    desc=f"  Evaluating {self.dataset_loader.dataset_name}",
                    unit="batch",
                    leave=False,
                )
            except ImportError:
                pass

        # Batch inference loop with torch.no_grad() to prevent VRAM memory accumulation
        with torch.no_grad():
            for batch_pixels, batch_targets, _ in batch_iter:
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

                # Slice logits to active target classes (e.g. 44 -> 43 for ViT GTSRB)
                if logits_tensor.shape[-1] > self.num_classes:
                    logits_tensor = logits_tensor[:, : self.num_classes]

                # Compute probabilities & predictions
                probs_tensor = torch.softmax(logits_tensor, dim=-1)
                preds_tensor = torch.argmax(probs_tensor, dim=-1)

                all_targets.extend(batch_targets.cpu().numpy().tolist())
                all_preds.extend(preds_tensor.cpu().numpy().tolist())
                all_probs_list.append(probs_tensor.cpu().numpy())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        y_true = np.array(all_targets, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_probs = np.concatenate(all_probs_list, axis=0)

        # Compute classification, confidence, entropy, and confusion matrix metrics
        metrics = MetricsCalculator.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            num_classes=self.num_classes,
        )

        device_str = str(getattr(self.adapter, "device", "cpu"))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    dataset_name: str = "bazyl/GTSRB",
    processor_name: str = "bazyl/gtsrb-model",
    split: str = "test",
    batch_size: int = 32,
    num_classes: int = 43,
    model_name: str = "GTSRB_Model",
    output_dir: Optional[Union[str, Path]] = "results/baseline",
    log_mlflow: bool = False,
) -> EvaluationResult:
    """
    Convenience function for performing clean baseline evaluation.

    Args:
        adapter: Module 1 model adapter.
        dataset_name: Hugging Face dataset identifier.
        processor_name: Hugging Face processor model identifier.
        split: Dataset split ('test' or 'train').
        batch_size: Evaluation batch size.
        num_classes: Number of target classes.
        model_name: Model identifier.
        output_dir: Output directory path to save JSON results.
        log_mlflow: Whether to log metrics to MLflow.

    Returns:
        EvaluationResult object.
    """
    loader = GTSRBDatasetLoader(
        dataset_name=dataset_name,
        processor_name=processor_name,
        split=split,
        batch_size=batch_size,
    )
    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,
        num_classes=num_classes,
        model_name=model_name,
    )
    return evaluator.evaluate(output_dir=output_dir, log_mlflow=log_mlflow)
