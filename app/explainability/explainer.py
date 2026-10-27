"""
Main XAI Coordinator (XAIExplainer) for AdverScan explainability module.
Coordinates multi-modal inputs, technique selection, explanations, and comparison.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from app.attack_engine.models import AttackResult
from app.explainability.comparison import compare_explanations
from app.explainability.explanation_result import ExplanationResult
from app.explainability.failure_analysis import analyze_failure
from app.explainability.input_utils import (
    extract_prediction_and_confidence,
    get_shape,
    predict_proba_np,
    safe_predict,
    to_numpy,
    to_tensor,
    validate_single_sample,
)
from app.explainability.modalities import get_modality_handler
from app.explainability.modality import Modality, detect_modality
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.registry import TechniqueRegistry, technique_registry
from app.explainability.techniques.shap_explainer import SHAPExplainer, TreeSHAPExplainer


class XAIExplainer:
    """
    Main coordinator for XAI explainability across modalities in AdverScan.

    Coordinates input validation and preparation, modality detection,
    technique-specific attribution generation (SHAP, TreeSHAP, LIME, Saliency, Integrated Gradients, Grad-CAM),
    clean vs. adversarial comparison, and failure mode analysis.
    """

    def __init__(
        self,
        default_technique: str = "shap",
        registry: Optional[TechniqueRegistry] = None,
    ):
        """
        Initialize XAIExplainer.

        Args:
            default_technique: Default XAI technique identifier ("shap", "treeshap", "lime", "saliency", etc.).
            registry: Optional custom TechniqueRegistry instance.
        """
        self.default_technique = default_technique.lower()
        self.registry = registry or technique_registry

        # Backward compatibility dictionary mapping
        self.techniques = {
            "shap": self.registry.get("shap"),
            "treeshap": self.registry.get("treeshap"),
            "lime": self.registry.get("lime"),
        }

    def _get_predictions_and_confidences(
        self,
        model: Any,
        inputs: Any,
        output_mode: str = "auto",
    ) -> Tuple[Any, float]:
        """
        Extract model predicted class label or scalar value and confidence score.

        Args:
            model: PyTorch module, PyTorchAdapter, or callable model.
            inputs: Input tensor, array, or strings.
            output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.

        Returns:
            Tuple of (predicted_class_or_value, confidence_score).
        """
        probs_or_scores = predict_proba_np(model, inputs, output_mode=output_mode)
        return extract_prediction_and_confidence(probs_or_scores, output_mode=output_mode)

    def explain(
        self,
        model: Any,
        clean_input: Any,
        adversarial_input: Any,
        true_label: Optional[Any] = None,
        technique: Optional[str] = None,
        attack_name: str = "adversarial_attack",
        assessment_result: Optional[Any] = None,
        modality: Optional[Union[str, Modality]] = None,
        target_class: Optional[Any] = None,
        output_mode: str = "auto",
        comparison_target: str = "clean_pred",
        compute_metrics: bool = False,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Generate complete XAI explanation for clean vs. adversarial inputs.
        Accepts exactly ONE sample. For batches, use explain_batch().

        Args:
            model: Target ML model or adapter.
            clean_input: Clean baseline input sample of shape (1, ...) or single instance.
            adversarial_input: Adversarial input sample of shape (1, ...) or single instance.
            true_label: Optional ground truth label for the sample.
            technique: XAI technique name ("shap", "treeshap", "lime", "saliency", "integrated_gradients", "gradcam").
            attack_name: Identifier of the attack executed.
            assessment_result: Optional AssessmentResult from M5 for context enrichment.
            modality: Explicit modality hint ("image", "text", "tabular", "timeseries").
            target_class: Optional explicit class index to explain.
            output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.
            comparison_target: Class used for clean/adv comparison ('clean_pred', 'adv_pred', 'ground_truth', 'both').
            compute_metrics: Whether to evaluate extra faithfulness/alignment metrics.

        Returns:
            Structured ExplanationResult DTO.
        """
        # 1. Modality detection & Handler selection
        mod = detect_modality(clean_input, hint=modality)
        handler = get_modality_handler(mod)

        # 2. Validate single sample (reject batches N > 1)
        validate_single_sample(clean_input, modality=mod.value)
        validate_single_sample(adversarial_input, modality=mod.value)

        # 3. Input preparation
        prep_clean, prep_adv = handler.prepare_batch(clean_input, adversarial_input)

        # 4. Select and validate technique
        tech_name = (technique or self.default_technique).lower()
        explainer = self.registry.select_technique(mod, model=model, preferred=tech_name, **kwargs)

        # 5. Get predictions and confidences
        clean_pred, clean_conf = self._get_predictions_and_confidences(model, prep_clean, output_mode=output_mode)
        adv_pred, adv_conf = self._get_predictions_and_confidences(model, prep_adv, output_mode=output_mode)

        # 6. Resolve target class for attribution comparison (apples-to-apples)
        is_classification = output_mode in ("classification", "auto")
        if is_classification:
            if target_class is not None:
                eval_target = target_class
            elif comparison_target == "adv_pred":
                eval_target = adv_pred
            elif comparison_target == "ground_truth" and true_label is not None:
                # Handle one-hot or scalar true_label
                if isinstance(true_label, (list, tuple, np.ndarray)) and len(true_label) > 1:
                    eval_target = int(np.argmax(true_label))
                else:
                    eval_target = int(true_label)
            else:  # default 'clean_pred'
                eval_target = clean_pred
        else:
            eval_target = None

        # 7. Run XAI technique with common target class
        clean_attr_result = explainer.explain(
            model,
            prep_clean,
            target_class=eval_target,
            modality=mod,
            output_mode=output_mode,
            **kwargs,
        )
        adv_attr_result = explainer.explain(
            model,
            prep_adv,
            target_class=eval_target,
            modality=mod,
            output_mode=output_mode,
            **kwargs,
        )

        # If comparison_target is 'both', also compute adversarial attribution w.r.t adv_pred
        adv_attr_adv_class_result = None
        if is_classification and comparison_target == "both" and clean_pred != adv_pred:
            adv_attr_adv_class_result = explainer.explain(
                model,
                prep_adv,
                target_class=adv_pred,
                modality=mod,
                output_mode=output_mode,
                **kwargs,
            )

        # 8. Postprocess attributions using modality handler
        clean_postprocessed = handler.postprocess_attribution(
            clean_attr_result.get("attribution"),
            clean_input,
            **kwargs,
        )
        adv_postprocessed = handler.postprocess_attribution(
            adv_attr_result.get("attribution"),
            adversarial_input,
            **kwargs,
        )

        clean_top_features = clean_postprocessed.get("top_features", [])
        adv_top_features = adv_postprocessed.get("top_features", [])

        clean_attr_result["postprocessed"] = clean_postprocessed
        clean_attr_result["top_features"] = clean_top_features
        adv_attr_result["postprocessed"] = adv_postprocessed
        adv_attr_result["top_features"] = adv_top_features

        attribution_data = {
            "technique": tech_name,
            "modality": mod.value,
            "comparison_target": comparison_target,
            "target_class_explained": eval_target,
            "clean": clean_attr_result,
            "adversarial": adv_attr_result,
        }
        if adv_attr_adv_class_result is not None:
            adv_attr_adv_class_post = handler.postprocess_attribution(
                adv_attr_adv_class_result.get("attribution"),
                adversarial_input,
                **kwargs,
            )
            adv_attr_adv_class_result["postprocessed"] = adv_attr_adv_class_post
            adv_attr_adv_class_result["top_features"] = adv_attr_adv_class_post.get("top_features", [])
            attribution_data["adversarial_adv_class"] = adv_attr_adv_class_result

        # 9. Perform clean vs. adversarial comparison
        clean_attr = clean_attr_result.get("attribution")
        adv_attr = adv_attr_result.get("attribution")

        comp_data = compare_explanations(
            clean_prediction=clean_pred,
            adversarial_prediction=adv_pred,
            clean_confidence=clean_conf,
            adversarial_confidence=adv_conf,
            clean_attribution=clean_attr,
            adv_attribution=adv_attr,
            clean_top_features=clean_top_features,
            adv_top_features=adv_top_features,
            target_class=target_class,
        )

        # Non-classification relative change handling
        if not is_classification:
            rel_change = abs(clean_pred - adv_pred) / (abs(clean_pred) + 1e-6)
            threshold = float(kwargs.get("change_threshold", 0.1))
            pred_changed = bool(rel_change > threshold)
            comp_data["prediction_changed"] = pred_changed
            comp_data["relative_change"] = float(rel_change)
            comp_data["threshold_crossed"] = pred_changed

        # 10. Perform failure analysis
        fail_data = analyze_failure(
            clean_prediction=clean_pred,
            adversarial_prediction=adv_pred,
            true_label=true_label,
            clean_attribution=clean_attr,
            adv_attribution=adv_attr,
            clean_top_features=clean_top_features,
            adv_top_features=adv_top_features,
            target_class=target_class,
        )

        # 11. Optional metrics (Perturbation Alignment)
        if compute_metrics:
            from app.explainability.metrics.perturbation_alignment import compute_perturbation_alignment
            comp_data["perturbation_alignment"] = compute_perturbation_alignment(
                clean_input,
                adversarial_input,
                clean_attr,
            )

        # 12. Metadata
        meta: Dict[str, Any] = {
            "attack_name": attack_name,
            "technique": tech_name,
            "modality": mod.value,
            "output_mode": output_mode,
            "comparison_target": comparison_target,
        }
        if assessment_result is not None:
            if hasattr(assessment_result, "to_dict"):
                meta["assessment_result"] = assessment_result.to_dict()
            elif isinstance(assessment_result, dict):
                meta["assessment_result"] = assessment_result

        feature_importance = {
            "clean_top_features": clean_top_features,
            "adversarial_top_features": adv_top_features,
        }

        # 13. Build standardized ExplanationResult DTO
        return ExplanationResult(
            attack_name=attack_name,
            technique=tech_name,
            true_label=true_label,
            clean_prediction=clean_pred,
            adversarial_prediction=adv_pred,
            clean_confidence=clean_conf,
            adversarial_confidence=adv_conf,
            prediction_changed=comp_data["prediction_changed"],
            attack_caused_failure=fail_data["attack_caused_failure"],
            attribution=attribution_data,
            comparison=comp_data,
            failure_analysis=fail_data,
            metadata=meta,
            modality=mod.value,
            feature_importance=feature_importance,
        )

    def explain_batch(
        self,
        model: Any,
        clean_inputs: Any,
        adversarial_inputs: Any,
        true_labels: Optional[Any] = None,
        technique: Optional[str] = None,
        attack_name: str = "adversarial_attack",
        modality: Optional[Union[str, Modality]] = None,
        **kwargs: Any,
    ) -> List[ExplanationResult]:
        """
        Generate explanations for a batch of samples by processing each sample individually.

        Args:
            model: Target ML model.
            clean_inputs: Batch of clean inputs (N, ...).
            adversarial_inputs: Batch of adversarial inputs (N, ...).
            true_labels: Optional sequence or tensor of ground truth labels of length N.
            technique: XAI technique name.
            attack_name: Identifier of the attack executed.
            modality: Explicit modality hint.
            **kwargs: Extra arguments passed to explain().

        Returns:
            List of ExplanationResult objects, one per sample.
        """
        mod = detect_modality(clean_inputs, hint=modality)

        # Determine batch size N
        if isinstance(clean_inputs, (list, tuple)) and all(isinstance(s, str) for s in clean_inputs):
            n_samples = len(clean_inputs)
            clean_list = clean_inputs
            adv_list = adversarial_inputs
        else:
            shape = get_shape(clean_inputs)
            n_samples = shape[0] if shape else 1
            clean_list = clean_inputs
            adv_list = adversarial_inputs

        results: List[ExplanationResult] = []
        for i in range(n_samples):
            # Slice sample i
            if isinstance(clean_list, (list, tuple)):
                sample_clean = clean_list[i]
                sample_adv = adv_list[i]
            elif isinstance(clean_list, torch.Tensor):
                sample_clean = clean_list[i : i + 1]
                sample_adv = adv_list[i : i + 1]
            elif isinstance(clean_list, np.ndarray):
                sample_clean = clean_list[i : i + 1]
                sample_adv = adv_list[i : i + 1]
            else:
                sample_clean = clean_list
                sample_adv = adv_list

            # Slice label i
            sample_label = None
            if true_labels is not None:
                if isinstance(true_labels, (list, tuple)):
                    sample_label = true_labels[i]
                elif isinstance(true_labels, (torch.Tensor, np.ndarray)):
                    sample_label = true_labels[i]
                    # Convert 1D one-hot vector to class index
                    if hasattr(sample_label, "ndim") and sample_label.ndim == 1 and len(sample_label) > 1:
                        sample_label = int(np.argmax(to_numpy(sample_label)))
                    elif hasattr(sample_label, "item"):
                        sample_label = sample_label.item()
                else:
                    sample_label = true_labels

            res = self.explain(
                model=model,
                clean_input=sample_clean,
                adversarial_input=sample_adv,
                true_label=sample_label,
                technique=technique,
                attack_name=attack_name,
                modality=mod,
                **kwargs,
            )
            results.append(res)

        return results

    def explain_attack_result(
        self,
        model: Any,
        attack_result: AttackResult,
        assessment_result: Optional[Any] = None,
        technique: Optional[str] = None,
        modality: Optional[Union[str, Modality]] = None,
        **kwargs: Any,
    ) -> Union[ExplanationResult, List[ExplanationResult]]:
        """
        Generate explanation directly from an M3 AttackResult contract.
        Automatically routes single samples to explain() and batches to explain_batch().

        Args:
            model: Target ML model.
            attack_result: M3 AttackResult instance.
            assessment_result: Optional M5 AssessmentResult instance.
            technique: XAI technique name.
            modality: Optional explicit modality hint.

        Returns:
            ExplanationResult (for single sample) or List[ExplanationResult] (for batches).
        """
        clean_input = attack_result.original_inputs
        adv_input = attack_result.adversarial_examples
        labels = attack_result.labels
        attack_name = (
            attack_result.metadata.attack_name
            if hasattr(attack_result, "metadata") and hasattr(attack_result.metadata, "attack_name")
            else "unknown_attack"
        )

        if clean_input is None:
            raise ValueError("AttackResult must contain 'original_inputs' for XAI explainability.")
        if adv_input is None:
            raise ValueError("AttackResult contains no 'adversarial_examples' for XAI explainability.")

        shape = get_shape(clean_input)
        is_batch = False
        if isinstance(clean_input, (list, tuple)) and len(clean_input) > 1 and all(isinstance(x, str) for x in clean_input):
            is_batch = True
        elif shape and shape[0] > 1:
            is_batch = True

        if is_batch:
            return self.explain_batch(
                model=model,
                clean_inputs=clean_input,
                adversarial_inputs=adv_input,
                true_labels=labels,
                technique=technique,
                attack_name=attack_name,
                modality=modality,
                assessment_result=assessment_result,
                **kwargs,
            )
        else:
            return self.explain(
                model=model,
                clean_input=clean_input,
                adversarial_input=adv_input,
                true_label=labels,
                technique=technique,
                attack_name=attack_name,
                assessment_result=assessment_result,
                modality=modality,
                **kwargs,
            )
