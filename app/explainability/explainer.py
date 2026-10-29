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
from app.explainability.utils import (
    extract_prediction_and_confidence,
    get_shape,
    predict_proba_np,
    safe_predict,
    to_numpy,
    to_tensor,
    validate_single_sample,
)
from app.explainability.domain_profiles import get_domain_profile, is_valid_adversarial
from app.explainability.modalities import get_modality_handler
from app.explainability.modality import Modality, detect_modality
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.registry import TechniqueRegistry, technique_registry
from app.explainability.techniques.shap_explainer import SHAPExplainer, TreeSHAPExplainer
from app.explainability.profiling import profile_explainer_call
from app.explainability.config import XAIConfig


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
        config: Optional[XAIConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize XAIExplainer.

        Args:
            default_technique: Default XAI technique identifier ("shap", "treeshap", "lime", "saliency", etc.).
            registry: Optional custom TechniqueRegistry instance.
            config: Optional XAIConfig defining evaluation mode, budgets, and parameters.
        """
        self.default_technique = default_technique.lower()
        self.registry = registry or technique_registry
        self.config = config or XAIConfig()

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
        domain: Optional[str] = None,
        config: Optional[XAIConfig] = None,
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
            assessment_result: Optional AssessmentResult from M3 for context enrichment.
            modality: Explicit modality hint ("image", "text", "tabular", "timeseries").
            target_class: Optional explicit class index to explain.
            output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.
            comparison_target: Class used for clean/adv comparison ('clean_pred', 'adv_pred', 'ground_truth', 'both').
            compute_metrics: Whether to evaluate extra faithfulness/alignment metrics.
            domain: Domain name preset ('fraud', 'ids', 'medical', 'maintenance') or custom.
            config: Optional XAIConfig overriding default explainer configuration.

        Returns:
            Structured ExplanationResult DTO.
        """
        # Resolve active configuration
        cfg = config or self.config

        # Merge config parameters into explainer kwargs
        explainer_kwargs = dict(kwargs)
        if "steps" not in explainer_kwargs and "ig_steps" not in explainer_kwargs:
            explainer_kwargs["steps"] = cfg.ig_steps
            explainer_kwargs["ig_steps"] = cfg.ig_steps
        if "num_samples" not in explainer_kwargs and "lime_num_samples" not in explainer_kwargs:
            explainer_kwargs["num_samples"] = cfg.lime_num_samples
            explainer_kwargs["lime_num_samples"] = cfg.lime_num_samples
        if "max_evals" not in explainer_kwargs and "shap_max_evals" not in explainer_kwargs:
            explainer_kwargs["max_evals"] = cfg.shap_max_evals
            explainer_kwargs["shap_max_evals"] = cfg.shap_max_evals
        if "batch_size" not in explainer_kwargs:
            explainer_kwargs["batch_size"] = cfg.batch_size
        if "max_text_length" not in explainer_kwargs:
            explainer_kwargs["max_text_length"] = cfg.max_text_length

        # 1. Modality detection & Handler selection
        mod = detect_modality(clean_input, hint=modality)
        handler = get_modality_handler(mod)

        # Domain profile resolution
        domain_name = domain or kwargs.get("domain", mod.value)
        try:
            domain_prof = get_domain_profile(domain_name)
        except Exception:
            domain_prof = None

        # 2. Validate single sample (reject batches N > 1)
        validate_single_sample(clean_input, modality=mod.value)
        validate_single_sample(adversarial_input, modality=mod.value)

        # 3. Input preparation
        prep_clean, prep_adv = handler.prepare_batch(clean_input, adversarial_input)

        # Domain validity check on adversarial sample
        validity_flags = {}
        if domain_prof is not None:
            _, validity_flags = is_valid_adversarial(prep_clean, prep_adv, domain_prof)

        # 4. Select and validate technique
        tech_name = (technique or (domain_prof.default_technique if domain_prof else self.default_technique)).lower()
        explainer = self.registry.select_technique(mod, model=model, preferred=tech_name, **explainer_kwargs)

        # 5. Get predictions, confidences, and probabilities using batched prediction
        clean_probs = predict_proba_np(
            model,
            prep_clean,
            output_mode=output_mode,
            batch_size=cfg.batch_size,
            max_text_length=cfg.max_text_length,
            tokenizer=kwargs.get("tokenizer"),
        )
        adv_probs = predict_proba_np(
            model,
            prep_adv,
            output_mode=output_mode,
            batch_size=cfg.batch_size,
            max_text_length=cfg.max_text_length,
            tokenizer=kwargs.get("tokenizer"),
        )
        clean_pred, clean_conf = extract_prediction_and_confidence(clean_probs, output_mode=output_mode)
        adv_pred, adv_conf = extract_prediction_and_confidence(adv_probs, output_mode=output_mode)

        # 6. Resolve target class for attribution comparison (apples-to-apples)
        is_classification = output_mode in ("classification", "auto")
        if is_classification:
            if target_class is not None:
                eval_target = target_class
            elif comparison_target == "adv_pred":
                eval_target = adv_pred
            elif comparison_target == "ground_truth" and true_label is not None:
                if isinstance(true_label, (list, tuple, np.ndarray)) and len(true_label) > 1:
                    eval_target = int(np.argmax(true_label))
                else:
                    eval_target = int(true_label)
            else:  # default 'clean_pred'
                eval_target = clean_pred
        else:
            eval_target = None

        # 7. Run XAI technique with common target class
        clean_attr_result, clean_stats = profile_explainer_call(
            explainer,
            model,
            prep_clean,
            target_class=eval_target,
            modality=mod,
            output_mode=output_mode,
            **explainer_kwargs,
        )
        adv_attr_result, adv_stats = profile_explainer_call(
            explainer,
            model,
            prep_adv,
            target_class=eval_target,
            modality=mod,
            output_mode=output_mode,
            **explainer_kwargs,
        )

        total_explain_seconds = clean_stats.explain_seconds + adv_stats.explain_seconds
        total_model_queries = clean_stats.num_model_queries + adv_stats.num_model_queries

        # Attribution status propagation
        attribution_status = clean_attr_result.get("status", "success")
        if clean_attr_result.get("status") in ("error", "unavailable"):
            attribution_status = clean_attr_result.get("status")
        elif adv_attr_result.get("status") in ("error", "unavailable"):
            attribution_status = adv_attr_result.get("status")

        # If comparison_target is 'both', also compute adversarial attribution w.r.t adv_pred
        adv_attr_adv_class_result = None
        if is_classification and comparison_target == "both" and clean_pred != adv_pred:
            adv_attr_adv_class_result, adv_class_stats = profile_explainer_call(
                explainer,
                model,
                prep_adv,
                target_class=adv_pred,
                modality=mod,
                output_mode=output_mode,
                **explainer_kwargs,
            )
            total_explain_seconds += adv_class_stats.explain_seconds
            total_model_queries += adv_class_stats.num_model_queries

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
            clean_probs=clean_probs,
            adv_probs=adv_probs,
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
            feature_names=domain_prof.feature_names if domain_prof else kwargs.get("feature_names"),
            sensor_names=domain_prof.sensor_names if domain_prof else kwargs.get("sensor_names"),
            modality=mod.value,
            top_k=kwargs.get("top_k", 5),
        )

        # 11. Optional metrics (Perturbation Alignment, Faithfulness, Stability)
        if compute_metrics:
            from app.explainability.metrics.faithfulness import compute_faithfulness
            from app.explainability.metrics.perturbation_alignment import compute_perturbation_alignment
            from app.explainability.metrics.stability import compute_stability

            comp_data["perturbation_alignment"] = compute_perturbation_alignment(
                clean_input,
                adversarial_input,
                clean_attr,
                top_k=kwargs.get("top_k", 5),
                background_std=kwargs.get("background_std"),
            )
            comp_data["faithfulness"] = compute_faithfulness(
                model=model,
                inputs=clean_input,
                attribution=clean_attr,
                top_k=kwargs.get("top_k", 5),
                background_inputs=kwargs.get("background_inputs"),
            )
            comp_data["stability"] = compute_stability(
                explainer=explainer,
                model=model,
                inputs=clean_input,
                target_class=eval_target if is_classification else 0,
                background_inputs=kwargs.get("background_inputs"),
            )

        # 12. Metadata
        meta: Dict[str, Any] = {
            "attack_name": attack_name,
            "technique": tech_name,
            "modality": mod.value,
            "output_mode": output_mode,
            "comparison_target": comparison_target,
            "domain": domain_prof.name if domain_prof else str(domain_name),
            "config": cfg.to_dict(),
            "explain_seconds": round(total_explain_seconds, 6),
            "num_model_queries": int(total_model_queries),
            "clean_explain_seconds": round(clean_stats.explain_seconds, 6),
            "clean_num_model_queries": int(clean_stats.num_model_queries),
            "adversarial_explain_seconds": round(adv_stats.explain_seconds, 6),
            "adversarial_num_model_queries": int(adv_stats.num_model_queries),
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
            domain=domain_prof.name if domain_prof else str(domain_name),
            feature_importance=feature_importance,
            feature_names=domain_prof.feature_names if domain_prof else kwargs.get("feature_names"),
            perturbation_metrics=comp_data.get("perturbation_alignment", {}),
            faithfulness=comp_data.get("faithfulness", {}),
            stability=comp_data.get("stability", {}),
            validity_flags=validity_flags,
            attribution_status=attribution_status,
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
        config: Optional[XAIConfig] = None,
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
            config: Optional XAIConfig for batch execution.
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
                config=config,
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
        Generate explanation directly from an M2 AttackResult contract.
        Automatically routes single samples to explain() and batches to explain_batch().

        Args:
            model: Target ML model.
            attack_result: M2 AttackResult instance.
            assessment_result: Optional M3 AssessmentResult instance.
            technique: XAI technique name.
            modality: Optional explicit modality hint.

        Returns:
            ExplanationResult (for single sample) or List[ExplanationResult] (for batches).
        """
        clean_input = attack_result.original_inputs
        adv_input = attack_result.adversarial_examples
        labels = attack_result.labels
        meta = getattr(attack_result, "metadata", None)
        raw_attack_name = getattr(meta, "attack_name", None) if meta is not None else None
        attack_name: str = raw_attack_name if raw_attack_name is not None else "unknown_attack"

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
