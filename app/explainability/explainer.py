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
from app.explainability.transformation_evidence import extract_attack_transformation
from app.explainability.input_utils import (
    extract_prediction_and_confidence,
    get_shape,
    predict_proba_np,
    safe_predict,
    to_numpy,
    to_tensor,
    unwrap_model,
    validate_single_sample,
)
from app.explainability.domain_profiles import get_domain_profile, is_valid_adversarial
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
        domain: Optional[str] = None,
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

        Returns:
            Structured ExplanationResult DTO.
        """
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

        # Resolve tokenizer if available
        tok = kwargs.get("tokenizer")
        if tok is None:
            if hasattr(model, "tokenizer") and model.tokenizer is not None:
                tok = model.tokenizer
            elif hasattr(model, "processor") and model.processor is not None:
                tok = getattr(model.processor, "tokenizer", model.processor)
                if hasattr(tok, "get_processor"):
                    tok = tok.get_processor()
            elif hasattr(unwrap_model(model), "tokenizer"):
                tok = unwrap_model(model).tokenizer

        # 3. Input preparation
        if mod == Modality.TEXT:
            prep_clean, prep_adv = handler.prepare_batch(clean_input, adversarial_input, tokenizer=tok, **kwargs)
        else:
            prep_clean, prep_adv = handler.prepare_batch(clean_input, adversarial_input)

        # Domain validity check on adversarial sample
        validity_flags = {}
        if domain_prof is not None:
            _, validity_flags = is_valid_adversarial(prep_clean, prep_adv, domain_prof)

        # 4. Select and validate technique
        tech_name = (technique or (domain_prof.default_technique if domain_prof else self.default_technique)).lower()
        explainer = self.registry.select_technique(mod, model=model, preferred=tech_name, **kwargs)

        # 5. Get predictions, confidences, and probabilities
        clean_probs = predict_proba_np(model, prep_clean, output_mode=output_mode)
        adv_probs = predict_proba_np(model, prep_adv, output_mode=output_mode)
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

        call_kwargs = dict(kwargs)
        if mod == Modality.TEXT and tok is not None:
            call_kwargs["tokenizer"] = tok

        # 7. Run XAI technique with common target class
        clean_attr_result = explainer.explain(
            model,
            prep_clean,
            target_class=eval_target,
            modality=mod,
            output_mode=output_mode,
            **call_kwargs,
        )
        adv_attr_result = explainer.explain(
            model,
            prep_adv,
            target_class=eval_target,
            modality=mod,
            output_mode=output_mode,
            **call_kwargs,
        )

        # Attribution status propagation
        attribution_status = clean_attr_result.get("status", "success")
        if clean_attr_result.get("status") in ("error", "unavailable"):
            attribution_status = clean_attr_result.get("status")
        elif adv_attr_result.get("status") in ("error", "unavailable"):
            attribution_status = adv_attr_result.get("status")

        # If comparison_target is 'both', also compute adversarial attribution w.r.t adv_pred
        adv_attr_adv_class_result = None
        if is_classification and comparison_target == "both" and clean_pred != adv_pred:
            adv_attr_adv_class_result = explainer.explain(
                model,
                prep_adv,
                target_class=adv_pred,
                modality=mod,
                output_mode=output_mode,
                **call_kwargs,
            )

        # 8. Postprocess attributions using modality handler
        target_clean_inp = prep_clean if mod == Modality.TEXT else clean_input
        target_adv_inp = prep_adv if mod == Modality.TEXT else adversarial_input

        clean_postprocessed = handler.postprocess_attribution(
            clean_attr_result.get("attribution"),
            target_clean_inp,
            **kwargs,
        )
        adv_postprocessed = handler.postprocess_attribution(
            adv_attr_result.get("attribution"),
            target_adv_inp,
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

        # 12. Empirical attack transformation evidence (cleanly decoupled from attributions)
        attack_meta = kwargs.get("attack_metadata") or (getattr(assessment_result, "attack_metadata", None) if assessment_result is not None else None)
        transformation_data = extract_attack_transformation(
            clean_input=prep_clean if mod == Modality.TEXT else clean_input,
            adversarial_input=prep_adv if mod == Modality.TEXT else adversarial_input,
            modality=mod.value,
            metadata=attack_meta,
            feature_names=domain_prof.feature_names if domain_prof else kwargs.get("feature_names"),
            tokenizer=tok,
        )

        # 13. Attribution validation & alignment evaluation
        clean_val = clean_postprocessed.get("attribution_validation") or {}
        adv_val = adv_postprocessed.get("attribution_validation") or {}
        comp_val = comp_data.get("attribution_validity") or {}

        overall_val_status = "valid"
        if clean_val.get("status") in ("empty", "non_finite", "near_zero"):
            overall_val_status = clean_val.get("status")
        elif adv_val.get("status") in ("empty", "non_finite", "near_zero"):
            overall_val_status = adv_val.get("status")
        elif comp_val.get("status") in ("non_finite", "empty", "near_zero", "shape_mismatch"):
            overall_val_status = comp_val.get("status")
        elif attribution_status in ("error", "unavailable"):
            overall_val_status = attribution_status

        attr_validation = {
            "status": overall_val_status,
            "is_finite": bool(clean_val.get("is_finite", True) and adv_val.get("is_finite", True)),
            "is_near_zero": bool(clean_val.get("is_near_zero", False) or adv_val.get("is_near_zero", False) or comp_val.get("clean_is_near_zero", False)),
            "clean_validation": clean_val,
            "adversarial_validation": adv_val,
            "token_alignment_status": clean_val.get("token_alignment_status", "not_applicable"),
            "saliency_detected": bool(clean_val.get("saliency_detected", True) or adv_val.get("saliency_detected", True)),
        }

        # 14. Structured Prediction Analysis
        pred_analysis = dict(comp_data.get("prediction_analysis") or {})
        pred_analysis.update({
            "clean_correct": fail_data.get("clean_correct"),
            "adversarial_correct": fail_data.get("adversarial_correct"),
            "attack_caused_failure": fail_data.get("attack_caused_failure"),
            "attack_objective": fail_data.get("attack_objective", "untargeted"),
        })

        # 15. Rich Explanation Metadata & Limitations
        unavailable_metrics = []
        if true_label is None:
            unavailable_metrics.append("ground_truth_accuracy")
            unavailable_metrics.append("verified_attack_causation")
        if clean_probs is None or adv_probs is None:
            unavailable_metrics.append("per_class_probabilities")
        if comp_data.get("attribution_l1") is None:
            unavailable_metrics.append("attribution_distance_metrics")

        explanation_limitations = [
            "Local surrogate approximation: Attribution weights explain local model sensitivity, not global network semantics.",
            "Non-causal interpretation: Attribution scores indicate feature importance/correlation under perturbation, not established causal necessity.",
            "Baseline dependence: Saliency scores are computed relative to the specified baseline/background reference.",
        ]

        baseline_strat = kwargs.get("background_strategy", kwargs.get("baseline", "default_modality_baseline"))
        masking_cfg = kwargs.get("masker", kwargs.get("masking", "default_partition_masker"))

        meta: Dict[str, Any] = {
            "attack_name": attack_name,
            "technique": tech_name,
            "modality": mod.value,
            "output_mode": output_mode,
            "comparison_target": comparison_target,
            "domain": domain_prof.name if domain_prof else str(domain_name),
            "target_class_explained": eval_target,
            "baseline_strategy": str(baseline_strat),
            "masking_configuration": str(masking_cfg),
            "unavailable_metrics": unavailable_metrics,
            "explanation_limitations": explanation_limitations,
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

        # 16. Build standardized ExplanationResult DTO
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
            attack_transformation=transformation_data,
            prediction_analysis=pred_analysis,
            attribution_validation=attr_validation,
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
        if clean_inputs is None or adversarial_inputs is None:
            return []
        if isinstance(clean_inputs, (list, tuple)) and len(clean_inputs) == 0:
            return []
        if hasattr(clean_inputs, "size") and getattr(clean_inputs, "size", 1) == 0:
            return []
        if hasattr(clean_inputs, "numel") and clean_inputs.numel() == 0:
            return []

        mod = detect_modality(clean_inputs, hint=modality)

        # Determine batch size N
        if isinstance(clean_inputs, (list, tuple)) and all(isinstance(s, str) for s in clean_inputs):
            n_samples = len(clean_inputs)
            clean_list = clean_inputs
            adv_list = adversarial_inputs
        elif isinstance(clean_inputs, (list, tuple)):
            n_samples = len(clean_inputs)
            clean_list = clean_inputs
            adv_list = adversarial_inputs
        else:
            shape = get_shape(clean_inputs)
            n_samples = shape[0] if shape else 1
            clean_list = clean_inputs
            adv_list = adversarial_inputs

        if n_samples == 0:
            return []

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
            elif isinstance(clean_list, dict) or hasattr(clean_list, "items"):
                sample_clean = {
                    k: (v[i : i + 1] if hasattr(v, "__getitem__") and len(v) > i else v)
                    for k, v in clean_list.items()
                }
                sample_adv = {
                    k: (v[i : i + 1] if hasattr(v, "__getitem__") and len(v) > i else v)
                    for k, v in adv_list.items()
                } if (isinstance(adv_list, dict) or hasattr(adv_list, "items")) else adv_list
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

        call_kwargs = dict(kwargs)
        if "attack_metadata" not in call_kwargs and meta is not None:
            call_kwargs["attack_metadata"] = meta

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
                **call_kwargs,
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
                **call_kwargs,
            )
