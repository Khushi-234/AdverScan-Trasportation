"""
Frontend service for M6 Explainable AI (XAI) root-cause and feature attribution analysis.
Coordinates backend XAIExplainer with multi-modal inputs and failure diagnostics.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.explainability.config import XAIConfig
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.aggregate import aggregate_explanations
from app.explainability.techniques.registry import technique_registry
from app.explainability.modality import Modality
from app.explainability.domain_profiles import get_domain_profile
from app.attack_engine.models import AttackResult as BackendAttackResult, AttackMetadata


TECHNIQUE_DISPLAY_NAMES: Dict[str, str] = {
    "shap": "SHAP (Kernel / Deep SHAP)",
    "treeshap": "TreeSHAP (Tree Ensemble)",
    "lime": "LIME (Local Interpretable Explanations)",
    "saliency": "Vanilla Saliency (Input Gradient)",
    "integrated_gradients": "Integrated Gradients (Path Integral)",
    "gradcam": "Grad-CAM (Activation Mapping)",
}


def get_available_xai_techniques(domain: str = "image", model: Any = None) -> List[str]:
    """
    Query backend TechniqueRegistry for genuinely available and compatible XAI techniques.
    Only returns techniques verified to be supported for the specified domain and model.
    """
    clean_domain = domain.strip().lower().replace("-", "_")
    try:
        mod = Modality.from_str(clean_domain)
        compatible = technique_registry.get_compatible_techniques(mod, model=model)
        if compatible:
            return compatible
        profile = get_domain_profile(mod)
        return [t for t in profile.compatible_techniques if t != "treeshap" or model is None]
    except Exception:
        return ["saliency", "integrated_gradients", "shap", "lime"]


def get_suggested_xai_technique(domain: str = "image", model: Any = None) -> str:
    """
    Determine the optimal and supported XAI technique for the given domain and model.
    """
    clean_domain = domain.strip().lower().replace("-", "_")
    compatible = get_available_xai_techniques(domain=clean_domain, model=model)
    if not compatible:
        return "saliency"
    try:
        mod = Modality.from_str(clean_domain)
        profile = get_domain_profile(mod)
        suggested = profile.default_technique
        if suggested in compatible:
            return suggested
        return compatible[0]
    except Exception:
        return compatible[0]


def explain_assessment(
    model: Any,
    attack_result: Union[Dict[str, Any], Any],
    vulnerability_result: Optional[Dict[str, Any]] = None,
    domain: str = "image",
    technique: Optional[str] = None,
    mode: str = "standard",
    budget: Optional[int] = None,
    target_class: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run authentic XAI explainability on attack inputs and model predictions.
    Gracefully handles unsupported models and missing dependencies without crashing.

    Args:
        model: Target ML model or adapter.
        attack_result: AttackResult dataclass or frontend result dict.
        vulnerability_result: Optional M5 vulnerability assessment/scoring result.
        domain: Input modality ("image", "tabular", "text", "timeseries").
        technique: Specific technique override ("shap", "treeshap", "lime", "saliency", "integrated_gradients", "gradcam").
        mode: Operational mode ('fast', 'standard', 'thorough').
        budget: Evaluation budget.
        target_class: Target class index to explain.

    Returns:
        Structured dictionary containing XAI explanation findings, attributions,
        feature rankings, and diagnostics.
    """
    clean_domain = domain.strip().lower().replace("-", "_")

    # Guard: missing model
    if model is None:
        return {
            "status": "unsupported",
            "executed": False,
            "technique": technique or "unknown",
            "message": "No model provided for explainability analysis.",
            "attribution": None,
            "local_explanation": {},
            "global_explanation": {},
        }

    # Guard: missing attack result
    if attack_result is None:
        return {
            "status": "unavailable",
            "executed": False,
            "technique": technique or "unknown",
            "message": "No attack results available to explain.",
            "attribution": None,
            "local_explanation": {},
            "global_explanation": {},
        }

    resolved_technique = technique or get_suggested_xai_technique(domain=clean_domain, model=model)

    try:
        xai_cfg = XAIConfig(
            mode=mode if mode in ("fast", "standard", "thorough") else "standard",
            technique=resolved_technique,
            budget=budget or 100,
        )
        explainer = XAIExplainer(config=xai_cfg)

        assessment_obj = None
        scoring_obj = None
        if isinstance(vulnerability_result, dict):
            assessment_obj = vulnerability_result.get("_assessment_obj") or vulnerability_result.get("assessment")
            scoring_obj = vulnerability_result.get("_scoring_obj") or vulnerability_result.get("scoring")

        target_attack_obj: Any = None
        if isinstance(attack_result, dict):
            cached_obj = attack_result.get("_attack_result_obj")
            if cached_obj is not None and isinstance(cached_obj, BackendAttackResult):
                target_attack_obj = cached_obj
            else:
                clean_in = attack_result.get("original_inputs")
                if clean_in is None:
                    clean_in = attack_result.get("_clean_inputs")
                if clean_in is None:
                    clean_in = attack_result.get("_original_inputs")
                if clean_in is None:
                    clean_in = kwargs.get("clean_inputs")

                adv_in = attack_result.get("adversarial_examples")
                if adv_in is None:
                    adv_in = attack_result.get("_adversarial_examples")
                if adv_in is None:
                    adv_in = kwargs.get("adversarial_inputs")

                labels = attack_result.get("labels")
                if labels is None:
                    labels = attack_result.get("_clean_labels")
                if labels is None:
                    labels = attack_result.get("_labels")
                if labels is None:
                    labels = kwargs.get("labels")
                raw_atk_name = (
                    attack_result.get("attack_name")
                    or attack_result.get("name")
                    or kwargs.get("attack_name", "adversarial_attack")
                )
                meta_obj = AttackMetadata(
                    name=str(raw_atk_name),
                    category=attack_result.get("category", "gradient"),
                    attack_type=attack_result.get("threat_model", "white_box"),
                )
                target_attack_obj = BackendAttackResult(
                    adversarial_examples=adv_in,
                    original_inputs=clean_in,
                    labels=labels,
                    metadata=meta_obj,
                    attack_name=str(raw_atk_name),
                    original_predictions=attack_result.get("original_predictions"),
                    adversarial_predictions=attack_result.get("adversarial_predictions"),
                    perturbation_metrics=attack_result.get("perturbation_metrics") or {},
                )
        else:
            target_attack_obj = attack_result

        # Fallback to active assessment session if original_inputs is still None
        if target_attack_obj is not None and getattr(target_attack_obj, "original_inputs", None) is None:
            try:
                from frontend.utils.session_state import get_active_assessment
                act_data = get_active_assessment()
                if act_data and act_data.get("test_inputs") is not None:
                    target_attack_obj.original_inputs = act_data["test_inputs"]
                    if getattr(target_attack_obj, "labels", None) is None:
                        target_attack_obj.labels = act_data.get("test_labels")
            except Exception:
                pass

        exp_output = explainer.explain_attack_result(
            model=model,
            attack_result=target_attack_obj,
            assessment_result=assessment_obj,
            scoring_result=scoring_obj,
            technique=resolved_technique,
            modality=clean_domain,
            target_class=target_class,
            raise_on_error=False,
            **kwargs,
        )

        # 1. Single sample explanation
        if isinstance(exp_output, ExplanationResult):
            res_dict = exp_output.to_dict()
            status = res_dict.get("attribution_status", "valid")
            executed = status in ("valid", "success")

            return {
                "status": "success" if executed else status,
                "executed": executed,
                "technique": exp_output.technique,
                "target_class": exp_output.target_class,
                "clean_prediction": exp_output.clean_prediction,
                "adversarial_prediction": exp_output.adversarial_prediction,
                "prediction_changed": exp_output.prediction_changed,
                "attack_caused_failure": exp_output.attack_caused_failure,
                "attribution": res_dict.get("attribution"),
                "clean_attribution": res_dict.get("clean_attribution"),
                "adversarial_attribution": res_dict.get("adversarial_attribution"),
                "feature_importance": res_dict.get("feature_importance", {}),
                "local_explanation": {
                    "clean_top_features": res_dict.get("feature_importance", {}).get("clean_top_features", []),
                    "adversarial_top_features": res_dict.get("feature_importance", {}).get("adversarial_top_features", []),
                    "attribution_drift": res_dict.get("comparison_data", {}).get("attribution_drift"),
                },
                "global_explanation": {},
                "decision": res_dict.get("xai_decision", {}),
                "message": f"Generated {exp_output.technique.upper()} explanation for class {exp_output.target_class}.",
                "_result_obj": exp_output,
            }

        # 2. Batch of explanations
        elif isinstance(exp_output, list) and exp_output:
            agg_summary = aggregate_explanations(exp_output)
            first_exp = exp_output[0]
            first_dict = first_exp.to_dict()

            return {
                "status": "success",
                "executed": True,
                "technique": first_exp.technique,
                "samples_explained": len(exp_output),
                "local_explanation": {
                    "sample_0_clean_top": first_dict.get("feature_importance", {}).get("clean_top_features", []),
                    "sample_0_adv_top": first_dict.get("feature_importance", {}).get("adversarial_top_features", []),
                },
                "global_explanation": {
                    "global_top_features": agg_summary.get("global_top_features", []),
                    "frequently_modified_features": agg_summary.get("frequently_modified_features", []),
                    "recurring_attribution_shifts": agg_summary.get("recurring_attribution_shifts", []),
                    "prediction_transitions": agg_summary.get("prediction_transitions", {}),
                },
                "attribution": first_dict.get("attribution"),
                "message": f"Generated batch {first_exp.technique.upper()} explanations ({len(exp_output)} samples).",
                "_result_objs": exp_output,
            }

        else:
            return {
                "status": "unavailable",
                "executed": False,
                "technique": technique or "unknown",
                "message": "XAI explanation generation did not return valid result objects.",
                "attribution": None,
                "local_explanation": {},
                "global_explanation": {},
            }

    except Exception as exc:
        return {
            "status": "error",
            "executed": False,
            "technique": technique or "unknown",
            "message": f"Explainability generation failed: {str(exc)}",
            "attribution": None,
            "local_explanation": {},
            "global_explanation": {},
        }
