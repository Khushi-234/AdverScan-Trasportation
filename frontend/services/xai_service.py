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


def get_available_xai_techniques(domain: str = "image", model: Any = None) -> List[str]:
    """
    Query backend TechniqueRegistry for genuinely available and compatible XAI techniques.
    """
    clean_domain = domain.strip().lower().replace("-", "_")
    try:
        mod = Modality.from_str(clean_domain)
        return technique_registry.get_compatible_techniques(mod, model=model)
    except Exception:
        return ["saliency", "integrated_gradients", "shap", "lime"]


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

    try:
        xai_cfg = XAIConfig(
            mode=mode if mode in ("fast", "standard", "thorough") else "standard",
            technique=technique,
            budget=budget or 100,
        )
        explainer = XAIExplainer(config=xai_cfg)

        assessment_obj = None
        scoring_obj = None
        if isinstance(vulnerability_result, dict):
            assessment_obj = vulnerability_result.get("_assessment_obj") or vulnerability_result.get("assessment")
            scoring_obj = vulnerability_result.get("_scoring_obj") or vulnerability_result.get("scoring")

        exp_output = explainer.explain_attack_result(
            model=model,
            attack_result=attack_result,
            assessment_result=assessment_obj,
            scoring_result=scoring_obj,
            technique=technique,
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
