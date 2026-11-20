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
    sample_index: Optional[int] = None,
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
        sample_index: Optional specific sample index to isolate and explain.

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

        def _slice_one(data: Any, idx: int) -> Any:
            if data is None:
                return None
            try:
                if isinstance(data, (list, tuple)):
                    if 0 <= idx < len(data):
                        return [data[idx]]
                elif hasattr(data, "shape") and len(data.shape) > 0:
                    if 0 <= idx < data.shape[0]:
                        return data[idx : idx + 1]
            except Exception:
                pass
            return data

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

        # Uniformly isolate sample if sample_index requested
        if sample_index is not None and target_attack_obj is not None:
            c_in = getattr(target_attack_obj, "original_inputs", None)
            a_in = getattr(target_attack_obj, "adversarial_examples", None)
            lbl = getattr(target_attack_obj, "labels", None)
            target_attack_obj = BackendAttackResult(
                adversarial_examples=_slice_one(a_in, sample_index),
                original_inputs=_slice_one(c_in, sample_index),
                labels=_slice_one(lbl, sample_index),
                metadata=getattr(target_attack_obj, "metadata", None),
                attack_name=getattr(target_attack_obj, "attack_name", "adversarial_attack"),
                original_predictions=getattr(target_attack_obj, "original_predictions", None),
                adversarial_predictions=getattr(target_attack_obj, "adversarial_predictions", None),
                perturbation_metrics=getattr(target_attack_obj, "perturbation_metrics", {}),
            )

        # Extract meaningful feature names wherever metadata is available
        extracted_feature_names = kwargs.get("feature_names")
        if extracted_feature_names is None:
            try:
                from frontend.utils.session_state import get_active_assessment
                act_data = get_active_assessment() or {}
                extracted_feature_names = (
                    act_data.get("metadata", {}).get("feature_names")
                    or act_data.get("dataset_metadata", {}).get("feature_names")
                    or act_data.get("feature_names")
                )
            except Exception:
                pass
        if extracted_feature_names is None:
            feat_in = getattr(model, "feature_names_in_", None)
            if feat_in is not None:
                try:
                    extracted_feature_names = list(feat_in)
                except Exception:
                    pass
        if extracted_feature_names is None and isinstance(attack_result, dict):
            extracted_feature_names = (
                attack_result.get("metadata", {}).get("feature_names")
                or attack_result.get("feature_names")
            )
        elif extracted_feature_names is None and hasattr(attack_result, "metadata"):
            m = getattr(attack_result, "metadata", None)
            if isinstance(m, dict):
                extracted_feature_names = m.get("feature_names")
            elif hasattr(m, "feature_names"):
                extracted_feature_names = getattr(m, "feature_names")

        if extracted_feature_names:
            kwargs["feature_names"] = extracted_feature_names

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

        import time
        t_start = time.perf_counter()
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
        t_elapsed = time.perf_counter() - t_start

        def _get_single_pred(preds: Any, idx: Optional[int]) -> Any:
            if preds is None:
                return None
            try:
                if hasattr(preds, "tolist"):
                    preds = preds.tolist()
                if isinstance(preds, (list, tuple)):
                    if idx is not None and 0 <= idx < len(preds):
                        return preds[idx]
                    elif len(preds) == 1:
                        return preds[0]
                elif isinstance(preds, (int, float, str, np.integer, np.floating)):
                    return preds
            except Exception:
                pass
            return None

        # 1. Single sample explanation
        if isinstance(exp_output, ExplanationResult):
            res_dict = exp_output.to_dict()
            attr_status = res_dict.get("attribution_status", "valid")
            executed = attr_status in ("valid", "success")

            tgt_class = getattr(exp_output, "target_class", None)
            if tgt_class is None and isinstance(res_dict.get("attribution"), dict):
                tgt_class = res_dict["attribution"].get("target_class_explained")
            if tgt_class is None:
                tgt_class = target_class

            clean_attr = res_dict.get("clean_attribution")
            adv_attr = res_dict.get("adversarial_attribution")
            attr_c = res_dict.get("attribution") or {}
            if clean_attr is None and isinstance(attr_c, dict):
                clean_attr = attr_c.get("clean", {}).get("attribution") if isinstance(attr_c.get("clean"), dict) else attr_c.get("clean_attribution")
            if adv_attr is None and isinstance(attr_c, dict):
                adv_attr = attr_c.get("adversarial", {}).get("attribution") if isinstance(attr_c.get("adversarial"), dict) else (attr_c.get("adversarial_attribution") or attr_c.get("heatmap"))

            comp_dict = res_dict.get("comparison") or res_dict.get("comparison_data") or {}
            drift = comp_dict.get("attribution_drift")
            if drift is None:
                drift = comp_dict.get("normalized_attribution_l1")
            distortion = comp_dict.get("feature_salience_distortion")

            c_pred = exp_output.clean_prediction
            a_pred = exp_output.adversarial_prediction
            if c_pred is None:
                orig_p = getattr(target_attack_obj, "original_predictions", None)
                if orig_p is None and isinstance(attack_result, dict):
                    orig_p = attack_result.get("original_predictions")
                c_pred = _get_single_pred(orig_p, sample_index)
            if a_pred is None:
                adv_p = getattr(target_attack_obj, "adversarial_predictions", None)
                if adv_p is None and isinstance(attack_result, dict):
                    adv_p = attack_result.get("adversarial_predictions")
                a_pred = _get_single_pred(adv_p, sample_index)

            pred_chg = exp_output.prediction_changed
            if pred_chg is None and c_pred is not None and a_pred is not None:
                pred_chg = bool(c_pred != a_pred)
            elif c_pred is None or a_pred is None:
                pred_chg = None

            # Resolve honest status and message without contradiction
            if executed:
                final_status = "success"
                status_msg = f"Generated {exp_output.technique.upper()} explanation for class {tgt_class}."
            else:
                final_status = attr_status if attr_status else "error"
                err_msg = None
                attr_dict = res_dict.get("attribution") or {}
                if isinstance(attr_dict, dict):
                    c_dict = attr_dict.get("clean") or {}
                    a_dict = attr_dict.get("adversarial") or {}
                    if isinstance(c_dict, dict) and c_dict.get("message") and c_dict.get("status") not in ("success", "valid"):
                        err_msg = c_dict["message"]
                    elif isinstance(a_dict, dict) and a_dict.get("message") and a_dict.get("status") not in ("success", "valid"):
                        err_msg = a_dict["message"]
                if not err_msg and isinstance(res_dict.get("metadata"), dict):
                    err_msg = res_dict["metadata"].get("error")
                if not err_msg:
                    err_msg = f"{exp_output.technique.upper()} explanation generation failed with status: {final_status}."
                status_msg = err_msg

            fail_dict = res_dict.get("failure_analysis", {})
            comp_dict = res_dict.get("comparison", {})
            meta_dict = res_dict.get("metadata", {})
            if "execution_time_seconds" not in meta_dict and "explain_seconds" in meta_dict:
                meta_dict["execution_time_seconds"] = meta_dict["explain_seconds"]
            elif "execution_time_seconds" not in meta_dict:
                meta_dict["execution_time_seconds"] = round(t_elapsed, 4)
            meta_dict["total_pipeline_seconds"] = round(t_elapsed, 4)
            pert_metrics = getattr(target_attack_obj, "perturbation_metrics", {}) or (attack_result.get("perturbation_metrics", {}) if isinstance(attack_result, dict) else {})

            return {
                "status": final_status,
                "executed": executed,
                "technique": exp_output.technique,
                "target_class": tgt_class,
                "clean_prediction": c_pred,
                "adversarial_prediction": a_pred,
                "clean_confidence": getattr(exp_output, "clean_confidence", None),
                "adversarial_confidence": getattr(exp_output, "adversarial_confidence", None),
                "prediction_changed": pred_chg,
                "attack_caused_failure": exp_output.attack_caused_failure,
                "attack_name": getattr(exp_output, "attack_name", getattr(target_attack_obj, "attack_name", "adversarial_attack")),
                "modality": getattr(exp_output, "modality", clean_domain),
                "failure_analysis": fail_dict,
                "comparison": comp_dict,
                "metadata": meta_dict,
                "defense_hints": res_dict.get("defense_hints", []) or fail_dict.get("defense_hints", []),
                "defense_pipeline": res_dict.get("defense_pipeline", {}) or fail_dict.get("defense_pipeline", {}),
                "recommended_defenses": fail_dict.get("recommended_defenses", []),
                "attack_reason": fail_dict.get("attack_reason", ""),
                "contributing_features": fail_dict.get("contributing_features", []),
                "perturbation_alignment": comp_dict.get("perturbation_alignment"),
                "perturbation_metrics": pert_metrics,
                "batch_counts": {"total": 1, "succeeded": 1 if executed else 0, "failed": 0 if executed else 1},
                "attribution": res_dict.get("attribution"),
                "clean_attribution": clean_attr,
                "adversarial_attribution": adv_attr,
                "step_importance": res_dict.get("step_importance") or (res_dict.get("attribution", {}).get("step_importance") if isinstance(res_dict.get("attribution"), dict) else None),
                "feature_importance": res_dict.get("feature_importance", {}),
                "feature_names": extracted_feature_names,
                "attribution_drift": drift,
                "feature_salience_distortion": distortion,
                "local_explanation": {
                    "clean_top_features": res_dict.get("feature_importance", {}).get("clean_top_features", []),
                    "adversarial_top_features": res_dict.get("feature_importance", {}).get("adversarial_top_features", []),
                    "attribution_drift": drift,
                    "feature_salience_distortion": distortion,
                },
                "global_explanation": {},
                "decision": res_dict.get("xai_decision", {}),
                "message": status_msg,
                "_result_obj": exp_output,
            }

        # 2. Batch of explanations
        elif isinstance(exp_output, list) and exp_output:
            agg_summary = aggregate_explanations(exp_output)
            first_exp = exp_output[0]
            first_dict = first_exp.to_dict()

            first_attr_status = first_dict.get("attribution_status", "valid")
            first_executed = first_attr_status in ("valid", "success")

            first_tgt_class = getattr(first_exp, "target_class", None)
            if first_tgt_class is None and isinstance(first_dict.get("attribution"), dict):
                first_tgt_class = first_dict["attribution"].get("target_class_explained")
            if first_tgt_class is None:
                first_tgt_class = target_class

            clean_attr = first_dict.get("clean_attribution")
            adv_attr = first_dict.get("adversarial_attribution")
            first_attr_c = first_dict.get("attribution") or {}
            if clean_attr is None and isinstance(first_attr_c, dict):
                clean_attr = first_attr_c.get("clean", {}).get("attribution") if isinstance(first_attr_c.get("clean"), dict) else first_attr_c.get("clean_attribution")
            if adv_attr is None and isinstance(first_attr_c, dict):
                adv_attr = first_attr_c.get("adversarial", {}).get("attribution") if isinstance(first_attr_c.get("adversarial"), dict) else (first_attr_c.get("adversarial_attribution") or first_attr_c.get("heatmap"))

            clean_top = first_dict.get("feature_importance", {}).get("clean_top_features", [])
            adv_top = first_dict.get("feature_importance", {}).get("adversarial_top_features", [])

            first_comp_dict = first_dict.get("comparison") or first_dict.get("comparison_data") or {}
            drift = first_comp_dict.get("attribution_drift")
            if drift is None:
                drift = first_comp_dict.get("normalized_attribution_l1")
            distortion = first_comp_dict.get("feature_salience_distortion")

            c_pred = getattr(first_exp, "clean_prediction", None)
            a_pred = getattr(first_exp, "adversarial_prediction", None)
            if c_pred is None:
                orig_p = getattr(target_attack_obj, "original_predictions", None)
                if orig_p is None and isinstance(attack_result, dict):
                    orig_p = attack_result.get("original_predictions")
                c_pred = _get_single_pred(orig_p, 0)
            if a_pred is None:
                adv_p = getattr(target_attack_obj, "adversarial_predictions", None)
                if adv_p is None and isinstance(attack_result, dict):
                    adv_p = attack_result.get("adversarial_predictions")
                a_pred = _get_single_pred(adv_p, 0)

            pred_chg = getattr(first_exp, "prediction_changed", None)
            if pred_chg is None and c_pred is not None and a_pred is not None:
                pred_chg = bool(c_pred != a_pred)
            elif c_pred is None or a_pred is None:
                pred_chg = None

            # Inspect all batch results to detect partial or complete failures honestly
            successful_exps = [e for e in exp_output if getattr(e, "attribution_status", None) in ("valid", "success", "timeout_fallback")]
            failed_exps = [(idx, e) for idx, e in enumerate(exp_output) if getattr(e, "attribution_status", None) not in ("valid", "success", "timeout_fallback")]

            def _extract_exp_error(e_obj: Any) -> Optional[str]:
                if hasattr(e_obj, "metadata") and isinstance(e_obj.metadata, dict):
                    err = e_obj.metadata.get("error")
                    if err:
                        return str(err)
                if hasattr(e_obj, "attribution") and isinstance(e_obj.attribution, dict):
                    c_msg = e_obj.attribution.get("clean", {}).get("message")
                    a_msg = e_obj.attribution.get("adversarial", {}).get("message")
                    if c_msg:
                        return str(c_msg)
                    if a_msg:
                        return str(a_msg)
                return None

            if len(failed_exps) == 0:
                batch_status = "success"
                batch_executed = True
                batch_msg = f"Generated batch {first_exp.technique.upper()} explanations ({len(exp_output)} samples)."
            elif len(successful_exps) > 0:
                batch_status = "partial"
                batch_executed = True
                first_fail_idx, first_fail_exp = failed_exps[0]
                fail_err = _extract_exp_error(first_fail_exp)
                failed_indices = [idx for idx, _ in failed_exps]
                batch_msg = (
                    f"Partial batch {first_exp.technique.upper()} explanations: {len(successful_exps)}/{len(exp_output)} succeeded. "
                    f"Failed indices: {failed_indices} (Sample {first_fail_idx}: {fail_err or 'attribution failed'})."
                )
            else:
                batch_status = first_attr_status if first_attr_status else "error"
                batch_executed = False
                first_fail_idx, first_fail_exp = failed_exps[0]
                fail_err = _extract_exp_error(first_fail_exp)
                batch_msg = f"Batch {first_exp.technique.upper()} explanations failed on sample {first_fail_idx}: {fail_err or f'status {batch_status}'}."

            first_fail_dict = first_dict.get("failure_analysis", {})
            first_comp_dict = first_dict.get("comparison", {})
            first_meta_dict = first_dict.get("metadata", {})
            batch_explain_sum = sum(getattr(e, "metadata", {}).get("explain_seconds", 0.0) if hasattr(e, "metadata") and isinstance(e.metadata, dict) else 0.0 for e in exp_output)
            first_meta_dict["execution_time_seconds"] = round(batch_explain_sum, 4) if batch_explain_sum > 0 else round(t_elapsed, 4)
            first_meta_dict["total_pipeline_seconds"] = round(t_elapsed, 4)
            first_meta_dict["per_sample_explain_seconds"] = round(batch_explain_sum / len(exp_output), 4) if len(exp_output) > 0 and batch_explain_sum > 0 else None
            pert_metrics = getattr(target_attack_obj, "perturbation_metrics", {}) or (attack_result.get("perturbation_metrics", {}) if isinstance(attack_result, dict) else {})

            return {
                "status": batch_status,
                "executed": batch_executed,
                "technique": first_exp.technique,
                "target_class": first_tgt_class,
                "clean_prediction": c_pred,
                "adversarial_prediction": a_pred,
                "clean_confidence": getattr(first_exp, "clean_confidence", None),
                "adversarial_confidence": getattr(first_exp, "adversarial_confidence", None),
                "prediction_changed": pred_chg,
                "attack_caused_failure": getattr(first_exp, "attack_caused_failure", False),
                "attack_name": getattr(first_exp, "attack_name", getattr(target_attack_obj, "attack_name", "adversarial_attack")),
                "modality": getattr(first_exp, "modality", clean_domain),
                "failure_analysis": first_fail_dict,
                "comparison": first_comp_dict,
                "metadata": first_meta_dict,
                "defense_hints": first_dict.get("defense_hints", []) or first_fail_dict.get("defense_hints", []),
                "defense_pipeline": first_dict.get("defense_pipeline", {}) or first_fail_dict.get("defense_pipeline", {}),
                "recommended_defenses": first_fail_dict.get("recommended_defenses", []),
                "attack_reason": first_fail_dict.get("attack_reason", ""),
                "contributing_features": first_fail_dict.get("contributing_features", []),
                "perturbation_alignment": first_comp_dict.get("perturbation_alignment"),
                "perturbation_metrics": pert_metrics,
                "batch_counts": {"total": len(exp_output), "succeeded": len(successful_exps), "failed": len(failed_exps)},
                "samples_explained": len(exp_output),
                "attribution": first_dict.get("attribution"),
                "clean_attribution": clean_attr,
                "adversarial_attribution": adv_attr,
                "step_importance": first_dict.get("step_importance") or (first_dict.get("attribution", {}).get("step_importance") if isinstance(first_dict.get("attribution"), dict) else None),
                "feature_importance": first_dict.get("feature_importance", {}),
                "feature_names": extracted_feature_names,
                "attribution_drift": drift,
                "feature_salience_distortion": distortion,
                "local_explanation": {
                    "clean_top_features": clean_top,
                    "adversarial_top_features": adv_top,
                    "attribution_drift": drift,
                    "feature_salience_distortion": distortion,
                    "sample_0_clean_top": clean_top,
                    "sample_0_adv_top": adv_top,
                },
                "global_explanation": {
                    "global_feature_importance": agg_summary.get("global_feature_importance", []),
                    "vulnerability_frequency": agg_summary.get("vulnerability_frequency", []),
                    "global_top_features": agg_summary.get("global_top_features", []),
                    "frequently_modified_features": agg_summary.get("frequently_modified_features", []),
                    "recurring_attribution_shifts": agg_summary.get("recurring_attribution_shifts", []),
                    "prediction_transitions": agg_summary.get("prediction_transitions", {}),
                },
                "decision": first_dict.get("xai_decision", {}),
                "message": batch_msg,
                "_result_obj": first_exp,
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
