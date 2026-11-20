"""
Aggregation module for dataset-level explanation and vulnerability pattern analysis.
"""

from collections import Counter, defaultdict
import re
from typing import Any, Dict, List, Optional
import numpy as np

from app.explainability.explanation_result import ExplanationResult


def _canonical_feature_name(name: str, modality: Optional[str] = None) -> str:
    """Group one-hot columns to parent feature and unroll time-series to sensor names without collapsing spatial image patches."""
    if not name:
        return name
    # Time-series: t{step}_{sensor} -> {sensor}
    m = re.match(r"^t\d+_(.+)$", name)
    if m:
        return m.group(1)
    # Image patches or coordinates: preserve spatial identity
    if name.startswith("patch_") or name.startswith("region_") or name.startswith("pixel_"):
        return name
    # Tabular one-hot: col_VALUE (e.g. type_TRANSFER, protocol_type_tcp) -> col
    if "_" in name and not name.startswith("feature_") and not name.startswith("f_") and not name.startswith("sensor_"):
        parts = name.split("_")
        if len(parts) >= 2 and (parts[1].isupper() or parts[0] in ("type", "protocol", "flag")):
            return parts[0]
    return name


def aggregate_explanations(
    results: List[ExplanationResult],
    top_k_features: int = 10,
) -> Dict[str, Any]:
    """
    Combine explanations across multiple samples to uncover dataset-level vulnerability patterns.
    Produces evidence-based summaries with explicitly defined denominators.

    Args:
        results: List of ExplanationResult objects.
        top_k_features: Number of globally influential features to report.

    Returns:
        Structured dictionary containing aggregated metrics, global feature ranking,
        prediction transitions, frequently modified features, recurring attribution shifts,
        and explanation failures with explicit denominators.
    """
    if not results:
        return {
            "total_samples": 0,
            "total_samples_evaluated": 0,
            "clean_correct_count": 0,
            "clean_incorrect_count": 0,
            "valid_samples_count": 0,
            "labeled_samples_count": 0,
            "unlabeled_samples_count": 0,
            "prediction_change_rate": 0.0,
            "attack_success_rate": 0.0,
            "attack_failure_rate": 0.0,  # Deprecated alias for backwards compatibility
            "attack_success_denominator": "total_samples_evaluated",
            "attack_success_denominator_count": 0,
            "mean_confidence_drop": 0.0,
            "global_feature_importance": [],
            "vulnerability_frequency": [],
            "global_top_features": [],
            "frequently_modified_features": [],
            "recurring_attribution_shifts": [],
            "prediction_transitions": {},
            "prediction_transition_rates": {},
            "explanation_failure_count": 0,
            "explanation_failure_rate": 0.0,
            "explanation_failure_denominator": "total_samples_evaluated",
            "explanation_failure_reasons": {},
            "exploitation_stats": {},
            "grouped_by_attack": {},
            "failure_mode_breakdown": {},
        }

    total = len(results)
    pred_changes = sum(1 for r in results if r.prediction_changed)

    # Denominators and sample counts
    labeled_samples = [r for r in results if r.true_label is not None]
    labeled_count = len(labeled_samples)
    unlabeled_count = total - labeled_count

    clean_correct_samples = [
        r for r in results
        if isinstance(r.failure_analysis, dict) and r.failure_analysis.get("clean_correct") is True
    ]
    clean_correct_count = len(clean_correct_samples)

    clean_incorrect_samples = [
        r for r in results
        if isinstance(r.failure_analysis, dict) and r.failure_analysis.get("clean_correct") is False
    ]
    clean_incorrect_count = len(clean_incorrect_samples)

    # Attack success rate with explicitly documented denominator
    if clean_correct_count > 0:
        successful_attacks = sum(1 for r in clean_correct_samples if r.attack_caused_failure is True)
        attack_success_rate = float(successful_attacks / clean_correct_count)
        attack_success_denominator = "clean_correct_samples"
        attack_success_denominator_count = clean_correct_count
    elif labeled_count > 0:
        successful_attacks = sum(1 for r in labeled_samples if r.attack_caused_failure is True)
        attack_success_rate = float(successful_attacks / labeled_count)
        attack_success_denominator = "labeled_samples"
        attack_success_denominator_count = labeled_count
    else:
        successful_attacks = pred_changes
        attack_success_rate = float(pred_changes / total)
        attack_success_denominator = "total_samples_evaluated"
        attack_success_denominator_count = total

    # Prediction transitions
    pred_transitions: Counter = Counter()
    for r in results:
        c_p = getattr(r, "clean_prediction", "unknown")
        a_p = getattr(r, "adversarial_prediction", "unknown")
        trans_key = f"{c_p} -> {a_p}"
        pred_transitions[trans_key] += 1

    pred_transition_rates = {k: float(v / total) for k, v in pred_transitions.items()}

    # Frequently modified features (from empirical attack transformations, NOT from feature saliency)
    feature_modifications: Counter = Counter()
    for r in results:
        tf = getattr(r, "attack_transformation", None)
        if isinstance(tf, dict):
            # Tabular/array changed features
            c_feats = tf.get("changed_features", [])
            for cf in c_feats:
                if isinstance(cf, dict) and cf.get("feature"):
                    feature_modifications[str(cf["feature"])] += 1
            # Text changed tokens
            c_toks = tf.get("changed_tokens", [])
            for ct in c_toks:
                if isinstance(ct, dict):
                    tok_name = ct.get("clean_token") or f"pos_{ct.get('position')}"
                    feature_modifications[str(tok_name)] += 1

    frequently_modified_features = []
    for fname, cnt in feature_modifications.most_common(top_k_features):
        frequently_modified_features.append({
            "feature": fname,
            "modification_count": cnt,
            "modification_rate": float(cnt / total),
            "denominator": total,
        })

    # Explanation failures tracking
    explanation_failures = 0
    failure_reasons: Counter = Counter()
    for r in results:
        status = getattr(r, "attribution_status", "success")
        val = getattr(r, "attribution_validation", {})
        val_status = val.get("status", "valid") if isinstance(val, dict) else "valid"
        if status != "success":
            explanation_failures += 1
            failure_reasons[f"status_{status}"] += 1
        elif val_status in ("non_finite", "empty", "error", "shape_mismatch"):
            explanation_failures += 1
            failure_reasons[f"validation_{val_status}"] += 1

    explanation_failure_rate = float(explanation_failures / total)

    conf_drops = []
    l1_drifts = []
    cos_sims = []
    failure_modes: Counter = Counter()

    # Track feature exploitation: successful vs failed
    feature_shifts_succ: Dict[str, List[float]] = defaultdict(list)
    feature_shifts_fail: Dict[str, List[float]] = defaultdict(list)
    feature_count_succ: Counter = Counter()
    feature_count_fail: Counter = Counter()
    all_feature_shifts: Dict[str, List[float]] = defaultdict(list)

    attack_names = set()
    modalities = set()
    techniques = set()

    # Per-attack tracking
    attack_groups: Dict[str, List[ExplanationResult]] = defaultdict(list)

    # Track sample-level absolute and signed feature attributions for global importance
    sample_abs_attributions: List[Dict[str, float]] = []
    sample_signed_attributions: List[Dict[str, float]] = []

    for r in results:
        att_name = r.attack_name or "unknown"
        mod_name = getattr(r, "modality", "unknown")
        attack_names.add(att_name)
        modalities.add(mod_name)
        techniques.add(r.technique)
        attack_groups[att_name].append(r)

        # Confidence and comparison
        if isinstance(r.comparison, dict):
            cd = r.comparison.get("clean_class_confidence_drop", r.comparison.get("confidence_difference"))
            if cd is not None:
                conf_drops.append(cd)

            l1 = r.comparison.get("normalized_attribution_l1", r.comparison.get("attribution_l1"))
            if l1 is not None:
                l1_drifts.append(l1)

            cos = r.comparison.get("attribution_cosine_similarity")
            if cos is not None:
                cos_sims.append(cos)

        # Failure modes
        if isinstance(r.failure_analysis, dict):
            mode = r.failure_analysis.get("failure_mode", "unknown")
            failure_modes[mode] += 1

        is_success = r.attack_caused_failure is True or (r.attack_caused_failure is None and r.prediction_changed)

        # Extract features contributing to this sample's failure/explanation
        contrib_features = []
        if isinstance(r.failure_analysis, dict) and r.failure_analysis.get("contributing_features"):
            contrib_features = r.failure_analysis.get("contributing_features", [])
        elif bool(getattr(r, "feature_importance", None)):
            contrib_features = r.feature_importance.get("clean_top_features") or r.feature_importance.get("adversarial_top_features") or []
        elif isinstance(r.attribution, dict):
            clean_attr = r.attribution.get("clean")
            adv_attr = r.attribution.get("adversarial")
            if isinstance(clean_attr, dict) and clean_attr.get("top_features"):
                contrib_features = clean_attr.get("top_features", [])
            elif isinstance(adv_attr, dict) and adv_attr.get("top_features"):
                contrib_features = adv_attr.get("top_features", [])

        # Extract per-sample full attribution values for global SHAP importance
        cur_sample_abs: Dict[str, float] = {}
        cur_sample_signed: Dict[str, float] = {}

        clean_attr_dict = r.attribution.get("clean") if isinstance(r.attribution, dict) else None
        if isinstance(clean_attr_dict, dict) and isinstance(clean_attr_dict.get("feature_importance_map"), dict):
            for raw_k, v in clean_attr_dict["feature_importance_map"].items():
                canon_k = _canonical_feature_name(str(raw_k), modality=mod_name)
                try:
                    val = float(v)
                    cur_sample_abs[canon_k] = abs(val)
                    cur_sample_signed[canon_k] = val
                except (ValueError, TypeError):
                    pass
        elif isinstance(clean_attr_dict, dict) and isinstance(clean_attr_dict.get("top_features"), list):
            for f in clean_attr_dict["top_features"]:
                if isinstance(f, dict) and f.get("feature"):
                    canon_k = _canonical_feature_name(str(f["feature"]), modality=mod_name)
                    val = float(f.get("importance", f.get("weight", 0.0)))
                    cur_sample_abs[canon_k] = abs(val)
                    cur_sample_signed[canon_k] = val
        elif getattr(r, "clean_attribution", None) is not None and getattr(r, "feature_names", None):
            try:
                c_arr = np.asarray(r.clean_attribution).flatten()
                f_names = list(r.feature_names)
                for f_n, val in zip(f_names, c_arr):
                    canon_k = _canonical_feature_name(str(f_n), modality=mod_name)
                    val_f = float(val)
                    cur_sample_abs[canon_k] = abs(val_f)
                    cur_sample_signed[canon_k] = val_f
            except Exception:
                pass
        elif isinstance(getattr(r, "feature_importance", None), dict):
            for f in r.feature_importance.get("clean_top_features", []):
                if isinstance(f, dict) and f.get("feature"):
                    canon_k = _canonical_feature_name(str(f["feature"]), modality=mod_name)
                    val = float(f.get("importance", f.get("weight", 0.0)))
                    cur_sample_abs[canon_k] = abs(val)
                    cur_sample_signed[canon_k] = val

        # Fallback to contrib_features if clean attribution was empty
        if not cur_sample_abs and contrib_features:
            for f in contrib_features:
                if isinstance(f, dict) and f.get("feature"):
                    canon_k = _canonical_feature_name(str(f["feature"]), modality=mod_name)
                    val = float(f.get("importance", f.get("weight", 0.0)))
                    cur_sample_abs[canon_k] = abs(val)
                    cur_sample_signed[canon_k] = val

        # Only aggregate attributions from valid, successfully computed explanations
        r_status = getattr(r, "attribution_status", "success")
        r_val = getattr(r, "attribution_validation", {})
        r_val_status = r_val.get("status", "valid") if isinstance(r_val, dict) else "valid"
        is_valid_exp = (r_status in ("success", "valid")) and (r_val_status not in ("non_finite", "empty", "error", "shape_mismatch"))

        if is_valid_exp and cur_sample_abs:
            sample_abs_attributions.append(cur_sample_abs)
            sample_signed_attributions.append(cur_sample_signed)

        for f in contrib_features:
            if isinstance(f, dict):
                raw_name = f.get("feature")
                if raw_name:
                    canon_name = _canonical_feature_name(str(raw_name), modality=mod_name)
                    shift = float(f.get("importance_shift", f.get("importance", 0.0)))
                    all_feature_shifts[canon_name].append(shift)
                    if is_success:
                        feature_shifts_succ[canon_name].append(shift)
                        feature_count_succ[canon_name] += 1
                    else:
                        feature_shifts_fail[canon_name].append(shift)
                        feature_count_fail[canon_name] += 1

    # 1. Global Feature Importance: mean absolute attribution (SHAP/gradient) across evaluated samples
    all_imp_features = set()
    for s in sample_abs_attributions:
        all_imp_features.update(s.keys())

    valid_denom = len(sample_abs_attributions) if sample_abs_attributions else (total if total > 0 else 1)
    global_feature_importance = []
    for fname in all_imp_features:
        sum_abs = sum(s.get(fname, 0.0) for s in sample_abs_attributions)
        sum_signed = sum(s.get(fname, 0.0) for s in sample_signed_attributions)
        mean_abs = float(sum_abs / valid_denom) if valid_denom > 0 else 0.0
        mean_signed = float(sum_signed / valid_denom) if valid_denom > 0 else 0.0
        sc = sum(1 for s in sample_abs_attributions if fname in s and s[fname] > 0.0)
        global_feature_importance.append({
            "feature": fname,
            "mean_abs_importance": round(mean_abs, 6),
            "mean_importance": round(mean_signed, 6),
            "sample_count": sc,
            "sample_ratio": float(sc / valid_denom) if valid_denom > 0 else 0.0,
            "importance": round(mean_abs, 6),
            "weight": round(mean_abs, 6),
        })

    global_feature_importance.sort(
        key=lambda x: (float(x["mean_abs_importance"]), x["sample_count"]),
        reverse=True,
    )
    for rank, entry in enumerate(global_feature_importance):
        entry["rank"] = rank + 1

    # 2. Adversarial Vulnerability Frequency: frequency of feature targeting and shifts in attacks
    all_features = set(feature_count_succ.keys()) | set(feature_count_fail.keys())
    ranked_features = []
    for fname in all_features:
        cnt_s = feature_count_succ[fname]
        shifts_s = feature_shifts_succ[fname]
        shifts_f = feature_shifts_fail[fname]
        mean_shift_s = float(np.mean(shifts_s)) if shifts_s else 0.0
        mean_shift_f = float(np.mean(shifts_f)) if shifts_f else 0.0

        tot_cnt = cnt_s + feature_count_fail[fname]
        ranked_features.append({
            "feature": fname,
            "frequency": tot_cnt,
            "frequency_ratio": float(tot_cnt / total),
            "success_frequency": cnt_s,
            "failure_frequency": feature_count_fail[fname],
            "total_frequency": tot_cnt,
            "mean_shift_successful": mean_shift_s,
            "mean_shift_failed": mean_shift_f,
        })

    # Sort primarily by success_frequency and magnitude of shift
    ranked_features.sort(key=lambda x: (int(x["success_frequency"]), abs(float(x["mean_shift_successful"]))), reverse=True)
    global_top_features = ranked_features[:top_k_features]
    for rank, entry in enumerate(global_top_features):
        entry["rank"] = rank + 1

    vulnerability_frequency = []
    for rank, entry in enumerate(ranked_features[:top_k_features]):
        vulnerability_frequency.append({
            "rank": rank + 1,
            "feature": entry["feature"],
            "frequency": entry["total_frequency"],
            "frequency_ratio": entry["frequency_ratio"],
            "success_frequency": entry["success_frequency"],
            "failure_frequency": entry["failure_frequency"],
            "total_frequency": entry["total_frequency"],
            "mean_shift_successful": entry["mean_shift_successful"],
            "mean_shift_failed": entry["mean_shift_failed"],
        })

    # Recurring attribution shifts summary
    recurring_shifts = []
    for fname, shifts in all_feature_shifts.items():
        if len(shifts) > 0:
            mean_s = float(np.mean(shifts))
            std_s = float(np.std(shifts)) if len(shifts) > 1 else 0.0
            recurring_shifts.append({
                "feature": fname,
                "sample_count": len(shifts),
                "frequency_ratio": float(len(shifts) / total),
                "mean_signed_shift": mean_s,
                "std_signed_shift": std_s,
            })
    recurring_shifts.sort(key=lambda x: (int(x["sample_count"]), abs(float(x["mean_signed_shift"]))), reverse=True)

    # Grouped metrics by attack_name
    grouped_by_attack = {}
    for att, group in sorted(attack_groups.items()):
        grp_changes = sum(1 for r in group if r.prediction_changed)
        grp_cc = [r for r in group if isinstance(r.failure_analysis, dict) and r.failure_analysis.get("clean_correct") is True]
        if grp_cc:
            grp_succ = sum(1 for r in grp_cc if r.attack_caused_failure is True)
            grp_asr = float(grp_succ / len(grp_cc))
        else:
            grp_asr = float(grp_changes / len(group))

        grouped_by_attack[att] = {
            "samples": len(group),
            "attack_success_rate": grp_asr,
            "prediction_change_rate": float(grp_changes / len(group)),
        }

    return {
        "total_samples": total,
        "total_samples_evaluated": total,
        "clean_correct_count": clean_correct_count,
        "clean_incorrect_count": clean_incorrect_count,
        "valid_samples_count": len(sample_abs_attributions),
        "labeled_samples_count": labeled_count,
        "unlabeled_samples_count": unlabeled_count,
        "attack_success_denominator": attack_success_denominator,
        "attack_success_denominator_count": attack_success_denominator_count,
        "prediction_transitions": dict(pred_transitions),
        "prediction_transition_rates": pred_transition_rates,
        "frequently_modified_features": frequently_modified_features,
        "recurring_attribution_shifts": recurring_shifts[:top_k_features],
        "explanation_failure_count": explanation_failures,
        "explanation_failure_rate": explanation_failure_rate,
        "explanation_failure_denominator": "total_samples_evaluated",
        "explanation_failure_reasons": dict(failure_reasons),
        "attacks": sorted(list(attack_names)),
        "modalities": sorted(list(modalities)),
        "techniques": sorted(list(techniques)),
        "prediction_change_rate": float(pred_changes / total),
        "attack_success_rate": attack_success_rate,
        "attack_failure_rate": attack_success_rate,  # Deprecated alias for backwards compatibility
        "mean_confidence_drop": float(np.mean(conf_drops)) if conf_drops else 0.0,
        "mean_attribution_l1_drift": float(np.mean(l1_drifts)) if l1_drifts else None,
        "mean_attribution_cosine_similarity": float(np.mean(cos_sims)) if cos_sims else None,
        "global_feature_importance": global_feature_importance[:top_k_features],
        "vulnerability_frequency": vulnerability_frequency,
        "global_top_features": global_top_features,
        "grouped_by_attack": grouped_by_attack,
        "failure_mode_breakdown": dict(failure_modes),
    }
