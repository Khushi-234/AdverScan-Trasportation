"""
Aggregation module for dataset-level explanation and vulnerability pattern analysis.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
import numpy as np

from app.explainability.explanation_result import ExplanationResult


def aggregate_explanations(
    results: List[ExplanationResult],
    top_k_features: int = 10,
) -> Dict[str, Any]:
    """
    Combine explanations across multiple samples to uncover dataset-level vulnerability patterns.

    Args:
        results: List of ExplanationResult objects.
        top_k_features: Number of globally influential features to report.

    Returns:
        Structured dictionary containing aggregated metrics, global feature ranking,
        and failure mode breakdown.
    """
    if not results:
        return {
            "total_samples": 0,
            "prediction_change_rate": 0.0,
            "attack_failure_rate": 0.0,
            "mean_confidence_drop": 0.0,
            "global_top_features": [],
            "failure_mode_breakdown": {},
        }

    total = len(results)
    pred_changes = sum(1 for r in results if r.prediction_changed)
    labeled_samples = sum(1 for r in results if r.attack_caused_failure is not None)
    failures = sum(1 for r in results if r.attack_caused_failure is True)

    conf_drops = []
    l1_drifts = []
    cos_sims = []
    failure_modes: Counter = Counter()

    # Track feature importance across dataset
    feature_importance_acc: Dict[str, float] = defaultdict(float)
    feature_count: Counter = Counter()

    attack_names = set()
    modalities = set()
    techniques = set()

    for r in results:
        attack_names.add(r.attack_name)
        modalities.add(getattr(r, "modality", "unknown"))
        techniques.add(r.technique)

        # Confidence and comparison
        if isinstance(r.comparison, dict):
            cd = r.comparison.get("confidence_difference")
            if cd is not None:
                conf_drops.append(cd)

            l1 = r.comparison.get("attribution_l1")
            if l1 is not None:
                l1_drifts.append(l1)

            cos = r.comparison.get("attribution_cosine_similarity")
            if cos is not None:
                cos_sims.append(cos)

        # Failure modes
        if isinstance(r.failure_analysis, dict):
            mode = r.failure_analysis.get("failure_mode", "unknown")
            failure_modes[mode] += 1

        # Extract top features from clean or adversarial attributions / feature_importance
        top_feats = []
        if hasattr(r, "feature_importance") and isinstance(r.feature_importance, dict):
            top_feats = r.feature_importance.get("adversarial_top_features") or r.feature_importance.get("clean_top_features") or []

        if not top_feats and isinstance(r.attribution, dict):
            adv_attr = r.attribution.get("adversarial")
            clean_attr = r.attribution.get("clean")
            if isinstance(adv_attr, dict):
                top_feats = adv_attr.get("top_features") or []
            if not top_feats and isinstance(clean_attr, dict):
                top_feats = clean_attr.get("top_features") or []

        for f in top_feats:
            if isinstance(f, dict):
                fname = f.get("feature")
                if fname:
                    feature_importance_acc[fname] += abs(f.get("importance", 0.0))
                    feature_count[fname] += 1

    # Compute global top features
    global_top_features = []
    if feature_importance_acc:
        sorted_feats = sorted(
            feature_importance_acc.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k_features]

        for rank, (name, total_imp) in enumerate(sorted_feats):
            cnt = feature_count[name]
            global_top_features.append({
                "feature": name,
                "total_importance": float(total_imp),
                "mean_importance": float(total_imp / cnt) if cnt > 0 else 0.0,
                "frequency": cnt,
                "frequency_ratio": float(cnt / total),
                "rank": rank + 1,
            })

    return {
        "total_samples": total,
        "attacks": list(attack_names),
        "modalities": list(modalities),
        "techniques": list(techniques),
        "prediction_change_rate": float(pred_changes / total),
        "attack_failure_rate": float(failures / labeled_samples) if labeled_samples > 0 else None,
        "mean_confidence_drop": float(np.mean(conf_drops)) if conf_drops else 0.0,
        "mean_attribution_l1_drift": float(np.mean(l1_drifts)) if l1_drifts else None,
        "mean_attribution_cosine_similarity": float(np.mean(cos_sims)) if cos_sims else None,
        "global_top_features": global_top_features,
        "failure_mode_breakdown": dict(failure_modes),
    }
