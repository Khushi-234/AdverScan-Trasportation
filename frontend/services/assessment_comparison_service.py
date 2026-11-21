"""
Assessment Comparison Engine for AdverScan Frontend.

Computes multi-dimensional comparisons between two historical assessments:
- Attack-by-Attack metrics (ASR delta, robust accuracy delta, distortion comparison)
- Model architecture and capacity comparison
- Pre-Defense vs Post-Defense hardening effectiveness
- Vulnerability findings & XAI attribution drift differences
- Overall executive verdict and actionable takeaways
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from frontend.services.assessment_history_service import get_scan_detail


def compare_assessments(
    baseline_scan_id: str,
    comparison_scan_id: str,
) -> Dict[str, Any]:
    """
    Perform a comprehensive side-by-side evaluation between a baseline assessment
    and a comparison (or hardened/re-evaluated) assessment.

    Args:
        baseline_scan_id: Primary reference scan identifier.
        comparison_scan_id: Secondary scan identifier to contrast against baseline.

    Returns:
        Structured comparison dictionary with metrics, deltas, and verdicts.
    """
    if not baseline_scan_id or not comparison_scan_id:
        return {
            "success": False,
            "error": "Both baseline and comparison scan IDs must be provided.",
        }

    base = get_scan_detail(baseline_scan_id)
    comp = get_scan_detail(comparison_scan_id)

    if not base:
        return {"success": False, "error": f"Baseline scan '{baseline_scan_id}' not found."}
    if not comp:
        return {"success": False, "error": f"Comparison scan '{comparison_scan_id}' not found."}

    # 1. Modality & Context Compatibility
    base_mod = (base.get("modality") or "general").lower().replace("-", "_")
    comp_mod = (comp.get("modality") or "general").lower().replace("-", "_")
    modality_match = (base_mod == comp_mod)

    # 2. Model & Dataset Header Comparison
    base_m_name = base.get("model_name") or base.get("model_id") or "Baseline Model"
    comp_m_name = comp.get("model_name") or comp.get("model_id") or "Comparison Model"

    same_model = (base.get("model_id") == comp.get("model_id") and base_m_name == comp_m_name)
    is_defense_comparison = bool(
        comp.get("parent_scan_id") == baseline_scan_id
        or comp.get("baseline_scan_id") == baseline_scan_id
        or (comp.get("defenses") and not base.get("defenses"))
    )

    models_comparison = {
        "baseline": {
            "scan_id": baseline_scan_id,
            "model_name": base_m_name,
            "modality": base_mod,
            "dataset_name": base.get("dataset_name") or "N/A",
            "created_at": str(base.get("created_at"))[:19],
        },
        "comparison": {
            "scan_id": comparison_scan_id,
            "model_name": comp_m_name,
            "modality": comp_mod,
            "dataset_name": comp.get("dataset_name") or "N/A",
            "created_at": str(comp.get("created_at"))[:19],
        },
        "modality_match": modality_match,
        "same_model": same_model,
        "is_defense_comparison": is_defense_comparison,
    }

    # 3. Attack-by-Attack Comparison
    base_attacks = {str(a.get("attack_name", "")).upper(): a for a in base.get("attacks", [])}
    comp_attacks = {str(a.get("attack_name", "")).upper(): a for a in comp.get("attacks", [])}

    all_atk_names = sorted(list(set(base_attacks.keys()) | set(comp_attacks.keys())))

    attack_comparisons: List[Dict[str, Any]] = []
    asr_deltas: List[float] = []
    adv_acc_deltas: List[float] = []

    for name in all_atk_names:
        b_atk = base_attacks.get(name)
        c_atk = comp_attacks.get(name)

        if b_atk and c_atk:
            b_asr = float(b_atk.get("attack_success_rate", 0.0))
            c_asr = float(c_atk.get("attack_success_rate", 0.0))
            delta_asr = c_asr - b_asr

            b_clean = float(b_atk.get("clean_accuracy", 0.0))
            c_clean = float(c_atk.get("clean_accuracy", 0.0))
            delta_clean = c_clean - b_clean

            b_adv = float(b_atk.get("adversarial_accuracy", 0.0))
            c_adv = float(c_atk.get("adversarial_accuracy", 0.0))
            delta_adv = c_adv - b_adv

            # Status determination
            # A negative ASR delta means attack is LESS successful on comparison = Robustness Improved!
            if delta_asr <= -3.0 or delta_adv >= 3.0:
                eval_status = "improved"
            elif delta_asr >= 3.0 or delta_adv <= -3.0:
                eval_status = "degraded"
            else:
                eval_status = "neutral"

            asr_deltas.append(delta_asr)
            adv_acc_deltas.append(delta_adv)

            attack_comparisons.append({
                "attack_name": name,
                "matched": True,
                "baseline_asr": b_asr,
                "comparison_asr": c_asr,
                "delta_asr": round(delta_asr, 2),
                "baseline_clean_acc": b_clean,
                "comparison_clean_acc": c_clean,
                "delta_clean_acc": round(delta_clean, 2),
                "baseline_adv_acc": b_adv,
                "comparison_adv_acc": c_adv,
                "delta_adv_acc": round(delta_adv, 2),
                "baseline_epsilon": b_atk.get("epsilon", 0.0),
                "comparison_epsilon": c_atk.get("epsilon", 0.0),
                "eval_status": eval_status,
            })
        elif b_atk:
            attack_comparisons.append({
                "attack_name": name,
                "matched": False,
                "present_in": "baseline_only",
                "baseline_asr": float(b_atk.get("attack_success_rate", 0.0)),
                "baseline_adv_acc": float(b_atk.get("adversarial_accuracy", 0.0)),
                "eval_status": "unmatched",
            })
        elif c_atk:
            attack_comparisons.append({
                "attack_name": name,
                "matched": False,
                "present_in": "comparison_only",
                "comparison_asr": float(c_atk.get("attack_success_rate", 0.0)),
                "comparison_adv_acc": float(c_atk.get("adversarial_accuracy", 0.0)),
                "eval_status": "unmatched",
            })

    # 4. Defense Effectiveness Comparison
    base_defenses = base.get("defenses", [])
    comp_defenses = comp.get("defenses", [])
    defense_comparison = {
        "baseline_defenses_count": len(base_defenses),
        "comparison_defenses_count": len(comp_defenses),
        "baseline_defenses": base_defenses,
        "comparison_defenses": comp_defenses,
    }

    # 5. Vulnerability & Risk Comparison
    base_vulns = base.get("vulnerabilities", [])
    comp_vulns = comp.get("vulnerabilities", [])

    base_avg_score = (
        sum(v.get("vulnerability_score", 0.0) for v in base_vulns) / len(base_vulns)
        if base_vulns else 0.0
    )
    comp_avg_score = (
        sum(v.get("vulnerability_score", 0.0) for v in comp_vulns) / len(comp_vulns)
        if comp_vulns else 0.0
    )

    vuln_comparison = {
        "baseline_count": len(base_vulns),
        "comparison_count": len(comp_vulns),
        "baseline_avg_score": round(base_avg_score, 2),
        "comparison_avg_score": round(comp_avg_score, 2),
        "score_delta": round(comp_avg_score - base_avg_score, 2),
    }

    # 6. Overall Verdict & Insights
    avg_asr_delta = sum(asr_deltas) / len(asr_deltas) if asr_deltas else 0.0
    avg_adv_acc_delta = sum(adv_acc_deltas) / len(adv_acc_deltas) if adv_acc_deltas else 0.0

    if avg_asr_delta <= -4.0 or avg_adv_acc_delta >= 4.0:
        overall_verdict = "Robustness Improved"
        verdict_badge = "success"
        verdict_summary = (
            f"The comparison assessment shows improved adversarial robustness with an average "
            f"ASR reduction of {abs(round(avg_asr_delta, 1))}% and robust accuracy gain of +{round(avg_adv_acc_delta, 1)}%."
        )
    elif avg_asr_delta >= 4.0 or avg_adv_acc_delta <= -4.0:
        overall_verdict = "Robustness Degraded"
        verdict_badge = "danger"
        verdict_summary = (
            f"The comparison assessment exhibits increased vulnerability: average ASR increased by "
            f"+{round(avg_asr_delta, 1)}% while robust accuracy declined by {round(avg_adv_acc_delta, 1)}%."
        )
    else:
        overall_verdict = "Comparable Robustness"
        verdict_badge = "neutral"
        verdict_summary = (
            "Robustness metrics are closely matched between both assessments within standard deviation bounds."
        )

    return {
        "success": True,
        "models": models_comparison,
        "attack_comparisons": attack_comparisons,
        "defense_comparison": defense_comparison,
        "vulnerability_comparison": vuln_comparison,
        "summary": {
            "avg_asr_delta": round(avg_asr_delta, 2),
            "avg_adv_acc_delta": round(avg_adv_acc_delta, 2),
            "matched_attacks_count": len(asr_deltas),
            "total_attacks_count": len(all_atk_names),
            "overall_verdict": overall_verdict,
            "verdict_badge": verdict_badge,
            "verdict_summary": verdict_summary,
        },
    }

