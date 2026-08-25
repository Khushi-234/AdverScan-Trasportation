"""
Main ReportGenerator coordinator for Module 9 (Report Generator) in AdverScan.

Collects, aggregates, and formats security evaluation results into a comprehensive,
structured security report with recommendations.
"""

from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional, Union

from .report_data import ReportData
from .report_result import ReportResult


class ReportGenerator:
    """
    Coordinator class for assembling AdverScan Security Reports.

    Aggregates findings across M1-M8 and generates model-agnostic security reports
    and recommendations without re-executing attacks or re-calculating scores.
    """

    def generate(self, data: Union[ReportData, Dict[str, Any]]) -> ReportResult:
        """
        Generate a comprehensive security report from input report data.

        Args:
            data: Instance of ReportData or a dictionary of module results.

        Returns:
            ReportResult containing aggregated report sections, recommendations,
            formatted markdown string, and metadata.
        """
        if isinstance(data, dict):
            report_data = ReportData.from_dict(data)
        elif isinstance(data, ReportData):
            report_data = data
        else:
            raise TypeError(f"Expected ReportData or dict, got {type(data).__name__}")

        report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine overall score & risk level
        risk_level = report_data.risk_level or "UNKNOWN"
        vulnerability_score = report_data.vulnerability_score

        if vulnerability_score is None:
            vulnerability_score = self._extract_composite_vulnerability_score(report_data.vulnerability_metrics)

        if risk_level == "UNKNOWN":
            risk_level = self._determine_risk_level(vulnerability_score)

        # Generate rule-based domain-agnostic recommendations
        recommendations = self._generate_recommendations(report_data, risk_level, vulnerability_score)

        # Build summary dict
        summary = self._build_summary(report_data, risk_level, vulnerability_score)

        # Format markdown text report
        formatted_report = self._format_text_report(
            report_id=report_id,
            timestamp=timestamp,
            report_data=report_data,
            risk_level=risk_level,
            vulnerability_score=vulnerability_score,
            recommendations=recommendations,
            summary=summary,
        )

        return ReportResult(
            report_id=report_id,
            timestamp=timestamp,
            status="SUCCESS",
            model_info=report_data.model_info,
            baseline_performance=report_data.baseline_performance,
            attack_results=report_data.attack_results,
            vulnerability_metrics=report_data.vulnerability_metrics,
            vulnerability_score=vulnerability_score,
            risk_level=risk_level,
            xai_findings=report_data.xai_findings,
            hardening_results=report_data.hardening_results,
            before_vs_after=report_data.before_vs_after,
            recommendations=recommendations,
            summary=summary,
            formatted_report=formatted_report,
            metadata=report_data.extra_metadata,
        )

    def _extract_composite_vulnerability_score(self, vuln_metrics: Dict[str, Any]) -> Optional[float]:
        """Extract or calculate average vulnerability score from metrics dictionary."""
        if not vuln_metrics or not isinstance(vuln_metrics, dict):
            return None

        scores = []
        for key, val in vuln_metrics.items():
            if isinstance(val, dict):
                scoring = val.get("scoring") or val.get("vulnerability_scoring") or {}
                if isinstance(scoring, dict) and "vulnerability_score" in scoring:
                    score_val = scoring["vulnerability_score"]
                    if score_val is not None:
                        scores.append(float(score_val))
                elif "vulnerability_score" in val and val["vulnerability_score"] is not None:
                    scores.append(float(val["vulnerability_score"]))

        if scores:
            return round(sum(scores) / len(scores), 2)
        return None

    def _determine_risk_level(self, vulnerability_score: Optional[float]) -> str:
        """Determine risk level based on vulnerability score."""
        if vulnerability_score is None:
            return "UNKNOWN"
        if vulnerability_score >= 75.0:
            return "CRITICAL"
        if vulnerability_score >= 50.0:
            return "HIGH"
        if vulnerability_score >= 25.0:
            return "MEDIUM"
        return "LOW"

    def _generate_recommendations(
        self,
        data: ReportData,
        risk_level: str,
        vulnerability_score: Optional[float],
    ) -> List[str]:
        """
        Generate model-agnostic and domain-agnostic security recommendations based on findings.
        """
        recs = []

        # Risk-based recommendations
        if risk_level in ("CRITICAL", "HIGH"):
            recs.append(
                f"Model presents a {risk_level} security risk level (Score: {vulnerability_score}). "
                "Immediate model hardening and defense deployment are recommended before production release."
            )
        elif risk_level == "MEDIUM":
            recs.append(
                f"Model presents a MEDIUM vulnerability score ({vulnerability_score}). "
                "Implement input sanitization or adversarial training to mitigate potential attack vectors."
            )
        elif risk_level == "LOW":
            recs.append(
                "Model exhibits strong baseline robustness against tested attack suites. "
                "Maintain continuous monitoring and periodic adversarial re-testing."
            )

        # Attack-specific recommendations
        attacks = data.attack_results or {}
        for attack_name in ["FGSM", "PGD", "DeepFool"]:
            # Case insensitive check
            matching_key = None
            for k in attacks.keys():
                if k.upper() == attack_name:
                    matching_key = k
                    break

            if matching_key:
                atk_info = attacks[matching_key]
                eval_info = atk_info.get("evaluation") if isinstance(atk_info, dict) else {}
                asr = None
                if isinstance(eval_info, dict):
                    metrics = eval_info.get("metrics", {})
                    asr = metrics.get("attack_success_rate") or metrics.get("asr")

                if asr is not None and asr > 0.5:
                    recs.append(
                        f"High vulnerability detected against {attack_name} attack (ASR: {round(asr * 100, 1)}%). "
                        f"Incorporate gradient-regularized adversarial training or feature denoising."
                    )

        # XAI Findings recommendation
        if data.xai_findings:
            recs.append(
                "Review XAI feature attribution maps to identify sensitive input features "
                "disproportionately targeted by adversarial perturbations."
            )

        # Hardening / Re-test recommendation
        if data.hardening_results:
            recs.append(
                "Hardening defense was applied. Verify post-hardening accuracy and residual risk "
                "using Module 8 Re-Test module."
            )

        if data.before_vs_after:
            improved = True
            for comp_key, comp_val in data.before_vs_after.items():
                if isinstance(comp_val, dict) and not comp_val.get("is_improved", True):
                    improved = False
                    recs.append(
                        f"Re-test indicates persistent vulnerability for attack vector '{comp_key}'. "
                        "Consider increasing adversarial training iterations or epsilon schedule."
                    )
            if improved:
                recs.append(
                    "Re-test results confirm overall improvement in model robustness across evaluated attack vectors."
                )

        if not recs:
            recs.append("Conduct comprehensive adversarial vulnerability scanning across standard perturbation bounds.")

        return recs

    def _build_summary(
        self,
        data: ReportData,
        risk_level: str,
        vulnerability_score: Optional[float],
    ) -> Dict[str, Any]:
        """Build executive summary metrics."""
        attacks_evaluated = list(data.attack_results.keys()) if data.attack_results else []
        xai_techniques_used = list(data.xai_findings.keys()) if data.xai_findings else []

        baseline_acc = None
        if isinstance(data.baseline_performance, dict):
            metrics = data.baseline_performance.get("metrics", {})
            if isinstance(metrics, dict):
                baseline_acc = metrics.get("accuracy")

        return {
            "model_name": data.model_info.get("model_name", "Unknown"),
            "risk_level": risk_level,
            "vulnerability_score": vulnerability_score,
            "baseline_accuracy": baseline_acc,
            "attacks_evaluated": attacks_evaluated,
            "num_attacks": len(attacks_evaluated),
            "xai_techniques": xai_techniques_used,
            "hardening_applied": bool(data.hardening_results),
            "retest_conducted": bool(data.before_vs_after),
        }

    def _format_text_report(
        self,
        report_id: str,
        timestamp: str,
        report_data: ReportData,
        risk_level: str,
        vulnerability_score: Optional[float],
        recommendations: List[str],
        summary: Dict[str, Any],
    ) -> str:
        """Format human-readable markdown security report."""
        lines = [
            "=" * 70,
            "                   ADVERSCAN SECURITY REPORT                   ",
            "=" * 70,
            f"Report ID : {report_id}",
            f"Timestamp : {timestamp}",
            f"Risk Level: {risk_level}",
            f"Vuln Score: {vulnerability_score if vulnerability_score is not None else 'N/A'}",
            "-" * 70,
            "",
            "1. MODEL INFORMATION",
            "-" * 30,
        ]

        # 1. Model Info
        if report_data.model_info:
            for k, v in report_data.model_info.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  No model information provided.")

        # 2. Baseline Performance
        lines.extend(["", "2. BASELINE PERFORMANCE", "-" * 30])
        if report_data.baseline_performance:
            metrics = report_data.baseline_performance.get("metrics", {})
            if isinstance(metrics, dict) and metrics:
                for mk, mv in metrics.items():
                    val_str = f"{round(mv * 100, 2)}%" if isinstance(mv, float) and mv <= 1.0 else str(mv)
                    lines.append(f"  - {mk}: {val_str}")
            else:
                for bk, bv in report_data.baseline_performance.items():
                    lines.append(f"  - {bk}: {bv}")
        else:
            lines.append("  No baseline performance data provided.")

        # 3. Attack Results
        lines.extend(["", "3. ATTACK RESULTS", "-" * 30])
        if report_data.attack_results:
            for atk_name, atk_info in report_data.attack_results.items():
                lines.append(f"  • Attack: {atk_name}")
                if isinstance(atk_info, dict):
                    params = atk_info.get("parameters", {})
                    if params:
                        lines.append(f"    - Parameters: {params}")
                    eval_info = atk_info.get("evaluation", {})
                    if isinstance(eval_info, dict) and "metrics" in eval_info:
                        for emk, emv in eval_info["metrics"].items():
                            val_str = f"{round(emv * 100, 2)}%" if isinstance(emv, float) and emv <= 1.0 else str(emv)
                            lines.append(f"    - {emk}: {val_str}")
        else:
            lines.append("  No attack results recorded.")

        # 4. Vulnerability Metrics & Scoring
        lines.extend(["", "4. VULNERABILITY METRICS & SCORING", "-" * 30])
        lines.append(f"  - Overall Vulnerability Score: {vulnerability_score if vulnerability_score is not None else 'N/A'}")
        lines.append(f"  - Risk Level: {risk_level}")
        if report_data.vulnerability_metrics:
            for vk, vv in report_data.vulnerability_metrics.items():
                lines.append(f"  • Vector: {vk}")
                if isinstance(vv, dict):
                    assess = vv.get("assessment", {})
                    if isinstance(assess, dict):
                        for amk, amv in assess.items():
                            if amk != "extra_metadata":
                                lines.append(f"    - {amk}: {amv}")

        # 5. XAI Findings
        lines.extend(["", "5. XAI FINDINGS", "-" * 30])
        if report_data.xai_findings:
            for xk, xv in report_data.xai_findings.items():
                lines.append(f"  • Technique: {xk}")
                if isinstance(xv, dict):
                    for xfk, xfv in xv.items():
                        if xfk not in ("heatmap", "attribution_map"):
                            lines.append(f"    - {xfk}: {xfv}")
        else:
            lines.append("  No XAI findings generated.")

        # 6. Hardening Results
        lines.extend(["", "6. HARDENING RESULTS", "-" * 30])
        if report_data.hardening_results:
            for hk, hv in report_data.hardening_results.items():
                lines.append(f"  - {hk}: {hv}")
        else:
            lines.append("  No hardening actions recorded.")

        # 7. Before vs After Comparison
        lines.extend(["", "7. BEFORE VS AFTER COMPARISON", "-" * 30])
        if report_data.before_vs_after:
            for ck, cv in report_data.before_vs_after.items():
                lines.append(f"  • Vector: {ck}")
                if isinstance(cv, dict):
                    for delta_k, delta_v in cv.items():
                        if delta_k.startswith("delta_") or delta_k in ("is_improved", "risk_level_changed"):
                            lines.append(f"    - {delta_k}: {delta_v}")
        else:
            lines.append("  No re-test comparison data provided.")

        # 8. Recommendations
        lines.extend(["", "8. RECOMMENDATIONS", "-" * 30])
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  [{i}] {rec}")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)
