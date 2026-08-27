"""
ReportGenerator — Main coordinator for Module 9 (Report Generator) in AdverScan.

Accepts a ReportData instance and produces a fully structured ReportResult
containing all 15 report sections, MITRE ATLAS mappings, and recommendations.
The actual file I/O is delegated to ReportWriter.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid

from .report_data import ReportData
from .report_result import ReportResult
from .execution_summary import ExecutionSummary


# ── MITRE ATLAS AML Tactic / Technique lookup ─────────────────────────────────
_MITRE_ATLAS: Dict[str, Dict[str, str]] = {
    "fgsm": {
        "tactic": "AML.TA0000 — ML Attack Staging",
        "technique": "AML.T0043 — Craft Adversarial Data (FGSM)",
        "mitigation": "AML.M0003 — Adversarial Input Detection",
    },
    "pgd": {
        "tactic": "AML.TA0000 — ML Attack Staging",
        "technique": "AML.T0043.001 — Craft Adversarial Data (PGD / Iterative)",
        "mitigation": "AML.M0003 — Adversarial Input Detection + AML.M0002 — Model Hardening",
    },
    "deepfool": {
        "tactic": "AML.TA0000 — ML Attack Staging",
        "technique": "AML.T0043.002 — Craft Adversarial Data (Minimal Perturbation)",
        "mitigation": "AML.M0003 — Adversarial Input Detection",
    },
    "cw": {
        "tactic": "AML.TA0000 — ML Attack Staging",
        "technique": "AML.T0043.003 — Craft Adversarial Data (Optimization-based, C&W)",
        "mitigation": "AML.M0002 — Model Hardening + AML.M0015 — Adversarial Training",
    },
}


class ReportGenerator:
    """
    Generates a complete 15-section AdverScan Security Assessment Report
    from a ReportData input without re-executing any attacks or scores.
    """

    def generate(
        self,
        data: Union[ReportData, Dict[str, Any]],
        scan_id: Optional[str] = None,
    ) -> ReportResult:
        """
        Generate a comprehensive security report.

        Args:
            data: ReportData instance or a plain dict that will be converted.
            scan_id: Optional unique identifier for this scan run.

        Returns:
            ReportResult with all 15 sections populated and formatted_report set.
        """
        if isinstance(data, dict):
            report_data = ReportData.from_dict(data)
        elif isinstance(data, ReportData):
            report_data = data
        else:
            raise TypeError(f"Expected ReportData or dict, got {type(data).__name__}")

        report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if scan_id is None:
            scan_id = f"SCAN-{uuid.uuid4().hex[:6].upper()}"

        # ── Scores & Risk ──────────────────────────────────────────────────────
        vuln_score = report_data.vulnerability_score
        if vuln_score is None:
            vuln_score = self._extract_composite_vulnerability_score(
                report_data.vulnerability_metrics
            )
        risk_level = report_data.risk_level or self._determine_risk_level(vuln_score)

        # ── Build Each Section ─────────────────────────────────────────────────
        s1_exec_summary = self._s1_executive_summary(
            report_data, risk_level, vuln_score, scan_id, timestamp
        )
        s8_mitre = self._s8_mitre_atlas(report_data.attack_results)
        s14_recs = self._s14_recommendations(report_data, risk_level, vuln_score)
        s15_final = self._s15_final_security_summary(
            report_data, risk_level, vuln_score, s14_recs
        )

        # ── Formatted Report ───────────────────────────────────────────────────
        formatted = self._format_report(
            report_id=report_id,
            scan_id=scan_id,
            timestamp=timestamp,
            report_data=report_data,
            risk_level=risk_level,
            vuln_score=vuln_score,
            executive_summary=s1_exec_summary,
            mitre_mapping=s8_mitre,
            recommendations=s14_recs,
            final_summary=s15_final,
        )

        return ReportResult(
            report_id=report_id,
            timestamp=timestamp,
            status="SUCCESS",
            scan_id=scan_id,
            # Section 1
            executive_summary=s1_exec_summary,
            # Section 2
            model_info=report_data.model_info,
            # Section 3
            dataset_config=report_data.dataset_config,
            # Section 4
            baseline_performance=report_data.baseline_performance,
            # Section 5
            attack_results=report_data.attack_results,
            # Section 6
            vulnerability_metrics=report_data.vulnerability_metrics,
            # Section 7
            vulnerability_score=vuln_score,
            risk_level=risk_level,
            # Section 8
            mitre_atlas_mapping=s8_mitre,
            # Section 9
            xai_findings=report_data.xai_findings,
            # Section 10
            hardening_results=report_data.hardening_results,
            # Section 11
            retest_results=report_data.retest_results,
            # Section 12
            before_vs_after=report_data.before_vs_after,
            # Section 13
            execution_summary=report_data.execution_summary,
            # Section 14
            recommendations=s14_recs,
            # Section 15
            final_security_summary=s15_final,
            formatted_report=formatted,
            metadata=report_data.extra_metadata,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Section Builders
    # ══════════════════════════════════════════════════════════════════════════

    def _s1_executive_summary(
        self,
        data: ReportData,
        risk_level: str,
        vuln_score: Optional[float],
        scan_id: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        attacks = list(data.attack_results.keys())
        baseline_acc = self._extract_baseline_accuracy(data.baseline_performance)
        return {
            "scan_id": scan_id,
            "timestamp": timestamp,
            "risk_level": risk_level,
            "vulnerability_score": vuln_score,
            "baseline_accuracy": baseline_acc,
            "attacks_evaluated": attacks,
            "num_attacks": len(attacks),
            "xai_enabled": bool(data.xai_findings),
            "hardening_applied": bool(data.hardening_results),
            "retest_conducted": bool(data.retest_results or data.before_vs_after),
            "execution_status": (
                data.execution_summary.overall_status
                if data.execution_summary
                else "UNKNOWN"
            ),
        }

    def _s8_mitre_atlas(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {}
        for attack_name in attack_results:
            key = attack_name.lower()
            if key in _MITRE_ATLAS:
                mapping[attack_name] = _MITRE_ATLAS[key]
            else:
                mapping[attack_name] = {
                    "tactic": "AML.TA0000 — ML Attack Staging",
                    "technique": f"AML.T0043 — Craft Adversarial Data ({attack_name.upper()})",
                    "mitigation": "AML.M0003 — Adversarial Input Detection",
                }
        return mapping

    def _s14_recommendations(
        self,
        data: ReportData,
        risk_level: str,
        vuln_score: Optional[float],
    ) -> List[str]:
        recs: List[str] = []

        # Risk-level recommendations
        if risk_level == "CRITICAL":
            recs.append(
                f"[CRITICAL] Vulnerability score {vuln_score} indicates critical exposure. "
                "Block all production inference endpoints and initiate immediate adversarial hardening."
            )
        elif risk_level == "HIGH":
            recs.append(
                f"[HIGH] Model presents a HIGH security risk (Score: {vuln_score}). "
                "Deploy adversarial training or certified defenses before production release."
            )
        elif risk_level == "MEDIUM":
            recs.append(
                f"[MEDIUM] Vulnerability score {vuln_score} warrants attention. "
                "Implement input sanitization and monitor inference traffic for anomalies."
            )
        elif risk_level == "LOW":
            recs.append(
                "Model exhibits strong baseline robustness against tested attack suites. "
                "Maintain continuous monitoring and conduct periodic adversarial re-testing."
            )

        # Attack-specific recommendations
        for attack_name, atk_info in data.attack_results.items():
            if not isinstance(atk_info, dict):
                continue
            eval_info = atk_info.get("evaluation") or {}
            metrics = eval_info.get("metrics") or atk_info
            asr = metrics.get("attack_success_rate") or metrics.get("asr")
            if asr is not None and float(asr) > 0.5:
                recs.append(
                    f"High susceptibility to {attack_name.upper()} (ASR: {round(float(asr) * 100, 1)}%). "
                    f"Apply gradient masking countermeasures or feature-space smoothing."
                )

        # XAI-based recommendation
        if data.xai_findings:
            recs.append(
                "XAI attribution maps are available. Review highlighted input regions "
                "disproportionately targeted by adversarial perturbations to guide robustness patches."
            )

        # Hardening / retest recommendation
        if data.hardening_results:
            recs.append(
                "Adversarial defense was applied. Validate post-hardening accuracy retention "
                "and conduct periodic re-tests to ensure defense durability."
            )
        if data.before_vs_after:
            for comp_key, comp_val in data.before_vs_after.items():
                if isinstance(comp_val, dict) and not comp_val.get("is_improved", True):
                    recs.append(
                        f"Re-test indicates persistent vulnerability for vector '{comp_key}'. "
                        "Increase adversarial training epochs or broaden epsilon schedules."
                    )

        # Execution-level recommendation
        if data.execution_summary and data.execution_summary.failed_modules:
            names = ", ".join(r.module_name for r in data.execution_summary.failed_modules)
            recs.append(
                f"Pipeline modules [{names}] encountered errors during this scan. "
                "Review error logs and re-run the pipeline after resolving the issues."
            )

        if not recs:
            recs.append(
                "Conduct comprehensive adversarial vulnerability scanning "
                "across standard perturbation bounds (ε = 4/255, 8/255, 16/255)."
            )
        return recs

    def _s15_final_security_summary(
        self,
        data: ReportData,
        risk_level: str,
        vuln_score: Optional[float],
        recommendations: List[str],
    ) -> Dict[str, Any]:
        baseline_acc = self._extract_baseline_accuracy(data.baseline_performance)
        # Mean adversarial accuracy across attacks
        adv_accs = []
        for atk_info in data.attack_results.values():
            if isinstance(atk_info, dict):
                ev = atk_info.get("evaluation") or atk_info
                acc = ev.get("accuracy") or (ev.get("metrics") or {}).get("accuracy")
                if acc is not None:
                    adv_accs.append(float(acc))
        mean_adv_acc = round(sum(adv_accs) / len(adv_accs), 4) if adv_accs else None

        return {
            "risk_level": risk_level,
            "vulnerability_score": vuln_score,
            "baseline_accuracy": baseline_acc,
            "mean_adversarial_accuracy": mean_adv_acc,
            "attacks_evaluated": list(data.attack_results.keys()),
            "hardening_applied": bool(data.hardening_results),
            "retest_conducted": bool(data.retest_results or data.before_vs_after),
            "total_recommendations": len(recommendations),
            "primary_recommendation": recommendations[0] if recommendations else "",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_baseline_accuracy(baseline: Dict[str, Any]) -> Optional[float]:
        if not baseline:
            return None
        metrics = baseline.get("metrics") or {}
        acc = metrics.get("accuracy") or baseline.get("accuracy")
        return float(acc) if acc is not None else None

    @staticmethod
    def _extract_composite_vulnerability_score(
        vuln_metrics: Dict[str, Any]
    ) -> Optional[float]:
        if not vuln_metrics or not isinstance(vuln_metrics, dict):
            return None
        scores = []
        for _, val in vuln_metrics.items():
            if isinstance(val, dict):
                scoring = val.get("scoring") or val.get("vulnerability_scoring") or {}
                if isinstance(scoring, dict) and "vulnerability_score" in scoring:
                    s = scoring["vulnerability_score"]
                    if s is not None:
                        scores.append(float(s))
                elif "vulnerability_score" in val and val["vulnerability_score"] is not None:
                    scores.append(float(val["vulnerability_score"]))
        return round(sum(scores) / len(scores), 2) if scores else None

    @staticmethod
    def _determine_risk_level(score: Optional[float]) -> str:
        if score is None:
            return "UNKNOWN"
        if score >= 75.0:
            return "CRITICAL"
        if score >= 50.0:
            return "HIGH"
        if score >= 25.0:
            return "MEDIUM"
        return "LOW"

    # ══════════════════════════════════════════════════════════════════════════
    # Report Formatter — all 15 sections
    # ══════════════════════════════════════════════════════════════════════════

    def _format_report(
        self,
        report_id: str,
        scan_id: str,
        timestamp: str,
        report_data: ReportData,
        risk_level: str,
        vuln_score: Optional[float],
        executive_summary: Dict[str, Any],
        mitre_mapping: Dict[str, Any],
        recommendations: List[str],
        final_summary: Dict[str, Any],
    ) -> str:
        W = 72
        SEP = "=" * W
        DIV = "-" * W
        SUB = "·" * W

        def pct(v: Any) -> str:
            if isinstance(v, float) and v <= 1.0:
                return f"{v * 100:.2f}%"
            return str(v) if v is not None else "N/A"

        lines: List[str] = [
            SEP,
            "          ADVERSCAN SECURITY ASSESSMENT REPORT           ",
            SEP,
            f"  Report ID   : {report_id}",
            f"  Scan ID     : {scan_id}",
            f"  Timestamp   : {timestamp}",
            f"  Risk Level  : {risk_level}",
            f"  Vuln. Score : {vuln_score if vuln_score is not None else 'N/A'}",
            SEP,
            "",
        ]

        # ── 1. Executive Summary ───────────────────────────────────────────────
        lines += [
            "1. EXECUTIVE SUMMARY", DIV,
            f"  Scan ID            : {executive_summary.get('scan_id')}",
            f"  Risk Level         : {risk_level}",
            f"  Vulnerability Score: {vuln_score if vuln_score is not None else 'N/A'}",
            f"  Baseline Accuracy  : {pct(executive_summary.get('baseline_accuracy'))}",
            f"  Attacks Evaluated  : {', '.join(executive_summary.get('attacks_evaluated', [])) or 'None'}",
            f"  XAI Enabled        : {executive_summary.get('xai_enabled')}",
            f"  Hardening Applied  : {executive_summary.get('hardening_applied')}",
            f"  Re-Test Conducted  : {executive_summary.get('retest_conducted')}",
            f"  Pipeline Status    : {executive_summary.get('execution_status')}",
            "",
        ]

        # ── 2. Model Information ───────────────────────────────────────────────
        lines += ["2. MODEL INFORMATION", DIV]
        if report_data.model_info:
            for k, v in report_data.model_info.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  No model information available.")
        lines.append("")

        # ── 3. Dataset / Evaluation Configuration ─────────────────────────────
        lines += ["3. DATASET / EVALUATION CONFIGURATION", DIV]
        if report_data.dataset_config:
            for k, v in report_data.dataset_config.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  No dataset configuration recorded.")
        lines.append("")

        # ── 4. Baseline Performance ────────────────────────────────────────────
        lines += ["4. BASELINE PERFORMANCE", DIV]
        if report_data.baseline_performance:
            metrics = report_data.baseline_performance.get("metrics") or {}
            source = metrics if metrics else report_data.baseline_performance
            for k, v in source.items():
                lines.append(f"  - {k}: {pct(v)}")
        else:
            lines.append("  No baseline performance data.")
        lines.append("")

        # ── 5. Adversarial Attack Results ─────────────────────────────────────
        lines += ["5. ADVERSARIAL ATTACK RESULTS", DIV]
        if report_data.attack_results:
            for atk, info in report_data.attack_results.items():
                lines.append(f"  ▶ {atk.upper()}")
                if isinstance(info, dict):
                    params = info.get("parameters") or {}
                    if params:
                        lines.append(f"    Parameters  : {params}")
                    eval_info = info.get("evaluation") or {}
                    m = eval_info.get("metrics") if isinstance(eval_info, dict) else {}
                    if not m and isinstance(info, dict):
                        m = {k: v for k, v in info.items()
                             if k not in ("parameters", "evaluation", "attack_name", "attack_class")}
                    for mk, mv in (m or {}).items():
                        lines.append(f"    {mk:<25}: {pct(mv)}")
                lines.append("")
        else:
            lines.append("  No attack results recorded.")
            lines.append("")

        # ── 6. Vulnerability Assessment ───────────────────────────────────────
        lines += ["6. VULNERABILITY ASSESSMENT", DIV]
        if report_data.vulnerability_metrics:
            for vec, vv in report_data.vulnerability_metrics.items():
                lines.append(f"  ▶ Vector: {vec}")
                if isinstance(vv, dict):
                    assess = vv.get("assessment") or {}
                    scoring = vv.get("scoring") or {}
                    for field_label, field_dict in [("Assessment", assess), ("Scoring", scoring)]:
                        if isinstance(field_dict, dict):
                            for k, v in field_dict.items():
                                if k != "extra_metadata":
                                    lines.append(f"    [{field_label}] {k}: {v}")
                lines.append("")
        else:
            lines.append("  No vulnerability metrics recorded.")
            lines.append("")

        # ── 7. Vulnerability Score & Risk Level ───────────────────────────────
        lines += [
            "7. VULNERABILITY SCORE & RISK LEVEL", DIV,
            f"  Overall Vulnerability Score : {vuln_score if vuln_score is not None else 'N/A'}",
            f"  Risk Level                  : {risk_level}",
            "",
        ]

        # ── 8. MITRE ATLAS Mapping ────────────────────────────────────────────
        lines += ["8. MITRE ATLAS MAPPING", DIV]
        if mitre_mapping:
            for atk, m in mitre_mapping.items():
                lines.append(f"  ▶ {atk.upper()}")
                for k, v in m.items():
                    lines.append(f"    {k:<12}: {v}")
                lines.append("")
        else:
            lines.append("  No MITRE ATLAS mappings available.")
            lines.append("")

        # ── 9. XAI Findings ───────────────────────────────────────────────────
        lines += ["9. XAI FINDINGS", DIV]
        if report_data.xai_findings:
            for tech, xv in report_data.xai_findings.items():
                lines.append(f"  ▶ Technique: {tech}")
                if isinstance(xv, dict):
                    for k, v in xv.items():
                        if k not in ("heatmap", "attribution_map"):
                            lines.append(f"    - {k}: {v}")
                lines.append("")
        else:
            lines.append("  No XAI findings generated.")
            lines.append("")

        # ── 10. Hardening ─────────────────────────────────────────────────────
        lines += ["10. HARDENING", DIV]
        if report_data.hardening_results:
            for k, v in report_data.hardening_results.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  No hardening actions recorded.")
        lines.append("")

        # ── 11. Re-Test Results ────────────────────────────────────────────────
        lines += ["11. RE-TEST RESULTS", DIV]
        if report_data.retest_results:
            for k, v in (report_data.retest_results or {}).items():
                if k not in ("comparisons",):
                    lines.append(f"  - {k}: {v}")
        else:
            lines.append("  No re-test results available.")
        lines.append("")

        # ── 12. Before vs After Comparison ────────────────────────────────────
        lines += ["12. BEFORE VS AFTER COMPARISON", DIV]
        if report_data.before_vs_after:
            for ck, cv in report_data.before_vs_after.items():
                lines.append(f"  ▶ Vector: {ck}")
                if isinstance(cv, dict):
                    for dk, dv in cv.items():
                        lines.append(f"    - {dk}: {dv}")
                lines.append("")
        else:
            lines.append("  No comparison data available.")
            lines.append("")

        # ── 13. Execution Performance ─────────────────────────────────────────
        lines += ["13. EXECUTION PERFORMANCE", DIV]
        es = report_data.execution_summary
        if es:
            lines.append(f"  Run Label    : {es.run_label}")
            lines.append(f"  Started At   : {es.run_timestamp}")
            lines.append(f"  Total Time   : {es.total_elapsed_seconds:.2f}s")
            lines.append(f"  Overall      : {es.overall_status}")
            lines.append("")
            lines.append(f"  {'MODULE':<30} {'STATUS':<12} {'TIME':>7}")
            lines.append("  " + SUB)
            for rec in es.modules:
                icon = {"SUCCESS": "✅", "FAILED": "❌", "SKIPPED": "⏭"}.get(rec.status, "❓")
                lines.append(
                    f"  {rec.module_name:<30} {icon} {rec.status:<10} {rec.elapsed_seconds:>6.2f}s"
                )
                for mk, mv in (rec.metrics or {}).items():
                    lines.append(f"    {'':>30}  └ {mk}: {pct(mv) if isinstance(mv, float) else mv}")
            lines.append("")
        else:
            lines.append("  No execution performance data captured.")
            lines.append("")

        # ── 14. Recommendations ────────────────────────────────────────────────
        lines += ["14. RECOMMENDATIONS", DIV]
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  [{i:02d}] {rec}")
        lines.append("")

        # ── 15. Final Security Summary ─────────────────────────────────────────
        lines += ["15. FINAL SECURITY SUMMARY", DIV]
        for k, v in final_summary.items():
            lines.append(f"  - {k}: {pct(v) if isinstance(v, float) else v}")
        lines.append("")

        lines.append(SEP)
        lines.append(f"  Generated by AdverScan — {timestamp}")
        lines.append(SEP)

        return "\n".join(lines)
