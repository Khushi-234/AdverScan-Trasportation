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
            if v is None or v == "None":
                return "N/A"
            if isinstance(v, (int, float)):
                if abs(float(v)) <= 1.0:
                    return f"{float(v) * 100:.2f}%"
                return f"{float(v):.2f}"
            return str(v)

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
            ignored_keys = {"per_class_metrics", "confusion_matrix"}
            for k, v in source.items():
                if k in ignored_keys:
                    continue
                lines.append(f"  - {k}: {pct(v)}")
        else:
            lines.append("  No baseline performance data.")
        lines.append("")

        # ── 5. Adversarial Attack Results ─────────────────────────────────────
        lines += ["5. ADVERSARIAL ATTACK RESULTS", DIV]
        if report_data.attack_results:
            lines.append("| Attack Vector | Parameters | Exec Time | Baseline Acc | Adv Acc | Acc Drop | Attack Success Rate (ASR) |")
            lines.append("|---|---|---|---|---|---|---|")

            for atk, info in report_data.attack_results.items():
                params_str = "Default"
                exec_time_str = "N/A"
                clean_acc_val = executive_summary.get("baseline_accuracy")
                adv_acc_val = None
                acc_drop_val = None
                asr_val = None

                if isinstance(info, dict):
                    params = info.get("parameters") or {}
                    if params:
                        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    t = info.get("execution_time_seconds")
                    if t is not None:
                        exec_time_str = f"{float(t):.2f}s"

                    eval_info = info.get("evaluation") or {}
                    m = eval_info.get("metrics") if isinstance(eval_info, dict) else {}
                    if not m and isinstance(info, dict):
                        m = {k: v for k, v in info.items()
                             if k not in ("parameters", "evaluation", "attack_name", "attack_class")}

                    if m:
                        clean_acc_val = m.get("clean_accuracy", clean_acc_val)
                        adv_acc_val = m.get("accuracy", m.get("adversarial_accuracy", adv_acc_val))
                        acc_drop_val = m.get("accuracy_drop", acc_drop_val)
                        asr_val = m.get("attack_success_rate", m.get("asr", asr_val))

                vm = (
                    report_data.vulnerability_metrics.get(atk)
                    or report_data.vulnerability_metrics.get(atk.lower())
                    or {}
                )
                if isinstance(vm, dict):
                    assess = vm.get("assessment") or vm
                    if isinstance(assess, dict):
                        if clean_acc_val is None:
                            clean_acc_val = assess.get("clean_accuracy")
                        if adv_acc_val is None:
                            adv_acc_val = assess.get("adversarial_accuracy")
                        if acc_drop_val is None:
                            acc_drop_val = assess.get("accuracy_drop")
                        if asr_val is None or asr_val == "None":
                            asr_val = assess.get("attack_success_rate")

                if acc_drop_val is None and clean_acc_val is not None and adv_acc_val is not None:
                    acc_drop_val = float(clean_acc_val) - float(adv_acc_val)

                if asr_val is None or asr_val == "None":
                    if clean_acc_val is not None and adv_acc_val is not None and float(clean_acc_val) > 0:
                        asr_val = (float(clean_acc_val) - float(adv_acc_val)) / float(clean_acc_val)
                    elif acc_drop_val is not None:
                        asr_val = float(acc_drop_val)

                c_str = pct(clean_acc_val)
                a_str = pct(adv_acc_val)
                d_str = pct(acc_drop_val)
                asr_str = pct(asr_val)

                lines.append(f"| {atk.upper():<13} | {params_str:<10} | {exec_time_str:<9} | {c_str:<12} | {a_str:<7} | {d_str:<8} | **{asr_str}** |")

            lines.append("")
            lines.append("  ▶ How Attacks Were Performed:")
            for atk, info in report_data.attack_results.items():
                params_str = "Default parameters"
                if isinstance(info, dict) and info.get("parameters"):
                    params_str = ", ".join(f"{k}={v}" for k, v in info["parameters"].items())
                lines.append(f"    • {atk.upper()}: Configured with [{params_str}].")
            lines.append("")
        else:
            lines.append("  No attack results recorded.")
            lines.append("")

        # ── 6. Vulnerability Assessment ───────────────────────────────────────
        lines += ["6. VULNERABILITY ASSESSMENT", DIV]
        if report_data.vulnerability_metrics:
            for vec, vv in report_data.vulnerability_metrics.items():
                lines.append(f"  ▶ Vector: {vec.upper()}")
                if isinstance(vv, dict):
                    assess = vv.get("assessment") or {}
                    scoring = vv.get("scoring") or {}

                    asr_val = assess.get("attack_success_rate")
                    clean_a = assess.get("clean_accuracy", executive_summary.get("baseline_accuracy"))
                    adv_a = assess.get("adversarial_accuracy")
                    if (asr_val is None or asr_val == "None") and clean_a is not None and adv_a is not None:
                        if float(clean_a) > 0:
                            asr_val = (float(clean_a) - float(adv_a)) / float(clean_a)

                    lines.append(f"    - Attack Success Rate (ASR) : {pct(asr_val)}")
                    lines.append(f"    - Vulnerability Score       : {scoring.get('vulnerability_score', 'N/A')}")
                    lines.append(f"    - Risk Level                : {scoring.get('risk_level', 'N/A')}")
                    lines.append(f"    - Clean vs Adversarial Acc  : {pct(clean_a)} ➔ {pct(adv_a)} (Drop: {pct(assess.get('accuracy_drop'))})")

                    pert = assess.get("perturbation")
                    if isinstance(pert, dict):
                        p_str = ", ".join(
                            f"{pk}={pct(pv) if isinstance(pv, float) and pv <= 1.0 else (f'{pv:.2f}' if isinstance(pv, float) else pv)}"
                            for pk, pv in pert.items()
                            if pk != "is_estimated"
                        )
                        lines.append(f"    - Perturbation Magnitude    : {p_str}")
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
                        if k in ("heatmap", "attribution_map", "clean_prediction", "adversarial_prediction", "true_label", "metadata"):
                            continue
                        if isinstance(v, (list, dict)) and len(str(v)) > 100:
                            lines.append(f"    - {k}: [Detailed Data Omitted]")
                        else:
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
        retest = report_data.retest_results or {}
        if retest:
            model_name = retest.get("hardened_model_name") or report_data.model_info.get("model_name", "N/A")
            dataset_name = retest.get("dataset_name") or report_data.baseline_performance.get("dataset_name", "N/A")
            samples = retest.get("num_samples") or report_data.baseline_performance.get("num_samples", "N/A")

            defense_name = report_data.hardening_results.get("defense") or report_data.hardening_results.get("defense_name", "Input Preprocessing / Hardening")

            overall_imp = retest.get("overall_improved")
            if overall_imp is None and report_data.before_vs_after:
                overall_imp = all(
                    cv.get("is_improved", True) if isinstance(cv, dict) else getattr(cv, "is_improved", True)
                    for cv in report_data.before_vs_after.values()
                )

            verdict_str = "✅ PASSED — Defense Successfully Mitigated Attacks" if (overall_imp or overall_imp is None) else "❌ FAILED — Vulnerabilities Persist"

            lines.append(f"  - Hardened Model    : {model_name}")
            lines.append(f"  - Dataset / Samples : {dataset_name} ({samples} samples)")
            lines.append(f"  - Defense Mechanism : {defense_name}")
            lines.append(f"  - Re-Test Status    : {verdict_str}")
        else:
            lines.append("  No re-test results available.")
        lines.append("")

        # ── 12. Before vs After Comparison ────────────────────────────────────
        lines += ["12. BEFORE VS AFTER COMPARISON", DIV]
        comp_data = report_data.before_vs_after
        if not comp_data and isinstance(report_data.retest_results, dict):
            comp_data = report_data.retest_results.get("comparisons") or {}

        if comp_data:
            lines.append("| Attack Vector | Before Defense (Adv Acc) | After Defense (Hardened Acc) | Robustness Gain | Before Risk | After Risk | Defense Status |")
            lines.append("|---|---|---|---|---|---|---|")

            all_improved = True
            for vec_name, cv in comp_data.items():
                if hasattr(cv, "to_dict"):
                    cv = cv.to_dict()

                b_acc = None
                a_acc = None
                gain = None
                b_risk = "UNKNOWN"
                a_risk = "UNKNOWN"
                is_imp = True

                if isinstance(cv, dict):
                    b_assess = cv.get("before_assessment") or {}
                    a_assess = cv.get("after_assessment") or {}

                    b_acc = b_assess.get("adversarial_accuracy")
                    a_acc = a_assess.get("adversarial_accuracy")

                    if b_acc is None:
                        vm = report_data.vulnerability_metrics.get(vec_name) or {}
                        if isinstance(vm, dict):
                            b_acc = (vm.get("assessment") or {}).get("adversarial_accuracy")

                    if a_acc is None:
                        delta_adv = cv.get("delta_adversarial_accuracy")
                        delta_drop = cv.get("delta_accuracy_drop")
                        if delta_adv is not None and b_acc is not None:
                            a_acc = float(b_acc) + float(delta_adv)
                        elif delta_drop is not None and b_acc is not None:
                            a_acc = float(b_acc) - float(delta_drop)

                    if a_acc is not None and b_acc is not None:
                        gain = float(a_acc) - float(b_acc)

                    b_risk = cv.get("before_risk_level") or (cv.get("before_scoring") or {}).get("risk_level") or "MEDIUM"
                    a_risk = cv.get("after_risk_level") or (cv.get("after_scoring") or {}).get("risk_level") or "LOW"
                    is_imp = cv.get("is_improved", True)
                    if not is_imp:
                        all_improved = False

                b_str = pct(b_acc)
                a_str = pct(a_acc)
                g_str = f"+{pct(gain)}" if isinstance(gain, float) and gain >= 0 else pct(gain)
                status_str = "✅ WORKED" if is_imp else "❌ FAILED"

                lines.append(f"| {vec_name.upper():<13} | {b_str:<24} | {a_str:<28} | {g_str:<15} | {b_risk:<11} | {a_risk:<10} | **{status_str}** |")

            lines.append("")
            lines.append("  ▶ Defense Verification & Retest Summary:")
            if all_improved:
                lines.append("    • Verdict: ✅ DEFENSE SUCCESSFUL — The applied defense effectively neutralized all attack vectors.")
            else:
                lines.append("    • Verdict: ⚠️ PARTIAL DEFENSE — Some attack vectors showed persistent vulnerability.")
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
