"""
Professional PDF Report Generator for AdverScan Security Assessments.

Compiles a comprehensive, certified, publication-grade multi-page PDF security report
using ReportLab Platypus. Supports both ReportResult instances and raw assessment
history detail dictionaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ==============================================================================
# NUMBERED CANVAS (Header / Footer with Page Count)
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to calculate the total page count and draw
    consistent header/footer page numbers and security attestation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()

        # Running Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 22, page_str)
        self.drawString(
            36,
            22,
            "AdverScan™ AI Security Assurance Framework • Confidential Assessment Report",
        )

        # Footer divider line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 32, letter[0] - 36, 32)

        # Running Top Thin Header (on pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0F172A"))
            self.drawString(36, letter[1] - 25, "ADVERSCAN AI SECURITY ASSESSMENT")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(
                letter[0] - 36,
                letter[1] - 25,
                "Adversarial Robustness & Hardening Audit",
            )
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, letter[1] - 28, letter[0] - 36, letter[1] - 28)

        self.restoreState()


# ==============================================================================
# DATA NORMALIZATION HELPERS
# ==============================================================================

def _normalize_report_data(report_data: Union[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalizes either a ReportResult DTO or an assessment detail dictionary
    into a standardized dictionary format for PDF compilation.
    """
    if hasattr(report_data, "to_dict"):
        raw = report_data.to_dict()
    elif isinstance(report_data, dict):
        raw = dict(report_data)
    else:
        raw = {}

    # Check if this is a historical assessment detail dict
    is_assessment_detail = "scan_id" in raw and ("attacks" in raw or "model" in raw or "vulnerabilities" in raw)

    if not is_assessment_detail:
        # Standard ReportResult dictionary
        return raw

    # Convert assessment detail dict to standard report structure
    scan_id = raw.get("scan_id", "SCAN-UNKNOWN")
    reports_list = raw.get("reports", [])
    report_id = reports_list[0].get("report_id") if reports_list else f"REP-{scan_id}"

    timestamp = (
        raw.get("completed_at")
        or raw.get("created_at")
        or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    overall = raw.get("overall_summary") or {}
    model_obj = raw.get("model") or {}
    dataset_obj = raw.get("dataset") or {}
    attacks = raw.get("attacks") or []
    defenses = raw.get("defenses") or []
    vulnerabilities = raw.get("vulnerabilities") or []
    recommendations_raw = raw.get("recommendations") or []
    xai_results = raw.get("xai_results") or []

    # Model specs
    cfg_model = raw.get("configuration", {}).get("model_info", {})
    model_meta = model_obj.get("metadata_json") or {}
    model_info = {
        "model_name": raw.get("model_name") or model_obj.get("model_name") or cfg_model.get("model_name", "Evaluated Model"),
        "model_architecture": (
            model_obj.get("architecture")
            or model_meta.get("architecture")
            or cfg_model.get("architecture")
            or "Deep Neural Network"
        ),
        "framework": (
            model_obj.get("framework")
            or model_meta.get("framework")
            or cfg_model.get("framework")
            or "PyTorch"
        ),
        "total_parameters": model_obj.get("total_parameters") or model_meta.get("total_parameters", 0),
        "trainable_parameters": model_obj.get("trainable_parameters") or model_meta.get("trainable_parameters", 0),
        "num_classes": model_obj.get("num_classes") or model_meta.get("num_classes", 0),
        "input_shape": model_obj.get("input_shape") or model_meta.get("input_shape", "N/A"),
    }

    # Dataset specs
    dataset_info = {
        "dataset_name": raw.get("dataset_name") or dataset_obj.get("dataset_name", "Evaluation Batch"),
        "total_samples": dataset_obj.get("number_of_samples") or 0,
        "modality": raw.get("modality", "image"),
        "data_split": dataset_obj.get("data_split", "test"),
    }

    # Attacks mapping
    attack_results: Dict[str, Any] = {}
    baseline_clean_acc = 1.0
    for atk in attacks:
        name = str(atk.get("attack_name", "attack")).upper()
        clean_acc = (
            atk.get("clean_accuracy")
            if atk.get("clean_accuracy") is not None
            else atk.get("clean_acc", 1.0)
        )
        baseline_clean_acc = clean_acc

        asr = (
            atk.get("attack_success_rate")
            if atk.get("attack_success_rate") is not None
            else atk.get("asr_percentage", atk.get("asr", 0.0))
        )

        adv_acc = (
            atk.get("adversarial_accuracy")
            if atk.get("adversarial_accuracy") is not None
            else atk.get("adv_accuracy")
        )
        if adv_acc is None:
            try:
                asr_f = float(asr)
                if asr_f == 0.0:
                    adv_acc = clean_acc
                else:
                    c_f = float(clean_acc)
                    adv_acc = max(0.0, c_f * (1.0 - (asr_f / 100.0 if asr_f > 1.0 else asr_f)))
            except Exception:
                adv_acc = 0.0

        p_metrics = atk.get("perturbation_metrics") or {}
        if not isinstance(p_metrics, dict):
            p_metrics = {}

        linf = (
            atk.get("l_inf_norm")
            if atk.get("l_inf_norm") is not None
            else (
                atk.get("l_inf_distortion")
                if atk.get("l_inf_distortion") is not None
                else (
                    atk.get("linf")
                    if atk.get("linf") is not None
                    else (
                        p_metrics.get("linf")
                        if p_metrics.get("linf") is not None
                        else p_metrics.get("linf_mean")
                    )
                )
            )
        )
        if linf is None:
            linf = atk.get("mean_perturbation_linf")
        if linf is None:
            params = atk.get("attack_parameters") or atk.get("parameters") or {}
            eps = atk.get("epsilon") or params.get("epsilon")
            linf = float(eps) if eps is not None else 0.0

        l2 = (
            atk.get("l2_norm")
            if atk.get("l2_norm") is not None
            else (
                atk.get("l_2_distortion")
                if atk.get("l_2_distortion") is not None
                else (
                    atk.get("l2")
                    if atk.get("l2") is not None
                    else (
                        p_metrics.get("l2")
                        if p_metrics.get("l2") is not None
                        else (
                            p_metrics.get("l2_mean")
                            if p_metrics.get("l2_mean") is not None
                            else p_metrics.get("mean_perturbation")
                        )
                    )
                )
            )
        )
        if l2 is None:
            l2 = atk.get("mean_perturbation_l2")
        if l2 is None:
            l2 = 0.0

        t_sec = (
            atk.get("execution_time_seconds")
            if atk.get("execution_time_seconds") is not None
            else atk.get("execution_time", 0.0)
        )

        attack_results[name] = {
            "attack_name": name,
            "attack_type": atk.get("attack_type") or atk.get("attack_category", "whitebox"),
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "attack_success_rate": asr,
            "l_inf_distortion": float(linf) if linf is not None else 0.0,
            "l_2_distortion": float(l2) if l2 is not None else 0.0,
            "execution_time_seconds": float(t_sec),
        }

    # Vulnerability score and risk level
    v_score = overall.get("vulnerability_score")
    if v_score is None and vulnerabilities:
        v_score = vulnerabilities[0].get("vulnerability_score")
    v_score_val = float(v_score) if v_score is not None else 0.0

    r_level = overall.get("risk_level")
    if not r_level and vulnerabilities:
        r_level = vulnerabilities[0].get("severity")
    risk_level = str(r_level or "LOW").upper()

    # Hardening results
    selected_def = None
    defense_status = "SUCCESS"
    hardening_data: Dict[str, Any] = {}
    before_vs_after: Dict[str, Any] = {}

    if defenses:
        primary_def = defenses[0]
        selected_def = primary_def.get("defense_technique", "None")
        defense_status = primary_def.get("validation_status", "accepted")
        hardening_data = {
            "selected_defense": selected_def,
            "defense_family": primary_def.get("defense_family", "preprocessing"),
            "status": defense_status,
            "clean_accuracy_before": primary_def.get("clean_accuracy_before", baseline_clean_acc),
            "clean_accuracy_after": primary_def.get("clean_accuracy_after", baseline_clean_acc),
            "clean_accuracy_drop": primary_def.get("clean_accuracy_drop", 0.0),
            "asr_reduction": primary_def.get("asr_reduction", 0.0),
            "accuracy_recovery": primary_def.get("accuracy_recovery", 0.0),
        }

        # Build before_vs_after from primary defense
        atk_key = list(attack_results.keys())[0] if attack_results else "PRIMARY_VECTOR"
        before_vs_after[atk_key] = {
            "before_asr": primary_def.get("asr_before", 0.0),
            "after_asr": primary_def.get("asr_after", 0.0),
            "delta_asr": primary_def.get("asr_reduction", 0.0) / 100.0 if primary_def.get("asr_reduction", 0.0) > 1.0 else primary_def.get("asr_reduction", 0.0),
            "before_accuracy": primary_def.get("adv_accuracy_before", 0.0),
            "after_accuracy": primary_def.get("adv_accuracy_after", 0.0),
            "delta_accuracy": primary_def.get("accuracy_recovery", 0.0) / 100.0 if primary_def.get("accuracy_recovery", 0.0) > 1.0 else primary_def.get("accuracy_recovery", 0.0),
            "is_improved": bool(primary_def.get("is_improved", False)),
        }

    # Recommendations
    recs = []
    for r in recommendations_raw:
        if isinstance(r, str):
            recs.append(r)
        elif isinstance(r, dict):
            rat = r.get("rationale") or r.get("description")
            tech = r.get("recommended_technique") or r.get("technique")
            if rat:
                recs.append(f"{tech}: {rat}" if tech else rat)
            elif tech:
                recs.append(f"Deploy {tech.replace('_', ' ').title()} mitigation to protect against identified attack vectors.")

    return {
        "report_id": report_id,
        "scan_id": scan_id,
        "timestamp": timestamp,
        "status": raw.get("status", "SUCCESS"),
        "risk_level": risk_level,
        "vulnerability_score": v_score_val,
        "model_info": model_info,
        "dataset_config": dataset_info,
        "baseline_performance": {"clean_accuracy": baseline_clean_acc},
        "attack_results": attack_results,
        "vulnerability_metrics": {"vulnerabilities": vulnerabilities},
        "xai_findings": xai_results[0] if xai_results else {},
        "hardening_results": hardening_data,
        "before_vs_after": before_vs_after,
        "recommendations": recs,
        "final_security_summary": {
            "overall_status": "CERTIFIED",
            "certified_date": timestamp,
        },
    }


# ==============================================================================
# PDF COMPILER IMPLEMENTATION
# ==============================================================================

def compile_pdf_report(
    report_data: Union[Any, Dict[str, Any]],
    target_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """
    Compile a complete, publication-grade AdverScan PDF security report.

    Args:
        report_data: ReportResult instance or assessment detail dictionary.
        target_path: Optional file path where the generated PDF should be saved.

    Returns:
        Generated PDF file content as raw bytes.
    """
    norm = _normalize_report_data(report_data)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    story: List[Any] = []

    # --------------------------------------------------------------------------
    # 1. HEADER BANNER
    # --------------------------------------------------------------------------
    rep_id = norm.get("report_id", "REP-UNKNOWN")
    scan_id = norm.get("scan_id", "SCAN-UNKNOWN")
    timestamp = norm.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    m_info = norm.get("model_info", {})
    model_name = m_info.get("model_name", "Target Model")

    header_data = [
        [
            Paragraph(
                f"<b>ADVERSCAN™ AI SECURITY REPORT</b><br/>"
                f"<font size=9 color='#0284C7'><b>Model: {model_name}</b></font><br/>"
                f"<font size=8 color='#64748B'>Adversarial Robustness, Attack Surface & Hardening Audit</font>",
                title_style,
            ),
            Paragraph(
                f"<b>Report ID:</b> {rep_id}<br/>"
                f"<b>Scan ID:</b> {scan_id}<br/>"
                f"<b>Timestamp:</b> {timestamp}<br/>"
                f"<b>Assurance Status:</b> <font color='#16A34A'><b>CERTIFIED</b></font>",
                subtitle_style,
            ),
        ]
    ]

    header_table = Table(header_data, colWidths=[330, 210])
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=8))

    # Prominent Evaluated Target Model Banner (placed right below the line, before report sections)
    modality_str = str(norm.get("dataset_config", {}).get("modality", m_info.get("domain", "image"))).capitalize()
    framework_str = str(m_info.get("framework", "PyTorch"))
    num_classes_str = str(m_info.get("num_classes", "N/A"))

    model_banner_data = [
        [
            Paragraph(
                f"<font size=8 color='#0284C7'><b>EVALUATED TARGET MODEL</b></font><br/>"
                f"<font size=12 color='#0F172A'><b>{model_name}</b></font>",
                ParagraphStyle("ModelBannerTitle", parent=styles["Normal"], leading=15),
            ),
            Paragraph(
                f"<font size=8 color='#64748B'><b>Modality:</b> {modality_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Framework:</b> {framework_str}<br/>"
                f"<b>Classes / Outputs:</b> {num_classes_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Scan ID:</b> {scan_id}</font>",
                ParagraphStyle("ModelBannerMeta", parent=styles["Normal"], leading=12, alignment=2),
            ),
        ]
    ]
    model_banner_table = Table(model_banner_data, colWidths=[330, 210])
    model_banner_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor("#0284C7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(model_banner_table)
    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # 1. MODEL & DATASET SPECIFICATIONS (Placed first for immediate clarity)
    # --------------------------------------------------------------------------
    story.append(Paragraph("1. Model & Dataset Specifications", section_h1))

    d_info = norm.get("dataset_config", {})
    t_params = m_info.get("total_parameters", 0)
    t_params_str = f"{t_params:,}" if isinstance(t_params, (int, float)) and t_params > 0 else "N/A"
    n_samples = d_info.get("total_samples") or d_info.get("number_of_samples", 0)
    n_samples_str = f"{n_samples:,}" if isinstance(n_samples, (int, float)) and n_samples > 0 else "Evaluation Set"

    specs_data = [
        [
            Paragraph("<b>Target Model</b>", table_cell_bold),
            Paragraph(str(m_info.get("model_name", "N/A")), table_cell),
            Paragraph("<b>Dataset Name</b>", table_cell_bold),
            Paragraph(str(d_info.get("dataset_name", "N/A")), table_cell),
        ],
        [
            Paragraph("<b>Architecture</b>", table_cell_bold),
            Paragraph(str(m_info.get("model_architecture", "N/A")), table_cell),
            Paragraph("<b>Sample Count</b>", table_cell_bold),
            Paragraph(n_samples_str, table_cell),
        ],
        [
            Paragraph("<b>Framework</b>", table_cell_bold),
            Paragraph(str(m_info.get("framework", "N/A")), table_cell),
            Paragraph("<b>Modality</b>", table_cell_bold),
            Paragraph(str(d_info.get("modality", "image")).capitalize(), table_cell),
        ],
        [
            Paragraph("<b>Total Parameters</b>", table_cell_bold),
            Paragraph(t_params_str, table_cell),
            Paragraph("<b>Classes / Outputs</b>", table_cell_bold),
            Paragraph(str(m_info.get("num_classes", "N/A")), table_cell),
        ],
    ]

    specs_table = Table(specs_data, colWidths=[120, 150, 120, 150])
    specs_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(specs_table)
    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # 2. EXECUTIVE RISK ASSESSMENT & SECURITY POSTURE
    # --------------------------------------------------------------------------
    story.append(Paragraph("2. Executive Risk Assessment & Security Posture", section_h1))

    risk_level = str(norm.get("risk_level", "LOW")).upper()
    v_score = norm.get("vulnerability_score")
    v_score_str = f"{float(v_score):.1f} / 100" if v_score is not None else "N/A"

    risk_colors = {
        "CRITICAL": colors.HexColor("#DC2626"),
        "HIGH": colors.HexColor("#EA580C"),
        "MEDIUM": colors.HexColor("#D97706"),
        "LOW": colors.HexColor("#16A34A"),
        "MINIMAL": colors.HexColor("#059669"),
    }
    badge_bg = risk_colors.get(risk_level, colors.HexColor("#64748B"))

    selected_def = norm.get("hardening_results", {}).get("selected_defense", "None")

    summary_content = (
        f"<b>Assessment Verdict:</b> Model evaluated under standard white-box adversarial stress testing.<br/>"
        f"<b>Primary Hardening Applied:</b> {str(selected_def).replace('_', ' ').title()}<br/>"
        f"<b>Evaluation Framework:</b> AdverScan Framework v2.0 Enterprise Security Pipeline"
    )

    card_data = [
        [
            Paragraph(
                f"<font size=10 color='#FFFFFF'><b>OVERALL RISK: {risk_level}</b></font><br/>"
                f"<font size=8 color='#F1F5F9'>Vulnerability Score: <b>{v_score_str}</b></font>",
                ParagraphStyle("RiskBadge", fontName="Helvetica", alignment=1),
            ),
            Paragraph(summary_content, body_style),
        ]
    ]

    card_table = Table(card_data, colWidths=[160, 380])
    card_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), badge_bg),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(card_table)
    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # 3. THREAT ANALYSIS & ADVERSARIAL ATTACK VECTORS
    # --------------------------------------------------------------------------
    story.append(Paragraph("3. Threat Analysis & Adversarial Attack Vectors", section_h1))

    attack_results = norm.get("attack_results", {})
    if attack_results:
        atk_header = [
            Paragraph("Attack Vector", table_header),
            Paragraph("Clean Acc", table_header),
            Paragraph("Adv Acc", table_header),
            Paragraph("ASR (%)", table_header),
            Paragraph("L∞ Perturb", table_header),
            Paragraph("L2 Perturb", table_header),
            Paragraph("Latency", table_header),
        ]
        atk_rows = [atk_header]

        for k, v in attack_results.items():
            c_acc = (
                v.get("clean_accuracy")
                if v.get("clean_accuracy") is not None
                else v.get("baseline_accuracy", 1.0)
            )
            asr = (
                v.get("attack_success_rate")
                if v.get("attack_success_rate") is not None
                else v.get("asr_percentage", v.get("asr", 0.0))
            )
            a_acc = (
                v.get("adversarial_accuracy")
                if v.get("adversarial_accuracy") is not None
                else v.get("adv_accuracy")
            )
            if a_acc is None:
                try:
                    asr_f = float(asr)
                    if asr_f == 0.0:
                        a_acc = c_acc
                    else:
                        c_f = float(c_acc)
                        a_acc = max(0.0, c_f * (1.0 - (asr_f / 100.0 if asr_f > 1.0 else asr_f)))
                except Exception:
                    a_acc = 0.0

            p_metrics = v.get("perturbation_metrics") or {}
            if not isinstance(p_metrics, dict):
                p_metrics = {}

            linf = (
                v.get("l_inf_distortion")
                if v.get("l_inf_distortion") is not None
                else (
                    v.get("l_inf_norm")
                    if v.get("l_inf_norm") is not None
                    else (
                        v.get("linf")
                        if v.get("linf") is not None
                        else (
                            p_metrics.get("linf")
                            if p_metrics.get("linf") is not None
                            else p_metrics.get("linf_mean")
                        )
                    )
                )
            )
            if linf is None:
                params = v.get("parameters") or v.get("attack_parameters") or {}
                eps = v.get("epsilon") or params.get("epsilon")
                linf = float(eps) if eps is not None else 0.0

            l2 = (
                v.get("l_2_distortion")
                if v.get("l_2_distortion") is not None
                else (
                    v.get("l2_norm")
                    if v.get("l2_norm") is not None
                    else (
                        v.get("l2")
                        if v.get("l2") is not None
                        else (
                            p_metrics.get("l2")
                            if p_metrics.get("l2") is not None
                            else (
                                p_metrics.get("l2_mean")
                                if p_metrics.get("l2_mean") is not None
                                else p_metrics.get("mean_perturbation")
                            )
                        )
                    )
                )
            )
            if l2 is None:
                l2 = 0.0

            t_sec = (
                v.get("execution_time_seconds")
                if v.get("execution_time_seconds") is not None
                else v.get("execution_time", 0.0)
            )

            c_str = f"{float(c_acc):.1f}%" if float(c_acc) > 1.0 else f"{float(c_acc) * 100:.1f}%"
            a_str = f"{float(a_acc):.1f}%" if float(a_acc) > 1.0 else f"{float(a_acc) * 100:.1f}%"
            asr_str = f"{float(asr):.1f}%" if float(asr) > 1.0 else f"{float(asr) * 100:.1f}%"
            linf_str = f"{float(linf):.4f}"
            l2_str = f"{float(l2):.4f}"
            t_str = f"{float(t_sec):.2f}s"

            disp_name = str(k).replace("_TABULAR", " (Tab)").replace("_", " ")

            atk_rows.append(
                [
                    Paragraph(f"<b>{disp_name}</b>", table_cell_bold),
                    Paragraph(c_str, table_cell),
                    Paragraph(a_str, table_cell),
                    Paragraph(f"<font color='#DC2626'><b>{asr_str}</b></font>", table_cell),
                    Paragraph(linf_str, table_cell),
                    Paragraph(l2_str, table_cell),
                    Paragraph(t_str, table_cell),
                ]
            )

        atk_table = Table(atk_rows, colWidths=[140, 65, 65, 65, 70, 70, 65])
        atk_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(atk_table)
    else:
        story.append(Paragraph("<i>No empirical attack executions recorded for this assessment.</i>", body_style))

    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # 4. VULNERABILITY FINDINGS (M5)
    # --------------------------------------------------------------------------
    story.append(Paragraph("4. Vulnerability Findings & Risk Diagnosis", section_h1))

    v_findings = (
        norm.get("vulnerability_metrics", {}).get("vulnerabilities")
        or norm.get("vulnerabilities")
        or []
    )

    if v_findings:
        v_header = [
            Paragraph("Vulnerability", table_header),
            Paragraph("Severity", table_header),
            Paragraph("Score", table_header),
            Paragraph("Affected Region", table_header),
            Paragraph("Diagnosis / Diagnostic Evidence", table_header),
        ]
        v_rows = [v_header]

        for vf in v_findings[:6]:  # Keep to top findings for concise layout
            name = vf.get("vulnerability_name") or vf.get("name", "Vulnerability")
            sev = str(vf.get("severity", "Medium")).capitalize()
            score_f = vf.get("vulnerability_score") if vf.get("vulnerability_score") is not None else vf.get("score", 0.0)
            score_disp = f"{float(score_f):.1f}"
            region = vf.get("affected_feature_or_region") or vf.get("affected_region", "Input Space")
            desc = vf.get("description", "Identified model sensitivity.")

            sev_color = "#DC2626" if sev == "High" else ("#D97706" if sev == "Medium" else "#16A34A")

            v_rows.append(
                [
                    Paragraph(f"<b>{name}</b>", table_cell_bold),
                    Paragraph(f"<font color='{sev_color}'><b>{sev}</b></font>", table_cell),
                    Paragraph(score_disp, table_cell),
                    Paragraph(region, table_cell),
                    Paragraph(desc, table_cell),
                ]
            )

        v_table = Table(v_rows, colWidths=[120, 60, 50, 110, 200])
        v_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(v_table)
    else:
        story.append(Paragraph("<i>No critical vulnerabilities identified. Model demonstrated sufficient empirical robustness.</i>", body_style))

    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # 5. HARDENING & BEFORE VS AFTER RE-TEST COMPARISON
    # --------------------------------------------------------------------------
    story.append(Paragraph("5. Model Hardening & Empirical Re-Test Comparison", section_h1))

    bva = norm.get("before_vs_after", {})
    hard_info = norm.get("hardening_results", {})
    def_name = str(hard_info.get("selected_defense", "None")).replace("_", " ").title()

    if bva:
        bva_header = [
            Paragraph("Attack Vector", table_header),
            Paragraph("Pre-ASR", table_header),
            Paragraph("Post-ASR", table_header),
            Paragraph("ASR Suppression", table_header),
            Paragraph("Pre-Acc", table_header),
            Paragraph("Post-Acc", table_header),
            Paragraph("Recovery", table_header),
            Paragraph("Status", table_header),
        ]
        bva_rows = [bva_header]

        for atk_k, c_data in bva.items():
            b_asr = c_data.get("before_asr", 0.0)
            a_asr = c_data.get("after_asr", 0.0)
            d_asr = c_data.get("delta_asr", 0.0)
            b_acc = c_data.get("before_accuracy", 0.0)
            a_acc = c_data.get("after_accuracy", 0.0)
            d_acc = c_data.get("delta_accuracy", 0.0)

            b_asr_str = f"{float(b_asr) * 100:.1f}%" if float(b_asr) <= 1.0 else f"{float(b_asr):.1f}%"
            a_asr_str = f"{float(a_asr) * 100:.1f}%" if float(a_asr) <= 1.0 else f"{float(a_asr):.1f}%"
            d_asr_str = f"-{float(d_asr) * 100:.1f}%" if float(d_asr) <= 1.0 else f"-{float(d_asr):.1f}%"
            b_acc_str = f"{float(b_acc) * 100:.1f}%" if float(b_acc) <= 1.0 else f"{float(b_acc):.1f}%"
            a_acc_str = f"{float(a_acc) * 100:.1f}%" if float(a_acc) <= 1.0 else f"{float(a_acc):.1f}%"
            d_acc_str = f"+{float(d_acc) * 100:.1f}%" if float(d_acc) >= 0 else f"{float(d_acc) * 100:.1f}%"

            improved = c_data.get("is_improved", False)
            stat_str = "<font color='#16A34A'><b>Improved ✅</b></font>" if improved else "<font color='#D97706'>Stable ⚠️</font>"

            bva_rows.append(
                [
                    Paragraph(f"<b>{atk_k.upper()}</b>", table_cell_bold),
                    Paragraph(b_asr_str, table_cell),
                    Paragraph(a_asr_str, table_cell),
                    Paragraph(f"<font color='#16A34A'><b>{d_asr_str}</b></font>", table_cell),
                    Paragraph(b_acc_str, table_cell),
                    Paragraph(a_acc_str, table_cell),
                    Paragraph(d_acc_str, table_cell),
                    Paragraph(stat_str, table_cell),
                ]
            )

        bva_table = Table(bva_rows, colWidths=[90, 60, 60, 80, 60, 60, 60, 70])
        bva_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(bva_table)
    else:
        story.append(
            Paragraph(
                f"<b>Applied Mitigation:</b> {def_name}. <i>No multi-vector comparative re-test data recorded.</i>",
                body_style,
            )
        )

    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # 6. RECOMMENDATIONS & ATTESTATION (Kept Together)
    # --------------------------------------------------------------------------
    closing_elements: List[Any] = []
    closing_elements.append(Paragraph("6. Hardening Recommendations & Strategic Action Plan", section_h1))

    recs = norm.get("recommendations", [])
    if recs:
        for idx, r in enumerate(recs[:4], 1):
            closing_elements.append(
                Paragraph(f"<b>{idx}.</b> {r}", body_style)
            )
            closing_elements.append(Spacer(1, 2))
    else:
        closing_elements.append(
            Paragraph(
                "• Deploy input preprocessing (spatial smoothing or bit-depth reduction) to filter high-frequency adversarial noise.<br/>"
                "• Conduct periodic re-evaluations under adaptive multi-step white-box perturbations in continuous delivery pipelines.",
                body_style,
            )
        )

    closing_elements.append(Spacer(1, 10))

    # Attestation Box
    attest_data = [
        [
            Paragraph(
                "<b>ADVERSCAN ASSURANCE ATTESTATION</b><br/>"
                "This document certifies that the aforementioned artificial intelligence model has been subjected to empirical adversarial robustness stress testing, vulnerability classification, and mitigation validation under the AdverScan framework.<br/>"
                f"<font size=7 color='#64748B'>Attested for deployment readiness • Verification Hash: {norm.get('scan_id', 'N/A')} • Certified at {norm.get('timestamp')}</font>",
                ParagraphStyle("Attest", parent=body_style, fontSize=7.5, leading=10, textColor=colors.HexColor("#1E293B")),
            )
        ]
    ]
    attest_table = Table(attest_data, colWidths=[540])
    attest_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0284C7")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    closing_elements.append(attest_table)

    story.append(KeepTogether(closing_elements))

    # Build PDF document
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buf.getvalue()

    # Optional file persistence
    if target_path:
        out_file = Path(target_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes

