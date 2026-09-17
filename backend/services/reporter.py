"""
FairEnough -- PDF Report Generator (v2 -- Professional Audit Edition)

Generates a dark-themed 5-page professional audit report readable by
non-technical HR directors or compliance officers.

Pages
-----
1  Cover + Executive Summary
2  Detailed Findings (per attribute)
3  Mitigation Results
4  Recommendations & Action Plan
5  Methodology & About
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# =====================================================================
# DESIGN SYSTEM
# =====================================================================
NAVY   = HexColor("#0F172A")
SURF   = HexColor("#1E293B")
CARD   = HexColor("#263349")
TEAL   = HexColor("#14B8A6")
PURPLE = HexColor("#818CF8")
WHITE  = HexColor("#F1F5F9")
GRAY   = HexColor("#94A3B8")
RED    = HexColor("#EF4444")
AMBER  = HexColor("#F59E0B")
GREEN  = HexColor("#22C55E")
ORANGE = HexColor("#F97316")
BORDER = HexColor("#2D3F55")

PAGE_W, PAGE_H = A4
MARGIN    = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# =====================================================================
# STYLE FACTORY
# =====================================================================
def _s(name, **kw) -> ParagraphStyle:
    kw.pop("parent", None)
    return ParagraphStyle(name, **kw)


_S_LOGO      = _s("logo",      fontName="Helvetica-Bold", fontSize=32, textColor=TEAL,   leading=38, spaceAfter=4)
_S_TITLE     = _s("title",     fontName="Helvetica-Bold", fontSize=22, textColor=WHITE,  leading=28, spaceAfter=4)
_S_H1        = _s("h1",        fontName="Helvetica-Bold", fontSize=16, textColor=TEAL,   leading=20, spaceBefore=8, spaceAfter=4)
_S_H2        = _s("h2",        fontName="Helvetica-Bold", fontSize=12, textColor=TEAL,   leading=16, spaceBefore=6, spaceAfter=3)
_S_H3        = _s("h3",        fontName="Helvetica-Bold", fontSize=9,  textColor=WHITE,  leading=13, spaceBefore=4, spaceAfter=2)
_S_BODY      = _s("body",      fontName="Helvetica",      fontSize=9,  textColor=WHITE,  leading=14, spaceAfter=3)
_S_GRAY      = _s("gray",      fontName="Helvetica",      fontSize=8,  textColor=GRAY,   leading=12, spaceAfter=2)
_S_GITAL     = _s("gital",     fontName="Helvetica-Oblique", fontSize=7, textColor=GRAY, leading=10, spaceAfter=2)
_S_SCRLBL    = _s("scrlbl",    fontName="Helvetica",      fontSize=8,  textColor=GRAY,   leading=10, alignment=1)
_S_EXHD      = _s("exhd",      fontName="Helvetica-Bold", fontSize=11, textColor=TEAL,   leading=14, spaceBefore=6, spaceAfter=3)
_S_FOOT      = _s("foot",      fontName="Helvetica",      fontSize=7,  textColor=GRAY,   leading=9)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def _f(val, dec=3) -> str:
    if val is None:
        return "\u2014"
    try:
        return f"{float(val):.{dec}f}"
    except (TypeError, ValueError):
        return str(val)


def _pct2(val) -> str:
    """val is 0-100."""
    if val is None:
        return "\u2014"
    try:
        return f"{float(val):.1f}%"
    except (TypeError, ValueError):
        return str(val)


def _comma(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _score_color(score: float) -> HexColor:
    if score > 84: return GREEN
    if score > 69: return AMBER
    if score > 49: return ORANGE
    return RED


def _grade_letter(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 50: return "C"
    return "F"


def _grade_label(score: float) -> str:
    if score >= 85: return "FAIR"
    if score >= 70: return "MINOR ISSUES"
    if score >= 50: return "MODERATE BIAS"
    return "HIGH BIAS \u2014 ACTION REQUIRED"


def _sev_color(sev: str) -> HexColor:
    return {"high": RED, "medium": AMBER, "low": GREEN}.get(str(sev).lower(), GRAY)


def _cause_color(cause: str) -> HexColor:
    return {"proxy": PURPLE, "underrepresentation": AMBER,
            "historical_skew": ORANGE, "none": GREEN}.get(str(cause).lower(), GRAY)


def _cause_label(cause: str) -> str:
    c = str(cause).lower()
    if c == "proxy":
        return "Proxy Feature Causing Indirect Discrimination"
    elif c == "underrepresentation":
        return "Insufficient Data for Disadvantaged Group"
    elif c == "historical_skew":
        return "Historical Patterns in Training Data"
    return "Root Cause Requires Further Investigation"


def _darrow(delta, lower=True) -> str:
    if delta is None: return "\u2014"
    try:
        d = float(delta)
    except (TypeError, ValueError):
        return "\u2014"
    good = (d < 0 and lower) or (d > 0 and not lower)
    col  = GREEN.hexval() if good else RED.hexval()
    arr  = "\u25bc" if d < 0 else "\u25b2"
    sign = "+" if d > 0 else ""
    return f'<font color="{col}">{arr} {sign}{d:.4f}</font>'


def _spd_ctx(spd, worst_group_name=None, gap_pct=None) -> str:
    if spd is None: return "\u2014"
    if worst_group_name and gap_pct is not None:
        return f"'{worst_group_name}' gets {gap_pct:.1f}% fewer positive outcomes"
    v = abs(float(spd))
    if v > 0.2: return f'<font color="{RED.hexval()}">HIGH \u2014 exceeds 0.2 threshold</font>'
    if v > 0.1: return f'<font color="{AMBER.hexval()}">MEDIUM \u2014 notable gap</font>'
    return f'<font color="{GREEN.hexval()}">LOW \u2014 within acceptable range</font>'


def _di_ctx(di) -> str:
    if di is None: return "\u2014"
    v = float(di)
    if v < 0.8: return f'<font color="{RED.hexval()}">FAILS legal 80% rule \u2014 regulatory risk</font>'
    return f'<font color="{GREEN.hexval()}">Passes legal threshold</font>'


def _perf_impact(delta) -> str:
    if delta is None: return "\u2014"
    d = float(delta)
    if abs(d) < 0.005: return f'<font color="{GREEN.hexval()}">Minimal impact</font>'
    if d < -0.05:  return f'<font color="{RED.hexval()}">Significant drop</font>'
    if d < 0:      return f'<font color="{AMBER.hexval()}">Small reduction</font>'
    return f'<font color="{GREEN.hexval()}">Improved</font>'


# =====================================================================
# TABLE BUILDERS
# =====================================================================
def _dark_table(data, col_widths=None, fs=8) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    n = len(data)
    cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  SURF),
        ("TEXTCOLOR",    (0,0), (-1,0),  TEAL),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), fs),
        ("TEXTCOLOR",    (0,1), (-1,-1), WHITE),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]
    for i in range(1, n):
        cmds.append(("BACKGROUND", (0,i), (-1,i), CARD if i%2==0 else SURF))
    t.setStyle(TableStyle(cmds))
    return t


def _alert_box(text: str, bg: HexColor, label: str = "") -> Table:
    txt = f"<b>{label}</b> {text}" if label else text
    cell = Paragraph(txt, _s(f"ab{id(text)}", fontName="Helvetica", fontSize=8, textColor=WHITE, leading=12))
    t = Table([[cell]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("BOX",          (0,0),(-1,-1), 0.8, BORDER),
    ]))
    return t


def _gm(d, k):
    """Get metric key case-insensitively."""
    if not d: return None
    return d.get(k.upper(), d.get(k.lower()))


# =====================================================================
# PAGE CALLBACK (background + header + footer)
# =====================================================================
def _on_page(canvas, doc):
    canvas.saveState()

    # Dark background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Teal top bar 4mm
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 4*mm, PAGE_W, 4*mm, fill=1, stroke=0)

    # Footer separator line
    fy = 10 * mm
    canvas.setFillColor(BORDER)
    canvas.rect(MARGIN, fy - 0.5, CONTENT_W, 0.5, fill=1, stroke=0)

    # Footer text
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, fy - 5, "Generated by FairEnough. For audit purposes only.")
    canvas.drawRightString(PAGE_W - MARGIN, fy - 5, f"Page {doc.page}")

    # Header (page 2+)
    if doc.page > 1:
        fname = getattr(doc, "_byus_fn", "")
        if fname:
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(GRAY)
            canvas.drawCentredString(PAGE_W/2, PAGE_H - 8*mm, fname)

    canvas.restoreState()


# =====================================================================
# REPORT GENERATOR
# =====================================================================
class ReportGenerator:
    """Generate a 5-page dark-themed professional PDF audit report."""

    def generate(self, session_data: dict[str, Any]) -> bytes:
        buf = io.BytesIO()

        # --- Extract fields ---
        filename         = session_data.get("filename", "Unknown Dataset")
        row_count        = session_data.get("row_count", 0)
        target_col       = session_data.get("target_col", "")
        sensitive_attrs  = session_data.get("sensitive_attrs", [])
        scenario_raw     = session_data.get("scenario", "Other")
        scenario         = scenario_raw if isinstance(scenario_raw, str) else scenario_raw.get("scenario", "Other")
        bias_results     = session_data.get("bias_results", {})
        validation       = session_data.get("validation", {})
        # Accept mitigation from either flat key or legacy key
        mitigation       = session_data.get("mitigation") or session_data.get("mitigation_results") or {}
        pattern_preds    = session_data.get("pattern_predictions", {})
        training_count   = session_data.get("training_examples_count", 20)

        audit_score      = float(
            session_data.get("audit_score") or bias_results.get("audit_score") or 0
        )
        metrics_per_attr = (
            session_data.get("metrics_per_attr") or
            bias_results.get("metrics_per_attr") or {}
        )
        engine           = validation.get("engine", "FairEnough Bias Engine")

        # Re-derive grade and severity from audit_score for guaranteed consistency
        if audit_score >= 85:
            overall_sev = "low";    computed_grade = "A"
        elif audit_score >= 70:
            overall_sev = "low";    computed_grade = "B"
        elif audit_score >= 50:
            overall_sev = "medium"; computed_grade = "C"
        else:
            overall_sev = "high";   computed_grade = "F"


        now = datetime.now(timezone.utc)
        generated_at = now.strftime("%d %b %Y, %H:%M UTC").lstrip("0")

        # --- Build doc ---
        doc = BaseDocTemplate(
            buf, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN + 6*mm, bottomMargin=MARGIN + 8*mm,
            title="FairEnough Bias Audit Report", author="FairEnough",
        )
        doc._byus_fn = filename

        frame = Frame(MARGIN, MARGIN + 8*mm, CONTENT_W,
                      PAGE_H - 2*MARGIN - 14*mm, id="main", showBoundary=0)
        doc.addPageTemplates([PageTemplate(id="dark", frames=[frame], onPage=_on_page)])

        story: list = []
        story += self._p1(filename, row_count, target_col, sensitive_attrs, scenario,
                          audit_score, overall_sev, engine, generated_at,
                          metrics_per_attr, mitigation, pattern_preds)
        story.append(PageBreak())
        story += self._p2(scenario, metrics_per_attr, pattern_preds, row_count)
        story.append(PageBreak())
        story += self._p3(sensitive_attrs, mitigation)
        story.append(PageBreak())
        story += self._p4(sensitive_attrs, metrics_per_attr, mitigation, pattern_preds)
        story.append(PageBreak())
        story += self._p5(training_count)

        doc.build(story)
        buf.seek(0)
        return buf.read()

    # =================================================================
    # PAGE 1 - COVER
    # =================================================================
    def _p1(self, filename, row_count, target_col, sensitive_attrs, scenario,
            audit_score, overall_sev, engine, generated_at,
            metrics_per_attr, mitigation, pattern_preds) -> list:
        story = []
        sc = _score_color(audit_score)
        gl = _grade_letter(audit_score)
        glab = _grade_label(audit_score)
        sev_col = _sev_color(overall_sev)

        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("FairEnough", _S_LOGO))
        story.append(Paragraph("BIAS AUDIT REPORT", _S_TITLE))
        story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=TEAL, spaceAfter=4))
        story.append(Paragraph(filename, _s("fn", fontName="Helvetica-Bold",
                                            fontSize=14, textColor=TEAL, leading=18)))
        story.append(Paragraph(f"Generated: {generated_at}", _S_GRAY))
        story.append(Spacer(1, 5*mm))

        # Score + info box
        score_sty = _s("bscore", fontName="Helvetica-Bold", fontSize=46,
                        textColor=sc, leading=52, alignment=1)
        grade_sty = _s("bgrade", fontName="Helvetica-Bold", fontSize=18,
                        textColor=sc, leading=22, alignment=1)
        glab_sty  = _s("bglab", fontName="Helvetica-Bold", fontSize=9,
                        textColor=sc, leading=12, alignment=1)
        sev_vsty  = _s("sevv", fontName="Helvetica-Bold", fontSize=9,
                        textColor=sev_col, leading=12)

        left_cell = [
            Paragraph("BIAS AUDIT SCORE", _S_SCRLBL),
            Paragraph(f"{audit_score:.1f}", score_sty),
            Paragraph(f"Grade: {gl}", grade_sty),
            Paragraph(glab, glab_sty),
        ]

        right_rows = [
            [Paragraph("Overall Severity", _S_GRAY), Paragraph(overall_sev.upper(), sev_vsty)],
            [Paragraph("Dataset", _S_GRAY),          Paragraph(filename, _S_BODY)],
            [Paragraph("Rows Analyzed", _S_GRAY),    Paragraph(_comma(row_count), _S_BODY)],
            [Paragraph("Target Variable", _S_GRAY),  Paragraph(target_col or "\u2014", _S_BODY)],
            [Paragraph("Sensitive Attributes", _S_GRAY),
             Paragraph(", ".join(sensitive_attrs) if sensitive_attrs else "\u2014", _S_BODY)],
            [Paragraph("Scenario", _S_GRAY),         Paragraph(str(scenario).capitalize(), _S_BODY)],
            [Paragraph("Analysis Engine", _S_GRAY),  Paragraph(str(engine), _S_BODY)],
            [Paragraph("Report Generated", _S_GRAY), Paragraph(generated_at, _S_BODY)],
        ]
        right_t = Table(right_rows, colWidths=[CONTENT_W*0.28, CONTENT_W*0.36])
        right_t.setStyle(TableStyle([
            ("GRID",          (0,0),(-1,-1), 0.3, BORDER),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [SURF, CARD]),
        ]))

        outer = Table([[left_cell, right_t]], colWidths=[CONTENT_W*0.32, CONTENT_W*0.68])
        outer.setStyle(TableStyle([
            ("BOX",          (0,0),(-1,-1), 1.5, TEAL),
            ("BACKGROUND",   (0,0),(-1,-1), SURF),
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",  (0,0),(-1,-1), 10),
            ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ]))
        story.append(outer)
        story.append(Spacer(1, 5*mm))

        # Executive summary
        story.append(Paragraph("EXECUTIVE SUMMARY", _S_EXHD))
        story.append(Paragraph(
            self._exec_summary(filename, row_count, scenario, sensitive_attrs,
                               metrics_per_attr, mitigation, audit_score, overall_sev),
            _S_BODY))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            "This report was generated by FairEnough automated bias analysis. "
            "Findings should be reviewed with qualified legal and compliance "
            "professionals before taking organizational action.",
            _S_GITAL))
        return story

    def _exec_summary(self, filename, row_count, scenario, sensitive_attrs,
                      metrics_per_attr, mitigation, audit_score, overall_sev) -> str:
        n = len(metrics_per_attr)
        high_attrs   = [a for a,m in metrics_per_attr.items()
                        if isinstance(m,dict) and m.get("severity")=="high"]
        legal_fails  = [a for a,m in metrics_per_attr.items()
                        if isinstance(m,dict) and m.get("legal_flag")]

        finding = ""
        for attr, m in metrics_per_attr.items():
            if not isinstance(m,dict) or "error" in m: continue
            gs = m.get("group_stats",{})
            if gs:
                sorted_groups = sorted(
                    gs.items(),
                    key=lambda x: x[1].get("positive_rate", 0)
                )
                worst_group_name  = str(sorted_groups[0][0])  if sorted_groups else "disadvantaged group"
                worst_group_rate  = sorted_groups[0][1].get("positive_rate", 0) if sorted_groups else 0
                best_group_name   = str(sorted_groups[-1][0]) if sorted_groups else "advantaged group"
                best_group_rate   = sorted_groups[-1][1].get("positive_rate", 0) if sorted_groups else 0
                gap_pct           = round(abs(best_group_rate - worst_group_rate) * 100, 1)

                finding = (f"The most significant finding is that '{best_group_name}' achieves a positive "
                           f"outcome rate of {best_group_rate*100:.1f}% compared to only {worst_group_rate*100:.1f}% for "
                           f"'{worst_group_name}' \u2014 a {gap_pct:.1f} percentage-point gap on the '{attr}' attribute.")
                break

        first_attr = sensitive_attrs[0] if sensitive_attrs else ""
        mit_for_attr = None
        if isinstance(mitigation, dict):
            if first_attr in mitigation and isinstance(mitigation[first_attr], dict) and "reweigh" in mitigation[first_attr]:
                mit_for_attr = mitigation[first_attr]
            elif "reweigh" in mitigation:
                mit_for_attr = mitigation

        winner = (mit_for_attr or {}).get("winner","")
        wr = {"reweigh":"Reweighing","threshold":"Threshold Adjustment"}.get(winner,"")
        w_eff = (mit_for_attr or {}).get("reweigh" if winner=="reweigh" else "threshold",{}).get("effects",{})
        br = w_eff.get("bias_reduction_pct",0) or 0
        mit = (f"Applying {wr} is recommended, achieving {br:.0f}% bias reduction while retaining model accuracy."
               if wr and br > 5 else "")

        legal = (f"Critically, {len(legal_fails)} attribute(s) ({', '.join(legal_fails)}) fail the "
                 f"80% Disparate Impact legal threshold, presenting potential regulatory risk."
                 if legal_fails else "All attributes pass the 80% Disparate Impact legal threshold.")

        high_s = (f"Of the {n} attribute(s) analyzed, {len(high_attrs)} show HIGH severity bias "
                  f"({', '.join(high_attrs)})." if high_attrs else
                  f"Of the {n} attribute(s) analyzed, none show high-severity bias.") if n else ""

        sc_clean = str(scenario).lower().replace(" ", "_")
        if "hiring" in sc_clean:
            opening = "This hiring audit analyzed"
        elif "lending" in sc_clean or "credit" in sc_clean:
            opening = "This credit/lending audit analyzed"
        elif "health" in sc_clean:
            opening = "This healthcare audit analyzed"
        elif "criminal" in sc_clean or "justice" in sc_clean:
            opening = "This criminal justice audit analyzed"
        elif "education" in sc_clean:
            opening = "This education dataset audit analyzed"
        elif "income" in sc_clean:
            opening = "This income classification audit analyzed"
        else:
            opening = "This dataset audit analyzed"

        parts = [
            f"{opening} <b>{_comma(row_count)}</b> records from <b>{filename}</b>. "
            f"The dataset was assessed across <b>{n}</b> sensitive attribute(s): "
            f"<b>{', '.join(sensitive_attrs)}</b>. "
            f"The overall Bias Audit Score is <b>{audit_score:.1f}/100</b> ({_grade_label(audit_score)}).",
            high_s, finding, legal, mit,
        ]
        return " ".join(p for p in parts if p)

    # =================================================================
    # PAGE 2 - DETAILED FINDINGS
    # =================================================================
    def _p2(self, scenario, metrics_per_attr, pattern_preds, row_count) -> list:
        story = []
        story.append(Paragraph("DETAILED FINDINGS", _S_H1))
        story.append(HRFlowable(width=CONTENT_W, thickness=1, color=TEAL, spaceAfter=5))

        if not metrics_per_attr:
            story.append(Paragraph("No analysis results found. Run /api/analyze first.", _S_BODY))
            return story

        for attr, m in metrics_per_attr.items():
            if not isinstance(m, dict): continue

            sev = m.get("severity","low")
            sev_col = _sev_color(sev)

            # Attribute header
            hdr_sty = _s(f"ah_{attr}", fontName="Helvetica-Bold", fontSize=13, textColor=TEAL, leading=16)
            bge_sty = _s(f"bg_{attr}", fontName="Helvetica-Bold", fontSize=9,  textColor=WHITE,
                          backColor=sev_col, leading=14)
            hdr_t = Table([[Paragraph(f"Attribute: <b>{attr}</b>", hdr_sty),
                            Paragraph(f" {sev.upper()} ", bge_sty)]],
                          colWidths=[CONTENT_W*0.8, CONTENT_W*0.2])
            hdr_t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), SURF),
                ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("LEFTPADDING",  (0,0),(-1,-1), 10),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("BOX",          (0,0),(-1,-1), 1, TEAL),
            ]))
            story.append(hdr_t)
            story.append(Spacer(1, 3*mm))

            if "error" in m:
                story.append(_alert_box(m["error"], RED, "Error:"))
                story.append(Spacer(1, 4*mm))
                continue

            spd   = m.get("SPD"); di  = m.get("DI")
            eod   = m.get("EOD"); aod = m.get("AOD")
            gs    = m.get("group_stats",{})
            sorted_groups = sorted(
                gs.items(),
                key=lambda x: x[1].get("positive_rate", 0)
            )
            worst_group_name  = str(sorted_groups[0][0])  if sorted_groups else (str(m.get("unprivileged_group", "")) or "disadvantaged group")
            worst_group_rate  = sorted_groups[0][1].get("positive_rate", 0) if sorted_groups else 0
            best_group_name   = str(sorted_groups[-1][0]) if sorted_groups else (str(m.get("privileged_group", "")) or "advantaged group")
            best_group_rate   = sorted_groups[-1][1].get("positive_rate", 0) if sorted_groups else 0
            gap_pct           = round(abs(best_group_rate - worst_group_rate) * 100, 1)

            priv  = best_group_name
            unpriv = worst_group_name
            ci    = m.get("bootstrapped_ci",{})
            prox  = m.get("proxy_features",[])
            pred  = pattern_preds.get(attr,{})
            cause = pred.get("predicted_cause","unknown")
            conf  = pred.get("confidence_pct",0)

            # A - What We Found
            story.append(Paragraph("A \u2014 What We Found", _S_H2))
            story.append(Paragraph(self._plain_finding(attr, scenario, sev, gs, priv, unpriv, spd), _S_BODY))
            story.append(Spacer(1, 2*mm))

            # B - The Numbers
            story.append(Paragraph("B \u2014 The Numbers", _S_H2))
            eod_val = _f(eod) if eod is not None else "N/A (upload model)"
            aod_val = _f(aod) if aod is not None else "N/A (upload model)"
            met_rows = [
                [Paragraph("Metric",_S_H3), Paragraph("Value",_S_H3),
                 Paragraph("Ideal",_S_H3),  Paragraph("What It Means",_S_H3)],
                [Paragraph("Statistical Parity Difference (SPD)",_S_BODY),
                 Paragraph(f"<b>{_f(spd)}</b>",_S_BODY), Paragraph("0",_S_BODY),
                 Paragraph(f"Outcome rate gap between groups<br/>{_spd_ctx(spd, worst_group_name, gap_pct)}",_S_BODY)],
                [Paragraph("Disparate Impact (DI)",_S_BODY),
                 Paragraph(f"<b>{_f(di)}</b>",_S_BODY), Paragraph("1.0",_S_BODY),
                 Paragraph(f"Ratio of positive outcomes<br/>{_di_ctx(di)}",_S_BODY)],
                [Paragraph("Equal Opportunity Diff (EOD)",_S_BODY),
                 Paragraph(f"<b>{eod_val}</b>",_S_BODY), Paragraph("0",_S_BODY),
                 Paragraph("True positive rate gap between groups",_S_BODY)],
                [Paragraph("Average Odds Diff (AOD)",_S_BODY),
                 Paragraph(f"<b>{aod_val}</b>",_S_BODY), Paragraph("0",_S_BODY),
                 Paragraph("Combined TPR and FPR gap",_S_BODY)],
            ]
            story.append(_dark_table(met_rows,
                col_widths=[CONTENT_W*0.30, CONTENT_W*0.13, CONTENT_W*0.09, CONTENT_W*0.48]))
            story.append(Spacer(1, 2*mm))

            # C - Group Breakdown
            if gs:
                story.append(Paragraph("C \u2014 Group Breakdown", _S_H2))
                avg_r = sum(g.get("positive_rate",0) for g in gs.values()) / max(len(gs),1)
                srt = sorted(gs.items(), key=lambda x: x[1].get("positive_rate",0), reverse=True)
                gr = [[Paragraph("Group",_S_H3), Paragraph("Count",_S_H3),
                       Paragraph("% of Dataset",_S_H3), Paragraph("Outcome Rate",_S_H3),
                       Paragraph("vs Average",_S_H3), Paragraph("Status",_S_H3)]]
                for gn, gd in srt:
                    pr2 = gd.get("positive_rate",0); cnt = gd.get("count",0); pct = gd.get("pct_of_total",0)
                    va  = (pr2 - avg_r)*100
                    vc  = GREEN.hexval() if va >= 0 else RED.hexval()
                    vs  = f'<font color="{vc}">{"+" if va>=0 else ""}{va:.1f}%</font>'
                    if str(gn)==str(priv):   stx = f'<font color="{GREEN.hexval()}">Advantaged</font>'
                    elif str(gn)==str(unpriv): stx = f'<font color="{RED.hexval()}">Disadvantaged</font>'
                    else:                    stx = f'<font color="{GRAY.hexval()}">Average</font>'
                    gr.append([Paragraph(str(gn),_S_BODY), Paragraph(_comma(cnt),_S_BODY),
                                Paragraph(f"{pct:.1f}%",_S_BODY), Paragraph(f"{pr2*100:.1f}%",_S_BODY),
                                Paragraph(vs,_S_BODY), Paragraph(stx,_S_BODY)])
                story.append(_dark_table(gr, col_widths=[
                    CONTENT_W*0.18, CONTENT_W*0.12, CONTENT_W*0.14,
                    CONTENT_W*0.15, CONTENT_W*0.14, CONTENT_W*0.27]))
                story.append(Spacer(1, 2*mm))

            # D - Why This Bias Exists
            story.append(Paragraph("D \u2014 Why This Bias Exists", _S_H2))
            cc = _cause_color(cause); cl = _cause_label(cause)
            conf_val = float(conf) if conf is not None else 0.0
            n_cases = int(pred.get("training_examples", 20)) or 20
            if conf_val > 0 and pred.get("learned", True):
                conf_text = f"Based on pattern analysis of {n_cases} reference datasets"
            else:
                conf_text = f"Rule-based analysis based on {n_cases} reference datasets"
            cause_table = Table([
                [Paragraph(f"<b>{cl}</b>", _s(f"cb_{attr}", fontName="Helvetica-Bold", fontSize=11, textColor=WHITE, leading=14, alignment=1))],
                [Paragraph(f'<font color="{GRAY.hexval()}">{conf_text}</font>', _s(f"cct_{attr}", fontName="Helvetica", fontSize=8, textColor=GRAY, leading=11, alignment=1))]
            ], colWidths=[CONTENT_W])
            cause_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), cc),
                ("TOPPADDING", (0, 0), (0, 0), 7),
                ("BOTTOMPADDING", (0, 0), (0, 0), 7),
                ("TOPPADDING", (0, 1), (0, 1), 4),
                ("BOTTOMPADDING", (0, 1), (0, 1), 2),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story.append(cause_table)
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(self._cause_para(cause, attr, priv, unpriv, gs, prox, spd, di), _S_BODY))

            if prox:
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph(f"Proxy Feature Analysis for '{attr}'", _S_H3))
                pf_rows = [[Paragraph("Feature",_S_H3),
                             Paragraph(f"Correlation with {attr}",_S_H3),
                             Paragraph("Strength",_S_H3), Paragraph("Risk",_S_H3)]]
                for pf in prox[:8]:
                    cr = pf.get("correlation",0); ft = pf.get("feature","")
                    if cr > 0.4:
                        st2=f'<font color="{RED.hexval()}">Strong</font>'
                        rk=f'<font color="{RED.hexval()}">High \u2014 acts as proxy</font>'
                    elif cr > 0.2:
                        st2=f'<font color="{AMBER.hexval()}">Moderate</font>'
                        rk=f'<font color="{AMBER.hexval()}">Moderate \u2014 monitor</font>'
                    else:
                        st2=f'<font color="{GREEN.hexval()}">Weak</font>'
                        rk=f'<font color="{GREEN.hexval()}">Low</font>'
                    pf_rows.append([Paragraph(ft,_S_BODY), Paragraph(f"{cr:.3f}",_S_BODY),
                                    Paragraph(st2,_S_BODY), Paragraph(rk,_S_BODY)])
                story.append(_dark_table(pf_rows,
                    col_widths=[CONTENT_W*0.35, CONTENT_W*0.20, CONTENT_W*0.18, CONTENT_W*0.27]))

            # E - Legal & Compliance
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("E \u2014 Legal & Compliance Context", _S_H2))
            di_v = float(di) if di is not None else 1.0
            if di_v < 0.8:
                story.append(_alert_box(
                    f"The Disparate Impact ratio of {_f(di)} falls below the 0.8 threshold "
                    f"established by the EEOC Uniform Guidelines on Employee Selection Procedures "
                    f"(29 CFR Part 1607) and similar EU anti-discrimination standards. "
                    f"A ratio below 0.8 means '{unpriv}' receives positive outcomes at less than "
                    f"80% the rate of '{priv}', which regulators consider evidence of adverse impact. "
                    f"We recommend immediate review by your legal and compliance team.",
                    HexColor("#3B0A0A"), "\u26a0  LEGAL RISK DETECTED \u2014"))
            else:
                story.append(_alert_box(
                    f"DI of {_f(di)} is above the 0.8 legal minimum. However, passing this "
                    f"threshold does not mean no bias exists \u2014 the SPD gap of {_f(spd)} "
                    f"still warrants attention and monitoring.",
                    HexColor("#0A2E1A"), "\u2713  PASSES LEGAL THRESHOLD \u2014"))

            story.append(Spacer(1, 2*mm))
            ci_l = ci.get("low_95"); ci_h = ci.get("high_95")
            sig  = m.get("statistically_significant", False)
            if sig:
                story.append(_alert_box(
                    f"This finding is statistically significant. 95% CI: [{_f(ci_l)}, {_f(ci_h)}]",
                    HexColor("#0A2E1A")))
            else:
                story.append(_alert_box(
                    f"\u26a0 This finding may not be statistically significant. "
                    f"95% CI: [{_f(ci_l)}, {_f(ci_h)}] \u2014 the interval crosses zero.",
                    HexColor("#2E2A0A")))

            story.append(Spacer(1, 6*mm))

        return story

    def _plain_finding(self, attr, scenario, sev, gs, priv, unpriv, spd) -> str:
        pd2  = gs.get(str(priv),{}); ud2 = gs.get(str(unpriv),{})
        pr   = pd2.get("positive_rate",0) or 0
        ur   = ud2.get("positive_rate",0) or 0
        gap  = round(abs(pr-ur)*100,1)
        return (
            f"In this <b>{str(scenario).lower()}</b> dataset, '<b>{attr}</b>' shows "
            f"<b>{sev.upper()}</b> bias. "
            f"'{priv}' achieves a positive outcome rate of <b>{pr*100:.1f}%</b>, "
            f"while '{unpriv}' achieves only <b>{ur*100:.1f}%</b> "
            f"\u2014 a <b>{gap:.1f} percentage-point</b> difference. "
            f"For every 100 people from '{unpriv}', approximately <b>{round(gap)}</b> fewer "
            f"receive a positive outcome compared to '{priv}'. "
            f"Someone from the '{unpriv}' group faces a structurally lower chance of a favorable "
            f"outcome in this system, purely because of their group membership."
        )

    def _cause_para(self, cause, attr, priv, unpriv, gs, prox, spd=None, di=None) -> str:
        pd2 = gs.get(str(priv),{}); ud2 = gs.get(str(unpriv),{})
        pc  = pd2.get("count",0); uc = ud2.get("count",0); up = ud2.get("pct_of_total",0)
        rat = round(pc / max(uc,1),1)
        tp  = prox[0] if prox else {}
        tf  = tp.get("feature","unknown feature"); tc = tp.get("correlation",0)
        if cause == "proxy":
            return (
                f"The most likely cause is <b>proxy discrimination</b>. "
                f"'<b>{tf}</b>' is strongly correlated with '{attr}' (correlation: {tc:.3f}). "
                f"This means the model can discriminate against '<b>{unpriv}</b>' "
                f"indirectly using this feature even if "
                f"'{attr}' is removed from the model entirely. Proxy discrimination is often "
                f"harder to detect than direct bias because the protected attribute is not used "
                f"directly \u2014 yet the same discriminatory outcome is produced through a substitute variable."
            )
        elif cause == "underrepresentation":
            return (
                f"The most likely cause is <b>data underrepresentation</b>. "
                f"'<b>{unpriv}</b>' has only <b>{_comma(uc)}</b> records (<b>{up:.1f}%</b> of the dataset) "
                f"compared to <b>{_comma(pc)}</b> for '{priv}'. The model has seen "
                f"<b>{rat}x</b> fewer examples from the disadvantaged group, reducing its ability "
                f"to make accurate and fair decisions for this group. When a group is underrepresented, "
                f"the model's performance suffers, leading to systematically worse outcomes."
            )
        elif cause == "historical_skew":
            return (
                f"The most likely cause is <b>historical bias</b>. The training data likely reflects "
                f"real-world systemic discrimination where '{attr}' influenced outcomes in the past "
                f"\u2014 for example, decisions made under discriminatory norms favoring '{priv}' over '{unpriv}'. "
                f"The model has learned these historical patterns and is now reproducing them in its "
                f"predictions, effectively automating and amplifying past injustices."
            )
        else:
            spd_val = abs(float(spd)) if spd is not None else 0.0
            di_val  = float(di) if di is not None else 1.0
            if spd_val > 0.1 or di_val < 0.8:
                return (
                    f"The bias pattern for '{attr}' does not clearly "
                    f"match known proxy, underrepresentation, or historical skew "
                    f"profiles from our training data. However, the metrics clearly "
                    f"show a {spd_val*100:.1f}% outcome gap (SPD={spd_val:.3f}) "
                    f"and a Disparate Impact of {di_val:.3f} which violates the legal "
                    f"0.8 threshold. The bias is real \u2014 the root cause may be a "
                    f"combination of factors or a pattern not yet in our training set."
                )
            else:
                return (
                    f"No significant bias pattern detected for '{attr}'. "
                    f"Fairness metrics are within acceptable ranges. "
                    f"Continue monitoring with quarterly audits."
                )

    # =================================================================
    # PAGE 3 - MITIGATION RESULTS
    # =================================================================
    def _p3(self, sensitive_attrs, mitigation) -> list:
        story = []
        story.append(Paragraph("BIAS MITIGATION RESULTS", _S_H1))
        story.append(HRFlowable(width=CONTENT_W, thickness=1, color=TEAL, spaceAfter=5))

        first_attr = sensitive_attrs[0] if sensitive_attrs else "the sensitive attribute"
        mit_for_attr = None
        if isinstance(mitigation, dict):
            if first_attr in mitigation and isinstance(mitigation[first_attr], dict) and "reweigh" in mitigation[first_attr]:
                mit_for_attr = mitigation[first_attr]
            elif "reweigh" in mitigation:
                mit_for_attr = mitigation

        if not mit_for_attr:
            story.append(_alert_box("No mitigation data found. Run POST /api/mitigate first.", SURF))
            return story

        winner     = mit_for_attr.get("winner","reweigh")
        win_reason = mit_for_attr.get("winner_reason","")
        rw = mit_for_attr.get("reweigh",{}); th = mit_for_attr.get("threshold",{})
        rw_b=rw.get("before",{}); rw_a=rw.get("after",{}); rw_e=rw.get("effects",{})
        th_b=th.get("before",{}); th_a=th.get("after",{}); th_e=th.get("effects",{})
        wr = "Reweighing" if winner=="reweigh" else "Threshold Adjustment"

        rw_red = rw_e.get("bias_reduction_pct", 0) or 0
        th_red = th_e.get("bias_reduction_pct", 0) or 0
        if rw_red <= 0 and th_red <= 0:
            story.append(_alert_box(
                "\u26a0 Note: Automated mitigation simulation showed limited results. "
                "This is likely because the model detected a data configuration issue "
                "during simulation. The recommended technique is still valid for your "
                "actual model training pipeline. See the Action Plan on the next page "
                "for specific guidance.",
                HexColor("#2E2A0A")
            ))
            story.append(Spacer(1, 4*mm))

        story.append(Paragraph(
            f"Two automated mitigation techniques were applied to address bias in '<b>{first_attr}</b>'. "
            f"The following results show what happens to fairness metrics and model performance. "
            f"<b>{wr}</b> is recommended for this dataset.", _S_BODY))
        story.append(Spacer(1, 4*mm))

        descs = {
            "reweigh": ("Reweighing", "Adjusts how much influence each training example has. "
                "Examples from underrepresented (group, outcome) combinations are given higher weight "
                "so the model learns from them more. Corrects bias at the data level before training."),
            "threshold": ("Threshold Adjustment", "Changes the decision cutoff separately for each "
                "demographic group so equally qualified people receive equal treatment. "
                "Corrects bias at the prediction level without retraining."),
        }

        for tk, (tname, tdesc) in descs.items():
            bef = rw_b if tk=="reweigh" else th_b
            aft = rw_a if tk=="reweigh" else th_a
            eff = rw_e if tk=="reweigh" else th_e
            is_w = (tk==winner)
            hbg  = TEAL if is_w else SURF; hfg = NAVY if is_w else WHITE

            ths = _s(f"th_{tk}", fontName="Helvetica-Bold", fontSize=11, textColor=hfg, leading=14)
            if is_w:
                row = [
                    Paragraph(tname, ths),
                    Paragraph("<b>★ RECOMMENDED</b>", _s(f"rb_{tk}", fontName="Helvetica-Bold", fontSize=8, textColor=NAVY, alignment=2, leading=12))
                ]
                ht = Table([row], colWidths=[CONTENT_W * 0.70, CONTENT_W * 0.30])
            else:
                row = [Paragraph(tname, ths)]
                ht = Table([row], colWidths=[CONTENT_W])
            ht.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), hbg),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("LEFTPADDING",  (0,0),(-1,-1), 10),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
                ("BOX",          (0,0),(-1,-1), 1, BORDER),
            ]))
            story.append(ht)
            story.append(Paragraph(tdesc, _S_GRAY))
            story.append(Spacer(1, 2*mm))

            # Fairness impact
            story.append(Paragraph("FAIRNESS IMPACT", _S_H3))
            def mrow(lbl, bv, av, dk, lwr=True):
                d = eff.get(dk); ds = _darrow(d, lower=lwr)
                rp = ""
                if bv is not None and av is not None:
                    try:
                        b2,a2 = abs(float(bv)),abs(float(av))
                        if b2>0:
                            rd = (b2-a2)/b2*100
                            rp = f" ({'+' if rd>=0 else ''}{rd:.0f}% {'reduction' if rd>=0 else 'increase'})"
                    except: pass
                return [Paragraph(lbl,_S_BODY), Paragraph(_f(bv),_S_BODY),
                        Paragraph(_f(av),_S_BODY), Paragraph(f"{ds}{rp}",_S_BODY)]

            spd_bv=_gm(bef,"spd"); di_bv=_gm(bef,"di")
            spd_av=_gm(aft,"spd"); di_av=_gm(aft,"di")
            eod_bv=_gm(bef,"eod"); eod_av=_gm(aft,"eod")
            aod_bv=_gm(bef,"aod"); aod_av=_gm(aft,"aod")

            fr = [
                [Paragraph("Metric",_S_H3), Paragraph("Before",_S_H3),
                 Paragraph("After",_S_H3),  Paragraph("Change",_S_H3)],
                mrow("SPD", spd_bv, spd_av, "spd_delta", True),
                mrow("DI",  di_bv,  di_av,  "di_delta",  False),
            ]
            if eod_bv is not None:
                fr.append(mrow("EOD", eod_bv, eod_av, "eod_delta", True))
            else:
                fr.append([Paragraph("EOD",_S_BODY), Paragraph("N/A",_S_BODY),
                            Paragraph("N/A",_S_BODY), Paragraph("\u2014",_S_BODY)])
            if aod_bv is not None:
                fr.append(mrow("AOD", aod_bv, aod_av, "aod_delta", True))
            else:
                fr.append([Paragraph("AOD",_S_BODY), Paragraph("N/A",_S_BODY),
                            Paragraph("N/A",_S_BODY), Paragraph("\u2014",_S_BODY)])

            story.append(_dark_table(fr,
                col_widths=[CONTENT_W*0.22, CONTENT_W*0.22, CONTENT_W*0.22, CONTENT_W*0.34]))
            story.append(Spacer(1, 2*mm))

            # Performance impact
            story.append(Paragraph("MODEL PERFORMANCE IMPACT", _S_H3))
            pr2 = [
                [Paragraph("Metric",_S_H3), Paragraph("Before",_S_H3), Paragraph("After",_S_H3),
                 Paragraph("Change",_S_H3), Paragraph("Impact",_S_H3)]
            ]
            for plbl, pk, dk in [
                ("Accuracy","accuracy","accuracy_delta"),
                ("Precision","precision","precision_delta"),
                ("Recall","recall","recall_delta"),
                ("F1 Score","f1","f1_delta"),
            ]:
                bv2=_gm(bef,pk); av2=_gm(aft,pk); dv2=eff.get(dk)
                pr2.append([Paragraph(plbl,_S_BODY),
                             Paragraph(_f(bv2) if bv2 is not None else "\u2014",_S_BODY),
                             Paragraph(_f(av2) if av2 is not None else "\u2014",_S_BODY),
                             Paragraph(_darrow(dv2, lower=False),_S_BODY),
                             Paragraph(_perf_impact(dv2),_S_BODY)])
            story.append(_dark_table(pr2,
                col_widths=[CONTENT_W*0.20, CONTENT_W*0.16, CONTENT_W*0.16,
                             CONTENT_W*0.20, CONTENT_W*0.28]))
            story.append(Spacer(1, 2*mm))

            br2 = eff.get("bias_reduction_pct",0)
            if br2 < 5:
                story.append(_alert_box(
                    "\u26a0  This technique achieved minimal bias reduction on this dataset. "
                    "This may indicate bias is driven by deep structural patterns requiring "
                    "more than weight adjustment.", HexColor("#2E2A0A")))
            elif br2 > 60:
                story.append(_alert_box(
                    f"\u2713  Strong bias reduction achieved ({br2:.0f}%) while maintaining "
                    "model performance.", HexColor("#0A2E1A")))
            story.append(Spacer(1, 4*mm))

        # Recommendation box
        story.append(Paragraph("RECOMMENDATION", _S_H2))
        w_bef = rw_b if winner=="reweigh" else th_b
        w_aft = rw_a if winner=="reweigh" else th_a
        w_eff = rw_e if winner=="reweigh" else th_e
        ab_v = _gm(w_bef,"accuracy"); aa_v = _gm(w_aft,"accuracy")
        ad   = w_eff.get("accuracy_delta",0) or 0
        br3  = w_eff.get("bias_reduction_pct",0)

        rec_content = [
            Paragraph(f"\u2605  RECOMMENDED: {wr.upper()}",
                _s("rh", fontName="Helvetica-Bold", fontSize=13, textColor=NAVY, leading=16)),
            Paragraph(win_reason, _s("rb2", fontName="Helvetica", fontSize=8, textColor=NAVY, leading=12)),
        ]
        if ab_v is not None and aa_v is not None:
            sign = "+" if ad >= 0 else ""
            imp = "improvement" if ad>0 else ("minimal change" if abs(ad)<0.005 else "reduction")
            rec_content.append(Paragraph(
                f"ACCURACY TRADE-OFF: {float(ab_v)*100:.1f}% \u2192 {float(aa_v)*100:.1f}% "
                f"({sign}{float(ad)*100:.1f}% \u2014 {imp})",
                _s("ra", fontName="Helvetica-Bold", fontSize=8, textColor=NAVY, leading=12)))
        # Stack each element in its own row inside the single-column table
        rec_t = Table([[p] for p in rec_content], colWidths=[CONTENT_W])
        rec_t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), TEAL),
            ("TOPPADDING",   (0,0),(-1,-1), 7),
            ("BOTTOMPADDING",(0,0),(-1,-1), 7),
            ("LEFTPADDING",  (0,0),(-1,-1), 12),
            ("RIGHTPADDING", (0,0),(-1,-1), 12),
            ("BOX",          (0,0),(-1,-1), 2, AMBER),
        ]))
        story.append(rec_t)
        story.append(Spacer(1, 4*mm))

        # Side-by-side comparison
        story.append(Paragraph("SIDE BY SIDE COMPARISON", _S_H2))
        def wt(rv, tv, lb=True):
            try:
                r2,t2 = abs(float(rv or 999)), abs(float(tv or 999))
                rw3 = r2<t2 if lb else r2>t2
            except: return "\u2014","\u2014"
            tick = f'<font color="{TEAL.hexval()}">\u2713</font>'
            return (tick,"\u2014") if rw3 else ("\u2014",tick)

        def gs2(d,k):
            # Returns raw signed value. abs() only used internally for winner comparison.
            v = (d or {}).get(k.upper(),(d or {}).get(k.lower()))
            return float(v) if v is not None else None

        sr = [[Paragraph("Metric",_S_H3), Paragraph("Reweighing",_S_H3),
               Paragraph("Threshold Adj.",_S_H3), Paragraph("Winner",_S_H3)]]
        cmps = [
            ("SPD After",        gs2(rw_a,"spd"),  gs2(th_a,"spd"),  True),
            ("DI After",         gs2(rw_a,"di"),   gs2(th_a,"di"),   False),
            ("EOD After",        gs2(rw_a,"eod"),  gs2(th_a,"eod"),  True),
            ("AOD After",        gs2(rw_a,"aod"),  gs2(th_a,"aod"),  True),
            ("Accuracy After",   gs2(rw_a,"accuracy"), gs2(th_a,"accuracy"), False),
            ("Bias Reduction %", rw_e.get("bias_reduction_pct"), th_e.get("bias_reduction_pct"), False),
            ("Accuracy Retained %", rw_e.get("accuracy_retained_pct"), th_e.get("accuracy_retained_pct"), False),
        ]
        for lbl2, rv2, tv2, lb2 in cmps:
            rt2, tt2 = wt(rv2, tv2, lb2)
            sr.append([Paragraph(lbl2,_S_BODY),
                        Paragraph(_f(rv2,3) if rv2 is not None else "\u2014",_S_BODY),
                        Paragraph(_f(tv2,3) if tv2 is not None else "\u2014",_S_BODY),
                        Paragraph(f"{rt2} / {tt2}",_S_BODY)])
        story.append(_dark_table(sr,
            col_widths=[CONTENT_W*0.34, CONTENT_W*0.22, CONTENT_W*0.22, CONTENT_W*0.22]))
        return story

    # =================================================================
    # PAGE 4 - RECOMMENDATIONS & ACTION PLAN
    # =================================================================
    def _p4(self, sensitive_attrs, metrics_per_attr, mitigation, pattern_preds) -> list:
        story = []
        story.append(Paragraph("RECOMMENDED ACTION PLAN", _S_H1))
        story.append(HRFlowable(width=CONTENT_W, thickness=1, color=TEAL, spaceAfter=5))
        story.append(Paragraph(
            "Based on the bias findings in this audit, the following specific actions are "
            "recommended, ordered by priority and urgency.", _S_BODY))
        story.append(Spacer(1, 3*mm))

        winner = (mitigation or {}).get("winner","reweigh")
        wr = "Reweighing" if winner=="reweigh" else "Threshold Adjustment"
        w_eff = ((mitigation or {}).get("reweigh" if winner=="reweigh" else "threshold",{})
                 .get("effects",{}))
        br4 = w_eff.get("bias_reduction_pct",0)
        ar4 = w_eff.get("accuracy_retained_pct",100)
        first_attr = sensitive_attrs[0] if sensitive_attrs else "the attribute"
        mit_for_attr = None
        if isinstance(mitigation, dict):
            if first_attr in mitigation and isinstance(mitigation[first_attr], dict) and "reweigh" in mitigation[first_attr]:
                mit_for_attr = mitigation[first_attr]
            elif "reweigh" in mitigation:
                mit_for_attr = mitigation

        winner = (mit_for_attr or {}).get("winner","reweigh") if mit_for_attr else (mitigation or {}).get("winner","reweigh")
        wr = "Reweighing" if winner=="reweigh" else "Threshold Adjustment"
        w_eff = ((mit_for_attr or {}).get("reweigh" if winner=="reweigh" else "threshold",{})
                 .get("effects",{}))
        br4 = w_eff.get("bias_reduction_pct",0) or 0
        ar4 = w_eff.get("accuracy_retained_pct",100) or 100

        top_proxy_name = "correlated features"
        if first_attr in metrics_per_attr:
            prx = metrics_per_attr[first_attr].get("proxy_features", [])
            if prx:
                top_proxy_name = prx[0].get("feature", "correlated features")

        steps = []
        n2 = 1

        # Step 1: Legal compliance
        for attr, m in metrics_per_attr.items():
            if isinstance(m,dict) and m.get("legal_flag"):
                di2 = m.get("DI")
                steps.append((n2, f"Address Legal Compliance Risk \u2014 {attr}",
                    f"The Disparate Impact of <b>{_f(di2)}</b> for '<b>{attr}</b>' falls below "
                    f"the legal 0.8 threshold. Before deploying or continuing to use this model, "
                    f"seek legal review. Document this finding and your remediation plan. "
                    f"Apply <b>{wr}</b> mitigation immediately.",
                    "IMMEDIATE", RED)); n2+=1; break

        # Step 2: Proxy feature
        for attr, m in metrics_per_attr.items():
            if not isinstance(m,dict): continue
            prx = m.get("proxy_features",[])
            if prx:
                tp2=prx[0]; tf2=tp2.get("feature",""); tc2=tp2.get("correlation",0)
                st3 = "strong" if tc2>0.4 else "moderate"
                steps.append((n2, f"Remove or Transform Proxy Feature '{tf2}'",
                    f"The feature '<b>{tf2}</b>' has a {st3} correlation "
                    f"(r=<b>{tc2:.3f}</b>) with '<b>{attr}</b>', meaning it can act as a "
                    f"hidden proxy for the protected attribute. Consider: (1) removing it, "
                    f"(2) replacing it with a neutral alternative, or (3) applying "
                    f"fairness-aware feature selection.",
                    "SHORT-TERM", AMBER)); n2+=1; break

        # Step 3: Apply mitigation
        if br4 > 5:
            step3_body = (
                f"Analysis shows <b>{wr}</b> achieves <b>{br4:.0f}%</b> bias reduction for "
                f"'<b>{first_attr}</b>' while retaining <b>{ar4:.0f}%</b> of model accuracy. "
                f"Apply this technique to your training pipeline."
            )
        else:
            step3_body = (
                f"Apply <b>{wr}</b> to your model training pipeline before retraining. "
                f"This technique rebalances group representation in the training data. "
                f"Note: automated simulation showed limited reduction in this session \u2014 "
                f"results may improve when applied to your full model training process with proper "
                f"feature engineering (e.g. removing proxy feature '<b>{top_proxy_name}</b>' first)."
            )

        steps.append((n2, f"Apply {wr} Before Model Retraining",
            step3_body, "SHORT-TERM", AMBER)); n2+=1

        # Step 4: Monitoring
        steps.append((n2, "Establish Ongoing Fairness Monitoring",
            "Set up quarterly bias audits using FairEnough to track fairness metrics over time. "
            "Bias can drift as the model is retrained on new data. Document audit results "
            "for compliance records and build a fairness review into your model deployment checklist.",
            "ONGOING", PURPLE)); n2+=1

        # Step 5: Underrepresentation
        for attr, m in metrics_per_attr.items():
            if not isinstance(m,dict): continue
            pred = pattern_preds.get(attr,{})
            if pred.get("predicted_cause")=="underrepresentation":
                gs2b = m.get("group_stats",{})
                sorted_g = sorted(gs2b.items(), key=lambda x: x[1].get("positive_rate", 0))
                unp2 = sorted_g[0][0] if sorted_g else (m.get("unprivileged_group","") or "disadvantaged group")
                ud3  = gs2b.get(str(unp2),{})
                cnt2 = ud3.get("count",0); pct2 = ud3.get("pct_of_total",0)
                steps.append((n2, f"Collect More Representative Data for '{unp2}'",
                    f"The group '<b>{unp2}</b>' is underrepresented with only "
                    f"<b>{_comma(cnt2)}</b> records (<b>{pct2:.1f}%</b>). Work to collect more "
                    f"data from this group, or use data augmentation techniques to improve "
                    f"model fairness for this demographic.",
                    "SHORT-TERM", AMBER)); break

        for num, title, body, prio, pcol in steps:
            ns = _s(f"sn_{num}", fontName="Helvetica-Bold", fontSize=14, textColor=TEAL,
                     leading=18, alignment=1)
            ps = _s(f"pp_{num}", fontName="Helvetica-Bold", fontSize=8, textColor=pcol, leading=12)
            step_t = Table([[Paragraph(str(num), ns),
                              [Paragraph(f"<b>{title}</b>", _S_H3),
                               Paragraph(body, _S_BODY),
                               Paragraph(f"Priority: {prio}", ps)]]],
                            colWidths=[CONTENT_W*0.07, CONTENT_W*0.93])
            step_t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), SURF),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ("RIGHTPADDING", (0,0),(-1,-1), 8),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("VALIGN",       (0,0),(-1,-1), "TOP"),
                ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
                ("LINEAFTER",    (0,0),(0,-1),  1, TEAL),
            ]))
            story.append(step_t)
            story.append(Spacer(1, 3*mm))

        # Glossary
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("GLOSSARY", _S_H2))
        gloss = [
            ("SPD", "The raw difference in positive outcome rates between groups. 0 = perfectly fair."),
            ("DI",  "The ratio of positive outcomes. Below 0.8 triggers legal scrutiny in most jurisdictions."),
            ("EOD", "Whether equally deserving people from different groups get equal positive decisions."),
            ("AOD", "A combined measure of both types of prediction errors across demographic groups."),
            ("Reweighing", "A technique that adjusts training data so all groups are represented fairly."),
            ("Threshold Adj.", "A technique that sets different decision cutoffs per group to equalize outcomes."),
            ("Proxy Discrimination", "When a non-protected feature encodes protected information, enabling indirect bias."),
        ]
        gr2 = [[Paragraph("Term",_S_H3), Paragraph("Plain English Definition",_S_H3)]]
        for t2,d2 in gloss:
            gr2.append([Paragraph(t2,_S_BODY), Paragraph(d2,_S_BODY)])
        story.append(_dark_table(gr2, col_widths=[CONTENT_W*0.22, CONTENT_W*0.78]))
        return story

    # =================================================================
    # PAGE 5 - METHODOLOGY & ABOUT
    # =================================================================
    def _p5(self, training_count) -> list:
        story = []
        story.append(Paragraph("METHODOLOGY", _S_H1))
        story.append(HRFlowable(width=CONTENT_W, thickness=1, color=TEAL, spaceAfter=5))

        story.append(Paragraph("How FairEnough Detects Bias", _S_H2))
        story.append(Paragraph(
            "FairEnough uses four legally-recognized fairness metrics to evaluate bias across demographic "
            "groups. These metrics are computed directly from the uploaded dataset and any provided "
            "ML model predictions. No data leaves your session \u2014 analysis runs entirely on your "
            "uploaded file.", _S_BODY))
        story.append(Spacer(1, 2*mm))

        mr = [
            [Paragraph("Metric",_S_H3), Paragraph("Formula",_S_H3),
             Paragraph("Flag When",_S_H3), Paragraph("Legal Standard",_S_H3)],
            [Paragraph("SPD",_S_BODY), Paragraph("P(Y=1|A=priv) \u2212 P(Y=1|A=unpriv)",_S_BODY),
             Paragraph(">0.1 medium, >0.2 high",_S_BODY), Paragraph("None established",_S_BODY)],
            [Paragraph("DI",_S_BODY), Paragraph("P(Y=1|A=unpriv) / P(Y=1|A=priv)",_S_BODY),
             Paragraph("<0.8 = legal risk",_S_BODY), Paragraph("EEOC 80% Rule (29 CFR 1607)",_S_BODY)],
            [Paragraph("EOD",_S_BODY), Paragraph("TPR_priv \u2212 TPR_unpriv",_S_BODY),
             Paragraph(">0.1 medium",_S_BODY), Paragraph("None established",_S_BODY)],
            [Paragraph("AOD",_S_BODY), Paragraph("(TPR_diff + FPR_diff) / 2",_S_BODY),
             Paragraph(">0.1 medium",_S_BODY), Paragraph("None established",_S_BODY)],
        ]
        story.append(_dark_table(mr, col_widths=[
            CONTENT_W*0.10, CONTENT_W*0.34, CONTENT_W*0.24, CONTENT_W*0.32]))
        story.append(Spacer(1, 3*mm))

        story.append(Paragraph("Audit Score Methodology", _S_H2))
        story.append(Paragraph(
            "The Bias Audit Score (0\u2013100) is a composite metric computed as 100 minus weighted "
            "penalties for SPD deviation, DI legal violations, and EOD (when available). "
            "Scores above 85 indicate a fair system. Scores below 50 require immediate remediation.",
            _S_BODY))
        story.append(Spacer(1, 2*mm))

        sc_rows = [
            [Paragraph("Grade",_S_H3), Paragraph("Score Range",_S_H3), Paragraph("Interpretation",_S_H3)],
            [Paragraph("A", _s("ga", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN,  leading=12)),
             Paragraph("85\u2013100",_S_BODY), Paragraph("Fair \u2014 minimal bias detected",_S_BODY)],
            [Paragraph("B", _s("gb", fontName="Helvetica-Bold", fontSize=9, textColor=AMBER,  leading=12)),
             Paragraph("70\u201384",_S_BODY), Paragraph("Minor issues \u2014 monitor and review",_S_BODY)],
            [Paragraph("C", _s("gc", fontName="Helvetica-Bold", fontSize=9, textColor=ORANGE, leading=12)),
             Paragraph("50\u201369",_S_BODY), Paragraph("Moderate bias \u2014 mitigation recommended",_S_BODY)],
            [Paragraph("F", _s("gf", fontName="Helvetica-Bold", fontSize=9, textColor=RED,    leading=12)),
             Paragraph("0\u201349",_S_BODY), Paragraph("High bias \u2014 immediate action required",_S_BODY)],
        ]
        story.append(_dark_table(sc_rows, col_widths=[CONTENT_W*0.08, CONTENT_W*0.20, CONTENT_W*0.72]))
        story.append(Spacer(1, 3*mm))

        story.append(Paragraph("Data Sources & Limitations", _S_H2))
        story.append(Paragraph(
            "This report is based solely on the uploaded dataset. Findings reflect patterns in "
            "historical data and may not capture all forms of bias. "
            "This report does not constitute legal advice. Consult qualified legal counsel before "
            "making compliance decisions based on this report.", _S_BODY))
        story.append(Spacer(1, 3*mm))

        story.append(Paragraph("About FairEnough", _S_H2))
        story.append(Paragraph(
            "FairEnough is an AI-powered bias detection and mitigation platform built to make AI fairness "
            "accessible to all organizations. It uses Google Gemini, Microsoft Fairlearn, "
            "and scikit-learn to surface hidden discrimination in automated decision systems.", _S_BODY))
        story.append(Spacer(1, 6*mm))

        story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=TEAL, spaceAfter=5))
        story.append(Paragraph("Generated by FairEnough. For audit purposes only.",
            _s("ff1", fontName="Helvetica-Bold", fontSize=8, textColor=TEAL, leading=12, alignment=1)))
        story.append(Paragraph("This report does not constitute legal advice.",
            _s("ff2", fontName="Helvetica-Oblique", fontSize=7, textColor=GRAY, leading=10, alignment=1)))
        return story

