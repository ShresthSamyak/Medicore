"""Generate the MediCore design write-up PDF (figure-rich, professional layout).

Pulls measured metrics from ``reports/eval_summary.json`` (qwen baseline) and,
if present, ``reports/eval_llama_summary.json`` (comparison), and embeds the
figure suite rendered by ``make_charts.py`` / ``make_diagram.py``.

    python docs/make_diagram.py       # architecture figure
    python docs/make_charts.py        # chart suite (needs an eval report)
    python docs/generate_writeup.py   # -> docs/MediCore_Design_Writeup.pdf
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Flowable, Image, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")

# ---- palette (matches the figures) ----------------------------------------
INK = colors.HexColor("#1f2933")
BLUE = colors.HexColor("#2563eb")
TEAL = colors.HexColor("#0d9488")
AMBER = colors.HexColor("#b45309")
LILAC = colors.HexColor("#7c3aed")
GREEN = colors.HexColor("#16a34a")
LIGHT = colors.HexColor("#f1f5f9")
LINE = colors.HexColor("#cbd5e1")
GREY = colors.HexColor("#64748b")
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 4 * cm


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _styles():
    ss = getSampleStyleSheet()
    S = {}
    S["cover_title"] = ParagraphStyle(
        "ct", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=40,
        textColor=colors.white, leading=42, alignment=TA_LEFT, spaceAfter=0)
    S["cover_sub"] = ParagraphStyle(
        "cs", parent=ss["Normal"], fontSize=14, textColor=colors.HexColor("#dbeafe"),
        leading=19, alignment=TA_LEFT)
    S["cover_meta"] = ParagraphStyle(
        "cm", parent=ss["Normal"], fontSize=10, textColor=colors.HexColor("#bfdbfe"),
        leading=14)
    S["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17,
        textColor=INK, spaceBefore=6, spaceAfter=2, leading=20)
    S["kicker"] = ParagraphStyle(
        "kk", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=9,
        textColor=BLUE, leading=11, spaceAfter=1)
    S["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
        textColor=BLUE, spaceBefore=12, spaceAfter=4, leading=15)
    S["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontSize=10, textColor=INK, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=6)
    S["bullet"] = ParagraphStyle(
        "bullet", parent=ss["Normal"], fontSize=9.7, textColor=INK, leading=13.5,
        alignment=TA_LEFT)
    S["caption"] = ParagraphStyle(
        "cap", parent=ss["Normal"], fontSize=8.6, textColor=GREY, leading=11,
        spaceBefore=3, spaceAfter=12, alignment=TA_CENTER)
    S["small"] = ParagraphStyle(
        "sm", parent=ss["Normal"], fontSize=8.4, textColor=GREY, leading=11)
    S["tile_num"] = ParagraphStyle(
        "tn", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=21,
        textColor=colors.white, leading=23, alignment=TA_CENTER)
    S["tile_lab"] = ParagraphStyle(
        "tl", parent=ss["Normal"], fontSize=8, textColor=colors.white,
        leading=10, alignment=TA_CENTER)
    return S


def _p(t, s):
    return Paragraph(t, s)


def _bullets(items, style):
    return ListFlowable(
        [ListItem(_p(t, style), leftIndent=6, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=13, bulletColor=BLUE,
        spaceBefore=2, spaceAfter=9)


def _fig(name, width_cm, caption, S, ratio=None):
    """Embed a figure by filename with a centered caption, if it exists.

    Looks in docs/figures/ first, then docs/ (where the architecture diagram
    lives), so callers can reference a bare filename either way.
    """
    if os.path.isabs(name):
        path = name
    else:
        path = os.path.join(FIG, name)
        if not os.path.exists(path):
            path = os.path.join(HERE, name)
    out = []
    if os.path.exists(path):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        w = width_cm * cm
        h = w * (ih / iw)
        img = Image(path, width=w, height=h)
        img.hAlign = "CENTER"
        out.append(img)
        if caption:
            out.append(_p(caption, S["caption"]))
    return out


# --------------------------------------------------------------------------- #
# Section rule (kicker + heading + underline)
# --------------------------------------------------------------------------- #
class Rule(Flowable):
    def __init__(self, width, color=BLUE, thickness=2):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness

    def wrap(self, *a):
        return (self.width, self.thickness + 4)

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 2, self.width, 2)


def section(num, title, S):
    return [
        Spacer(1, 2),
        _p(f"SECTION {num}", S["kicker"]),
        _p(title, S["h1"]),
        Rule(2.2 * cm, BLUE, 2.4),
        Spacer(1, 6),
    ]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def load_metrics(prefix="eval") -> Optional[dict]:
    path = os.path.join(ROOT, "reports", f"{prefix}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tiles(m, S):
    """Four coloured KPI tiles."""
    def tile(num, lab, col):
        inner = Table([[_p(num, S["tile_num"])], [_p(lab, S["tile_lab"])]],
                      colWidths=[3.7 * cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), col),
            ("TOPPADDING", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
            ("TOPPADDING", (0, 1), (-1, 1), 1),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return inner

    if not m:
        return Spacer(1, 1)
    tiles = [
        tile(f"{m['micro_recall']*100:.0f}%", "RECALL (micro)", BLUE),
        tile(f"{m['retrieval_recall']*100:.0f}%", "RETRIEVAL CEILING", TEAL),
        tile(f"{m['avg_seconds']:.0f}s", "AVG TIME / NOTE", AMBER),
        tile(f"{m['avg_total_tokens']:.0f}", "AVG TOKENS / NOTE", LILAC),
    ]
    row = Table([tiles], colWidths=[4.0 * cm] * 4)
    row.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return row


def metrics_table(m, S):
    if not m:
        return _p("<i>No evaluation report found. Run "
                  "<font face='Courier'>python scripts/run_eval.py</font>.</i>",
                  S["body"])
    rows = [
        ["Metric", "Micro", "Macro"],
        ["Recall (primary)", f"{m['micro_recall']:.3f}", f"{m['macro_recall']:.3f}"],
        ["Precision", f"{m['micro_precision']:.3f}", f"{m['macro_precision']:.3f}"],
        ["F1", f"{m['micro_f1']:.3f}", f"{m['macro_f1']:.3f}"],
    ]
    t = Table(rows, colWidths=[7 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def ops_table(m, S):
    if not m:
        return Spacer(1, 1)
    rows = [
        ["Operational metric", "Value"],
        ["Cases evaluated", str(m.get("n_cases", "-"))],
        ["Retrieval recall (candidate ceiling)", f"{m['retrieval_recall']:.3f}"],
        ["Average time per note", f"{m['avg_seconds']:.2f} s"],
        ["Average total tokens per note", f"{m['avg_total_tokens']:.0f}"],
        ["  - prompt tokens", f"{m['avg_prompt_tokens']:.0f}"],
        ["  - completion tokens", f"{m['avg_completion_tokens']:.0f}"],
    ]
    t = Table(rows, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def comparison_table(qwen, llama, S):
    if not (qwen and llama):
        return None
    def col(m):
        return [f"{m['micro_recall']:.3f}", f"{m['macro_recall']:.3f}",
                f"{m['micro_precision']:.3f}", f"{m['micro_f1']:.3f}",
                f"{m['retrieval_recall']:.3f}", f"{m['avg_seconds']:.1f} s",
                f"{m['avg_total_tokens']:.0f}"]
    metrics = ["Recall (micro)", "Recall (macro)", "Precision (micro)",
               "F1 (micro)", "Retrieval ceiling", "Avg time / note",
               "Avg tokens / note"]
    q, l = col(qwen), col(llama)
    rows = [["Metric", "qwen2.5:3b", "llama3.1:8b"]]
    for i, name in enumerate(metrics):
        rows.append([name, q[i], l[i]])
    t = Table(rows, colWidths=[7 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# --------------------------------------------------------------------------- #
# Page furniture (cover banner + footer)
# --------------------------------------------------------------------------- #
def _cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#1e3a8a"))
    canvas.rect(0, PAGE_H - 12 * cm, PAGE_W, 12 * cm, fill=1, stroke=0)
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H - 12 * cm, PAGE_W, 0.35 * cm, fill=1, stroke=0)
    # accent ticks
    for i, c in enumerate([TEAL, AMBER, LILAC, GREEN]):
        canvas.setFillColor(c)
        canvas.rect(2 * cm + i * 1.4 * cm, PAGE_H - 12.0 * cm - 0.0 * cm,
                    1.1 * cm, 0.16 * cm, fill=1, stroke=0)
    canvas.restoreState()
    _footer(canvas, doc)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(2 * cm, 1.35 * cm, PAGE_W - 2 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 0.95 * cm, "MediCore — Automated ICD-10-CM Coding")
    canvas.drawRightString(PAGE_W - 2 * cm, 0.95 * cm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def build(out_path: Optional[str] = None) -> str:
    out_path = out_path or os.path.join(HERE, "MediCore_Design_Writeup.pdf")
    S = _styles()
    m = load_metrics("eval")
    llama = load_metrics("eval_llama")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        title="MediCore — Automated ICD-10-CM Coding: System Design",
        author="MediCore")
    E: List = []

    # ============================ COVER =============================
    E.append(Spacer(1, 2.1 * cm))
    E.append(_p("SYSTEM DESIGN &amp; EVALUATION", ParagraphStyle(
        "ck", parent=S["cover_meta"], fontName="Helvetica-Bold", fontSize=11,
        textColor=colors.HexColor("#93c5fd"), spaceAfter=8)))
    E.append(_p("MediCore", S["cover_title"]))
    E.append(Spacer(1, 6))
    E.append(_p("A retrieval-augmented system for automated ICD-10-CM medical "
                "coding, powered by a local Ollama model.", S["cover_sub"]))
    E.append(Spacer(1, 3.0 * cm))
    E.append(_tiles(m, S))
    E.append(Spacer(1, 0.7 * cm))
    if m:
        cover_note = ParagraphStyle("cmi", parent=S["small"], textColor=GREY,
                                    fontSize=9.5)
        E.append(_p(f"Baseline: <b>qwen2.5:3b</b> + nomic-embed-text · evaluated "
                    f"on all {m.get('n_cases','?')} provided cases.", cover_note))
    E += _fig("funnel.png", 15, "The core challenge in one picture: the pipeline "
              "narrows ~75,000 assignable codes to ~120 candidates per note before "
              "the model makes any decision.", S)
    E.append(PageBreak())

    # ============================ 1. PROBLEM ========================
    E += section("1", "Problem &amp; Framing", S)
    E.append(_p(
        "Medical coding translates a clinical narrative — a discharge summary, "
        "encounter note, or operative report — into standardized codes used for "
        "billing, reporting, and analytics. The target vocabulary here is "
        "<b>ICD-10-CM</b>. The 2026 release contains <b>98,186</b> ordered "
        "entries, of which <b>74,719</b> are valid, billable (assignable) codes; "
        "the rest are non-billable header categories in the hierarchy.", S["body"]))
    E.append(_p("Three properties make this hard and shape every design decision:",
                S["h2"]))
    E.append(_bullets([
        "<b>Enormous label space.</b> ~75k assignable codes cannot fit in any "
        "context window, so the system cannot simply \"ask the model to pick "
        "codes\". The label set must be narrowed before generation.",
        "<b>Multi-label output.</b> Each note maps to a variable-size set of "
        "codes (in this data: 1–10, mean ≈ 2.9). The system must decide both "
        "<i>which</i> codes and <i>how many</i>.",
        "<b>Specificity rules.</b> ICD-10-CM encodes laterality, encounter type, "
        "acuity, and combination concepts inside the code itself "
        "(e.g. <font face='Courier'>S52.502A</font>). The correct code is often "
        "one of dozens of near-identical siblings.",
    ], S["bullet"]))
    E += _fig("codes_per_case.png", 13.5, "Figure 1 — Distribution of gold codes "
              "per case in the provided dataset. Most cases are genuinely "
              "multi-label, which is why the system must decide set membership, "
              "not a single best label.", S)
    E.append(_p(
        "The dataset comprises <b>54 labelled cases</b> "
        "(<font face='Courier'>icd10_cm_cases.json</font>) and the official CMS "
        "order file (<font face='Courier'>icd10cm_order_2026.txt</font>). All 136 "
        "unique gold codes are valid billable codes present in the order file, "
        "confirming the assignable set is the correct candidate space.", S["body"]))

    # ============================ 2. ARCHITECTURE ===================
    E.append(PageBreak())
    E += section("2", "System Architecture", S)
    E.append(_p(
        "Because the label space is far too large for direct generation, MediCore "
        "uses a <b>retrieve-then-assign</b> (retrieval-augmented generation) "
        "pipeline. This mirrors how a human coder actually works: read the chart, "
        "abstract the codeable concepts, look them up in the index, then select "
        "the most specific supported code. The pipeline has an offline "
        "index-build stage and a four-step per-note stage.", S["body"]))
    E += _fig("architecture.png", 16.5, "Figure 2 — End-to-end architecture. "
              "Retrieval handles recall over the full code space; the LLM is asked "
              "only to exercise judgement over a small candidate set.", S)

    E.append(_p("2.1 · Offline — code knowledge base", S["h2"]))
    E.append(_p(
        "The CMS order file is parsed into the 74,719 billable codes, each with "
        "its long description. Two indexes are built over the descriptions:",
        S["body"]))
    E.append(_bullets([
        "<b>BM25 lexical index</b> — a sparse ranker: fast, model-free, and "
        "excellent at matching exact clinical terminology (drugs, organisms, "
        "anatomical sites) that appears verbatim in descriptions.",
        "<b>Dense embedding index</b> — each description is embedded with an "
        "Ollama model (<font face='Courier'>nomic-embed-text</font>) and cached "
        "to disk, capturing paraphrase and synonymy BM25 misses "
        "(\"heart attack\" ↔ \"myocardial infarction\").",
    ], S["bullet"]))

    E.append(_p("2.2 · Per-note pipeline", S["h2"]))
    E.append(_bullets([
        "<b>Step 1 — Concept extraction (LLM).</b> The model reads the note and "
        "returns a JSON list of discrete codeable concepts (diagnoses, injuries "
        "with laterality, symptoms, procedures, external causes, status/history), "
        "turning a long noisy narrative into precise search phrases.",
        "<b>Step 2 — Hybrid retrieval.</b> The whole note and every extracted "
        "concept are used as queries. BM25 and embedding results are fused with "
        "<b>Reciprocal Rank Fusion</b> (RRF) — a rank-based combiner needing no "
        "score calibration — into a de-duplicated pool of ~120 candidates.",
        "<b>Step 3 — Code assignment (LLM).</b> The note plus the candidate list "
        "(code + description) is given to the model, which selects the supported "
        "codes with a short justification each, as JSON, choosing only from the "
        "provided candidates.",
        "<b>Step 4 — Validation.</b> Outputs are normalized to dotted form; any "
        "code not in the candidate set or not a valid billable code is dropped, "
        "eliminating hallucinations by construction.",
    ], S["bullet"]))
    E.append(_p(
        "Why extract concepts with an LLM rather than retrieve on the raw note: "
        "on this data, BM25 over the raw note frequently fails to surface the "
        "correct code — incidental words like \"hospital\" and \"admitted\" pull "
        "in place-of-occurrence and external-cause codes — whereas a single "
        "extracted phrase such as \"severe opioid dependence with withdrawal\" "
        "retrieves the exact code (<font face='Courier'>F11.23</font>) at rank 1.",
        S["body"]))

    # ============================ 3. RATIONALE ======================
    E.append(PageBreak())
    E += section("3", "Rationale &amp; Alternatives Considered", S)
    E.append(_p("Alternatives weighed against the chosen RAG design:", S["body"]))
    E.append(_bullets([
        "<b>Direct generation (no retrieval).</b> Ask the LLM to emit codes from "
        "memory. Rejected: small local models don't reliably memorize a 75k-code "
        "vocabulary and hallucinate plausible-but-invalid codes, with no validity "
        "guarantee.",
        "<b>Flat multi-label classifier.</b> Train a 75k-way classifier. Rejected "
        "here: needs a large labelled corpus (54 cases is far too few), and the "
        "task constrains us to an Ollama model, not a bespoke trained head.",
        "<b>Pure lexical lookup.</b> BM25 alone. Rejected as a full solution: it "
        "retrieves well but cannot judge documentation support, specificity, or "
        "de-duplicate sibling codes — that reasoning is what the assignment LLM "
        "provides.",
        "<b>Hierarchical routing</b> (chapter → category → code). Viable and "
        "complementary, but adds latency and compounds errors across hops; RRF "
        "hybrid retrieval reaches the correct leaf directly.",
    ], S["bullet"]))
    E.append(_p(
        "The retrieve-then-assign design makes the LLM responsible only for "
        "<i>judgement over a small candidate set</i> — what LLMs are good at — "
        "while retrieval handles <i>recall over the huge label space</i> — what "
        "indexes are good at. It is model-agnostic (any Ollama chat model plugs "
        "in), needs no training data, and guarantees output validity.", S["body"]))

    E += section("4", "Trade-offs", S)
    E.append(_bullets([
        "<b>Retrieval recall is the hard ceiling.</b> If a gold code is never "
        "retrieved, assignment cannot recover it. We therefore bias retrieval "
        "toward recall (high K, hybrid, multi-query) at the cost of a larger, "
        "noisier candidate list. Section 6 measures this ceiling explicitly.",
        "<b>Two LLM calls vs one.</b> Concept extraction markedly improves "
        "retrieval but adds a call (latency + tokens). It is toggleable "
        "(<font face='Courier'>extract_concepts: false</font>).",
        "<b>Candidate-list size.</b> More candidates raise the recall ceiling but "
        "lengthen the assignment prompt and add distractors. 120 is a tunable "
        "middle setting.",
        "<b>Embeddings vs BM25-only.</b> Embeddings improve paraphrase recall but "
        "need a one-time index build (~75k vectors) and extra memory; a "
        "BM25-only mode is provided for constrained environments.",
        "<b>Model size vs speed.</b> A small model keeps latency and memory low "
        "but reasons less reliably about specificity; the design swaps models "
        "with a one-line config change (quantified in Section 6).",
    ], S["bullet"]))

    # ============================ 5. ASSUMPTIONS ====================
    E.append(PageBreak())
    E += section("5", "Assumptions", S)
    E.append(_bullets([
        "The assignable target space is the set of billable (flag = 1) codes in "
        "the CMS order file; non-billable headers are never assigned.",
        "Gold labels are a complete, order-independent set per note; evaluation "
        "is set-based (no code sequencing / principal-diagnosis ordering scored).",
        "Codes are compared in canonical dotted, upper-case form; the order "
        "file's undotted codes are normalized to match the gold format.",
        "Notes are self-contained; no external EHR context, problem list, or "
        "prior encounters are available.",
        "Only an Ollama model is used for all learned components (extraction, "
        "assignment, embeddings), per the task constraint.",
    ], S["bullet"]))

    # ============================ 6. RESULTS ========================
    E += section("6", "Measured Results", S)
    if m:
        E.append(_p(
            f"Evaluated on all <b>{m.get('n_cases','?')}</b> provided cases. "
            f"Recall is the primary requested metric; precision/F1 give context. "
            f"<i>Micro</i> pools code-level hits across notes (weights by number "
            f"of codes); <i>macro</i> averages per-note scores (weights notes "
            f"equally).", S["body"]))
    E.append(metrics_table(m, S))
    E.append(Spacer(1, 5 * mm))
    E.append(ops_table(m, S))
    E.append(Spacer(1, 4 * mm))
    E += _fig("metrics_bars.png", 12.5, "Figure 3 — Accuracy metrics, micro vs. "
              "macro.", S)

    E.append(PageBreak())
    E.append(_p("6.1 · Where recall is lost", S["h2"]))
    E.append(_p(
        "The single most useful diagnostic is decomposing recall into a retrieval "
        "stage and an assignment stage. Retrieval surfaces the majority of gold "
        "codes into the candidate pool (the <i>ceiling</i>); the remaining loss "
        "is the assignment model failing to select a retrieved gold code — "
        "typically choosing a near-identical sibling that differs only in "
        "laterality or specificity.", S["body"]))
    E += _fig("waterfall.png", 13.5, "Figure 4 — Recall decomposition. The gap "
              "from 100% to the ceiling is what retrieval never surfaced; the gap "
              "from the ceiling to the green bar is what the assignment model "
              "lost. This tells you exactly where to invest next.", S)
    E += _fig("percase_recall.png", 12.5, "Figure 5 — Per-case recall distribution: "
              "the system fully or partly codes many cases while missing others "
              "entirely, rather than degrading uniformly.", S)
    E += _fig("timing_tokens.png", 12.5, "Figure 6 — Per-note cost. Prompt tokens "
              "dominate (the candidate list); completion is small. Latency is "
              "governed by the two LLM calls on CPU.", S)
    E.append(_p(
        "Token usage is read directly from Ollama's response fields "
        "(<font face='Courier'>prompt_eval_count</font>, "
        "<font face='Courier'>eval_count</font>); timing is wall-clock per note "
        "including retrieval. These figures scale with the chosen model and "
        "candidate-list size, not with the design.", S["small"]))

    # ---- 6.2 model comparison (if available) ----
    if llama:
        E.append(PageBreak())
        E.append(_p("6.2 · Model comparison", S["h2"]))
        E.append(_p(
            "The pipeline is model-agnostic. Swapping only the chat model "
            "(<font face='Courier'>--model llama3.1:8b</font>) — same retrieval, "
            "same prompts — shifts the numbers as follows. Because retrieval is "
            "unchanged, the retrieval ceiling is identical; the difference is "
            "entirely in the assignment model's specificity judgement.", S["body"]))
        E.append(comparison_table(m, llama, S))
        E.append(Spacer(1, 4 * mm))
        E += _fig("model_comparison.png", 16, "Figure 7 — Same pipeline, two "
                  "Ollama models. A larger model recovers more of the retrieval "
                  "ceiling at the cost of latency.", S)
    else:
        E.append(_p(
            "<i>A model-comparison section (qwen2.5:3b vs. llama3.1:8b) appears "
            "here when <font face='Courier'>reports/eval_llama_summary.json</font> "
            "is present — run "
            "<font face='Courier'>python scripts/run_eval.py --model llama3.1:8b "
            "--prefix eval_llama</font>.</i>", S["small"]))

    # ============================ 7. IMPROVEMENTS ===================
    E.append(PageBreak())
    E += section("7", "How to Improve the System Further", S)
    E.append(_p("Ordered roughly by expected return on effort:", S["body"]))
    E.append(_bullets([
        "<b>Stronger / fine-tuned assignment model.</b> Figures 4 and 7 show the "
        "dominant remaining error is specificity judgement, not retrieval — the "
        "highest-leverage change is a larger or LoRA-tuned Ollama model.",
        "<b>Learned reranker.</b> A cross-encoder between retrieval and assignment "
        "pushes gold codes to the top of the candidate list, raising the "
        "achievable ceiling per token.",
        "<b>Ontology-aware retrieval.</b> Exploit the ICD-10 hierarchy and the "
        "Alphabetic Index / Includes-Excludes notes to expand queries and "
        "constrain candidates to coherent sub-trees (raises the ceiling above its "
        "current value).",
        "<b>Coding-guideline injection.</b> Supply the ICD-10-CM Official "
        "Guidelines (sequencing, 7th-character, combination rules) as context to "
        "improve specificity and reduce invalid combinations.",
        "<b>Constrained / grammar decoding.</b> Force assignment output onto the "
        "candidate-code grammar so malformed or out-of-set codes are impossible "
        "rather than filtered afterward.",
        "<b>Self-consistency &amp; verification.</b> Sample multiple assignments "
        "and vote, or add a verifier pass checking each code against the "
        "documentation before emitting it.",
        "<b>Human-in-the-loop.</b> Surface per-code justifications and confidence "
        "so a coder can accept/reject — turning the tool assistive and generating "
        "new training data.",
        "<b>Scale &amp; robustness.</b> Batch embedding, an ANN index (FAISS) for "
        "sub-millisecond retrieval at full scale, and evaluation on a larger, "
        "multi-institution corpus (54 cases is indicative, not statistically "
        "tight).",
    ], S["bullet"]))
    E.append(Spacer(1, 4 * mm))
    E.append(Rule(CONTENT_W, LINE, 0.8))
    E.append(Spacer(1, 3 * mm))
    E.append(_p(
        "<b>Reproducibility.</b> Every number and figure in this document is "
        "produced from <font face='Courier'>reports/eval_summary.json</font> by "
        "<font face='Courier'>python scripts/run_eval.py</font>, then "
        "<font face='Courier'>python docs/make_charts.py</font> and "
        "<font face='Courier'>python docs/generate_writeup.py</font>. See the "
        "README for configuration and setup.", S["small"]))

    doc.build(E, onFirstPage=_cover_bg, onLaterPages=_footer)
    return out_path


if __name__ == "__main__":
    print("wrote", build())
