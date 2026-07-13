"""Generate the MediCore design write-up PDF.

Pulls measured metrics from ``reports/eval_summary.json`` (if present) so the
results section always reflects the latest evaluation run. Uses reportlab
(pure Python) plus the matplotlib architecture diagram, so no system-level PDF
tooling is required.

    python docs/generate_writeup.py            # -> docs/MediCore_Design_Writeup.pdf
"""
from __future__ import annotations

import json
import os
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INK = colors.HexColor("#1f2933")
BLUE = colors.HexColor("#2563eb")
TEAL = colors.HexColor("#0d9488")
AMBER = colors.HexColor("#b45309")
LIGHT = colors.HexColor("#f1f5f9")
GREY = colors.HexColor("#64748b")


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontName="Helvetica-Bold",
        fontSize=24, textColor=INK, spaceAfter=6, leading=28)
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontSize=12, textColor=GREY,
        spaceAfter=18, leading=16)
    styles["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
        textColor=BLUE, spaceBefore=16, spaceAfter=7, leading=18)
    styles["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12,
        textColor=INK, spaceBefore=10, spaceAfter=4, leading=15)
    styles["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontSize=10, textColor=INK, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=6)
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=ss["Normal"], fontSize=10, textColor=INK, leading=14,
        alignment=TA_LEFT)
    styles["small"] = ParagraphStyle(
        "small", parent=ss["Normal"], fontSize=8.5, textColor=GREY, leading=11)
    styles["caption"] = ParagraphStyle(
        "caption", parent=ss["Normal"], fontSize=8.5, textColor=GREY,
        leading=11, spaceBefore=3, spaceAfter=10, alignment=TA_LEFT)
    styles["code"] = ParagraphStyle(
        "code", parent=ss["Normal"], fontName="Courier", fontSize=8.5,
        textColor=INK, leading=12, backColor=LIGHT, borderPadding=6,
        spaceAfter=8)
    return styles


def _p(text, style):
    return Paragraph(text, style)


def _bullets(items, style, bullet="•"):
    return ListFlowable(
        [ListItem(_p(t, style), leftIndent=8, value=bullet) for t in items],
        bulletType="bullet", start=bullet, leftIndent=14, spaceBefore=2,
        spaceAfter=8,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def load_metrics(reports_dir: str) -> Optional[dict]:
    path = os.path.join(reports_dir, "eval_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, pct=False, dec=3):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.{dec}f}"


def metrics_table(m: Optional[dict], styles):
    """Build the results table; falls back to a placeholder note if unmeasured."""
    if not m:
        return _p(
            "<i>No evaluation report found yet. Run "
            "<font face='Courier'>python scripts/run_eval.py</font> to populate "
            "measured recall, timing, and token usage; then regenerate this "
            "document.</i>", styles["body"])

    header = ["Metric", "Micro", "Macro"]
    rows = [
        header,
        ["Recall (primary)", _fmt(m.get("micro_recall")), _fmt(m.get("macro_recall"))],
        ["Precision", _fmt(m.get("micro_precision")), _fmt(m.get("macro_precision"))],
        ["F1", _fmt(m.get("micro_f1")), _fmt(m.get("macro_f1"))],
    ]
    t = Table(rows, colWidths=[7 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def ops_table(m: Optional[dict], styles):
    if not m:
        return Spacer(1, 1)
    rows = [
        ["Operational metric", "Value"],
        ["Cases evaluated", str(m.get("n_cases", "—"))],
        ["Retrieval recall (candidate ceiling)", _fmt(m.get("retrieval_recall"))],
        ["Average time per note", f"{m.get('avg_seconds', 0):.2f} s"],
        ["Average tokens per note (total)", f"{m.get('avg_total_tokens', 0):.0f}"],
        ["  · prompt tokens", f"{m.get('avg_prompt_tokens', 0):.0f}"],
        ["  · completion tokens", f"{m.get('avg_completion_tokens', 0):.0f}"],
    ]
    t = Table(rows, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def build(out_path: Optional[str] = None) -> str:
    out_path = out_path or os.path.join(HERE, "MediCore_Design_Writeup.pdf")
    s = _styles()
    m = load_metrics(os.path.join(ROOT, "reports"))

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="MediCore — Automated ICD-10-CM Coding: System Design",
        author="MediCore",
    )
    E = []  # flowables

    # ---- Title ----------------------------------------------------------
    E.append(_p("MediCore", s["title"]))
    E.append(_p("A Retrieval-Augmented System for Automated ICD-10-CM "
                "Medical Coding with a Local Ollama Model", s["subtitle"]))
    E.append(_p(
        "This document describes the design of a system that reads free-text "
        "clinical documentation and assigns the corresponding ICD-10-CM "
        "diagnosis codes. It covers the problem framing, the chosen "
        "architecture and why it was chosen, the trade-offs and assumptions "
        "made, measured results on the provided dataset, and a roadmap for "
        "improving the system further.", s["body"]))

    # ---- 1. Problem framing --------------------------------------------
    E.append(_p("1. Problem &amp; Framing", s["h1"]))
    E.append(_p(
        "Medical coding is the task of translating a clinical narrative "
        "(discharge summary, encounter note, operative report) into a set of "
        "standardized codes used for billing, reporting, and analytics. The "
        "target vocabulary here is <b>ICD-10-CM</b>. The 2026 code set contains "
        "<b>98,186</b> ordered entries, of which <b>74,719</b> are valid, "
        "billable (assignable) codes; the remainder are non-billable header "
        "categories in the code hierarchy.", s["body"]))
    E.append(_p(
        "Three properties make this hard and shape every design decision:", s["h2"]))
    E.append(_bullets([
        "<b>Enormous label space.</b> ~75k assignable codes cannot fit in any "
        "model context window, so the system cannot simply \"ask the model to "
        "pick codes\". The label set must be narrowed before generation.",
        "<b>Multi-label output.</b> Each note maps to a variable-size set of "
        "codes (in the provided data: 1–10, mean ≈ 2.9). The system must decide "
        "both <i>which</i> and <i>how many</i>.",
        "<b>Specificity rules.</b> ICD-10-CM encodes laterality, encounter type "
        "(initial/subsequent/sequela), acuity, and combination concepts in the "
        "code itself (e.g. <font face='Courier'>S52.502A</font>). The correct "
        "code is often one of dozens of near-identical siblings.",
    ], s["bullet"]))
    E.append(_p(
        "The dataset provided consists of <b>54 labelled cases</b> "
        "(<font face='Courier'>icd10_cm_cases.json</font>) and the official CMS "
        "order file (<font face='Courier'>icd10cm_order_2026.txt</font>). All "
        "136 unique gold codes are valid billable codes present in the order "
        "file, confirming the assignable set is the correct candidate space.",
        s["body"]))

    # ---- 2. Architecture ------------------------------------------------
    E.append(_p("2. System Architecture", s["h1"]))
    E.append(_p(
        "Because the label space is far too large for direct generation, "
        "MediCore uses a <b>retrieve-then-assign</b> (retrieval-augmented "
        "generation) pipeline. This mirrors how a human coder actually works: "
        "read the chart, abstract the codeable concepts, look them up in the "
        "index, then select the most specific supported code. The pipeline has "
        "an offline index-build stage and a four-step per-note stage.",
        s["body"]))

    diagram = os.path.join(HERE, "architecture.png")
    if os.path.exists(diagram):
        E.append(Image(diagram, width=16 * cm, height=16 * cm * 0.655))
        E.append(_p("Figure 1 — MediCore end-to-end architecture. The candidate "
                    "space is compressed from ~75k codes to ~120 per note before "
                    "the assignment LLM is invoked.", s["caption"]))

    E.append(_p("2.1 Offline — code knowledge base", s["h2"]))
    E.append(_p(
        "The CMS order file is parsed into the 74,719 billable codes, each with "
        "its long description. Two indexes are built over the descriptions:",
        s["body"]))
    E.append(_bullets([
        "<b>BM25 lexical index</b> — a classic sparse ranker. It is fast, needs "
        "no model, and is excellent at matching exact clinical terminology "
        "(drug names, organisms, anatomical sites) that appears verbatim in "
        "descriptions.",
        "<b>Dense embedding index</b> — each description is embedded with an "
        "Ollama embedding model (<font face='Courier'>nomic-embed-text</font>) "
        "and cached to disk. This captures paraphrase and synonymy that BM25 "
        "misses (\"heart attack\" ↔ \"myocardial infarction\").",
    ], s["bullet"]))

    E.append(_p("2.2 Per-note pipeline", s["h2"]))
    E.append(_bullets([
        "<b>Step 1 — Concept extraction (LLM).</b> The model reads the note and "
        "returns a JSON list of discrete codeable concepts (diagnoses, "
        "injuries with laterality, symptoms, procedures, external causes, "
        "status/history). This converts a long noisy narrative into precise "
        "search phrases — critical, because BM25 on the raw note is dominated "
        "by irrelevant tokens.",
        "<b>Step 2 — Hybrid retrieval.</b> The whole note and every extracted "
        "concept are used as queries. BM25 and embedding results are fused with "
        "<b>Reciprocal Rank Fusion</b> (RRF), a rank-based combiner that needs "
        "no score calibration, yielding a de-duplicated candidate pool of "
        "~120 codes.",
        "<b>Step 3 — Code assignment (LLM).</b> The note plus the candidate "
        "list (code + description) is given to the model, which selects the "
        "codes the documentation supports, each with a short justification, as "
        "JSON. The model chooses only from the provided candidates.",
        "<b>Step 4 — Validation.</b> Outputs are normalized to dotted form and "
        "any code not in the candidate set / not a valid billable code is "
        "dropped, eliminating hallucinated codes.",
    ], s["bullet"]))

    E.append(_p(
        "Why concept extraction is done with an LLM rather than raw retrieval: "
        "on the provided data, BM25 over the raw note frequently fails to "
        "surface the correct code (the narrative's incidental words — "
        "\"hospital\", \"admitted\" — pull in place-of-occurrence and "
        "external-cause codes), whereas a single extracted phrase such as "
        "\"severe opioid dependence with withdrawal\" retrieves the exact code "
        "(<font face='Courier'>F11.23</font>) at rank 1.", s["body"]))

    # ---- 3. Why this architecture --------------------------------------
    E.append(PageBreak())
    E.append(_p("3. Rationale &amp; Alternatives Considered", s["h1"]))
    E.append(_p("Alternatives weighed against the chosen RAG design:", s["body"]))
    E.append(_bullets([
        "<b>Direct generation (no retrieval).</b> Ask the LLM to emit ICD-10 "
        "codes from parametric memory. Rejected: small local models do not "
        "reliably memorize a 75k-code vocabulary and hallucinate plausible but "
        "invalid codes; there is also no way to guarantee validity.",
        "<b>Flat multi-label classifier.</b> Train a 75k-way classifier. "
        "Rejected here: needs a large labelled corpus (54 cases is far too "
        "few), and the task constrains us to an Ollama model rather than a "
        "bespoke trained head.",
        "<b>Pure lexical lookup.</b> BM25 alone. Rejected as a full solution: "
        "it retrieves candidates well but cannot judge documentation support, "
        "specificity, or de-duplicate sibling codes — that reasoning is what "
        "the assignment LLM provides.",
        "<b>Hierarchical / two-level routing</b> (chapter → category → code). "
        "Viable and complementary, but adds latency and error-compounding "
        "across hops; RRF hybrid retrieval reaches the correct leaf directly.",
    ], s["bullet"]))
    E.append(_p(
        "The retrieve-then-assign design keeps the LLM responsible only for "
        "<i>judgement over a small candidate set</i> — the thing LLMs are good "
        "at — while retrieval handles <i>recall over the huge label space</i> — "
        "the thing indexes are good at. It is model-agnostic (any Ollama chat "
        "model plugs in), needs no training data, and guarantees output "
        "validity by construction.", s["body"]))

    # ---- 4. Trade-offs --------------------------------------------------
    E.append(_p("4. Trade-offs", s["h1"]))
    E.append(_bullets([
        "<b>Retrieval recall is the hard ceiling.</b> If a gold code is never "
        "retrieved, the assignment stage cannot recover it. We therefore bias "
        "retrieval toward recall (high K, hybrid, multi-query) at the cost of a "
        "larger, noisier candidate list. The reported <i>retrieval recall</i> "
        "metric makes this ceiling explicit.",
        "<b>Two LLM calls vs one.</b> Concept extraction markedly improves "
        "retrieval but adds one call (latency + tokens). It can be disabled "
        "(<font face='Courier'>extract_concepts: false</font>) to trade "
        "accuracy for speed.",
        "<b>Candidate-list size.</b> More candidates → higher recall ceiling "
        "but a longer, costlier assignment prompt and more distractors for the "
        "model. 120 is a middle setting, tunable in config.",
        "<b>Embeddings vs BM25-only.</b> Embeddings improve paraphrase recall "
        "but require a one-time index build (~75k vectors) and extra memory. "
        "A BM25-only mode is provided for constrained environments.",
        "<b>Model size vs speed.</b> A small 3B model keeps latency and memory "
        "low but reasons less reliably about specificity than a larger model; "
        "the design lets you swap models without code changes.",
    ], s["bullet"]))

    # ---- 5. Assumptions -------------------------------------------------
    E.append(_p("5. Assumptions", s["h1"]))
    E.append(_bullets([
        "The assignable target space is the set of billable (flag = 1) codes in "
        "the CMS order file; non-billable headers are never assigned.",
        "Gold labels are treated as a complete, order-independent set per note; "
        "evaluation is set-based (no code sequencing / principal-diagnosis "
        "ordering is scored).",
        "Codes are compared in canonical dotted, upper-case form; the order "
        "file's undotted codes are normalized to match the gold format.",
        "Notes are self-contained; no external EHR context, problem list, or "
        "prior encounters are available.",
        "Only an Ollama model is used for all learned components (extraction, "
        "assignment, embeddings), per the task constraint.",
    ], s["bullet"]))

    # ---- 6. Results -----------------------------------------------------
    E.append(PageBreak())
    E.append(_p("6. Measured Results", s["h1"]))
    if m:
        E.append(_p(
            f"Evaluated on all <b>{m.get('n_cases','?')}</b> provided cases with "
            f"the configured Ollama models. Recall is the primary requested "
            f"metric; precision/F1 are reported for context. <i>Micro</i> pools "
            f"code-level hits across all notes (weights by number of codes); "
            f"<i>macro</i> averages per-note scores (weights notes equally).",
            s["body"]))
    E.append(metrics_table(m, s))
    E.append(Spacer(1, 6 * mm))
    E.append(ops_table(m, s))
    if m:
        E.append(Spacer(1, 4 * mm))
        E.append(_p(
            "Reading the numbers: the gap between <i>retrieval recall</i> and "
            "final <i>recall</i> is what the assignment LLM loses (it failed to "
            "select a retrieved gold code); the gap between retrieval recall and "
            "1.0 is what retrieval never surfaced. This decomposition tells you "
            "where to invest next — a bigger/better assignment model vs. deeper "
            "retrieval.", s["body"]))
        E.append(_p(
            "Token usage is captured directly from the Ollama response fields "
            "(<font face='Courier'>prompt_eval_count</font>, "
            "<font face='Courier'>eval_count</font>); timing is wall-clock per "
            "note including retrieval. Numbers scale with the chosen model and "
            "candidate-list size, not with the design itself.", s["body"]))
    E.append(_p(
        "The 54-case set is small, so these figures are indicative rather than "
        "statistically tight; they establish a working baseline and a "
        "measurement harness, not a production benchmark.", s["small"]))

    # ---- 7. Improvements ------------------------------------------------
    E.append(_p("7. How to Improve the System Further", s["h1"]))
    E.append(_bullets([
        "<b>Ontology-aware retrieval.</b> Exploit the ICD-10 hierarchy and the "
        "Alphabetic Index / Includes-Excludes notes to expand queries and "
        "constrain candidates to coherent sub-trees.",
        "<b>Learned reranker.</b> Insert a cross-encoder rerank step between "
        "retrieval and assignment to push gold codes into the top of the "
        "candidate list, raising the achievable ceiling per token.",
        "<b>Coding-guideline injection.</b> Provide ICD-10-CM Official "
        "Guidelines (sequencing, combination codes, 7th-character rules) as "
        "system context to improve specificity and reduce invalid combinations.",
        "<b>Constrained / grammar decoding.</b> Force the assignment output "
        "onto the candidate code grammar so malformed or out-of-set codes are "
        "impossible rather than filtered afterward.",
        "<b>Stronger / fine-tuned model.</b> A larger Ollama model, or one "
        "LoRA-tuned on coded charts, would improve the specificity judgement "
        "that dominates the remaining error.",
        "<b>Self-consistency &amp; verification.</b> Sample multiple assignments "
        "and vote, or add a verifier pass that checks each code against the "
        "documentation before emitting it.",
        "<b>Human-in-the-loop.</b> Surface per-code justifications and "
        "confidence so a coder can accept/reject — turning the system into an "
        "assistive tool and generating new training data.",
        "<b>Scale &amp; robustness.</b> Batch embedding, an ANN index (FAISS) "
        "for sub-millisecond retrieval at full scale, and evaluation on a "
        "larger, multi-institution corpus.",
    ], s["bullet"]))

    E.append(Spacer(1, 6 * mm))
    E.append(_p(
        "Appendix — reproducibility. All metrics in Section 6 are produced by "
        "<font face='Courier'>python scripts/run_eval.py</font>, which writes "
        "<font face='Courier'>reports/eval_summary.json</font> and a per-case "
        "CSV. See the README for configuration and setup.", s["small"]))

    doc.build(E)
    return out_path


if __name__ == "__main__":
    print("wrote", build())
