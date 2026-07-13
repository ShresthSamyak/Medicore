"""Generate the figure suite for the MediCore write-up.

Reads the evaluation report(s) under ``reports/`` and renders a set of clean,
consistent, colour-blind-safe charts into ``docs/figures/``. All figures share
one visual system (palette, typography, spare gridlines) so the PDF reads as a
single designed document.

    python docs/make_charts.py
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figures")

# ---- shared visual system --------------------------------------------------
INK = "#1f2933"
BLUE = "#2563eb"
TEAL = "#0d9488"
AMBER = "#b45309"
LILAC = "#7c3aed"
SLATE = "#64748b"
ROSE = "#e11d48"
GREEN = "#16a34a"
GRID = "#e2e8f0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.9,
    "figure.dpi": 170,
})


def _style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def _load(prefix: str) -> Optional[dict]:
    path = os.path.join(ROOT, "reports", f"{prefix}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1. Label-space funnel
# --------------------------------------------------------------------------- #
def funnel(m: dict):
    per = m.get("per_case", [])
    avg_pred = (sum(len(c["predicted"]) for c in per) / len(per)) if per else 3
    stages = [
        ("All ICD-10-CM entries (2026)", 98186, SLATE),
        ("Billable / assignable codes", 74719, TEAL),
        ("Retrieved candidates / note", 120, BLUE),
        ("Assigned codes / note", round(avg_pred, 1), LILAC),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    maxw = stages[0][1]
    for i, (label, val, color) in enumerate(stages):
        y = len(stages) - 1 - i
        w = max(val / maxw, 0.004)
        ax.barh(y, w, height=0.62, color=color, zorder=3,
                edgecolor="white", linewidth=1.5)
        ax.text(w + 0.012, y, f"{val:,}".rstrip("0").rstrip(".") if isinstance(val, float) else f"{val:,}",
                va="center", ha="left", fontweight="bold", fontsize=11, color=color)
        ax.text(-0.012, y, label, va="center", ha="right", fontsize=10.5)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.axis("off")
    ax.set_title("Compressing a 98k-code space to a handful per note",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    return _save(fig, "funnel.png")


# --------------------------------------------------------------------------- #
# 2. Recall waterfall — where recall is lost
# --------------------------------------------------------------------------- #
def waterfall(m: dict):
    retr = m.get("retrieval_recall", 0)
    final = m.get("micro_recall", 0)
    retr_loss = 1.0 - retr
    assign_loss = retr - final

    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = ["All gold\ncodes", "Not retrieved", "Missed by\nassignment", "Correctly\nassigned"]
    # floating bars
    ax.bar(0, 1.0, color=SLATE, width=0.62, zorder=3)
    ax.bar(1, retr_loss, bottom=retr, color=ROSE, width=0.62, zorder=3)
    ax.bar(2, assign_loss, bottom=final, color=AMBER, width=0.62, zorder=3)
    ax.bar(3, final, color=GREEN, width=0.62, zorder=3)

    vals = [1.0, retr_loss, assign_loss, final]
    tops = [1.0, 1.0, retr, final]
    for i, (v, t) in enumerate(zip(vals, tops)):
        ax.text(i, t + 0.02, f"{v*100:.0f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11.5)
    # connector lines
    for i, y in [(0, 1.0), (1, retr), (2, final)]:
        ax.plot([i + 0.31, i + 0.69], [y, y], color=SLATE, lw=1.0, ls=":", zorder=2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of gold codes")
    _style(ax)
    ax.set_title("Where recall is lost: retrieval ceiling vs. assignment",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    return _save(fig, "waterfall.png")


# --------------------------------------------------------------------------- #
# 3. Metrics: micro vs macro grouped bars
# --------------------------------------------------------------------------- #
def metrics_bars(m: dict):
    cats = ["Recall", "Precision", "F1"]
    micro = [m.get("micro_recall", 0), m.get("micro_precision", 0), m.get("micro_f1", 0)]
    macro = [m.get("macro_recall", 0), m.get("macro_precision", 0), m.get("macro_f1", 0)]
    x = range(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.3))
    b1 = ax.bar([i - w / 2 for i in x], micro, w, label="Micro (code-weighted)",
                color=BLUE, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], macro, w, label="Macro (note-weighted)",
                color=TEAL, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, max(micro + macro) * 1.25 + 0.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _style(ax)
    ax.set_title("Accuracy metrics (micro vs. macro)", fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    return _save(fig, "metrics_bars.png")


# --------------------------------------------------------------------------- #
# 4. Gold codes per case (dataset characteristic)
# --------------------------------------------------------------------------- #
def codes_per_case(m: dict):
    per = m.get("per_case", [])
    counts = [len(c["gold"]) for c in per]
    from collections import Counter
    c = Counter(counts)
    xs = sorted(c)
    ys = [c[k] for k in xs]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.bar(xs, ys, color=LILAC, width=0.7, zorder=3)
    for xi, yi in zip(xs, ys):
        ax.text(xi, yi + 0.15, str(yi), ha="center", va="bottom", fontsize=9.5)
    ax.set_xlabel("Number of gold codes in a case")
    ax.set_ylabel("Cases")
    ax.set_xticks(xs)
    mean = sum(counts) / len(counts)
    ax.axvline(mean, color=AMBER, ls="--", lw=1.6, zorder=4)
    ax.text(mean + 0.12, max(ys) * 0.9, f"mean = {mean:.1f}", color=AMBER,
            fontsize=9.5, fontweight="bold")
    _style(ax)
    ax.set_title("Multi-label difficulty: codes per case", fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    return _save(fig, "codes_per_case.png")


# --------------------------------------------------------------------------- #
# 5. Per-case recall distribution
# --------------------------------------------------------------------------- #
def percase_recall(m: dict):
    per = m.get("per_case", [])
    recalls = [c["recall"] for c in per]
    bins = [(-0.01, 0.001, "0%"), (0.001, 0.334, "1–33%"),
            (0.334, 0.667, "34–66%"), (0.667, 0.999, "67–99%"),
            (0.999, 1.01, "100%")]
    labels, vals, cols = [], [], [ROSE, AMBER, "#ca8a04", TEAL, GREEN]
    for lo, hi, lab in bins:
        labels.append(lab)
        vals.append(sum(1 for r in recalls if lo < r <= hi))
    fig, ax = plt.subplots(figsize=(8, 3.9))
    bars = ax.bar(labels, vals, color=cols, width=0.72, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                str(int(b.get_height())), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Per-case recall")
    ax.set_ylabel("Cases")
    _style(ax)
    ax.set_title("How recall is distributed across cases", fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    return _save(fig, "percase_recall.png")


# --------------------------------------------------------------------------- #
# 6. Timing & tokens per case (scatter with means)
# --------------------------------------------------------------------------- #
def timing_tokens(m: dict):
    per = m.get("per_case", [])
    secs = [c["seconds"] for c in per]
    toks = [c["total_tokens"] for c in per]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.scatter(toks, secs, s=42, color=BLUE, alpha=0.75, zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.axhline(m.get("avg_seconds", 0), color=AMBER, ls="--", lw=1.4, zorder=2)
    ax.axvline(m.get("avg_total_tokens", 0), color=TEAL, ls="--", lw=1.4, zorder=2)
    ax.text(ax.get_xlim()[1], m.get("avg_seconds", 0), f"  avg {m.get('avg_seconds',0):.1f}s",
            va="center", ha="left", color=AMBER, fontsize=9, fontweight="bold")
    ax.text(m.get("avg_total_tokens", 0), ax.get_ylim()[1],
            f" avg {m.get('avg_total_tokens',0):.0f} tok", va="top", ha="left",
            color=TEAL, fontsize=9, fontweight="bold")
    ax.set_xlabel("Total tokens per note")
    ax.set_ylabel("Seconds per note")
    _style(ax, grid_axis="both")
    ax.set_title("Cost per note: latency vs. token usage", fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    return _save(fig, "timing_tokens.png")


# --------------------------------------------------------------------------- #
# 7. Model comparison (optional; needs both reports)
# --------------------------------------------------------------------------- #
def model_comparison(models: Dict[str, dict]):
    if len(models) < 2:
        return None
    names = list(models.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    palette = [BLUE, TEAL, AMBER, LILAC]

    def grouped(ax, values, title, fmt, ymax=None, pct=False):
        x = range(len(names))
        bars = ax.bar(x, values, color=palette[:len(names)], width=0.6, zorder=3)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                    fmt(v), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=9)
        if ymax:
            ax.set_ylim(0, ymax)
        if pct:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        _style(ax)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")

    grouped(axes[0], [models[n].get("micro_recall", 0) for n in names],
            "Recall (micro)", lambda v: f"{v:.2f}",
            ymax=max(models[n].get("micro_recall", 0) for n in names) * 1.3 + 0.05, pct=True)
    grouped(axes[1], [models[n].get("avg_seconds", 0) for n in names],
            "Avg time / note (s)", lambda v: f"{v:.1f}",
            ymax=max(models[n].get("avg_seconds", 0) for n in names) * 1.3)
    grouped(axes[2], [models[n].get("avg_total_tokens", 0) for n in names],
            "Avg tokens / note", lambda v: f"{v:.0f}",
            ymax=max(models[n].get("avg_total_tokens", 0) for n in names) * 1.3)
    fig.suptitle("Model comparison — same pipeline, different Ollama model",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "model_comparison.png")


def build_all() -> List[str]:
    m = _load("eval")
    if not m:
        print("[charts] no reports/eval_summary.json — run scripts/run_eval.py first")
        return []
    made = [funnel(m), waterfall(m), metrics_bars(m), codes_per_case(m),
            percase_recall(m), timing_tokens(m)]

    models = {}
    qwen = _load("eval")
    if qwen:
        models["qwen2.5:3b"] = qwen
    llama = _load("eval_llama")
    if llama:
        models["llama3.1:8b"] = llama
    mc = model_comparison(models)
    if mc:
        made.append(mc)
    made = [p for p in made if p]
    print(f"[charts] wrote {len(made)} figures to {FIG_DIR}")
    return made


if __name__ == "__main__":
    build_all()
