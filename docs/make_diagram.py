"""Render the MediCore architecture diagram to docs/architecture.png.

Kept separate so the diagram can be regenerated independently of the PDF.
Uses only matplotlib (no graphviz/system deps) for portability on Windows.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette (colour-blind safe, works in print)
INK = "#1f2933"
BLUE = "#2563eb"
TEAL = "#0d9488"
AMBER = "#b45309"
SLATE = "#475569"
LILAC = "#7c3aed"
BG_OFF = "#f1f5f9"


def _box(ax, x, y, w, h, text, fc, ec=INK, tc="white", fs: float = 9, bold=True):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal",
            zorder=3, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=SLATE, style="-|>", lw=1.6,
           ls: "Union[str, Tuple]" = "-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
        shrinkA=2, shrinkB=2,
    ))


def build(path: Optional[str] = None) -> str:
    path = path or os.path.join(HERE, "architecture.png")
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---- Offline: knowledge-base construction (top band) ----------------
    ax.text(0.2, 8.7, "OFFLINE  ·  build once, cached", fontsize=10,
            color=AMBER, fontweight="bold")
    _box(ax, 0.2, 7.4, 2.5, 1.0,
         "ICD-10-CM order file\n(74,719 billable codes)", BG_OFF, tc=INK, fs=8.5)
    _box(ax, 3.1, 7.7, 2.4, 0.7, "BM25 lexical index", TEAL, fs=9)
    _box(ax, 3.1, 6.8, 2.4, 0.7, "Embedding index\n(Ollama, cached)", TEAL, fs=8.5)
    _arrow(ax, 2.7, 7.9, 3.1, 8.05)
    _arrow(ax, 2.7, 7.7, 3.1, 7.15)

    # ---- Online: per-note pipeline (main flow) --------------------------
    ax.text(0.2, 5.9, "PER-NOTE PIPELINE", fontsize=10, color=BLUE,
            fontweight="bold")

    _box(ax, 0.2, 4.7, 2.1, 1.0, "Medical note\n(free text)", "#e2e8f0", tc=INK, fs=9)
    _box(ax, 2.9, 4.7, 2.3, 1.0,
         "1. Concept\nextraction\n(LLM · JSON)", BLUE, fs=9)
    _box(ax, 5.8, 4.7, 2.3, 1.0,
         "2. Hybrid\nretrieval\n(BM25 + embed · RRF)", TEAL, fs=8.7)
    _box(ax, 8.7, 4.7, 2.3, 1.0,
         "3. Code\nassignment\n(LLM · JSON)", BLUE, fs=9)

    _arrow(ax, 2.3, 5.2, 2.9, 5.2, color=BLUE)
    _arrow(ax, 5.2, 5.2, 5.8, 5.2, color=TEAL)
    _arrow(ax, 8.1, 5.2, 8.7, 5.2, color=BLUE)

    # note also feeds retrieval + assignment directly
    _arrow(ax, 1.25, 4.7, 6.5, 3.7, color=SLATE, ls=(0, (4, 3)), lw=1.1)
    ax.text(3.4, 3.5, "raw note also queried / shown to both LLM stages",
            fontsize=7.5, color=SLATE, style="italic")

    # KB feeds retrieval
    _arrow(ax, 4.3, 6.8, 6.6, 5.7, color=TEAL, ls=(0, (2, 2)), lw=1.2)

    # ---- Validation + output --------------------------------------------
    _box(ax, 8.7, 3.0, 2.3, 0.9,
         "4. Validate\n(drop hallucinations,\nnormalize dotted)", LILAC, fs=8.3)
    _arrow(ax, 9.85, 4.7, 9.85, 3.9, color=LILAC)

    _box(ax, 8.7, 1.7, 2.3, 0.9, "Assigned ICD-10-CM\ncodes + rationale",
         "#e2e8f0", tc=INK, fs=8.7)
    _arrow(ax, 9.85, 3.0, 9.85, 2.6, color=LILAC)

    # ---- Evaluation harness ---------------------------------------------
    _box(ax, 0.2, 1.7, 6.0, 0.9,
         "Evaluation harness\nrecall · precision · F1 · retrieval-recall\n"
         "avg time · avg tokens", AMBER, fs=8.5)
    _arrow(ax, 8.7, 2.15, 6.2, 2.15, color=AMBER, style="-|>")

    ax.text(6.05, 0.75,
            "Candidate space is compressed from ~75k codes to ~120 per note, "
            "so the LLM never sees the full label set.",
            ha="center", fontsize=8.5, color=INK, style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("wrote", build())
