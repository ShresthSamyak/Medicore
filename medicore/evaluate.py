"""Evaluation harness: recall (primary), precision/F1, timing, token usage.

Definitions (multi-label, set-based per note):
  TP = |predicted ∩ gold|,  FP = |predicted − gold|,  FN = |gold − predicted|

  Micro  = pool TP/FP/FN across all notes, then divide (weights by #codes).
  Macro  = per-note recall/precision, then average (weights each note equally).
  Retrieval recall (candidate hit rate) = fraction of gold codes that appeared
      in the retrieved candidate pool — the ceiling the assignment stage can reach.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

from .data import Case
from .pipeline import CodingResult


@dataclass
class PerCase:
    index: int
    gold: List[str]
    predicted: List[str]
    tp: int
    fp: int
    fn: int
    recall: float
    precision: float
    f1: float
    retrieval_recall: float
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _prf(tp: int, fp: int, fn: int):
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return recall, precision, f1


def score_case(case: Case, result: CodingResult) -> PerCase:
    gold = set(case.gold_codes)
    pred = set(result.predicted_codes)
    cand = set(result.candidate_codes)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    recall, precision, f1 = _prf(tp, fp, fn)
    retr = (len(gold & cand) / len(gold)) if gold else 1.0
    u = result.usage
    return PerCase(
        index=case.index,
        gold=sorted(gold),
        predicted=sorted(pred),
        tp=tp, fp=fp, fn=fn,
        recall=recall, precision=precision, f1=f1,
        retrieval_recall=retr,
        seconds=result.seconds,
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        total_tokens=u.total_tokens,
    )


@dataclass
class Summary:
    n_cases: int = 0
    micro_recall: float = 0.0
    micro_precision: float = 0.0
    micro_f1: float = 0.0
    macro_recall: float = 0.0
    macro_precision: float = 0.0
    macro_f1: float = 0.0
    retrieval_recall: float = 0.0
    avg_seconds: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    per_case: List[PerCase] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_case"}
        d["per_case"] = [pc.__dict__ for pc in self.per_case]
        return d


def summarize(per_cases: List[PerCase]) -> Summary:
    n = len(per_cases)
    s = Summary(n_cases=n, per_case=per_cases)
    if n == 0:
        return s
    s.total_tp = sum(p.tp for p in per_cases)
    s.total_fp = sum(p.fp for p in per_cases)
    s.total_fn = sum(p.fn for p in per_cases)
    s.micro_recall, s.micro_precision, s.micro_f1 = _prf(
        s.total_tp, s.total_fp, s.total_fn
    )
    s.macro_recall = sum(p.recall for p in per_cases) / n
    s.macro_precision = sum(p.precision for p in per_cases) / n
    s.macro_f1 = sum(p.f1 for p in per_cases) / n
    s.retrieval_recall = sum(p.retrieval_recall for p in per_cases) / n
    s.avg_seconds = sum(p.seconds for p in per_cases) / n
    s.avg_prompt_tokens = sum(p.prompt_tokens for p in per_cases) / n
    s.avg_completion_tokens = sum(p.completion_tokens for p in per_cases) / n
    s.avg_total_tokens = sum(p.total_tokens for p in per_cases) / n
    return s


def save_reports(summary: Summary, reports_dir: str, prefix: str = "eval") -> Dict[str, str]:
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, f"{prefix}_summary.json")
    csv_path = os.path.join(reports_dir, f"{prefix}_per_case.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.as_dict(), f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "gold", "predicted", "tp", "fp", "fn",
                    "recall", "precision", "f1", "retrieval_recall",
                    "seconds", "prompt_tokens", "completion_tokens", "total_tokens"])
        for p in summary.per_case:
            w.writerow([p.index, ";".join(p.gold), ";".join(p.predicted),
                        p.tp, p.fp, p.fn,
                        f"{p.recall:.3f}", f"{p.precision:.3f}", f"{p.f1:.3f}",
                        f"{p.retrieval_recall:.3f}",
                        f"{p.seconds:.2f}", p.prompt_tokens,
                        p.completion_tokens, p.total_tokens])
    return {"json": json_path, "csv": csv_path}


def print_summary(summary: Summary) -> None:
    s = summary
    print("\n" + "=" * 60)
    print(f"  MediCore evaluation - {s.n_cases} cases")
    print("=" * 60)
    print(f"  Recall     (micro / macro):  {s.micro_recall:.3f} / {s.macro_recall:.3f}")
    print(f"  Precision  (micro / macro):  {s.micro_precision:.3f} / {s.macro_precision:.3f}")
    print(f"  F1         (micro / macro):  {s.micro_f1:.3f} / {s.macro_f1:.3f}")
    print(f"  Retrieval recall (ceiling):  {s.retrieval_recall:.3f}")
    print("-" * 60)
    print(f"  Avg time / note:             {s.avg_seconds:.2f} s")
    print(f"  Avg tokens / note (total):   {s.avg_total_tokens:.0f}")
    print(f"     prompt / completion:      {s.avg_prompt_tokens:.0f} / {s.avg_completion_tokens:.0f}")
    print(f"  Totals TP/FP/FN:             {s.total_tp}/{s.total_fp}/{s.total_fn}")
    print("=" * 60 + "\n")
