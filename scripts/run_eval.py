"""Run the full evaluation over the labelled cases.

Usage:
    python scripts/run_eval.py [--config config.yaml] [--limit N] [--no-embeddings]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medicore.config import load_config
from medicore.data import load_cases
from medicore.evaluate import print_summary, save_reports, score_case, summarize
from medicore.kb import CodeKB
from medicore.llm import OllamaClient
from medicore.pipeline import CodingPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N cases")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="force BM25-only retrieval for this run")
    ap.add_argument("--prefix", default="eval", help="report filename prefix")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_embeddings:
        cfg.retrieval.use_embeddings = False

    client = OllamaClient(cfg.ollama)
    if not client.is_up():
        sys.exit(f"[error] Ollama not reachable at {cfg.ollama.base_url}. "
                 f"Start it with `ollama serve`.")
    models = client.available_models()
    print(f"[eval] Ollama up. LLM={cfg.ollama.llm_model} "
          f"embed={cfg.ollama.embed_model} | installed: {models}")

    print("[eval] building code knowledge base...")
    kb = CodeKB(cfg)
    print(f"[eval] candidate code space: {len(kb.codes)} codes "
          f"({'billable only' if cfg.retrieval.billable_only else 'all'})")
    if cfg.retrieval.use_embeddings:
        kb.ensure_embeddings(client)

    cases = load_cases(cfg.paths.cases_file)
    if args.limit:
        cases = cases[: args.limit]
    print(f"[eval] scoring {len(cases)} cases...\n")

    pipeline = CodingPipeline(cfg, kb, client)
    per_cases = []
    for case in cases:
        result = pipeline.code_note(case.note)
        pc = score_case(case, result)
        per_cases.append(pc)
        print(f"  case {case.index:>2}: "
              f"R={pc.recall:.2f} P={pc.precision:.2f} "
              f"gold={pc.gold} pred={pc.predicted} "
              f"({pc.seconds:.1f}s, {pc.total_tokens} tok)")

    summary = summarize(per_cases)
    print_summary(summary)
    paths = save_reports(summary, cfg.paths.reports_dir, prefix=args.prefix)
    print(f"[eval] reports written:\n  {paths['json']}\n  {paths['csv']}")


if __name__ == "__main__":
    main()
