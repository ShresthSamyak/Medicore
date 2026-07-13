"""Code a single medical note from a file or stdin.

Usage:
    python scripts/code_note.py --file note.txt
    echo "patient note ..." | python scripts/code_note.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medicore.config import load_config
from medicore.kb import CodeKB
from medicore.llm import OllamaClient
from medicore.pipeline import CodingPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--file", help="path to a text file with the note")
    args = ap.parse_args()

    note = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    if not note.strip():
        sys.exit("[error] empty note")

    cfg = load_config(args.config)
    client = OllamaClient(cfg.ollama)
    if not client.is_up():
        sys.exit(f"[error] Ollama not reachable at {cfg.ollama.base_url}.")

    kb = CodeKB(cfg)
    if cfg.retrieval.use_embeddings:
        kb.ensure_embeddings(client)
    pipeline = CodingPipeline(cfg, kb, client)

    result = pipeline.code_note(note)
    print("\nExtracted concepts:")
    for c in result.concepts:
        print(f"  - {c}")
    print(f"\nRetrieved {len(result.candidate_codes)} candidate codes.")
    print("\nAssigned ICD-10-CM codes:")
    for a in result.assignments:
        print(f"  {a['code']:<10} {kb.describe(a['code'])}")
        if a.get("reason"):
            print(f"             ↳ {a['reason']}")
    print(f"\n({result.seconds:.1f}s, {result.usage.total_tokens} tokens)")


if __name__ == "__main__":
    main()
