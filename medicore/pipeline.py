"""End-to-end coding pipeline: note -> concepts -> candidates -> codes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from .assign import assign_codes
from .config import Config
from .extract import extract_concepts
from .kb import CodeKB
from .llm import OllamaClient, Usage
from .retrieval import Candidate, Retriever


@dataclass
class CodingResult:
    predicted_codes: List[str]
    assignments: List[Dict[str, str]]
    concepts: List[str]
    n_candidates: int
    candidate_hit: bool = False        # was the correct set retrievable? (eval-only)
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)


class CodingPipeline:
    def __init__(self, cfg: Config, kb: CodeKB, client: OllamaClient):
        self.cfg = cfg
        self.kb = kb
        self.client = client
        self.retriever = Retriever(cfg, kb, client)

    def code_note(self, note: str) -> CodingResult:
        t0 = time.time()
        self.client.reset_usage()

        # Stage 1 — concept extraction (optional).
        concepts: List[str] = []
        if self.cfg.pipeline.extract_concepts:
            concepts = extract_concepts(
                self.client, note, self.cfg.pipeline.max_note_chars
            )

        # Stage 2 — retrieve candidates on the note + each concept.
        queries = [note] + concepts
        candidates: List[Candidate] = self.retriever.retrieve(queries)

        # Stage 3 — LLM assignment from the candidate list, then validate.
        assignments = assign_codes(
            self.client,
            self.kb,
            note,
            candidates,
            self.cfg.pipeline.max_note_chars,
        )
        predicted = [a["code"] for a in assignments]

        return CodingResult(
            predicted_codes=predicted,
            assignments=assignments,
            concepts=concepts,
            n_candidates=len(candidates),
            seconds=time.time() - t0,
            usage=self.client.total_usage(),
            candidate_hit=False,
        )
