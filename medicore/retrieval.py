"""Candidate retrieval over the code knowledge base.

Given one or more query strings (the whole note plus each extracted concept),
return a ranked, de-duplicated list of candidate codes. Two retrievers are
combined with Reciprocal Rank Fusion (RRF):

  score(code) = sum_over_query_lists( 1 / (rrf_k + rank) )

RRF is rank-based (not score-based), so it fuses BM25 scores and cosine
similarities without needing to calibrate their very different scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import Config
from .kb import CodeKB
from .llm import OllamaClient
from .utils import tokenize


@dataclass
class Candidate:
    code: str
    desc: str
    score: float


class Retriever:
    def __init__(self, cfg: Config, kb: CodeKB, client: OllamaClient):
        self.cfg = cfg
        self.kb = kb
        self.client = client

    # -- individual retrievers --------------------------------------------
    def _bm25_ranklist(self, query: str, k: int) -> List[int]:
        scores = self.kb.bm25.get_scores(tokenize(query))
        if k >= len(scores):
            idx = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        return [int(i) for i in idx if scores[i] > 0][:k]

    def _embed_ranklist(self, query: str, k: int) -> List[int]:
        mat = self.kb.embeddings
        if mat is None:
            return []
        q = np.array(self.client.embed(query, stage="embed_query"), dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        sims = mat @ q
        if k >= len(sims):
            idx = np.argsort(sims)[::-1]
        else:
            idx = np.argpartition(sims, -k)[-k:]
            idx = idx[np.argsort(sims[idx])[::-1]]
        return [int(i) for i in idx][:k]

    # -- fusion ------------------------------------------------------------
    def retrieve(self, queries: List[str]) -> List[Candidate]:
        """Fuse BM25 (+ optional embeddings) across all queries via RRF."""
        rc = self.cfg.retrieval
        fused: Dict[int, float] = {}

        def add(ranklist: List[int]):
            for rank, doc_idx in enumerate(ranklist):
                fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (rc.rrf_k + rank)

        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            add(self._bm25_ranklist(q, rc.bm25_top_k))
            if rc.use_embeddings:
                add(self._embed_ranklist(q, rc.embed_top_k))

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        ranked = ranked[: rc.final_candidates]
        out: List[Candidate] = []
        for doc_idx, score in ranked:
            e = self.kb.entries[doc_idx]
            out.append(Candidate(code=e.code, desc=e.long_desc, score=score))
        return out
