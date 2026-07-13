"""Code knowledge base.

Loads the ICD-10-CM code entries, keeps the assignable (billable) subset as the
candidate space, and builds/caches the retrieval indexes:

  * BM25 lexical index over each code's description text (always built).
  * Optional dense embedding matrix (Ollama), cached to disk so it is computed
    once (~75k codes) and reused across runs.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Config
from .data import CodeEntry, load_code_entries
from .llm import OllamaClient
from .utils import tokenize


def _doc_text(e: CodeEntry) -> str:
    """Text a code is indexed/embedded under."""
    if e.long_desc and e.long_desc != e.short_desc:
        return f"{e.code} {e.long_desc}"
    return f"{e.code} {e.short_desc}"


class CodeKB:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        entries = load_code_entries(cfg.paths.code_order_file)
        if cfg.retrieval.billable_only:
            entries = [e for e in entries if e.billable]
        self.entries: List[CodeEntry] = entries
        self.docs: List[str] = [_doc_text(e) for e in entries]
        self.codes: List[str] = [e.code for e in entries]
        self.by_code = {e.code: e for e in entries}

        # BM25 over tokenized descriptions.
        self._tokenized = [tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(self._tokenized)

        self._embeddings: Optional[np.ndarray] = None  # (N, D), L2-normalized

    # -- descriptions ------------------------------------------------------
    def describe(self, code: str) -> str:
        e = self.by_code.get(code)
        return e.long_desc if e else ""

    def is_valid(self, code: str) -> bool:
        return code in self.by_code

    # -- embeddings --------------------------------------------------------
    def _cache_path(self) -> str:
        os.makedirs(self.cfg.paths.cache_dir, exist_ok=True)
        # Cache key ties to model + corpus size so stale caches are ignored.
        key = f"{self.cfg.ollama.embed_model}:{len(self.codes)}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return os.path.join(self.cfg.paths.cache_dir, f"embeddings_{h}.pkl")

    def ensure_embeddings(self, client: OllamaClient, verbose: bool = True) -> None:
        """Build (or load) the dense embedding matrix for all codes."""
        if self._embeddings is not None:
            return
        path = self._cache_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("codes") == self.codes:
                self._embeddings = blob["matrix"]
                if verbose:
                    print(f"[kb] loaded cached embeddings: {path}")
                return

        if verbose:
            print(f"[kb] embedding {len(self.docs)} codes with "
                  f"'{self.cfg.ollama.embed_model}' (one-time)...")
        vecs: List[List[float]] = []
        try:
            from tqdm import tqdm
            iterator = tqdm(self.docs, desc="embedding codes")
        except Exception:
            iterator = self.docs
        for d in iterator:
            vecs.append(client.embed(d, stage="embed_index"))
        mat = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self._embeddings = mat
        with open(path, "wb") as f:
            pickle.dump({"codes": self.codes, "matrix": mat}, f)
        if verbose:
            print(f"[kb] cached embeddings -> {path}")

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        return self._embeddings
