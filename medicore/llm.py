"""Ollama client wrapper.

Thin HTTP client over the Ollama REST API for (a) chat completions and
(b) embeddings. Every call records latency and token counts so the evaluation
harness can report average time and token usage without extra plumbing.

Only the standard Ollama endpoints are used:
  POST /api/chat       -> chat completion (returns prompt_eval_count, eval_count)
  POST /api/embeddings -> single-text embedding
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .config import OllamaCfg


@dataclass
class Usage:
    """Accumulates token + timing telemetry across many calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int, seconds: float) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.calls += 1
        self.seconds += seconds

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.seconds += other.seconds

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
        }


class OllamaClient:
    def __init__(self, cfg: OllamaCfg):
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")
        self.session = requests.Session()
        if cfg.api_key:
            self.session.headers.update({"Authorization": f"Bearer {cfg.api_key}"})
        # Telemetry buckets, keyed by logical stage name.
        self.usage: Dict[str, Usage] = {}

    # -- internal ----------------------------------------------------------
    def _record(self, stage: str, prompt: int, completion: int, seconds: float):
        self.usage.setdefault(stage, Usage()).add(prompt, completion, seconds)

    # -- health ------------------------------------------------------------
    def is_up(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def available_models(self) -> List[str]:
        try:
            r = self.session.get(f"{self.base}/api/tags", timeout=10)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    # -- chat --------------------------------------------------------------
    def chat_json(
        self,
        system: str,
        user: str,
        stage: str = "chat",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Chat call that requests a JSON object back (``format=json``).

        Returns ``{"data": <parsed json or {}>, "raw": <str>, "usage": Usage}``.
        Robust to models that wrap JSON in prose or code fences.
        """
        model = model or self.cfg.llm_model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.cfg.temperature,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        t0 = time.time()
        r = self.session.post(
            f"{self.base}/api/chat", json=payload, timeout=self.cfg.request_timeout
        )
        elapsed = time.time() - t0
        r.raise_for_status()
        body = r.json()
        content = body.get("message", {}).get("content", "") or ""
        self._record(
            stage,
            body.get("prompt_eval_count", 0),
            body.get("eval_count", 0),
            elapsed,
        )
        data = _loads_lenient(content)
        return {"data": data, "raw": content, "seconds": elapsed}

    # -- embeddings --------------------------------------------------------
    def embed(self, text: str, stage: str = "embed") -> List[float]:
        payload = {"model": self.cfg.embed_model, "prompt": text}
        t0 = time.time()
        r = self.session.post(
            f"{self.base}/api/embeddings", json=payload, timeout=self.cfg.request_timeout
        )
        elapsed = time.time() - t0
        r.raise_for_status()
        vec = r.json().get("embedding", [])
        # Embedding endpoints don't report token counts; log time + a call only.
        self._record(stage, 0, 0, elapsed)
        return vec

    def embed_batch(self, texts: List[str], stage: str = "embed_batch") -> List[List[float]]:
        """Batched embeddings via the /api/embed endpoint (much faster on CPU)."""
        payload = {"model": self.cfg.embed_model, "input": texts}
        t0 = time.time()
        r = self.session.post(
            f"{self.base}/api/embed", json=payload, timeout=self.cfg.request_timeout
        )
        elapsed = time.time() - t0
        r.raise_for_status()
        self._record(stage, 0, 0, elapsed)
        return r.json().get("embeddings", [])

    # -- telemetry helpers -------------------------------------------------
    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.usage.values():
            total.merge(u)
        return total

    def reset_usage(self) -> None:
        self.usage = {}


def _loads_lenient(text: str) -> Dict[str, Any]:
    """Best-effort JSON parse tolerant to code fences / surrounding prose."""
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` fences.
    if "```" in text:
        inner = text.split("```")
        for chunk in inner:
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    # Grab the first {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}
