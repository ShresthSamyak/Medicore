"""Stage 1 — abstract a note into codeable clinical concepts (LLM call)."""
from __future__ import annotations

from typing import List

from .llm import OllamaClient
from .prompts import EXTRACT_SYSTEM, EXTRACT_USER


def extract_concepts(client: OllamaClient, note: str, max_chars: int) -> List[str]:
    """Return a list of short concept phrases. Falls back to [] on failure.

    The caller always also retrieves on the raw note, so an empty concept list
    degrades gracefully to plain note-level retrieval.
    """
    note = note[:max_chars]
    res = client.chat_json(
        system=EXTRACT_SYSTEM,
        user=EXTRACT_USER.format(note=note),
        stage="extract",
    )
    data = res.get("data") or {}
    concepts = data.get("concepts", [])
    out: List[str] = []
    seen = set()
    for c in concepts:
        if isinstance(c, str):
            phrase = c.strip()
        elif isinstance(c, dict):  # tolerate {"concept": "..."} shapes
            phrase = str(c.get("concept") or c.get("phrase") or "").strip()
        else:
            phrase = ""
        key = phrase.lower()
        if phrase and key not in seen:
            seen.add(key)
            out.append(phrase)
    return out
