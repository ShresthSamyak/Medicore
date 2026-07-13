"""Stage 2 — select final codes from the candidate list (LLM call) + validate."""
from __future__ import annotations

from typing import Dict, List

from .kb import CodeKB
from .llm import OllamaClient
from .prompts import ASSIGN_SYSTEM, ASSIGN_USER
from .retrieval import Candidate
from .utils import normalize_code


def _format_candidates(cands: List[Candidate]) -> str:
    return "\n".join(f"{c.code} — {c.desc}" for c in cands)


def assign_codes(
    client: OllamaClient,
    kb: CodeKB,
    note: str,
    candidates: List[Candidate],
    max_chars: int,
) -> List[Dict[str, str]]:
    """Return validated assignments: [{"code","reason"}], hallucinations dropped.

    Validation keeps only codes that (a) are valid entries in the KB and
    (b) were actually offered in the candidate list, guarding against the model
    fabricating a plausible-looking code.
    """
    allowed = {c.code for c in candidates}
    res = client.chat_json(
        system=ASSIGN_SYSTEM,
        user=ASSIGN_USER.format(
            note=note[:max_chars], candidates=_format_candidates(candidates)
        ),
        stage="assign",
    )
    data = res.get("data") or {}
    raw = data.get("assignments", [])

    out: List[Dict[str, str]] = []
    seen = set()
    for item in raw:
        if isinstance(item, str):
            code, reason = item, ""
        elif isinstance(item, dict):
            code = item.get("code", "")
            reason = str(item.get("reason", ""))
        else:
            continue
        code = normalize_code(code)
        if not code or code in seen:
            continue
        if code not in allowed or not kb.is_valid(code):
            continue  # drop hallucinated / out-of-candidate codes
        seen.add(code)
        out.append({"code": code, "reason": reason})
    return out
