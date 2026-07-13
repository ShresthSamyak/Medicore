"""Dataset loading: the CMS ICD-10-CM order file and the labelled cases."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from .utils import normalize_code


@dataclass
class CodeEntry:
    code: str          # dotted, canonical (e.g. "I21.19")
    billable: bool     # True when the CMS "valid HIPAA code" flag is 1
    short_desc: str
    long_desc: str

    @property
    def chapter_letter(self) -> str:
        return self.code[0] if self.code else ""


@dataclass
class Case:
    index: int
    note: str
    gold_codes: List[str]   # dotted, canonical


# ---- CMS order file fixed-width layout -------------------------------------
# cols 0:5   order number
# cols 6:13  code (undotted, left-justified, 7 wide)
# col  14    valid/billable flag (0 header, 1 billable leaf)
# cols 16:76 short description (60 wide)
# cols 77:   long description
# ---------------------------------------------------------------------------

def load_code_entries(path: str) -> List[CodeEntry]:
    """Parse the fixed-width CMS order file into a list of :class:`CodeEntry`."""
    entries: List[CodeEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_code = line[6:13].strip()
            flag = line[14:15].strip()
            short_desc = line[16:76].strip()
            long_desc = line[77:].strip()
            if not raw_code:
                continue
            entries.append(
                CodeEntry(
                    code=normalize_code(raw_code),
                    billable=(flag == "1"),
                    short_desc=short_desc,
                    long_desc=long_desc or short_desc,
                )
            )
    return entries


def load_cases(path: str) -> List[Case]:
    """Load labelled cases; gold codes are normalized to dotted form."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases: List[Case] = []
    for i, c in enumerate(raw.get("cases", [])):
        gold = [normalize_code(x) for x in c.get("icd10_cm", {}).get("codes", [])]
        cases.append(Case(index=i, note=c.get("medical_note", ""), gold_codes=gold))
    return cases


def build_code_index(entries: List[CodeEntry]) -> Dict[str, CodeEntry]:
    """Map dotted code -> entry for O(1) validation/lookup."""
    return {e.code: e for e in entries}
