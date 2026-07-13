"""Small shared helpers: ICD-10 code normalization and text tokenization."""
from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_code(code: str) -> str:
    """Return the canonical *dotted* ICD-10-CM form (upper-case, no spaces).

    ICD-10-CM places a decimal point after the 3rd character for codes longer
    than 3 characters. The CMS order file stores codes *undotted* (``I2119``),
    whereas the case gold labels are *dotted* (``I21.19``). We standardise on
    the dotted form everywhere so comparisons are apples-to-apples.

    >>> normalize_code("i2119")
    'I21.19'
    >>> normalize_code("I10")
    'I10'
    >>> normalize_code("A00.0")
    'A00.0'
    """
    if code is None:
        return ""
    c = code.strip().upper().replace(".", "").replace(" ", "")
    if len(c) <= 3:
        return c
    return c[:3] + "." + c[3:]


def undot_code(code: str) -> str:
    """Return the undotted form used by the CMS order file (``I21.19`` -> ``I2119``)."""
    return code.strip().upper().replace(".", "").replace(" ", "")


def tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric tokenization used by the BM25 index."""
    return _TOKEN_RE.findall(text.lower())
