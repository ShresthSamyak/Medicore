"""Configuration loading.

Reads ``config.yaml`` into a light dataclass tree so the rest of the codebase
can use attribute access (``cfg.ollama.llm_model``) instead of dict lookups.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class OllamaCfg:
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    llm_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    request_timeout: int = 600
    temperature: float = 0.0
    num_ctx: int = 8192


@dataclass
class RetrievalCfg:
    use_embeddings: bool = True
    bm25_top_k: int = 40
    embed_top_k: int = 40
    final_candidates: int = 120
    rrf_k: int = 60
    billable_only: bool = True


@dataclass
class PipelineCfg:
    extract_concepts: bool = True
    max_note_chars: int = 6000


@dataclass
class PathsCfg:
    code_order_file: str = "data/icd10cm_order_2026.txt"
    cases_file: str = "data/icd10_cm_cases.json"
    cache_dir: str = ".cache"
    reports_dir: str = "reports"


@dataclass
class Config:
    ollama: OllamaCfg = field(default_factory=OllamaCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)


def _merge(section_cls, data: Dict[str, Any]):
    """Instantiate a dataclass, ignoring unknown keys and keeping defaults."""
    known = {f for f in section_cls.__dataclass_fields__}
    return section_cls(**{k: v for k, v in (data or {}).items() if k in known})


def load_config(path: str = "config.yaml") -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy/adjust the shipped config.yaml."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        ollama=_merge(OllamaCfg, raw.get("ollama", {})),
        retrieval=_merge(RetrievalCfg, raw.get("retrieval", {})),
        pipeline=_merge(PipelineCfg, raw.get("pipeline", {})),
        paths=_merge(PathsCfg, raw.get("paths", {})),
    )

    # Environment override: OLLAMA_API_KEY wins over the file (avoids committing secrets).
    env_key = os.environ.get("OLLAMA_API_KEY")
    if env_key:
        cfg.ollama.api_key = env_key
    env_url = os.environ.get("OLLAMA_BASE_URL")
    if env_url:
        cfg.ollama.base_url = env_url

    return cfg
