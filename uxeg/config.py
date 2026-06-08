"""Configuration loading.

Configuration comes from two places, merged in this order (later wins for the
LM Studio / embedding settings):

1. ``config.yaml`` (project defaults, checked into git)
2. ``.env`` / real environment variables (machine-specific, not in git)

This keeps secrets and per-machine model names out of the repository while
still giving sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


@dataclass
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    chat_model: str = "change-this-to-your-loaded-model"
    embedding_model: str = "change-this-to-your-embedding-model"


@dataclass
class EmbeddingConfig:
    provider: str = "lmstudio"
    fallback_provider: str = "sentence_transformers"


@dataclass
class RetrievalConfig:
    top_k_chunks: int = 8
    top_k_nodes: int = 10
    max_graph_depth: int = 2
    include_pending_edges: bool = True
    minimum_edge_confidence: float = 0.45


@dataclass
class ReviewConfig:
    default_edge_status: str = "pending"
    use_rejected_edges: bool = False


@dataclass
class Config:
    """Full project configuration."""

    project_name: str = "ux-evidence-graph"
    database_path: str = "data/processed/uxeg.sqlite"
    chunk_size_words: int = 900
    chunk_overlap_words: int = 120
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    config_path: str = "config.yaml"

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Config":
        """Load configuration from a YAML file, then overlay environment vars."""

        # Load .env so os.environ has the user's local model names / keys.
        load_dotenv()

        raw: Dict[str, Any] = {}
        path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        lm_raw = raw.get("lmstudio", {}) or {}
        emb_raw = raw.get("embedding", {}) or {}
        ret_raw = raw.get("retrieval", {}) or {}
        rev_raw = raw.get("review", {}) or {}

        lmstudio = LMStudioConfig(
            base_url=os.getenv("LMSTUDIO_BASE_URL", lm_raw.get("base_url", "http://localhost:1234/v1")),
            api_key=os.getenv("LMSTUDIO_API_KEY", lm_raw.get("api_key", "lm-studio")),
            chat_model=os.getenv("LMSTUDIO_CHAT_MODEL", lm_raw.get("chat_model", "change-this-to-your-loaded-model")),
            embedding_model=os.getenv(
                "LMSTUDIO_EMBEDDING_MODEL",
                lm_raw.get("embedding_model", "change-this-to-your-embedding-model"),
            ),
        )

        embedding = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", emb_raw.get("provider", "lmstudio")),
            fallback_provider=emb_raw.get("fallback_provider", "sentence_transformers"),
        )

        retrieval = RetrievalConfig(
            top_k_chunks=int(ret_raw.get("top_k_chunks", 8)),
            top_k_nodes=int(ret_raw.get("top_k_nodes", 10)),
            max_graph_depth=int(ret_raw.get("max_graph_depth", 2)),
            include_pending_edges=bool(ret_raw.get("include_pending_edges", True)),
            minimum_edge_confidence=float(ret_raw.get("minimum_edge_confidence", 0.45)),
        )

        review = ReviewConfig(
            default_edge_status=str(rev_raw.get("default_edge_status", "pending")),
            use_rejected_edges=bool(rev_raw.get("use_rejected_edges", False)),
        )

        return cls(
            project_name=raw.get("project_name", "ux-evidence-graph"),
            database_path=raw.get("database_path", "data/processed/uxeg.sqlite"),
            chunk_size_words=int(raw.get("chunk_size_words", 900)),
            chunk_overlap_words=int(raw.get("chunk_overlap_words", 120)),
            lmstudio=lmstudio,
            embedding=embedding,
            retrieval=retrieval,
            review=review,
            config_path=str(path),
        )
