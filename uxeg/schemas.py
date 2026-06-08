"""Pydantic models that define the UX evidence graph schema.

These models are the single source of truth for what a valid node, edge, and
extraction result looks like. They are used to validate the JSON returned by
the local model and to keep the SQLite tables consistent.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------
class NodeType(str, Enum):
    """Allowed node types. Keep this list closed so the graph stays analysable."""

    USER_GROUP = "USER_GROUP"
    PARTICIPANT = "PARTICIPANT"
    TASK = "TASK"
    FEATURE = "FEATURE"
    PAIN_POINT = "PAIN_POINT"
    CAUSE = "CAUSE"
    BEHAVIOR = "BEHAVIOR"
    EMOTION = "EMOTION"
    OUTCOME = "OUTCOME"
    METRIC = "METRIC"
    RECOMMENDATION = "RECOMMENDATION"
    CLAIM = "CLAIM"
    EVIDENCE = "EVIDENCE"
    DECISION = "DECISION"
    CONCEPT = "CONCEPT"


class EdgeType(str, Enum):
    """Allowed edge/relation types."""

    EXPERIENCES = "EXPERIENCES"
    STRUGGLES_WITH = "STRUGGLES_WITH"
    CAUSES = "CAUSES"
    LEADS_TO = "LEADS_TO"
    INCREASES = "INCREASES"
    REDUCES = "REDUCES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    APPEARS_IN = "APPEARS_IN"
    ADDRESSES = "ADDRESSES"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    EVIDENCED_BY = "EVIDENCED_BY"


class ReviewStatus(str, Enum):
    """Human review state for nodes and edges."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


NODE_TYPE_VALUES = [t.value for t in NodeType]
EDGE_TYPE_VALUES = [t.value for t in EdgeType]
REVIEW_STATUS_VALUES = [s.value for s in ReviewStatus]


def new_id(prefix: str) -> str:
    """Generate a short, unique id with a readable prefix (e.g. ``node_ab12cd``)."""

    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# What the model is asked to return (extraction payload)
# ---------------------------------------------------------------------------
class ExtractedNode(BaseModel):
    """A node as produced by the extraction model (before we assign ids)."""

    label: str
    normalized_label: Optional[str] = None
    type: NodeType
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def label_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("node label must not be empty")
        return value.strip()


class ExtractedEdge(BaseModel):
    """An edge as produced by the extraction model (before we assign ids)."""

    source_label: str
    target_label: str
    relation_type: EdgeType
    description: str = ""
    evidence_text: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("source_label", "target_label")
    @classmethod
    def label_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("edge labels must not be empty")
        return value.strip()


class ExtractionResult(BaseModel):
    """Top-level object the model must return: a list of nodes and edges."""

    nodes: List[ExtractedNode] = Field(default_factory=list)
    edges: List[ExtractedEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# What we persist (full records with ids and review metadata)
# ---------------------------------------------------------------------------
class Node(BaseModel):
    """A fully-formed graph node stored in the database."""

    id: str = Field(default_factory=lambda: new_id("node"))
    label: str
    normalized_label: str
    type: NodeType
    description: str = ""
    source_doc: Optional[str] = None
    source_chunk: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_by_model: Optional[str] = None
    review_status: ReviewStatus = ReviewStatus.PENDING


class Edge(BaseModel):
    """A fully-formed graph edge stored in the database."""

    id: str = Field(default_factory=lambda: new_id("edge"))
    source_node_id: str
    target_node_id: str
    source_label: str
    target_label: str
    relation_type: EdgeType
    description: str = ""
    evidence_text: str = ""
    source_doc: Optional[str] = None
    source_chunk: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_note: str = ""


class Document(BaseModel):
    """A source document stored in the database."""

    document_id: str = Field(default_factory=lambda: new_id("doc"))
    file_name: str
    file_path: str
    file_type: str
    raw_text: str
    created_at: Optional[str] = None
    metadata_json: str = "{}"


class Chunk(BaseModel):
    """A chunk of a document used as the unit for extraction and embeddings."""

    chunk_id: str = Field(default_factory=lambda: new_id("chunk"))
    document_id: str
    file_name: str
    chunk_text: str
    start_word: int = 0
    end_word: int = 0
