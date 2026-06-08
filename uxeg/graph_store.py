"""SQLite persistence + NetworkX graph construction.

This module owns the database. All tables are created here, and all reads /
writes of documents, chunks, nodes, edges, embeddings, answers and review logs
go through :class:`GraphStore`.

The NetworkX graph is built on demand from approved + (optionally) pending
edges. Rejected edges are never included.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from .schemas import (
    Chunk,
    Document,
    Edge,
    EdgeType,
    Node,
    NodeType,
    ReviewStatus,
)
from .utils import ensure_dir, now_iso


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    file_name TEXT,
    file_path TEXT,
    file_type TEXT,
    raw_text TEXT,
    created_at TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,
    file_name TEXT,
    chunk_text TEXT,
    start_word INTEGER,
    end_word INTEGER
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    label TEXT,
    normalized_label TEXT,
    type TEXT,
    description TEXT,
    source_doc TEXT,
    source_chunk TEXT,
    confidence REAL,
    created_by_model TEXT,
    review_status TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_node_id TEXT,
    target_node_id TEXT,
    source_label TEXT,
    target_label TEXT,
    relation_type TEXT,
    description TEXT,
    evidence_text TEXT,
    source_doc TEXT,
    source_chunk TEXT,
    confidence REAL,
    weight REAL,
    review_status TEXT,
    review_note TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,
    file_name TEXT,
    chunk_text TEXT,
    embedding_model TEXT,
    vector BLOB
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id TEXT PRIMARY KEY,
    question TEXT,
    answer_markdown TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS review_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT,
    entity_id TEXT,
    field TEXT,
    old_value TEXT,
    new_value TEXT,
    created_at TEXT
);
"""


class GraphStore:
    """All database access for the project."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        ensure_dir(Path(self.database_path).parent)
        # check_same_thread=False keeps things simple for CLI + tests.
        self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init_db(self) -> None:
        """Create all tables if they do not already exist."""

        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _ensure_initialized(self) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'"
        )
        if cur.fetchone() is None:
            raise RuntimeError(
                "Database not initialized. Run 'python main.py init' first."
            )

    # ------------------------------------------------------------------
    # Documents & chunks
    # ------------------------------------------------------------------
    def add_document(self, doc: Document) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO documents
               (document_id, file_name, file_path, file_type, raw_text, created_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.document_id,
                doc.file_name,
                doc.file_path,
                doc.file_type,
                doc.raw_text,
                doc.created_at or now_iso(),
                doc.metadata_json,
            ),
        )
        self.conn.commit()

    def add_chunk(self, chunk: Chunk) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, document_id, file_name, chunk_text, start_word, end_word)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.file_name,
                chunk.chunk_text,
                chunk.start_word,
                chunk.end_word,
            ),
        )
        self.conn.commit()

    def get_documents(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM documents")]

    def get_chunks(self) -> List[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks")
        return [
            Chunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                file_name=r["file_name"],
                chunk_text=r["chunk_text"],
                start_word=r["start_word"],
                end_word=r["end_word"],
            )
            for r in rows
        ]

    def count(self, table: str) -> int:
        # Table name is internal/controlled, never user input.
        cur = self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}")
        return int(cur.fetchone()["c"])

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, label, normalized_label, type, description, source_doc,
                source_chunk, confidence, created_by_model, review_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id,
                node.label,
                node.normalized_label,
                node.type.value,
                node.description,
                node.source_doc,
                node.source_chunk,
                node.confidence,
                node.created_by_model,
                node.review_status.value,
            ),
        )
        self.conn.commit()

    def find_node_by_normalized(self, normalized_label: str, node_type: str) -> Optional[Node]:
        """Return an existing node with the same normalized label + type."""

        row = self.conn.execute(
            "SELECT * FROM nodes WHERE normalized_label = ? AND type = ? LIMIT 1",
            (normalized_label, node_type),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_nodes(self, statuses: Optional[List[str]] = None) -> List[Node]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.conn.execute(
                f"SELECT * FROM nodes WHERE review_status IN ({placeholders})", statuses
            )
        else:
            rows = self.conn.execute("SELECT * FROM nodes")
        return [self._row_to_node(r) for r in rows]

    def get_node(self, node_id: str) -> Optional[Node]:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def update_node_fields(self, node_id: str, fields: Dict) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [node_id]
        self.conn.execute(f"UPDATE nodes SET {assignments} WHERE id = ?", values)
        self.conn.commit()

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            label=row["label"],
            normalized_label=row["normalized_label"],
            type=NodeType(row["type"]),
            description=row["description"] or "",
            source_doc=row["source_doc"],
            source_chunk=row["source_chunk"],
            confidence=row["confidence"] if row["confidence"] is not None else 0.5,
            created_by_model=row["created_by_model"],
            review_status=ReviewStatus(row["review_status"]),
        )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def add_edge(self, edge: Edge) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO edges
               (id, source_node_id, target_node_id, source_label, target_label,
                relation_type, description, evidence_text, source_doc, source_chunk,
                confidence, weight, review_status, review_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.id,
                edge.source_node_id,
                edge.target_node_id,
                edge.source_label,
                edge.target_label,
                edge.relation_type.value,
                edge.description,
                edge.evidence_text,
                edge.source_doc,
                edge.source_chunk,
                edge.confidence,
                edge.weight,
                edge.review_status.value,
                edge.review_note,
            ),
        )
        self.conn.commit()

    def get_edges(self, statuses: Optional[List[str]] = None) -> List[Edge]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.conn.execute(
                f"SELECT * FROM edges WHERE review_status IN ({placeholders})", statuses
            )
        else:
            rows = self.conn.execute("SELECT * FROM edges")
        return [self._row_to_edge(r) for r in rows]

    def update_edge_fields(self, edge_id: str, fields: Dict) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [edge_id]
        self.conn.execute(f"UPDATE edges SET {assignments} WHERE id = ?", values)
        self.conn.commit()

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            source_label=row["source_label"],
            target_label=row["target_label"],
            relation_type=EdgeType(row["relation_type"]),
            description=row["description"] or "",
            evidence_text=row["evidence_text"] or "",
            source_doc=row["source_doc"],
            source_chunk=row["source_chunk"],
            confidence=row["confidence"] if row["confidence"] is not None else 0.5,
            weight=row["weight"] if row["weight"] is not None else 0.5,
            review_status=ReviewStatus(row["review_status"]),
            review_note=row["review_note"] or "",
        )

    # ------------------------------------------------------------------
    # Answers & review logs
    # ------------------------------------------------------------------
    def add_answer(self, answer_id: str, question: str, answer_markdown: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO answers (answer_id, question, answer_markdown, created_at) VALUES (?, ?, ?, ?)",
            (answer_id, question, answer_markdown, now_iso()),
        )
        self.conn.commit()

    def log_review(self, entity_type: str, entity_id: str, field: str, old_value: str, new_value: str) -> None:
        self.conn.execute(
            """INSERT INTO review_logs (entity_type, entity_id, field, old_value, new_value, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entity_type, entity_id, field, str(old_value), str(new_value), now_iso()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # NetworkX graph construction
    # ------------------------------------------------------------------
    def build_graph(self, include_pending: bool = True, use_rejected: bool = False, min_confidence: float = 0.0) -> nx.MultiDiGraph:
        """Build a NetworkX graph from stored nodes and edges.

        Rejected edges are excluded unless ``use_rejected`` is True (not
        recommended). Approved edges are marked so retrieval can prioritise
        them.
        """

        statuses = [ReviewStatus.APPROVED.value, ReviewStatus.EDITED.value]
        if include_pending:
            statuses.append(ReviewStatus.PENDING.value)
        if use_rejected:
            statuses.append(ReviewStatus.REJECTED.value)

        graph = nx.MultiDiGraph()

        # Add all non-rejected nodes.
        node_statuses = [s for s in statuses if s != ReviewStatus.REJECTED.value] or None
        for node in self.get_nodes(statuses=node_statuses):
            graph.add_node(
                node.id,
                label=node.label,
                normalized_label=node.normalized_label,
                type=node.type.value,
                description=node.description,
                confidence=node.confidence,
                review_status=node.review_status.value,
                source_doc=node.source_doc,
            )

        for edge in self.get_edges(statuses=statuses):
            if edge.confidence < min_confidence:
                continue
            # Skip edges whose endpoints were filtered out.
            if edge.source_node_id not in graph or edge.target_node_id not in graph:
                continue
            graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                key=edge.id,
                id=edge.id,
                relation_type=edge.relation_type.value,
                description=edge.description,
                evidence_text=edge.evidence_text,
                confidence=edge.confidence,
                weight=edge.weight,
                review_status=edge.review_status.value,
                source_doc=edge.source_doc,
                source_chunk=edge.source_chunk,
            )

        return graph
