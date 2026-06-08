"""CSV-based human review workflow.

There is no UI. Instead the researcher exports edges (or nodes) to a CSV,
edits the review columns in Excel / Numbers / Google Sheets, and imports the
CSV back. Changes are written to the database and logged in ``review_logs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .graph_store import GraphStore
from .schemas import EDGE_TYPE_VALUES, NODE_TYPE_VALUES, REVIEW_STATUS_VALUES
from .utils import ensure_dir, success, warn


# Columns the researcher may safely change for edges.
EDGE_EDITABLE = ["review_status", "weight", "confidence", "relation_type", "description", "review_note"]
NODE_EDITABLE = ["review_status", "label", "type", "description", "confidence"]

EDGE_COLUMNS = [
    "edge_id",
    "source_label",
    "relation_type",
    "target_label",
    "description",
    "evidence_text",
    "source_doc",
    "source_chunk",
    "confidence",
    "weight",
    "review_status",
    "review_note",
]

NODE_COLUMNS = [
    "node_id",
    "label",
    "type",
    "description",
    "source_doc",
    "source_chunk",
    "confidence",
    "review_status",
]


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def export_edges(store: GraphStore, output_path: str | Path) -> int:
    """Write all edges to a review CSV. Returns the number of rows written."""

    edges = store.get_edges()
    rows = []
    for e in edges:
        rows.append(
            {
                "edge_id": e.id,
                "source_label": e.source_label,
                "relation_type": e.relation_type.value,
                "target_label": e.target_label,
                "description": e.description,
                "evidence_text": e.evidence_text,
                "source_doc": e.source_doc,
                "source_chunk": e.source_chunk,
                "confidence": e.confidence,
                "weight": e.weight,
                "review_status": e.review_status.value,
                "review_note": e.review_note,
            }
        )
    df = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    ensure_dir(Path(output_path).parent)
    df.to_csv(output_path, index=False)
    return len(rows)


def import_edges(store: GraphStore, input_path: str | Path) -> int:
    """Read an edited edge CSV and apply allowed changes. Returns rows updated."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}")

    df = pd.read_csv(path)
    if "edge_id" not in df.columns:
        raise ValueError("Review CSV must contain an 'edge_id' column.")

    updated = 0
    for _, row in df.iterrows():
        edge_id = str(row["edge_id"])
        existing = next((e for e in store.get_edges() if e.id == edge_id), None)
        if existing is None:
            warn(f"Skipping unknown edge_id '{edge_id}'.")
            continue

        fields = {}
        for col in EDGE_EDITABLE:
            if col not in df.columns or pd.isna(row[col]):
                continue
            new_value = row[col]
            old_value = getattr(existing, col)
            old_value = old_value.value if hasattr(old_value, "value") else old_value

            # Validate constrained fields.
            if col == "review_status" and str(new_value) not in REVIEW_STATUS_VALUES:
                warn(f"Edge {edge_id}: invalid review_status '{new_value}'; ignored.")
                continue
            if col == "relation_type" and str(new_value) not in EDGE_TYPE_VALUES:
                warn(f"Edge {edge_id}: invalid relation_type '{new_value}'; ignored.")
                continue
            if col in ("weight", "confidence"):
                try:
                    new_value = max(0.0, min(1.0, float(new_value)))
                except (TypeError, ValueError):
                    warn(f"Edge {edge_id}: invalid {col} '{new_value}'; ignored.")
                    continue

            if str(old_value) != str(new_value):
                fields[col] = new_value
                store.log_review("edge", edge_id, col, old_value, new_value)

        # If a human changed something but left status as pending, mark edited.
        if fields and fields.get("review_status") is None and existing.review_status.value == "pending":
            content_changed = any(c in fields for c in ("weight", "confidence", "relation_type", "description"))
            if content_changed:
                fields["review_status"] = "edited"

        if fields:
            store.update_edge_fields(edge_id, fields)
            updated += 1

    return updated


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def export_nodes(store: GraphStore, output_path: str | Path) -> int:
    """Write all nodes to a review CSV. Returns the number of rows written."""

    nodes = store.get_nodes()
    rows = []
    for n in nodes:
        rows.append(
            {
                "node_id": n.id,
                "label": n.label,
                "type": n.type.value,
                "description": n.description,
                "source_doc": n.source_doc,
                "source_chunk": n.source_chunk,
                "confidence": n.confidence,
                "review_status": n.review_status.value,
            }
        )
    df = pd.DataFrame(rows, columns=NODE_COLUMNS)
    ensure_dir(Path(output_path).parent)
    df.to_csv(output_path, index=False)
    return len(rows)


def import_nodes(store: GraphStore, input_path: str | Path) -> int:
    """Read an edited node CSV and apply allowed changes. Returns rows updated."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}")

    df = pd.read_csv(path)
    if "node_id" not in df.columns:
        raise ValueError("Node review CSV must contain a 'node_id' column.")

    from .utils import normalize_label

    updated = 0
    for _, row in df.iterrows():
        node_id = str(row["node_id"])
        existing = store.get_node(node_id)
        if existing is None:
            warn(f"Skipping unknown node_id '{node_id}'.")
            continue

        fields = {}
        for col in NODE_EDITABLE:
            if col not in df.columns or pd.isna(row[col]):
                continue
            new_value = row[col]
            old_value = getattr(existing, col)
            old_value = old_value.value if hasattr(old_value, "value") else old_value

            if col == "review_status" and str(new_value) not in REVIEW_STATUS_VALUES:
                warn(f"Node {node_id}: invalid review_status '{new_value}'; ignored.")
                continue
            if col == "type" and str(new_value) not in NODE_TYPE_VALUES:
                warn(f"Node {node_id}: invalid type '{new_value}'; ignored.")
                continue
            if col == "confidence":
                try:
                    new_value = max(0.0, min(1.0, float(new_value)))
                except (TypeError, ValueError):
                    warn(f"Node {node_id}: invalid confidence '{new_value}'; ignored.")
                    continue

            if str(old_value) != str(new_value):
                fields[col] = new_value
                store.log_review("node", node_id, col, old_value, new_value)

        # Keep normalized_label in sync if the label changed.
        if "label" in fields:
            fields["normalized_label"] = normalize_label(str(fields["label"]))

        if fields:
            store.update_node_fields(node_id, fields)
            updated += 1

    return updated
