"""Node de-duplication helpers.

Two stages:

1. Exact merge: nodes with the same ``normalized_label`` and ``type`` are the
   same concept. This is handled during extraction/storage already, but we also
   provide a sweep to merge any that slipped through.
2. Near-duplicate suggestions: use simple string similarity to *suggest* merges
   to the researcher via a CSV. We never auto-merge fuzzy matches — the human
   decides.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from .graph_store import GraphStore
from .schemas import Node


def string_similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity ratio between two strings."""

    return SequenceMatcher(None, a, b).ratio()


def suggest_merges(store: GraphStore, threshold: float = 0.82) -> List[Dict]:
    """Suggest near-duplicate node pairs of the same type.

    Returns a list of dict rows suitable for writing to a review CSV. Each row
    proposes merging ``duplicate_node_id`` into ``keep_node_id``; the researcher
    edits the ``approve_merge`` column before importing.
    """

    nodes = store.get_nodes()
    by_type: Dict[str, List[Node]] = {}
    for node in nodes:
        by_type.setdefault(node.type.value, []).append(node)

    suggestions: List[Dict] = []
    seen_pairs: set[Tuple[str, str]] = set()

    for node_type, group in by_type.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.normalized_label == b.normalized_label:
                    # Exact normalized match: high-confidence suggestion.
                    sim = 1.0
                else:
                    sim = string_similarity(a.normalized_label, b.normalized_label)
                if sim < threshold:
                    continue

                pair_key = tuple(sorted((a.id, b.id)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Keep the higher-confidence node by default.
                keep, dup = (a, b) if a.confidence >= b.confidence else (b, a)
                suggestions.append(
                    {
                        "keep_node_id": keep.id,
                        "keep_label": keep.label,
                        "duplicate_node_id": dup.id,
                        "duplicate_label": dup.label,
                        "type": node_type,
                        "similarity": round(sim, 3),
                        "approve_merge": "no",  # researcher changes to "yes"
                    }
                )

    suggestions.sort(key=lambda r: r["similarity"], reverse=True)
    return suggestions


def apply_merge(store: GraphStore, keep_node_id: str, duplicate_node_id: str) -> bool:
    """Merge ``duplicate_node_id`` into ``keep_node_id``.

    All edges pointing at the duplicate are repointed to the kept node, then the
    duplicate node is removed. Returns True if the merge happened.
    """

    if keep_node_id == duplicate_node_id:
        return False
    keep = store.get_node(keep_node_id)
    dup = store.get_node(duplicate_node_id)
    if keep is None or dup is None:
        return False

    # Repoint edges from the duplicate to the kept node.
    for edge in store.get_edges():
        fields = {}
        if edge.source_node_id == duplicate_node_id:
            fields["source_node_id"] = keep_node_id
            fields["source_label"] = keep.label
        if edge.target_node_id == duplicate_node_id:
            fields["target_node_id"] = keep_node_id
            fields["target_label"] = keep.label
        if fields:
            store.update_edge_fields(edge.id, fields)

    # Delete the duplicate node row.
    store.conn.execute("DELETE FROM nodes WHERE id = ?", (duplicate_node_id,))
    store.conn.commit()
    store.log_review("node", duplicate_node_id, "merged_into", duplicate_node_id, keep_node_id)
    return True
