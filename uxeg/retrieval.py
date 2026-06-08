"""GraphSearch + hybrid retrieval.

Two retrieval paths are combined when answering a question:

* Path A — vector search over chunks (semantic text similarity).
* Path B — GraphSearch over nodes and edges (structural reasoning).

The combined :class:`RetrievalResult` is what the answer generator reasons over.
Rejected edges are never used. Approved edges are prioritised over pending ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from .config import Config
from .graph_store import GraphStore
from .schemas import Edge, Node
from .utils import normalize_label
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Bundle of everything retrieved for a question."""

    relevant_nodes: List[Dict] = field(default_factory=list)
    relevant_edges: List[Dict] = field(default_factory=list)
    graph_paths: List[List[Dict]] = field(default_factory=list)
    source_docs: List[str] = field(default_factory=list)
    chunks: List[Dict] = field(default_factory=list)
    confidence_summary: Dict = field(default_factory=dict)


class GraphSearch:
    """Structural search over the evidence graph."""

    def __init__(self, store: GraphStore, config: Config):
        self.store = store
        self.config = config
        self.graph: nx.MultiDiGraph = store.build_graph(
            include_pending=config.retrieval.include_pending_edges,
            use_rejected=config.review.use_rejected_edges,
            min_confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Node finders
    # ------------------------------------------------------------------
    def find_nodes_by_keyword(self, query: str, limit: int = 10) -> List[str]:
        """Return node ids whose label/description matches query keywords."""

        terms = [t for t in normalize_label(query).split() if len(t) > 2]
        scored = []
        for node_id, data in self.graph.nodes(data=True):
            haystack = normalize_label(f"{data.get('label', '')} {data.get('description', '')}")
            score = sum(1 for t in terms if t in haystack)
            if score > 0:
                scored.append((score, node_id))
        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:limit]]

    def neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """Return node ids within ``depth`` hops (treating graph as undirected)."""

        if node_id not in self.graph:
            return []
        undirected = self.graph.to_undirected(as_view=True)
        lengths = nx.single_source_shortest_path_length(undirected, node_id, cutoff=depth)
        return [n for n in lengths if n != node_id]

    def shortest_paths(self, source_ids: List[str], max_depth: int = 3, max_paths: int = 5) -> List[List[str]]:
        """Find short paths between pairs of relevant source nodes."""

        undirected = self.graph.to_undirected(as_view=True)
        paths: List[List[str]] = []
        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                a, b = source_ids[i], source_ids[j]
                if a not in self.graph or b not in self.graph:
                    continue
                try:
                    path = nx.shortest_path(undirected, a, b)
                    if len(path) <= max_depth + 1:
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
                if len(paths) >= max_paths:
                    return paths
        return paths

    def high_degree_nodes(self, top_n: int = 10) -> List[str]:
        """Return the most connected node ids (hub concepts)."""

        degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        return [n for n, d in degrees[:top_n] if d > 0]

    def repeated_concepts(self, min_docs: int = 2) -> List[str]:
        """Nodes that are evidenced across multiple source documents."""

        result = []
        nodes = {n.id: n for n in self.store.get_nodes()}
        for node_id in self.graph.nodes():
            docs = set()
            node = nodes.get(node_id)
            if node and node.source_doc:
                docs.add(node.source_doc)
            for _, _, data in self.graph.in_edges(node_id, data=True):
                if data.get("source_doc"):
                    docs.add(data["source_doc"])
            for _, _, data in self.graph.out_edges(node_id, data=True):
                if data.get("source_doc"):
                    docs.add(data["source_doc"])
            if len(docs) >= min_docs:
                result.append(node_id)
        return result

    def weak_links(self, max_confidence: float = 0.45) -> List[Dict]:
        """Edges with low confidence — reasoning that needs human checking."""

        weak = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("confidence", 1.0) <= max_confidence:
                weak.append(self._edge_payload(u, v, data))
        return weak

    def contradictions(self) -> List[Dict]:
        """Edges explicitly marked as CONTRADICTS."""

        out = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation_type") == "CONTRADICTS":
                out.append(self._edge_payload(u, v, data))
        return out

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------
    def _node_payload(self, node_id: str) -> Dict:
        data = self.graph.nodes[node_id]
        return {
            "id": node_id,
            "label": data.get("label"),
            "type": data.get("type"),
            "description": data.get("description"),
            "confidence": data.get("confidence"),
            "review_status": data.get("review_status"),
            "source_doc": data.get("source_doc"),
        }

    def _edge_payload(self, u: str, v: str, data: Dict) -> Dict:
        return {
            "id": data.get("id"),
            "source_id": u,
            "target_id": v,
            "source_label": self.graph.nodes.get(u, {}).get("label", u),
            "target_label": self.graph.nodes.get(v, {}).get("label", v),
            "relation_type": data.get("relation_type"),
            "description": data.get("description"),
            "evidence_text": data.get("evidence_text"),
            "confidence": data.get("confidence"),
            "weight": data.get("weight"),
            "review_status": data.get("review_status"),
            "source_doc": data.get("source_doc"),
            "source_chunk": data.get("source_chunk"),
        }

    def edges_for_nodes(self, node_ids: List[str]) -> List[Dict]:
        """Return edges that touch any of the given nodes."""

        node_set = set(node_ids)
        out = []
        seen = set()
        for u, v, data in self.graph.edges(data=True):
            if u in node_set or v in node_set:
                eid = data.get("id")
                if eid in seen:
                    continue
                seen.add(eid)
                out.append(self._edge_payload(u, v, data))
        return out

    def path_to_payload(self, path: List[str]) -> List[Dict]:
        return [self._node_payload(n) for n in path]


def _edge_priority(edge: Dict) -> tuple:
    """Sort key: approved first, then by confidence then weight (descending)."""

    status_rank = {"approved": 0, "edited": 1, "pending": 2}.get(edge.get("review_status"), 3)
    return (status_rank, -(edge.get("confidence") or 0), -(edge.get("weight") or 0))


def hybrid_retrieve(
    question: str,
    store: GraphStore,
    config: Config,
    vector_store: Optional[VectorStore] = None,
) -> RetrievalResult:
    """Run both retrieval paths and combine them into a single result."""

    result = RetrievalResult()

    # --- Path B: GraphSearch ---
    gs = GraphSearch(store, config)
    keyword_nodes = gs.find_nodes_by_keyword(question, limit=config.retrieval.top_k_nodes)

    # Expand with neighbours so we capture reasoning chains.
    expanded = set(keyword_nodes)
    for node_id in keyword_nodes:
        expanded.update(gs.neighbors(node_id, depth=config.retrieval.max_graph_depth))

    relevant_node_ids = list(expanded)
    result.relevant_nodes = [gs._node_payload(n) for n in relevant_node_ids if n in gs.graph]

    edges = gs.edges_for_nodes(relevant_node_ids)
    edges.sort(key=_edge_priority)
    result.relevant_edges = edges

    # Graph paths between the keyword-matched concepts.
    for path in gs.shortest_paths(keyword_nodes, max_depth=config.retrieval.max_graph_depth + 1):
        result.graph_paths.append(gs.path_to_payload(path))

    # --- Path A: Vector search over chunks ---
    if vector_store is not None:
        chunks = vector_store.search(question, top_k=config.retrieval.top_k_chunks)
        result.chunks = chunks

    # --- Combine source docs ---
    docs = set()
    for n in result.relevant_nodes:
        if n.get("source_doc"):
            docs.add(n["source_doc"])
    for e in result.relevant_edges:
        if e.get("source_doc"):
            docs.add(e["source_doc"])
    for c in result.chunks:
        if c.get("file_name"):
            docs.add(c["file_name"])
    result.source_docs = sorted(docs)

    # --- Confidence summary ---
    approved = sum(1 for e in result.relevant_edges if e.get("review_status") in ("approved", "edited"))
    pending = sum(1 for e in result.relevant_edges if e.get("review_status") == "pending")
    weak = sum(1 for e in result.relevant_edges if (e.get("confidence") or 0) < config.retrieval.minimum_edge_confidence)
    result.confidence_summary = {
        "total_edges": len(result.relevant_edges),
        "approved_edges": approved,
        "pending_edges": pending,
        "weak_edges": weak,
        "num_chunks": len(result.chunks),
        "num_source_docs": len(result.source_docs),
        "num_graph_paths": len(result.graph_paths),
    }

    return result
