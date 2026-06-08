"""Graph and table exports.

Produces, under ``outputs/graphs/``:

* graph.json        — node-link JSON
* graph.graphml     — GraphML for tools like Gephi/yEd
* nodes.csv         — node table
* edges.csv         — edge table
* adjacency.csv     — adjacency matrix
* graph_summary.md  — human-readable text summary

Optionally renders a static PNG via matplotlib (best-effort, never required).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

from .config import Config
from .graph_store import GraphStore
from .retrieval import GraphSearch
from .utils import ensure_dir, success, warn


def _graph_for_export(store: GraphStore, config: Config) -> nx.MultiDiGraph:
    return store.build_graph(
        include_pending=config.retrieval.include_pending_edges,
        use_rejected=config.review.use_rejected_edges,
    )


def export_json(store: GraphStore, config: Config, out_dir: str | Path = "outputs/graphs") -> Path:
    graph = _graph_for_export(store, config)
    out_dir = ensure_dir(out_dir)
    data = nx.node_link_data(graph, edges="links")
    path = out_dir / "graph.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def export_graphml(store: GraphStore, config: Config, out_dir: str | Path = "outputs/graphs") -> Path:
    graph = _graph_for_export(store, config)
    out_dir = ensure_dir(out_dir)
    path = out_dir / "graph.graphml"
    # GraphML cannot store None; replace with empty strings.
    clean = nx.MultiDiGraph()
    for n, d in graph.nodes(data=True):
        clean.add_node(n, **{k: ("" if v is None else v) for k, v in d.items()})
    for u, v, d in graph.edges(data=True):
        clean.add_edge(u, v, **{k: ("" if val is None else val) for k, val in d.items()})
    nx.write_graphml(clean, path)
    return path


def export_tables(store: GraphStore, out_dir: str | Path = "outputs/graphs") -> dict:
    """Export nodes.csv, edges.csv, and adjacency.csv."""

    out_dir = ensure_dir(out_dir)
    nodes = store.get_nodes()
    edges = store.get_edges()

    nodes_df = pd.DataFrame(
        [
            {
                "id": n.id,
                "label": n.label,
                "normalized_label": n.normalized_label,
                "type": n.type.value,
                "description": n.description,
                "source_doc": n.source_doc,
                "confidence": n.confidence,
                "review_status": n.review_status.value,
            }
            for n in nodes
        ]
    )
    edges_df = pd.DataFrame(
        [
            {
                "id": e.id,
                "source_node_id": e.source_node_id,
                "target_node_id": e.target_node_id,
                "source_label": e.source_label,
                "target_label": e.target_label,
                "relation_type": e.relation_type.value,
                "confidence": e.confidence,
                "weight": e.weight,
                "review_status": e.review_status.value,
                "source_doc": e.source_doc,
            }
            for e in edges
        ]
    )

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    # Adjacency matrix keyed by label.
    labels = [n.label for n in nodes]
    adjacency = pd.DataFrame(0, index=labels, columns=labels)
    label_by_id = {n.id: n.label for n in nodes}
    for e in edges:
        s = label_by_id.get(e.source_node_id)
        t = label_by_id.get(e.target_node_id)
        if s in adjacency.index and t in adjacency.columns:
            adjacency.at[s, t] += 1
    adjacency_path = out_dir / "adjacency.csv"
    adjacency.to_csv(adjacency_path)

    return {"nodes": nodes_path, "edges": edges_path, "adjacency": adjacency_path}


def export_summary(store: GraphStore, config: Config, out_dir: str | Path = "outputs/graphs") -> Path:
    """Write a human-readable markdown summary of the graph."""

    out_dir = ensure_dir(out_dir)
    gs = GraphSearch(store, config)
    graph = gs.graph

    n_docs = store.count("documents")
    n_chunks = store.count("chunks")
    n_nodes = store.count("nodes")
    n_edges = store.count("edges")

    node_types = Counter(d.get("type") for _, d in graph.nodes(data=True))
    edge_types = Counter(d.get("relation_type") for _, _, d in graph.edges(data=True))

    top_connected = gs.high_degree_nodes(top_n=10)
    repeated = gs.repeated_concepts(min_docs=2)
    weak = gs.weak_links(max_confidence=config.retrieval.minimum_edge_confidence)
    contradictions = gs.contradictions()

    # Find a few sample paths between high-degree nodes for illustration.
    sample_paths = gs.shortest_paths(top_connected[:5], max_depth=4, max_paths=5)

    # Repeated pain points specifically.
    pain_nodes = [
        gs.graph.nodes[n].get("label")
        for n in repeated
        if gs.graph.nodes.get(n, {}).get("type") == "PAIN_POINT"
    ]

    lines = ["# Graph summary", ""]
    lines.append(f"- Documents: {n_docs}")
    lines.append(f"- Chunks: {n_chunks}")
    lines.append(f"- Nodes: {n_nodes}")
    lines.append(f"- Edges: {n_edges}")

    lines.append("\n## Top node types")
    for t, c in node_types.most_common(10):
        lines.append(f"- {t}: {c}")

    lines.append("\n## Top edge types")
    for t, c in edge_types.most_common(10):
        lines.append(f"- {t}: {c}")

    lines.append("\n## Top connected concepts")
    for node_id in top_connected:
        data = graph.nodes[node_id]
        lines.append(f"- {data.get('label')} (degree {graph.degree(node_id)}, type {data.get('type')})")

    lines.append("\n## Repeated pain points (across multiple documents)")
    if pain_nodes:
        for label in pain_nodes:
            lines.append(f"- {label}")
    else:
        lines.append("- (none detected)")

    lines.append("\n## Low-confidence links (need human review)")
    if weak:
        for e in weak[:20]:
            lines.append(
                f"- {e['source_label']} --{e['relation_type']}--> {e['target_label']} "
                f"(confidence {e['confidence']}, status {e['review_status']})"
            )
    else:
        lines.append("- (none)")

    lines.append("\n## Contradictions")
    if contradictions:
        for e in contradictions:
            lines.append(f"- {e['source_label']} CONTRADICTS {e['target_label']} (status {e['review_status']})")
    else:
        lines.append("- (none detected)")

    lines.append("\n## Sample graph paths")
    if sample_paths:
        for path in sample_paths:
            chain = " -> ".join(gs.graph.nodes[n].get("label", n) for n in path)
            lines.append(f"- {chain}")
    else:
        lines.append("- (no multi-step paths found)")

    path = out_dir / "graph_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_image(store: GraphStore, config: Config, out_dir: str | Path = "outputs/graphs") -> Optional[Path]:
    """Best-effort static PNG of the graph using matplotlib. Optional."""

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend, no GUI needed
        import matplotlib.pyplot as plt
    except ImportError:
        warn("matplotlib not installed; skipping image export.")
        return None

    graph = _graph_for_export(store, config)
    if graph.number_of_nodes() == 0:
        warn("Graph is empty; skipping image export.")
        return None

    out_dir = ensure_dir(out_dir)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, k=0.5, seed=42)
    labels = {n: d.get("label", n) for n, d in graph.nodes(data=True)}
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="#9ecae1")
    nx.draw_networkx_edges(graph, pos, alpha=0.3, arrows=True)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    path = out_dir / "graph.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def export_merge_suggestions(rows: list, output_path: str | Path = "outputs/review/node_merge_suggestions.csv") -> Path:
    """Write node merge suggestions to a CSV for human review."""

    ensure_dir(Path(output_path).parent)
    df = pd.DataFrame(
        rows,
        columns=[
            "keep_node_id",
            "keep_label",
            "duplicate_node_id",
            "duplicate_label",
            "type",
            "similarity",
            "approve_merge",
        ],
    )
    df.to_csv(output_path, index=False)
    return Path(output_path)
