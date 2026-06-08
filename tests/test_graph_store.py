"""Tests for SQLite persistence, review CSV round-trip, and graph construction."""

import pytest

from uxeg.config import Config
from uxeg.graph_store import GraphStore
from uxeg.review import export_edges, import_edges
from uxeg.schemas import (
    Chunk,
    Document,
    Edge,
    EdgeType,
    Node,
    NodeType,
    ReviewStatus,
)


@pytest.fixture()
def store(tmp_path):
    db_path = tmp_path / "test.sqlite"
    gs = GraphStore(db_path)
    gs.init_db()
    yield gs
    gs.close()


def _make_two_nodes(store):
    a = Node(label="Onboarding confusion", normalized_label="onboarding confusion", type=NodeType.PAIN_POINT)
    b = Node(label="Trust", normalized_label="trust", type=NodeType.CONCEPT)
    store.add_node(a)
    store.add_node(b)
    return a, b


def test_node_insertion_and_lookup(store):
    a, b = _make_two_nodes(store)
    assert store.count("nodes") == 2
    found = store.find_node_by_normalized("trust", "CONCEPT")
    assert found is not None
    assert found.id == b.id


def test_edge_insertion(store):
    a, b = _make_two_nodes(store)
    edge = Edge(
        source_node_id=a.id,
        target_node_id=b.id,
        source_label=a.label,
        target_label=b.label,
        relation_type=EdgeType.REDUCES,
        confidence=0.7,
        weight=0.6,
    )
    store.add_edge(edge)
    assert store.count("edges") == 1
    edges = store.get_edges()
    assert edges[0].relation_type == EdgeType.REDUCES
    assert edges[0].review_status == ReviewStatus.PENDING


def test_document_and_chunk_insertion(store):
    doc = Document(file_name="f.txt", file_path="/tmp/f.txt", file_type="txt", raw_text="hello world")
    store.add_document(doc)
    chunk = Chunk(document_id=doc.document_id, file_name="f.txt", chunk_text="hello world", start_word=0, end_word=2)
    store.add_chunk(chunk)
    assert store.count("documents") == 1
    assert store.count("chunks") == 1


def test_review_csv_round_trip(store, tmp_path):
    a, b = _make_two_nodes(store)
    edge = Edge(
        source_node_id=a.id,
        target_node_id=b.id,
        source_label=a.label,
        target_label=b.label,
        relation_type=EdgeType.RELATED_TO,
        confidence=0.4,
        weight=0.4,
    )
    store.add_edge(edge)

    csv_path = tmp_path / "edges.csv"
    n = export_edges(store, csv_path)
    assert n == 1

    # Simulate a human edit: approve the edge and bump weight.
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.loc[0, "review_status"] = "approved"
    df.loc[0, "weight"] = 0.9
    df.to_csv(csv_path, index=False)

    updated = import_edges(store, csv_path)
    assert updated == 1
    reloaded = store.get_edges()[0]
    assert reloaded.review_status == ReviewStatus.APPROVED
    assert reloaded.weight == 0.9


def test_build_graph_excludes_rejected(store):
    a, b = _make_two_nodes(store)
    good = Edge(
        source_node_id=a.id, target_node_id=b.id, source_label=a.label, target_label=b.label,
        relation_type=EdgeType.REDUCES, review_status=ReviewStatus.APPROVED,
    )
    bad = Edge(
        source_node_id=a.id, target_node_id=b.id, source_label=a.label, target_label=b.label,
        relation_type=EdgeType.RELATED_TO, review_status=ReviewStatus.REJECTED,
    )
    store.add_edge(good)
    store.add_edge(bad)

    graph = store.build_graph(include_pending=True, use_rejected=False)
    assert graph.number_of_nodes() == 2
    # Only the approved edge should be present.
    assert graph.number_of_edges() == 1


def test_database_not_initialized_raises(tmp_path):
    gs = GraphStore(tmp_path / "fresh.sqlite")
    with pytest.raises(RuntimeError):
        gs._ensure_initialized()
    gs.close()
