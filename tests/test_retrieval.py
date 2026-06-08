"""Tests for GraphSearch and hybrid retrieval (without calling LM Studio)."""

import pytest

from uxeg.config import Config
from uxeg.graph_store import GraphStore
from uxeg.retrieval import GraphSearch, hybrid_retrieve
from uxeg.schemas import Edge, EdgeType, Node, NodeType, ReviewStatus


@pytest.fixture()
def populated_store(tmp_path):
    gs = GraphStore(tmp_path / "test.sqlite")
    gs.init_db()

    onboarding = Node(label="Onboarding confusion", normalized_label="onboarding confusion",
                      type=NodeType.PAIN_POINT, source_doc="interview_01.txt", confidence=0.8)
    trust = Node(label="Trust", normalized_label="trust", type=NodeType.CONCEPT,
                 source_doc="interview_02.txt", confidence=0.9)
    verification = Node(label="Extra verification", normalized_label="extra verification",
                        type=NodeType.BEHAVIOR, source_doc="survey_comments.csv", confidence=0.7)
    for n in (onboarding, trust, verification):
        gs.add_node(n)

    gs.add_edge(Edge(
        source_node_id=onboarding.id, target_node_id=trust.id,
        source_label=onboarding.label, target_label=trust.label,
        relation_type=EdgeType.REDUCES, confidence=0.8, weight=0.7,
        review_status=ReviewStatus.APPROVED, source_doc="interview_01.txt",
    ))
    gs.add_edge(Edge(
        source_node_id=trust.id, target_node_id=verification.id,
        source_label=trust.label, target_label=verification.label,
        relation_type=EdgeType.CAUSES, confidence=0.3, weight=0.4,
        review_status=ReviewStatus.PENDING, source_doc="interview_02.txt",
    ))

    yield gs
    gs.close()


def test_keyword_search_finds_node(populated_store):
    config = Config()
    gs = GraphSearch(populated_store, config)
    ids = gs.find_nodes_by_keyword("trust")
    labels = [gs.graph.nodes[i]["label"] for i in ids]
    assert "Trust" in labels


def test_neighbors(populated_store):
    config = Config()
    gs = GraphSearch(populated_store, config)
    trust_id = gs.find_nodes_by_keyword("trust")[0]
    neighbors = gs.neighbors(trust_id, depth=1)
    assert len(neighbors) >= 1


def test_weak_links_detected(populated_store):
    config = Config()
    gs = GraphSearch(populated_store, config)
    weak = gs.weak_links(max_confidence=0.45)
    # The pending CAUSES edge has confidence 0.3.
    assert any(e["relation_type"] == "CAUSES" for e in weak)


def test_hybrid_retrieve_without_vectors(populated_store):
    config = Config()
    result = hybrid_retrieve("Which pain points are connected to trust?", populated_store, config, vector_store=None)
    assert len(result.relevant_nodes) >= 1
    assert result.confidence_summary["total_edges"] >= 1
    # Approved edge should sort before pending edge.
    statuses = [e["review_status"] for e in result.relevant_edges]
    if "approved" in statuses and "pending" in statuses:
        assert statuses.index("approved") < statuses.index("pending")


def test_shortest_path_between_concepts(populated_store):
    config = Config()
    gs = GraphSearch(populated_store, config)
    onboarding_id = gs.find_nodes_by_keyword("onboarding")[0]
    verification_id = gs.find_nodes_by_keyword("verification")[0]
    paths = gs.shortest_paths([onboarding_id, verification_id], max_depth=3)
    assert len(paths) >= 1
    # Path should connect onboarding -> trust -> verification.
    assert len(paths[0]) == 3
