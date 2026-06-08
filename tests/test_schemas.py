"""Tests for pydantic schema validation."""

import pytest
from pydantic import ValidationError

from uxeg.schemas import (
    Edge,
    EdgeType,
    ExtractedEdge,
    ExtractionResult,
    Node,
    NodeType,
    ReviewStatus,
)


def test_node_defaults():
    node = Node(label="Pricing confusion", normalized_label="pricing confusion", type=NodeType.PAIN_POINT)
    assert node.id.startswith("node_")
    assert node.review_status == ReviewStatus.PENDING
    assert 0.0 <= node.confidence <= 1.0


def test_edge_defaults_to_pending():
    edge = Edge(
        source_node_id="n1",
        target_node_id="n2",
        source_label="A",
        target_label="B",
        relation_type=EdgeType.CAUSES,
    )
    assert edge.review_status == ReviewStatus.PENDING
    assert edge.id.startswith("edge_")


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ExtractedEdge(
            source_label="A",
            target_label="B",
            relation_type=EdgeType.RELATED_TO,
            confidence=1.5,
        )


def test_invalid_node_type_rejected():
    with pytest.raises(ValidationError):
        Node(label="x", normalized_label="x", type="NOT_A_TYPE")


def test_extraction_result_parses_valid_json():
    payload = """
    {
      "nodes": [
        {"label": "Onboarding confusion", "normalized_label": "onboarding confusion",
         "type": "PAIN_POINT", "description": "users unsure of first step", "confidence": 0.8}
      ],
      "edges": [
        {"source_label": "Onboarding confusion", "target_label": "Trust",
         "relation_type": "REDUCES", "description": "confusion lowers trust",
         "evidence_text": "the confusing onboarding lowered my trust", "confidence": 0.7, "weight": 0.6}
      ]
    }
    """
    result = ExtractionResult.model_validate_json(payload)
    assert len(result.nodes) == 1
    assert result.nodes[0].type == NodeType.PAIN_POINT
    assert len(result.edges) == 1
    assert result.edges[0].relation_type == EdgeType.REDUCES


def test_empty_label_rejected():
    with pytest.raises(ValidationError):
        ExtractedEdge(source_label="  ", target_label="B", relation_type=EdgeType.RELATED_TO)
