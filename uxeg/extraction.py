"""Send chunks to the local model and parse structured nodes/edges.

The model is asked to return JSON only. We validate with pydantic; if the
first attempt is not valid JSON we retry once with a short repair prompt. If
it still fails, the raw response is saved under ``outputs/errors/`` and we move
on so one bad chunk does not stop the whole run.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .lmstudio_client import LMStudioClient
from .schemas import (
    Chunk,
    Edge,
    EdgeType,
    ExtractionResult,
    Node,
    NodeType,
    ReviewStatus,
)
from .schemas import NODE_TYPE_VALUES, EDGE_TYPE_VALUES
from .utils import ensure_dir, extract_json_block, normalize_label, now_iso, warn


EXTRACTION_SYSTEM_PROMPT = f"""You are helping a UX researcher structure qualitative evidence.

Extract only information that is supported by the provided text.
Do not invent missing information.
Return valid JSON only.
Extract nodes and edges.
Use only the allowed node types and edge types.
Each edge must include evidence text from the source chunk.
Assign confidence from 0 to 1.
Assign weight from 0 to 1 based on the strength of the relationship.
Use lower confidence when a relationship is implied rather than explicit.
Prefer meaningful analytical relationships over broad semantic similarity.
When unsure, use RELATED_TO with lower confidence.

Allowed node types: {", ".join(NODE_TYPE_VALUES)}
Allowed edge types: {", ".join(EDGE_TYPE_VALUES)}

Return JSON with exactly this shape:
{{
  "nodes": [
    {{"label": "string", "normalized_label": "string", "type": "ONE_OF_NODE_TYPES", "description": "string", "confidence": 0.0}}
  ],
  "edges": [
    {{"source_label": "string", "target_label": "string", "relation_type": "ONE_OF_EDGE_TYPES", "description": "string", "evidence_text": "string", "confidence": 0.0, "weight": 0.0}}
  ]
}}
Return JSON only. No markdown, no commentary."""


REPAIR_SYSTEM_PROMPT = """You returned text that was not valid JSON.
Return ONLY a valid JSON object with keys "nodes" and "edges" matching the
requested schema. Do not include any prose, explanation, or markdown fences."""


def _build_user_prompt(chunk: Chunk) -> str:
    return (
        f"Source document: {chunk.file_name}\n"
        f"Chunk id: {chunk.chunk_id}\n\n"
        f"Text to analyse:\n\"\"\"\n{chunk.chunk_text}\n\"\"\"\n\n"
        "Extract the nodes and edges as JSON."
    )


def _parse_result(raw: str) -> Optional[ExtractionResult]:
    """Try to parse a model response into a validated ExtractionResult."""

    json_str = extract_json_block(raw)
    if not json_str:
        return None
    try:
        return ExtractionResult.model_validate_json(json_str)
    except Exception:  # noqa: BLE001 - any validation issue means "not usable"
        return None


def extract_from_chunk(
    client: LMStudioClient,
    chunk: Chunk,
    errors_dir: str | Path = "outputs/errors",
) -> Optional[ExtractionResult]:
    """Extract structured nodes/edges from one chunk, with one repair retry."""

    user_prompt = _build_user_prompt(chunk)

    raw = client.chat(EXTRACTION_SYSTEM_PROMPT, user_prompt, temperature=0.1)
    result = _parse_result(raw)
    if result is not None:
        return result

    # Repair attempt: hand the bad output back and ask for clean JSON.
    repair_user = (
        "The following response was not valid JSON. Convert it into a valid "
        "JSON object with keys 'nodes' and 'edges'.\n\n" + raw
    )
    raw2 = client.chat(REPAIR_SYSTEM_PROMPT, repair_user, temperature=0.0)
    result = _parse_result(raw2)
    if result is not None:
        return result

    # Give up on this chunk; persist the failure for debugging.
    errors_path = ensure_dir(errors_dir)
    fail_file = errors_path / f"failed_{chunk.chunk_id}.txt"
    fail_file.write_text(
        f"chunk_id: {chunk.chunk_id}\nfile_name: {chunk.file_name}\ntime: {now_iso()}\n\n"
        f"--- attempt 1 ---\n{raw}\n\n--- attempt 2 (repair) ---\n{raw2}\n",
        encoding="utf-8",
    )
    warn(f"Could not parse extraction for chunk {chunk.chunk_id}; saved to {fail_file}")
    return None


def result_to_records(
    result: ExtractionResult,
    chunk: Chunk,
    model_name: str,
    default_status: ReviewStatus,
    label_to_node: dict,
) -> Tuple[List[Node], List[Edge]]:
    """Convert an ExtractionResult into Node/Edge records ready for storage.

    ``label_to_node`` maps ``(normalized_label, type)`` -> existing Node so that
    repeated concepts within a run reuse the same node id. Callers should pass a
    dict that persists across chunks (and may be pre-seeded from the database).
    """

    nodes: List[Node] = []
    edges: List[Edge] = []

    def get_or_create_node(label: str, node_type: NodeType, description: str, confidence: float) -> Node:
        norm = normalize_label(label)
        key = (norm, node_type.value)
        if key in label_to_node:
            return label_to_node[key]
        node = Node(
            label=label,
            normalized_label=norm,
            type=node_type,
            description=description,
            source_doc=chunk.file_name,
            source_chunk=chunk.chunk_id,
            confidence=confidence,
            created_by_model=model_name,
            review_status=ReviewStatus.PENDING,
        )
        label_to_node[key] = node
        nodes.append(node)
        return node

    # Create nodes first so edges can reference them.
    for en in result.nodes:
        get_or_create_node(en.label, en.type, en.description, en.confidence)

    for ee in result.edges:
        # Edge endpoints may reference labels not explicitly listed in nodes;
        # create lightweight CONCEPT nodes for those so the graph stays valid.
        src = _resolve_label_node(ee.source_label, result, get_or_create_node)
        tgt = _resolve_label_node(ee.target_label, result, get_or_create_node)

        edge = Edge(
            source_node_id=src.id,
            target_node_id=tgt.id,
            source_label=src.label,
            target_label=tgt.label,
            relation_type=ee.relation_type,
            description=ee.description,
            evidence_text=ee.evidence_text,
            source_doc=chunk.file_name,
            source_chunk=chunk.chunk_id,
            confidence=ee.confidence,
            weight=ee.weight,
            review_status=default_status,
        )
        edges.append(edge)

    return nodes, edges


def _resolve_label_node(label: str, result: ExtractionResult, get_or_create_node) -> Node:
    """Find the declared node type for a label, defaulting to CONCEPT."""

    norm = normalize_label(label)
    for en in result.nodes:
        if normalize_label(en.label) == norm:
            return get_or_create_node(en.label, en.type, en.description, en.confidence)
    # Not declared explicitly; treat as a generic concept.
    return get_or_create_node(label, NodeType.CONCEPT, "", 0.4)
