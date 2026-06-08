"""Generate evidence-grounded answers from retrieved context.

The model is told to use ONLY the retrieved chunks and graph relationships, to
distinguish strong vs weak evidence, to flag pending/unreviewed links, and to
cite document names and chunk ids. The graph is explicitly not treated as
ground truth.

Every answer is saved as a markdown file under ``outputs/answers/``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config
from .graph_store import GraphStore
from .lmstudio_client import LMStudioClient
from .retrieval import RetrievalResult
from .schemas import new_id
from .utils import ensure_dir, now_iso


ANSWER_SYSTEM_PROMPT = """You are helping a UX researcher reason over structured qualitative evidence.

Use only the provided retrieved chunks and graph relationships.
Do not invent evidence.
Distinguish between strong evidence, possible patterns, and weak connections.
Mention when a connection is based on pending or unreviewed links.
Cite document names and chunk ids.
Do not treat the graph as ground truth.
Include practical interpretation for UX research.

Structure your answer with exactly these markdown sections:

## Answer
## Evidence used
## Graph relationships used
## Human checks needed
## Confidence level
"""


def _format_context(retrieval: RetrievalResult) -> str:
    """Turn the retrieval result into a compact text context for the model."""

    lines = []

    lines.append("### Retrieved text chunks (semantic search)")
    if retrieval.chunks:
        for c in retrieval.chunks:
            snippet = c["chunk_text"][:700].replace("\n", " ")
            lines.append(
                f"- [{c['file_name']} | chunk {c['chunk_id']} | similarity {c['similarity']:.2f}]: {snippet}"
            )
    else:
        lines.append("- (no chunks retrieved)")

    lines.append("\n### Graph nodes (concepts)")
    if retrieval.relevant_nodes:
        for n in retrieval.relevant_nodes[:30]:
            lines.append(
                f"- {n['label']} (type={n['type']}, confidence={n['confidence']}, status={n['review_status']})"
            )
    else:
        lines.append("- (no nodes retrieved)")

    lines.append("\n### Graph relationships (edges)")
    if retrieval.relevant_edges:
        for e in retrieval.relevant_edges[:40]:
            lines.append(
                f"- {e['source_label']} --{e['relation_type']}--> {e['target_label']} "
                f"(confidence={e['confidence']}, weight={e['weight']}, status={e['review_status']}, "
                f"source={e.get('source_doc')} chunk={e.get('source_chunk')})"
                + (f" | evidence: \"{(e.get('evidence_text') or '')[:200]}\"" if e.get("evidence_text") else "")
            )
    else:
        lines.append("- (no edges retrieved)")

    lines.append("\n### Graph paths (possible reasoning chains)")
    if retrieval.graph_paths:
        for path in retrieval.graph_paths:
            chain = " -> ".join(p["label"] for p in path)
            lines.append(f"- {chain}")
    else:
        lines.append("- (no multi-step paths found)")

    lines.append("\n### Source documents")
    lines.append(", ".join(retrieval.source_docs) if retrieval.source_docs else "(none)")

    lines.append("\n### Confidence summary")
    for k, v in retrieval.confidence_summary.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def generate_answer(
    question: str,
    retrieval: RetrievalResult,
    client: LMStudioClient,
    store: Optional[GraphStore] = None,
    answers_dir: str | Path = "outputs/answers",
) -> str:
    """Generate a markdown answer, persist it, and return the markdown text."""

    context = _format_context(retrieval)
    user_prompt = (
        f"Researcher question:\n{question}\n\n"
        f"Retrieved evidence and graph context:\n{context}\n\n"
        "Write the answer using the required sections. Remember: only use the "
        "evidence above, cite document names and chunk ids, and clearly mark "
        "anything based on pending/unreviewed links."
    )

    answer_body = client.chat(ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0.2)

    # Prepend a small header with metadata for traceability.
    header = (
        f"# Question\n\n{question}\n\n"
        f"*Generated: {now_iso()}*\n\n"
        f"*Confidence summary: {retrieval.confidence_summary}*\n\n"
        "---\n\n"
    )
    full_markdown = header + answer_body

    # Persist to disk.
    out_dir = ensure_dir(answers_dir)
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:50] or "question"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_{slug}.md"
    out_path.write_text(full_markdown, encoding="utf-8")

    if store is not None:
        store.add_answer(new_id("ans"), question, full_markdown)

    return full_markdown
