"""ux-evidence-graph: turn unstructured UX text into a reviewable evidence graph.

High-level usage::

    from uxeg import EvidenceGraphProject

    project = EvidenceGraphProject.from_config("config.yaml")
    project.init()
    project.ingest("data/sample")
    project.extract()
    project.export_review("outputs/review/edges_for_review.csv")
    answer = project.ask("Which pain points are connected to trust?")
    print(answer)

This is a lightweight research prototype, not a validated automated analysis
tool. The graph suggests possible reasoning paths; humans validate the links.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import Config
from .graph_store import GraphStore
from .lmstudio_client import LMStudioClient
from .retrieval import RetrievalResult, hybrid_retrieve
from .vector_store import VectorStore
from .utils import info, success, warn

__all__ = ["EvidenceGraphProject", "Config"]
__version__ = "0.1.0"


class EvidenceGraphProject:
    """Facade over the full pipeline so the project is usable as a library."""

    def __init__(self, config: Config):
        self.config = config
        self.store = GraphStore(config.database_path)
        self._client: Optional[LMStudioClient] = None
        self._vector_store: Optional[VectorStore] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> "EvidenceGraphProject":
        return cls(Config.from_yaml(path))

    @property
    def client(self) -> LMStudioClient:
        if self._client is None:
            self._client = LMStudioClient(self.config)
        return self._client

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(self.config, self.store, self.client)
        return self._vector_store

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def init(self) -> None:
        """Create the database and tables."""

        self.store.init_db()

    def ingest(self, input_folder: str | Path = "data/raw") -> int:
        """Read files from ``input_folder``, chunk them, and store everything."""

        from .chunking import chunk_text
        from .ingestion import ingest_folder

        documents = ingest_folder(input_folder)
        if not documents:
            warn(f"No supported documents found in '{input_folder}'.")
            return 0

        n_chunks = 0
        for doc in documents:
            self.store.add_document(doc)
            chunks = chunk_text(
                doc.raw_text,
                doc.document_id,
                doc.file_name,
                self.config.chunk_size_words,
                self.config.chunk_overlap_words,
            )
            for chunk in chunks:
                self.store.add_chunk(chunk)
                n_chunks += 1

        info(f"Ingested {len(documents)} document(s) into {n_chunks} chunk(s).")
        return n_chunks

    def extract(self) -> int:
        """Run extraction on all stored chunks, persisting nodes and edges."""

        from .extraction import extract_from_chunk, result_to_records
        from .schemas import ReviewStatus

        chunks = self.store.get_chunks()
        if not chunks:
            warn("No chunks found. Run ingest first.")
            return 0

        if not self.client.health_check(verbose=True):
            warn("LM Studio does not appear reachable. Extraction may fail.")

        default_status = ReviewStatus(self.config.review.default_edge_status)
        model_name = self.config.lmstudio.chat_model

        # Seed the label->node map from existing nodes so repeated concepts
        # across runs reuse the same node id.
        label_to_node = {}
        for node in self.store.get_nodes():
            label_to_node[(node.normalized_label, node.type.value)] = node

        total_nodes = 0
        total_edges = 0
        for i, chunk in enumerate(chunks, start=1):
            info(f"Extracting chunk {i}/{len(chunks)} ({chunk.file_name})...")
            result = extract_from_chunk(self.client, chunk)
            if result is None:
                continue
            nodes, edges = result_to_records(result, chunk, model_name, default_status, label_to_node)
            for node in nodes:
                self.store.add_node(node)
                total_nodes += 1
            for edge in edges:
                self.store.add_edge(edge)
                total_edges += 1

        success(f"Extraction complete: {total_nodes} new node(s), {total_edges} edge(s).")
        return total_edges

    def embed(self) -> int:
        """Create and store embeddings for all chunks."""

        return self.vector_store.embed_chunks()

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------
    def export_review(self, output_path: str = "outputs/review/edges_for_review.csv") -> int:
        from .review import export_edges

        n = export_edges(self.store, output_path)
        success(f"Exported {n} edge(s) to {output_path}")
        return n

    def import_review(self, input_path: str = "outputs/review/edges_for_review.csv") -> int:
        from .review import import_edges

        n = import_edges(self.store, input_path)
        success(f"Updated {n} edge(s) from {input_path}")
        return n

    def export_nodes_review(self, output_path: str = "outputs/review/nodes_for_review.csv") -> int:
        from .review import export_nodes

        n = export_nodes(self.store, output_path)
        success(f"Exported {n} node(s) to {output_path}")
        return n

    def import_nodes_review(self, input_path: str = "outputs/review/nodes_for_review.csv") -> int:
        from .review import import_nodes

        n = import_nodes(self.store, input_path)
        success(f"Updated {n} node(s) from {input_path}")
        return n

    def suggest_merges(self, output_path: str = "outputs/review/node_merge_suggestions.csv") -> int:
        from .export import export_merge_suggestions
        from .normalization import suggest_merges

        rows = suggest_merges(self.store)
        export_merge_suggestions(rows, output_path)
        success(f"Wrote {len(rows)} merge suggestion(s) to {output_path}")
        return len(rows)

    def import_merges(self, input_path: str = "outputs/review/node_merge_suggestions.csv") -> int:
        import pandas as pd

        from .normalization import apply_merge

        df = pd.read_csv(input_path)
        applied = 0
        for _, row in df.iterrows():
            if str(row.get("approve_merge", "no")).strip().lower() in ("yes", "y", "true", "1"):
                if apply_merge(self.store, str(row["keep_node_id"]), str(row["duplicate_node_id"])):
                    applied += 1
        success(f"Applied {applied} merge(s).")
        return applied

    # ------------------------------------------------------------------
    # Retrieval + answering
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        """Run hybrid retrieval (graph + vector) for a question."""

        # Ensure embeddings exist for the vector path.
        if self.store.count("embeddings") == 0 and self.store.count("chunks") > 0:
            info("No embeddings found yet; creating them now...")
            self.embed()

        return hybrid_retrieve(question, self.store, self.config, self.vector_store)

    def ask(self, question: str) -> str:
        """Retrieve evidence and generate a grounded markdown answer."""

        from .answer_generation import generate_answer

        retrieval = self.retrieve(question)
        answer = generate_answer(question, retrieval, self.client, self.store)
        return answer

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def export_graph(self, fmt: str = "json") -> Path:
        from . import export as export_mod

        if fmt == "json":
            return export_mod.export_json(self.store, self.config)
        if fmt == "graphml":
            return export_mod.export_graphml(self.store, self.config)
        raise ValueError(f"Unknown graph format '{fmt}'. Use 'json' or 'graphml'.")

    def export_tables(self) -> dict:
        from . import export as export_mod

        return export_mod.export_tables(self.store)

    def export_summary(self) -> Path:
        from . import export as export_mod

        return export_mod.export_summary(self.store, self.config)

    def export_image(self) -> Optional[Path]:
        from . import export as export_mod

        return export_mod.export_image(self.store, self.config)

    def health_check(self) -> bool:
        return self.client.health_check(verbose=True)

    def close(self) -> None:
        self.store.close()
