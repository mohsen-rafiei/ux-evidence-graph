"""Chunk embeddings + cosine similarity search.

Embeddings are stored as float32 BLOBs in the SQLite ``embeddings`` table so
everything stays in one local file. Two providers are supported:

* ``lmstudio`` — uses the LM Studio embeddings endpoint (default).
* ``sentence_transformers`` — local fallback using all-MiniLM-L6-v2.

If the configured provider fails (e.g. no embedding model loaded in LM Studio),
we automatically fall back to sentence-transformers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .graph_store import GraphStore
from .lmstudio_client import LMStudioClient, LMStudioError
from .schemas import Chunk
from .utils import info, warn


class VectorStore:
    """Create, persist, and search chunk embeddings."""

    def __init__(self, config: Config, store: GraphStore, client: Optional[LMStudioClient] = None):
        self.config = config
        self.store = store
        self.client = client
        self._st_model = None  # lazily loaded sentence-transformers model
        self._active_model_name = "unknown"

    # ------------------------------------------------------------------
    # Embedding providers
    # ------------------------------------------------------------------
    def _load_sentence_transformer(self):
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is not installed. Install it with "
                    "'pip install sentence-transformers'."
                ) from exc
            info("Loading local fallback embedding model 'all-MiniLM-L6-v2' (first run may download it)...")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model

    def _embed_texts(self, texts: List[str]) -> Tuple[np.ndarray, str]:
        """Embed texts using the configured provider, falling back if needed."""

        provider = self.config.embedding.provider

        if provider == "lmstudio" and self.client is not None:
            try:
                vectors = self.client.embed(texts)
                self._active_model_name = self.config.lmstudio.embedding_model
                return np.array(vectors, dtype=np.float32), self._active_model_name
            except LMStudioError as exc:
                warn(f"LM Studio embeddings unavailable ({exc}). Falling back to sentence-transformers.")

        # Fallback (or explicitly configured) sentence-transformers.
        model = self._load_sentence_transformer()
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=False)
        self._active_model_name = "all-MiniLM-L6-v2"
        return np.array(vectors, dtype=np.float32), self._active_model_name

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def embed_chunks(self, chunks: Optional[List[Chunk]] = None, batch_size: int = 16) -> int:
        """Embed and store vectors for the given chunks (or all stored chunks)."""

        if chunks is None:
            chunks = self.store.get_chunks()
        if not chunks:
            warn("No chunks to embed. Run ingest + extract first.")
            return 0

        count = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.chunk_text for c in batch]
            vectors, model_name = self._embed_texts(texts)
            for chunk, vector in zip(batch, vectors):
                self._store_vector(chunk, vector, model_name)
                count += 1
        info(f"Embedded {count} chunks using '{self._active_model_name}'.")
        return count

    def _store_vector(self, chunk: Chunk, vector: np.ndarray, model_name: str) -> None:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self.store.conn.execute(
            """INSERT OR REPLACE INTO embeddings
               (chunk_id, document_id, file_name, chunk_text, embedding_model, vector)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk.chunk_id, chunk.document_id, chunk.file_name, chunk.chunk_text, model_name, blob),
        )
        self.store.conn.commit()

    def _load_all_vectors(self) -> List[dict]:
        rows = self.store.conn.execute("SELECT * FROM embeddings").fetchall()
        results = []
        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            results.append(
                {
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "file_name": r["file_name"],
                    "chunk_text": r["chunk_text"],
                    "embedding_model": r["embedding_model"],
                    "vector": vec,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def search(self, query: str, top_k: int = 8) -> List[dict]:
        """Return the top_k most similar chunks to ``query`` by cosine similarity."""

        stored = self._load_all_vectors()
        if not stored:
            warn("No embeddings found. Run 'python main.py embed' (or extract) first.")
            return []

        query_vec, _ = self._embed_texts([query])
        query_vec = query_vec[0]

        scored = []
        for item in stored:
            score = self._cosine(query_vec, item["vector"])
            scored.append({**item, "similarity": score})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        # Drop the raw vector from the returned payload to keep it light.
        for item in scored:
            item.pop("vector", None)
        return scored[:top_k]
