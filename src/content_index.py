"""
Content index for semantic search over file chunks.

Uses ChromaDB for persistent or in-memory vector storage. Caller provides
chunks and precomputed embeddings; search embeds the query via EmbeddingGenerator
and returns top-k hits. All processing is local.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chroma collection name
COLLECTION_NAME = "fileloop_content"


@dataclass
class Hit:
    """A single search hit: path, chunk text, score, and metadata."""
    path: str
    text: str
    score: float
    chunk_index: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "text": self.text,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one vector and many (each row of b)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(bn, an)


class ContentIndex:
    """
    Vector index over content chunks. Uses ChromaDB when available (and index_dir
    may be set for persistence); otherwise falls back to an in-memory numpy store.
    Search uses the same embedding model as the one used to produce chunk embeddings.
    """

    def __init__(
        self,
        embedding_generator: Optional[Any] = None,
        index_dir: Optional[str] = None,
        path_prefix_for_embedding: bool = True,
    ):
        """
        Args:
            embedding_generator: Used to embed the query in search(). If None,
                a default EmbeddingGenerator is created (lazy).
            index_dir: If set and Chroma is available, index persists under this directory.
                If Chroma is not available, index is in-memory only and this is ignored.
            path_prefix_for_embedding: If True, chunk text is stored as
                "file: {path}\\n{text}" for path-aware similarity (we still return raw text in Hit).
        """
        self._embedding_generator = embedding_generator
        self._index_dir = index_dir
        self._path_prefix = path_prefix_for_embedding
        self._client = None
        self._collection = None
        self._collection_name = COLLECTION_NAME
        self._use_chroma = None
        self._memory_embeddings: Optional[np.ndarray] = None
        self._memory_records: List[Dict[str, Any]] = []

    def _get_embedding_generator(self):
        if self._embedding_generator is None:
            try:
                from .embeddings import EmbeddingGenerator
            except ImportError:
                from embeddings import EmbeddingGenerator
            self._embedding_generator = EmbeddingGenerator()
        return self._embedding_generator

    def _chroma_available(self) -> bool:
        if self._use_chroma is None:
            try:
                import chromadb  # noqa: F401
                self._use_chroma = True
            except ImportError:
                self._use_chroma = False
        return self._use_chroma

    def _get_client(self):
        if self._client is None and self._chroma_available():
            import chromadb
            if self._index_dir:
                path = Path(self._index_dir).resolve()
                path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(path))
                logger.info("Using persistent Chroma at %s", path)
            else:
                self._client = chromadb.EphemeralClient()
                logger.info("Using in-memory Chroma")
        return self._client

    def _get_collection(self, create: bool = True):
        if not self._chroma_available():
            return None
        client = self._get_client()
        if client is None:
            return None
        if create:
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            try:
                self._collection = client.get_collection(name=self._collection_name)
            except Exception:
                self._collection = None
        return self._collection

    def _parse_chunk(self, chunk: Any) -> tuple:
        path = chunk.path if hasattr(chunk, "path") else chunk["path"]
        chunk_index = chunk.chunk_index if hasattr(chunk, "chunk_index") else chunk["chunk_index"]
        text = chunk.text if hasattr(chunk, "text") else chunk["text"]
        start_char = chunk.start_char if hasattr(chunk, "start_char") else chunk.get("start_char", 0)
        end_char = chunk.end_char if hasattr(chunk, "end_char") else chunk.get("end_char", 0)
        meta = chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {})
        return path, chunk_index, text, start_char, end_char, meta

    def add_chunks(
        self,
        chunks: List[Any],
        embeddings: np.ndarray,
        replace: bool = True,
    ) -> None:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks: List of ContentChunk (or dicts with path, chunk_index, text,
                start_char, end_char, metadata).
            embeddings: Array of shape (len(chunks), embedding_dim).
            replace: If True, clear the collection before adding (default).
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks length ({len(chunks)}) must match embeddings length ({len(embeddings)})"
            )

        if self._chroma_available():
            self._add_chunks_chroma(chunks, embeddings, replace)
        else:
            self._add_chunks_memory(chunks, embeddings, replace)

    def _add_chunks_memory(self, chunks: List[Any], embeddings: np.ndarray, replace: bool) -> None:
        if replace:
            self._memory_embeddings = None
            self._memory_records = []
        records = []
        for chunk, _ in zip(chunks, embeddings):
            path, chunk_index, text, start_char, end_char, meta = self._parse_chunk(chunk)
            records.append({
                "path": path,
                "text": text,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
                "metadata": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
            })
        emb_arr = np.asarray(embeddings, dtype=np.float64)
        if self._memory_embeddings is None:
            self._memory_embeddings = emb_arr
        else:
            self._memory_embeddings = np.vstack([self._memory_embeddings, emb_arr])
        self._memory_records.extend(records)
        logger.info("Indexed %d chunks (in-memory)", len(chunks))

    def _add_chunks_chroma(self, chunks: List[Any], embeddings: np.ndarray, replace: bool) -> None:
        coll = self._get_collection(create=True)
        if replace:
            try:
                self._get_client().delete_collection(self._collection_name)
            except Exception:
                pass
            coll = self._get_client().get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        ids = []
        documents = []
        metadatas = []

        for chunk, emb in zip(chunks, embeddings):
            path, chunk_index, text, start_char, end_char, meta = self._parse_chunk(chunk)
            doc_id = f"{path}:{chunk_index}:{uuid.uuid4().hex[:8]}"
            ids.append(doc_id)
            doc_text = f"file: {path}\n{text}" if self._path_prefix else text
            documents.append(doc_text)
            meta_record = {"path": path, "chunk_index": chunk_index, "start_char": start_char, "end_char": end_char}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta_record[k] = v
            metadatas.append(meta_record)

        coll.add(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)
        self._collection = coll
        logger.info("Indexed %d chunks (Chroma)", len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 10,
        path_filter: Optional[str] = None,
    ) -> List[Hit]:
        """
        Search the index by semantic similarity to the query string.

        Returns:
            List of Hit (path, text, score, chunk_index, metadata). Score is similarity (higher is better).
        """
        if self._chroma_available() and self._get_collection(create=False) is not None:
            return self._search_chroma(query, top_k, path_filter)
        return self._search_memory(query, top_k, path_filter)

    def _search_memory(
        self, query: str, top_k: int, path_filter: Optional[str]
    ) -> List[Hit]:
        if self._memory_embeddings is None or len(self._memory_records) == 0:
            return []
        gen = self._get_embedding_generator()
        query_emb = gen.generate_text_embeddings([query]).ravel()
        scores = _cosine_similarity(query_emb, self._memory_embeddings)
        order = np.argsort(-scores)
        hits = []
        for i in order:
            rec = self._memory_records[i]
            if path_filter and path_filter not in rec["path"]:
                continue
            if len(hits) >= top_k:
                break
            hits.append(Hit(
                path=rec["path"],
                text=rec["text"],
                score=float(scores[i]),
                chunk_index=rec["chunk_index"],
                metadata=rec.get("metadata", {}),
            ))
        return hits

    def _search_chroma(
        self, query: str, top_k: int, path_filter: Optional[str]
    ) -> List[Hit]:
        coll = self._get_collection(create=False)
        if coll is None:
            return []

        gen = self._get_embedding_generator()
        query_embedding = gen.generate_text_embeddings([query])
        query_list = query_embedding.tolist()

        n_results = coll.count()
        if n_results == 0:
            return []
        request_n = min(n_results, top_k * 20 if path_filter else top_k)

        result = coll.query(
            query_embeddings=query_list,
            n_results=request_n,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not result or not result["ids"] or not result["ids"][0]:
            return hits

        distances = result["distances"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]

        for dist, doc, meta in zip(distances, docs, metas):
            if path_filter and path_filter not in meta.get("path", ""):
                continue
            if len(hits) >= top_k:
                break
            path = meta.get("path", "")
            chunk_index = meta.get("chunk_index", 0)
            if self._path_prefix and doc and doc.startswith("file: "):
                first_nl = doc.find("\n")
                text = doc[first_nl + 1:] if first_nl != -1 else doc
            else:
                text = doc or ""
            score = 1.0 - float(dist) if dist is not None else 0.0
            hits.append(
                Hit(
                    path=path,
                    text=text,
                    score=score,
                    chunk_index=chunk_index,
                    metadata={k: v for k, v in meta.items() if k not in ("path", "chunk_index", "start_char", "end_char")},
                )
            )
        return hits

    def count(self) -> int:
        """Return the number of chunks in the index."""
        if self._chroma_available():
            coll = self._get_collection(create=False)
            if coll is not None:
                return coll.count()
        if self._memory_records is not None:
            return len(self._memory_records)
        return 0

    def clear(self) -> None:
        """Remove all documents from the index."""
        if self._chroma_available():
            client = self._get_client()
            if client is not None:
                try:
                    client.delete_collection(self._collection_name)
                except Exception:
                    pass
            self._collection = None
        self._memory_embeddings = None
        self._memory_records = []


def index_directory(
    root_path: str,
    excluded_dirs: Optional[List[str]] = None,
    index_dir: Optional[str] = None,
    embedding_generator: Optional[Any] = None,
) -> tuple[ContentIndex, int]:
    """
    Build a content index for a directory: extract chunks, embed them, and add to index.

    Args:
        root_path: Directory to index.
        excluded_dirs: Directory names to exclude from scanning.
        index_dir: If set, persist Chroma under this path. If None, in-memory.
        embedding_generator: Optional EmbeddingGenerator; created if None.

    Returns:
        (ContentIndex instance, number of chunks indexed).
    """
    try:
        from .content import ContentExtractor
    except ImportError:
        from content import ContentExtractor

    extractor = ContentExtractor()
    chunks = extractor.extract_directory(root_path, excluded_dirs=excluded_dirs)
    if not chunks:
        idx = ContentIndex(embedding_generator=embedding_generator, index_dir=index_dir)
        return idx, 0

    gen = embedding_generator
    if gen is None:
        try:
            from .embeddings import EmbeddingGenerator
        except ImportError:
            from embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()

    texts = [c.text for c in chunks]
    if ContentIndex(None)._path_prefix:
        texts_for_embed = [f"file: {c.path}\n{c.text}" for c in chunks]
    else:
        texts_for_embed = texts
    embeddings = gen.generate_text_embeddings(texts_for_embed)

    idx = ContentIndex(embedding_generator=gen, index_dir=index_dir)
    idx.add_chunks(chunks, embeddings, replace=True)
    return idx, len(chunks)
