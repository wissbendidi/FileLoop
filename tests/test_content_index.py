"""
Tests for the content index (vector store and search).
"""

import os
import tempfile
import shutil

import numpy as np
import pytest

from src.content import ContentChunk, ContentExtractor
from src.content_index import ContentIndex, Hit, index_directory


class TestHit:
    def test_to_dict(self):
        h = Hit(path="/a/b.txt", text="snippet", score=0.9, chunk_index=1, metadata={})
        d = h.to_dict()
        assert d["path"] == "/a/b.txt"
        assert d["text"] == "snippet"
        assert d["score"] == 0.9
        assert d["chunk_index"] == 1


class TestContentIndex:
    """Test ContentIndex add_chunks and search."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = ContentExtractor(chunk_size=500, chunk_overlap=50)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_chunks(self, *texts):
        path = os.path.join(self.temp_dir, "doc.txt")
        return [
            ContentChunk(path=path, chunk_index=i, text=t, start_char=0, end_char=len(t), metadata={})
            for i, t in enumerate(texts)
        ]

    def test_add_chunks_and_search_in_memory(self):
        try:
            from src.embeddings import EmbeddingGenerator
        except ImportError:
            from embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        chunks = self._make_chunks(
            "Authentication is handled by the login module.",
            "The API configuration uses environment variables.",
            "Database connection string is in config.",
        )
        embeddings = gen.generate_text_embeddings([c.text for c in chunks])

        idx = ContentIndex(embedding_generator=gen, index_dir=None, path_prefix_for_embedding=False)
        idx.add_chunks(chunks, embeddings, replace=True)
        assert idx.count() == 3

        hits = idx.search("Where is authentication handled?", top_k=2)
        assert len(hits) <= 2
        assert len(hits) >= 1
        assert all(isinstance(h, Hit) for h in hits)
        assert hits[0].path == chunks[0].path
        assert "Authentication" in hits[0].text or "login" in hits[0].text

    def test_search_empty_index(self):
        gen = None
        idx = ContentIndex(embedding_generator=gen, index_dir=None)
        hits = idx.search("anything", top_k=5)
        assert hits == []

    def test_add_chunks_length_mismatch_raises(self):
        try:
            from src.embeddings import EmbeddingGenerator
        except ImportError:
            from embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        chunks = self._make_chunks("one", "two")
        embeddings = np.zeros((3, 384))
        idx = ContentIndex(embedding_generator=gen, index_dir=None)
        with pytest.raises(ValueError, match="must match"):
            idx.add_chunks(chunks, embeddings)

    def test_index_directory_integration(self):
        with open(os.path.join(self.temp_dir, "readme.txt"), "w", encoding="utf-8") as f:
            f.write("This project uses Python and Streamlit for the dashboard.")
        idx, n = index_directory(
            self.temp_dir,
            excluded_dirs=[],
            index_dir=None,
        )
        assert n >= 1
        assert idx.count() >= 1
        hits = idx.search("What is the dashboard built with?", top_k=3)
        assert len(hits) >= 1
        assert any("Streamlit" in h.text or "Python" in h.text for h in hits)
