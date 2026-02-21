"""
Content extraction and chunking for file contents.

This module extracts text from supported file types (plain text, code, PDF, DOCX)
and splits it into chunks for indexing and semantic search. All processing is local.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Max file size to process (10 MB) to avoid OOM
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Chunking defaults: ~2000 chars with 100 char overlap (~512 tokens equivalent)
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 100

# Supported extensions: plain text / code
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".htm", ".css",
    ".json", ".yaml", ".yml", ".csv", ".xml", ".toml", ".ini", ".cfg", ".conf",
    ".sh", ".bash", ".zsh", ".rb", ".go", ".rs", ".java", ".kt", ".c", ".cpp",
    ".h", ".hpp", ".php", ".r", ".sql", ".graphql", ".vue", ".svelte",
}

# Extensions that need special extraction
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}

ALL_INDEXABLE_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS


@dataclass
class ContentChunk:
    """A single chunk of extracted file content for indexing."""
    path: str
    chunk_index: int
    text: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class ContentExtractor:
    """Extract and chunk text content from files in a directory."""

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        chunk_size: int = CHUNK_SIZE_CHARS,
        chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    ):
        self.max_file_size = max_file_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_directory(
        self,
        root_path: str,
        excluded_dirs: Optional[List[str]] = None,
    ) -> List[ContentChunk]:
        """
        Extract content from all indexable files under root_path and return chunks.

        Args:
            root_path: Directory to scan.
            excluded_dirs: Directory names to skip (e.g. .git, node_modules).

        Returns:
            List of ContentChunk for all extracted content.
        """
        try:
            from .scanner import FileScanner
        except ImportError:
            from scanner import FileScanner

        scanner = FileScanner(excluded_dirs=excluded_dirs or [])
        file_metadata = scanner.scan_directory(root_path)

        chunks: List[ContentChunk] = []
        for f in file_metadata:
            if f.is_directory:
                continue
            ext = f.extension.lower()
            if ext not in ALL_INDEXABLE_EXTENSIONS:
                continue
            if f.size > self.max_file_size:
                logger.debug("Skipping large file %s (%d bytes)", f.path, f.size)
                continue
            try:
                file_chunks = self.extract_file(f.path)
                chunks.extend(file_chunks)
            except Exception as e:
                logger.warning("Could not extract %s: %s", f.path, e)
                continue

        return chunks

    def extract_file(self, file_path: str) -> List[ContentChunk]:
        """
        Extract and chunk content from a single file.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List of ContentChunk for this file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file type is not supported.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.stat().st_size > self.max_file_size:
            return []

        ext = path.suffix.lower()
        if ext in PDF_EXTENSIONS:
            text = self._extract_pdf(str(path))
        elif ext in DOCX_EXTENSIONS:
            text = self._extract_docx(str(path))
        elif ext in TEXT_EXTENSIONS:
            text = self._extract_text(str(path))
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        if not text or not text.strip():
            return []

        return self._chunk_text(text, str(path))

    def _extract_text(self, file_path: str) -> str:
        """Read plain text/code file with encoding fallback."""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        logger.warning("Could not decode %s with utf-8 or latin-1", file_path)
        return ""

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.warning("PyPDF2 not available; cannot extract PDF")
            return ""

        parts = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    parts.append(text)
        except Exception as e:
            logger.warning("PDF extraction failed for %s: %s", file_path, e)
            return ""
        return "\n\n".join(parts)

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document
        except ImportError:
            logger.warning("python-docx not available; cannot extract DOCX")
            return ""

        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.warning("DOCX extraction failed for %s: %s", file_path, e)
            return ""

    def _chunk_text(self, text: str, path: str) -> List[ContentChunk]:
        """
        Split text into overlapping chunks. Prefer paragraph/newline and
        code boundaries (def, class, function) when possible.
        """
        path_str = str(Path(path).resolve())
        # Prefer splitting on double newline (paragraphs), then single newline
        segments = re.split(r"\n\n+", text)
        if len(segments) == 1 and len(text) <= self.chunk_size:
            return [
                ContentChunk(
                    path=path_str,
                    chunk_index=0,
                    text=text.strip(),
                    start_char=0,
                    end_char=len(text),
                    metadata={},
                )
            ]

        # For code, try to split by def/class/function to keep units together
        is_likely_code = any(
            path_str.lower().endswith(ext)
            for ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".go", ".java")
        )
        if is_likely_code:
            chunks_raw = self._chunk_by_code_units(text)
        else:
            chunks_raw = self._chunk_by_paragraphs_or_fixed(segments)

        result = []
        for i, (start, end, chunk_text) in enumerate(chunks_raw):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            result.append(
                ContentChunk(
                    path=path_str,
                    chunk_index=i,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={},
                )
            )
        return result

    def _chunk_by_code_units(self, text: str) -> List[tuple]:
        """Split by def/class/function boundaries when possible; else fixed size."""
        pattern = re.compile(
            r"^(\s*(?:def|class|function|func)\s+\w+.*)$",
            re.MULTILINE,
        )
        positions = [0]
        for m in pattern.finditer(text):
            positions.append(m.start())
        positions.append(len(text))

        # Merge positions that are too close so we don't get tiny chunks
        merged = [positions[0]]
        for p in positions[1:-1]:
            if p - merged[-1] >= self.chunk_size // 2:
                merged.append(p)
        merged.append(len(text))

        chunks = []
        for i in range(len(merged) - 1):
            start, end = merged[i], merged[i + 1]
            chunk_text = text[start:end]
            if len(chunk_text) > self.chunk_size:
                # Sub-split this segment by fixed size with overlap
                sub = self._fixed_size_chunks(chunk_text, start)
                chunks.extend(sub)
            else:
                chunks.append((start, end, chunk_text))
        return chunks

    def _chunk_by_paragraphs_or_fixed(
        self, segments: List[str]
    ) -> List[tuple]:
        """Build chunks from paragraph segments; use fixed size if a segment is huge."""
        chunks = []
        current = []
        current_len = 0
        start_offset = 0

        for seg in segments:
            seg_len = len(seg) + 2  # +2 for \n\n
            if seg_len > self.chunk_size and not current:
                # Single segment too large: fixed-size split
                full_text = seg
                for s, e, t in self._fixed_size_chunks(full_text, 0):
                    chunks.append((s, e, t))
                continue
            if current_len + seg_len > self.chunk_size and current:
                text = "\n\n".join(current)
                end_offset = start_offset + len(text)
                chunks.append((start_offset, end_offset, text))
                # Overlap: keep last part of current
                overlap_text = ""
                for c in reversed(current):
                    if len(overlap_text) + len(c) + 2 <= self.chunk_overlap:
                        overlap_text = c + ("\n\n" if overlap_text else "") + overlap_text
                    else:
                        break
                start_offset = end_offset - len(overlap_text)
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            current.append(seg)
            current_len += seg_len

        if current:
            text = "\n\n".join(current)
            end_offset = start_offset + len(text)
            chunks.append((start_offset, end_offset, text))
        return chunks

    def _fixed_size_chunks(self, text: str, global_start: int) -> List[tuple]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Try to break at newline near end
            if end < len(text):
                last_nl = text.rfind("\n", start, end + 1)
                if last_nl > start + self.chunk_size // 2:
                    end = last_nl + 1
            chunk_text = text[start:end]
            chunks.append((global_start + start, global_start + end, chunk_text))
            start = end - self.chunk_overlap
            if start >= end:
                start = end
        return chunks


# Example usage
if __name__ == "__main__":
    extractor = ContentExtractor()
    try:
        chunks = extractor.extract_directory(".", excluded_dirs=[".git", "__pycache__"])
        print(f"Extracted {len(chunks)} chunks")
        for c in chunks[:3]:
            print(f"  {c.path} #{c.chunk_index}: {c.text[:80]}...")
    except Exception as e:
        print(f"Error: {e}")
