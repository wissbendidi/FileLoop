"""
FileLoop web app: FastAPI backend with a single chat-style UI.

Run with: uvicorn src.app:app --reload
Or: python -m uvicorn src.app:app --host 0.0.0.0 --port 8501
"""

import os
from pathlib import Path
from typing import Optional, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# In-process state (one folder / one index at a time for local use)
_content_index: Any = None
_indexed_path: Optional[str] = None
_index_chunk_count: int = 0

# Default excluded dirs for indexing
DEFAULT_EXCLUDED = [".git", "__pycache__", "node_modules", ".venv", ".env"]


def _imports():
    try:
        from src.content_index import ContentIndex, index_directory
        from src.content import ContentExtractor
        from src.ollama_client import check_ollama_available, list_models, generate as ollama_generate
    except ImportError:
        from content_index import ContentIndex, index_directory
        from content import ContentExtractor
        from ollama_client import check_ollama_available, list_models, generate as ollama_generate
    return ContentIndex, index_directory, ContentExtractor, check_ollama_available, list_models, ollama_generate


app = FastAPI(title="FileLoop", description="Chat about your files locally.")

# Mount static files (created next)
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class IndexRequest(BaseModel):
    folder_path: str
    excluded_dirs: Optional[List[str]] = None


class ChatRequest(BaseModel):
    message: str
    use_ollama: bool = False
    ollama_model: Optional[str] = None
    ollama_base_url: Optional[str] = None


class ListDirQuery(BaseModel):
    path: Optional[str] = None


@app.get("/")
async def root():
    """Serve the single-page chat UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "FileLoop API. Serve static/index.html for the chat UI."}


@app.get("/api/status")
async def api_status():
    """Return current index status."""
    global _indexed_path, _index_chunk_count
    return {
        "indexed_path": _indexed_path,
        "chunk_count": _index_chunk_count,
        "ready": _content_index is not None and _index_chunk_count > 0,
    }


@app.post("/api/index")
async def api_index(req: IndexRequest):
    """Build or refresh the content index for the given folder."""
    global _content_index, _indexed_path, _index_chunk_count
    path = Path(req.folder_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail="Path does not exist")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    excluded = req.excluded_dirs or DEFAULT_EXCLUDED
    try:
        ContentIndex, index_directory, *_ = _imports()
        idx, n = index_directory(str(path), excluded_dirs=excluded, index_dir=None)
        _content_index = idx
        _indexed_path = str(path)
        _index_chunk_count = n
        return {"chunk_count": n, "indexed_path": _indexed_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Run semantic search and optionally Ollama answer over the current index."""
    global _content_index, _index_chunk_count
    if _content_index is None or _index_chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No index built. Set a folder and build the index first.",
        )
    try:
        _, _, _, check_ollama_available, _, ollama_generate = _imports()
        hits = _content_index.search(req.message, top_k=15)
        hits_payload = [
            {"path": h.path, "text": h.text, "score": round(h.score, 4), "chunk_index": h.chunk_index}
            for h in hits
        ]
        answer = None
        if req.use_ollama and check_ollama_available(req.ollama_base_url or "http://localhost:11434"):
            base = req.ollama_base_url or "http://localhost:11434"
            model = req.ollama_model or "llama3.2"
            excerpts = "\n\n---\n\n".join(
                [f"From {Path(h.path).name}:\n{h.text[:800]}" for h in hits[:8]]
            )
            system = "Answer concisely based only on the following excerpts from the user's files."
            prompt_text = f"Excerpts:\n\n{excerpts}\n\nUser question: {req.message}"
            answer = ollama_generate(prompt_text, model=model, base_url=base, system=system)
        return {"answer": answer, "hits": hits_payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/list_dir")
async def api_list_dir(path: Optional[str] = None):
    """List directories and files for browsing. Returns entries for the given path or cwd."""
    base = Path(path).expanduser().resolve() if path else Path.cwd()
    if not base.exists():
        raise HTTPException(status_code=400, detail="Path does not exist")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    try:
        entries = []
        for p in sorted(base.iterdir()):
            try:
                entries.append({
                    "name": p.name,
                    "path": str(p.resolve()),
                    "is_dir": p.is_dir(),
                })
            except (PermissionError, OSError):
                continue
        return {"path": str(base), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ollama")
async def api_ollama():
    """Check if Ollama is available and return model list."""
    try:
        _, _, _, check_ollama_available, list_models, _ = _imports()
        base = "http://localhost:11434"
        ok = check_ollama_available(base)
        models = list_models(base) if ok else []
        return {"available": ok, "models": models}
    except Exception:
        return {"available": False, "models": []}


def main():
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8501, reload=True)
