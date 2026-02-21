# FileLoop - AI-Powered Local File Cleaner

A privacy-focused, local AI-powered file cleaner that helps you organize and clean up your computer files without uploading anything to the cloud.

##  Features

-  **Chat about your files**: One chat interface (ChatGPT-style). Set a folder, build the index, then ask questions in natural language.
-  **Content-aware search**: Results are based on the actual content of files (text, code, PDF, DOCX), not just metadata.
-  **Browse or type path**: Enter a folder path or browse directories to choose what to index.
-  **Privacy-first**: All processing runs locally; no cloud APIs, no file uploads.
-  **Optional Ollama**: Use a local LLM (Ollama) for short answers in addition to relevant snippets.


##  Quick Start

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies and run:**
```bash
uv sync
uv run fileloop
```

3. **Open your browser** at **http://localhost:8501**

### Option 2: Using pip

```bash
pip install -e .
fileloop
```

Then open **http://localhost:8501**.

##  Usage Guide

### Chat workflow

1. **Set folder**: Enter the path to the folder you want to analyze, or click **Browse** to pick it from a directory list.
2. **Build index**: Click **Build index** to scan and index file contents (text, code, PDFs, DOCX). Wait until you see "Indexed N chunks".
3. **Ask questions**: Type in the chat box (e.g. "Where is authentication handled?", "What config is used for the API?") and press Enter or **Send**.
4. **Optional**: Check **Use Ollama for answers** to get a short natural-language answer from a local Ollama model in addition to the file snippets.

All processing runs locally; nothing is sent to the cloud.

##  Privacy & Safety

- **100% Local Processing**: All AI analysis happens on your machine
- **No Data Upload**: Files never leave your computer
- **Safe Operations**: Files are moved to quarantine, never deleted directly
- **Review Before Action**: Always review files before any cleanup operations



### Running Tests

**Using uv (Recommended):**
```bash
# Run tests with coverage
uv run pytest

# Or use the test script
./scripts/test.sh
```

**Using pip:**
```bash
pytest tests/
```

### Adding New Features
1. Follow the modular architecture in `src/`
2. Add type hints and docstrings to all functions
3. Write tests for new functionality
4. Update this README with new features

##  Current Status

**Completed:**
- Content extraction (text, code, PDF, DOCX) and chunking
- Local content index (in-memory or Chroma) and semantic search
- Single-page chat UI (FastAPI + vanilla JS) with folder path and browse
- Optional Ollama integration for natural-language answers
- File scanning, embeddings, duplicate detection, and quarantine (library code; UI focused on chat)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

See LICENSE file for details.
