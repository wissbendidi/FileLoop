# FileLoop - AI-Powered Local File Cleaner

A privacy-focused, local AI-powered file cleaner that helps you organize and clean up your computer files without uploading anything to the cloud.

##  Features

-  **Smart File Scanning**: Collects metadata for all files (path, size, timestamps, etc.)
-  **AI-Powered Classification**: Uses local embeddings to cluster and classify files
-  **Interactive Visualization**: Beautiful treemap and sunburst charts to explore your files
-  **Privacy-First**: All processing happens locally - no cloud APIs, no file uploads
-  **Safe Operations**: Files are moved to quarantine folders, never deleted directly
-  **Duplicate Detection**: Find duplicate files using SHA256/MD5 hashing


##  Quick Start

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies and run:**
```bash
# Sync dependencies
uv sync

# Run the application
uv run streamlit run src/ui.py
```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

### Option 2: Using pip (Legacy)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
streamlit run src/ui.py
```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

##  Usage Guide

### Basic Workflow

1. **Select Directory**: Enter the path to the directory you want to analyze
2. **Configure Options**: Choose which directories to exclude and clustering settings
3. **Scan Files**: Click "Scan Directory" to collect file metadata
4. **AI Analysis**: Click "Run AI Analysis" to cluster and classify files
5. **Find Duplicates**: Click "Find Duplicates" to identify duplicate files
6. **Explore Visualizations**: Use the interactive charts to understand your file organization
7. **Clean Up**: Move files to quarantine for safe review (coming soon!)

### Features Overview

- **File Statistics**: See total files, size, and category breakdown
- **Treemap View**: Hierarchical visualization of your file organization
- **Sunburst Chart**: Circular view of file categories and clusters
- **Cluster Analysis**: AI-powered grouping of similar files
- **Duplicate Detection**: Find and manage duplicate files safely

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
- File scanning and metadata collection
- Local AI embedding generation
- File clustering and classification
- Interactive visualizations (treemap, sunburst, scatter plots)
- Duplicate file detection
- Streamlit web interface
- Basic safety features (quarantine system)

**In Progress:**
- Content extraction for PDFs and documents
- Enhanced safety features
- Export functionality (CSV/JSON)
- Search functionality
- Performance optimizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

See LICENSE file for details.
