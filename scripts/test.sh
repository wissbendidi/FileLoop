#!/bin/bash
# Test script for FileLoop

set -e

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "🧪 Running FileLoop tests..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies
uv sync --dev

# Run tests with coverage
echo "Running tests with coverage..."
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

echo "✅ Tests completed! Coverage report generated in htmlcov/"
