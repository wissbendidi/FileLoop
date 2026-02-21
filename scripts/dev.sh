#!/bin/bash
# Development script for FileLoop

set -e

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "🚀 Starting FileLoop development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies
echo "📦 Syncing dependencies..."
uv sync --dev

# Run tests
echo "🧪 Running tests..."
uv run pytest

# Start the application
echo "🎯 Starting FileLoop application..."
uv run fileloop
