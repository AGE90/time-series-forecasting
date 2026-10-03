#!/bin/bash
# Install all dependency groups and the pre-commit hooks.
# Git, DVC, .env and the directory layout are created by the template itself.
set -euo pipefail

command -v uv >/dev/null || { echo "❌ Error: uv is not installed (https://docs.astral.sh/uv/)"; exit 1; }

uv sync --all-groups
uv run pre-commit install

echo "✅ Setup complete. Run 'make test' to verify."
