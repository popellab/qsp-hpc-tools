#!/bin/bash
# Setup Python virtual environment on HPC using uv
#
# This script uses uv to create a virtual environment with Python 3.11+
# and installs required packages for Parquet I/O and test statistics derivation.
#
# Prerequisites:
#   - uv must be installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#
# Usage:
#   bash scripts/setup_hpc_venv.sh

set -e  # Exit on error

VENV_DIR="$HOME/qspio_venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.11"

echo "🐍 Setting up Python virtual environment for QSP on HPC (using uv)"
echo "   Location: $VENV_DIR"
echo "   Python: $PYTHON_VERSION"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo ""
    echo "❌ Error: uv is not installed"
    echo ""
    echo "Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   source ~/.bashrc  # or ~/.zshrc"
    echo ""
    exit 1
fi

echo "   ✓ uv is installed: $(uv --version)"

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "   ✓ Virtual environment already exists"
    echo "   → Will update packages if needed"
else
    echo "   → Creating new virtual environment with Python $PYTHON_VERSION..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    echo "   ✓ Virtual environment created"
fi

# Install/update packages using uv
echo "   → Installing required packages with uv..."
if [ -f "$PROJECT_ROOT/requirements_hpc.txt" ]; then
    uv pip install --python "$VENV_DIR/bin/python" -r "$PROJECT_ROOT/requirements_hpc.txt"
    echo "   ✓ Packages installed"
else
    echo "   ⚠️  requirements_hpc.txt not found, installing manually..."
    uv pip install --python "$VENV_DIR/bin/python" numpy pandas pyarrow scipy
    echo "   ✓ Packages installed"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
"$VENV_DIR/bin/python" -c "import numpy, pandas, pyarrow, scipy; print('   ✓ All packages imported successfully')"
"$VENV_DIR/bin/python" --version | sed 's/^/   Python: /'

echo ""
echo "✅ HPC Python environment ready!"
echo ""
echo "To use this environment:"
echo "   source $VENV_DIR/bin/activate"
