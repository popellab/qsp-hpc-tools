#!/bin/bash
# Setup Python virtual environment on HPC using uv
#
# This script uses uv to create a virtual environment with Python 3.11+
# and installs qsp-hpc-tools package (which includes all dependencies).
#
# Prerequisites:
#   - uv must be installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#
# Usage:
#   bash scripts/hpc/setup_hpc_venv.sh [QSP_HPC_TOOLS_SOURCE]
#
# Arguments:
#   QSP_HPC_TOOLS_SOURCE: Where to install qsp-hpc-tools from
#                         Default: git+https://github.com/jeliason/qsp-hpc-tools.git@main

set -e  # Exit on error

VENV_DIR="$HOME/qspio_venv"
PYTHON_VERSION="3.11"

# Package source (from argument or default to GitHub main)
QSP_HPC_TOOLS_SOURCE="${1:-git+https://github.com/jeliason/qsp-hpc-tools.git@main}"

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

# Install qsp-hpc-tools (includes all dependencies)
echo "   → Installing qsp-hpc-tools from: $QSP_HPC_TOOLS_SOURCE"
uv pip install --python "$VENV_DIR/bin/python" "$QSP_HPC_TOOLS_SOURCE"
echo "   ✓ qsp-hpc-tools installed"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
"$VENV_DIR/bin/python" -c "import qsp_hpc; print('   ✓ qsp-hpc-tools imported successfully')"
"$VENV_DIR/bin/python" -c "import numpy, pandas, pyarrow; print('   ✓ Dependencies available')"
"$VENV_DIR/bin/python" --version | sed 's/^/   Python: /'

echo ""
echo "✅ HPC Python environment ready!"
echo ""
echo "To use this environment:"
echo "   source $VENV_DIR/bin/activate"
