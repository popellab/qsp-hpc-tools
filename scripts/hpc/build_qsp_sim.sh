#!/usr/bin/env bash
# Build the C++ qsp_sim binary on an HPC cluster (e.g. Rockfish).
#
# Prerequisites:
#   - CMake >= 3.14 and a C++17 compiler (module load cmake gcc)
#   - SUNDIALS with CVODE installed (module load sundials, or build from source)
#
# Usage:
#   # First time — clone the repo and build:
#   bash build_qsp_sim.sh /home/$USER/SPQSP_PDAC
#
#   # Subsequent runs — pulls latest and rebuilds:
#   bash build_qsp_sim.sh /home/$USER/SPQSP_PDAC
#
# The built binary lands at:
#   <repo>/PDAC/qsp/sim/build/qsp_sim
#
# Point credentials.yaml at it:
#   cpp:
#     binary_path: /home/<user>/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim
#     template_path: /home/<user>/SPQSP_PDAC/PDAC/sim/resource/param_all.xml

set -euo pipefail

REPO_DIR="${1:?Usage: build_qsp_sim.sh <SPQSP_PDAC_ROOT>}"
REPO_URL="${SPQSP_PDAC_REPO_URL:-git@github.com:jeliason/SPQSP_PDAC.git}"
BRANCH="${SPQSP_PDAC_BRANCH:-cpp-sweep-binary-io}"

echo "=== Building qsp_sim ==="
echo "  Repo dir:  $REPO_DIR"
echo "  Branch:    $BRANCH"

# Clone or update
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning $REPO_URL ..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "Updating existing clone ..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
fi

BUILD_DIR="$REPO_DIR/PDAC/qsp/sim/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with CMake ..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building ..."
cmake --build . --target qsp_sim -j "$(nproc)"

BINARY="$BUILD_DIR/qsp_sim"
if [ -x "$BINARY" ]; then
    echo "=== Build succeeded ==="
    echo "  Binary: $BINARY"
    echo "  Template: $REPO_DIR/PDAC/sim/resource/param_all.xml"
    echo ""
    echo "Add to ~/.config/qsp-hpc/credentials.yaml:"
    echo "  cpp:"
    echo "    binary_path: $BINARY"
    echo "    template_path: $REPO_DIR/PDAC/sim/resource/param_all.xml"
else
    echo "ERROR: Binary not found at $BINARY" >&2
    exit 1
fi
