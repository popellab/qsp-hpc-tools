#!/usr/bin/env bash
# Build the C++ qsp_sim binary on Rockfish (JHU HPC).
#
# qsp_sim is the pure-C++/CVODE driver used by qsp-hpc-tools' C++ backend
# (see qsp_hpc/batch/cpp_batch_worker.py). It lives in PDAC/qsp/sim/, NOT
# in PDAC/sim/ (that's the full spatial GPU sim and is built separately).
#
# Dependencies:
#   - Boost 1.70+ (serialization)       — Rockfish module Boost/1.82.0-GCC-12.3.0
#   - SUNDIALS 7+ (cvode + nvecserial)  — no Rockfish module; auto-fetched by
#                                         FetchContent (git clone + build, ~1 min)
#
# Usage (from a Rockfish login node):
#   scripts/hpc/build_qsp_sim.sh                            # build in place
#   scripts/hpc/build_qsp_sim.sh --clone ~/SPQSP_PDAC       # clone/update first
#
# The built binary lands at:
#   <repo>/PDAC/qsp/sim/build/qsp_sim
#
# Wire it into ~/.config/qsp-hpc/credentials.yaml:
#   cpp:
#     binary_path: /home/<user>/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim
#     template_path: /home/<user>/SPQSP_PDAC/PDAC/sim/resource/param_all.xml

set -euo pipefail

REPO_DIR=""
REPO_URL="${SPQSP_PDAC_REPO_URL:-git@github.com:popellab/SPQSP_PDAC.git}"
BRANCH="${SPQSP_PDAC_BRANCH:-cpp-sweep-binary-io}"
CLONE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --clone)
            CLONE=1
            REPO_DIR="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [ $CLONE -eq 1 ]; then
    if [ ! -d "$REPO_DIR/.git" ]; then
        echo "Cloning $REPO_URL (branch $BRANCH) into $REPO_DIR ..."
        git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        echo "Updating existing clone at $REPO_DIR ..."
        cd "$REPO_DIR"
        git fetch origin
        git checkout "$BRANCH"
        git pull --ff-only origin "$BRANCH"
    fi
    cd "$REPO_DIR/PDAC/qsp/sim"
else
    # Expect to be run from somewhere inside a SPQSP_PDAC checkout; chdir to
    # the qsp_sim subdir regardless of where we were invoked from.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Script is in qsp-hpc-tools/scripts/hpc/; find SPQSP_PDAC sibling.
    # Walk up until we find a directory that contains SPQSP_PDAC.
    SEARCH="$SCRIPT_DIR"
    while [ "$SEARCH" != "/" ]; do
        if [ -d "$SEARCH/../SPQSP_PDAC/PDAC/qsp/sim" ]; then
            cd "$SEARCH/../SPQSP_PDAC/PDAC/qsp/sim"
            break
        fi
        SEARCH="$(dirname "$SEARCH")"
    done
    if [ ! -f "CMakeLists.txt" ]; then
        echo "ERROR: could not find SPQSP_PDAC/PDAC/qsp/sim. Use --clone." >&2
        exit 1
    fi
fi

echo "=== Loading modules ==="
# GCC 13.2.0: binary depends on GLIBCXX_3.4.32 at runtime, which is in the
# GCC 13 libstdc++. GCC 12.3.0's libstdc++ tops out at 3.4.30 and would
# cause "GLIBCXX_3.4.32 not found" at run time.
module purge
module load GCC/13.2.0 cmake/3.27.7 Boost/1.83.0-GCC-13.2.0 git/2.42.0
module list 2>&1 | sed 's/^/  /'

mkdir -p build
cd build

echo ""
echo "=== Configuring (FetchContent will pull SUNDIALS 7.6.0) ==="
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "=== Building qsp_sim ==="
time cmake --build . --target qsp_sim -j "$(nproc)"

BINARY="$PWD/qsp_sim"
TEMPLATE="$(cd ../../../sim/resource && pwd)/param_all.xml"

if [ -x "$BINARY" ]; then
    echo ""
    echo "=== Build succeeded ==="
    echo "  Binary:   $BINARY"
    echo "  Template: $TEMPLATE"
    echo ""
    echo "Add to ~/.config/qsp-hpc/credentials.yaml:"
    echo ""
    echo "  cpp:"
    echo "    binary_path: $BINARY"
    echo "    template_path: $TEMPLATE"
    echo "    subtree: QSP"
else
    echo "ERROR: binary not found at $BINARY" >&2
    exit 1
fi
