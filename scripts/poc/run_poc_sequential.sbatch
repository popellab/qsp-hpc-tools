#!/bin/bash
#SBATCH --job-name=qsp_poc_seq
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/poc_seq_%j.out
#SBATCH --error=logs/poc_seq_%j.err
#
# Fire the compiled exe 4 times back-to-back on the same node. If Runtime
# init is cold-cache-bound, runs 2-4 should show a much smaller wall time
# than run 1. If it's genuine MCR startup, all four are ~equal.

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

MATLAB_ROOT=/data/apps/extern/matlab/R2024a
export LD_LIBRARY_PATH="$MATLAB_ROOT/runtime/glnxa64:$MATLAB_ROOT/bin/glnxa64:$MATLAB_ROOT/sys/os/glnxa64:$MATLAB_ROOT/sys/opengl/lib/glnxa64:${LD_LIBRARY_PATH:-}"
export XAPPLRESDIR="$MATLAB_ROOT/X11/app-defaults"

echo "[seq] node: $SLURMD_NODENAME"

for i in 1 2 3 4; do
    echo ">>> run $i"
    t0=$(date +%s.%N)
    ./build/qsp_poc_main 20
    t1=$(date +%s.%N)
    echo ">>> run $i wall: $(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")s"
    echo ""
done
