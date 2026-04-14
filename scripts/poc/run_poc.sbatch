#!/bin/bash
#SBATCH --job-name=qsp_poc_run
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/poc_run_%j.out
#SBATCH --error=logs/poc_run_%j.err
#
# Run the mcc-built exe against the existing MATLAB R2024a runtime libs.
# This is the deployment path we're probing — no `module load matlab`,
# no MATLAB session, no license seat consumed at runtime.
#
# Args:
#   $1 = number of sims to run inside the exe (default 5)
#
# Sanity check: the exe should start and finish in seconds, not the ~80s
# floor we measured for `matlab -batch` per task.

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

N_SIMS="${1:-5}"

# Point at the system MATLAB install's runtime libs (no separate MCR install).
MATLAB_ROOT=/data/apps/extern/matlab/R2024a
export LD_LIBRARY_PATH="$MATLAB_ROOT/runtime/glnxa64:$MATLAB_ROOT/bin/glnxa64:$MATLAB_ROOT/sys/os/glnxa64:$MATLAB_ROOT/sys/opengl/lib/glnxa64:${LD_LIBRARY_PATH:-}"
export XAPPLRESDIR="$MATLAB_ROOT/X11/app-defaults"

echo "[run] node: $SLURMD_NODENAME"
echo "[run] launching ./build/qsp_poc_main $N_SIMS at $(date)"
t0=$(date +%s.%N)

./build/qsp_poc_main "$N_SIMS"

t1=$(date +%s.%N)
echo "[run] wall: $(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")s"
