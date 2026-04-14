#!/bin/bash
#SBATCH --job-name=qsp_poc_build
#SBATCH --partition=shared
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/poc_build_%j.out
#SBATCH --error=logs/poc_build_%j.err
#
# One-session build: SimBiology model + SimFunction + accelerate + mcc.
# Must be one session because sf.DependentFiles points to session-scoped
# codegen dirs that MATLAB cleans up on exit.
#
# Consumes two licenses for the duration: SimBiology + MATLAB Compiler.
# Everything after this is license-free at runtime.

set -euo pipefail
mkdir -p logs build
cd "$SLURM_SUBMIT_DIR"

module load matlab/R2024a

t0=$(date +%s)
matlab -nodisplay -nosplash -batch "build_deployed"
t1=$(date +%s)
echo "[wall] build+mcc complete in $((t1 - t0))s"
ls -lh build/
