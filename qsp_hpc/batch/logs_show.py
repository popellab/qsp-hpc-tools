#!/usr/bin/env python3
"""
Show logs from HPC batch job derivation tasks.

This utility fetches and displays logs from the latest HPC SLURM array job
for test statistic derivation.

Usage:
    # Show task 0 from latest job
    python -m qsp_hpc.batch.logs_show

    # Show task 3 from latest job
    python -m qsp_hpc.batch.logs_show 3

    # Show last 100 lines instead of 50
    python -m qsp_hpc.batch.logs_show 3 --lines 100
"""

import argparse
import sys

from qsp_hpc.batch.hpc_job_manager import HPCJobManager
from qsp_hpc.utils.security import build_safe_ssh_command


def show_logs(array_task_id: int = 0, lines: int = 50):
    """
    Show logs from latest HPC derivation job.

    Args:
        array_task_id: Array task ID to show logs for (default: 0)
        lines: Number of lines to show from end of log (default: 50)
    """
    print("📋 Fetching HPC derivation logs...")
    print(f"   Task: {array_task_id}")

    # Initialize job manager
    job_manager = HPCJobManager()

    # Build log directory path
    log_dir = f"{job_manager.config.remote_project_path}/batch_jobs/logs"

    # Find the latest derivation log file
    print("   → Finding latest qsp_derive logs...")

    # Use safe command construction
    list_cmd = build_safe_ssh_command(
        ["sh", "-c", "ls -t qsp_derive_*.out 2>/dev/null | head -1"], cwd=log_dir
    )
    status, output = job_manager.transport.exec(list_cmd)

    if status != 0 or not output.strip():
        print(f"   ✗ No qsp_derive logs found in {log_dir}")
        sys.exit(1)

    latest_log = output.strip()
    # Extract job ID from filename
    # Format: qsp_derive_12345_6.out
    parts = latest_log.replace(".out", "").split("_")
    if len(parts) < 3:
        print(f"   ✗ Could not parse job ID from: {latest_log}")
        sys.exit(1)

    job_id = parts[-2]
    print(f"   ✓ Found: Job {job_id}")

    # Build log file paths
    out_log = f"qsp_derive_{job_id}_{array_task_id}.out"
    err_log = f"qsp_derive_{job_id}_{array_task_id}.err"

    # Show stdout log
    print(f"\n{'='*80}")
    print(f"📄 STDOUT: {out_log}")
    print(f"{'='*80}")

    # Use safe command construction
    tail_cmd = build_safe_ssh_command(
        [
            "sh",
            "-c",
            f'if [ -f {out_log} ]; then tail -{lines} {out_log}; else echo "(Log file not found)"; fi',
        ],
        cwd=log_dir,
    )
    status, output = job_manager.transport.exec(tail_cmd)

    if status == 0:
        print(output)
    else:
        print("   ✗ Failed to fetch stdout log")

    # Show stderr log
    print(f"\n{'='*80}")
    print(f"📄 STDERR: {err_log}")
    print(f"{'='*80}")

    # Use safe command construction
    tail_cmd = build_safe_ssh_command(
        [
            "sh",
            "-c",
            f'if [ -f {err_log} ]; then if [ -s {err_log} ]; then tail -{lines} {err_log}; else echo "(Empty - no errors)"; fi; else echo "(Log file not found)"; fi',
        ],
        cwd=log_dir,
    )
    status, output = job_manager.transport.exec(tail_cmd)

    if status == 0:
        print(output)
    else:
        print("   ✗ Failed to fetch stderr log")

    # Show summary of all tasks for this job
    print(f"\n{'='*80}")
    print(f"📊 All tasks for job {job_id}:")
    print(f"{'='*80}")

    # Use safe command construction
    summary_cmd = build_safe_ssh_command(
        [
            "sh",
            "-c",
            f'echo "Available tasks:"; ls qsp_derive_{job_id}_*.out 2>/dev/null | sed "s/.*_{job_id}_//;s/.out$//" | sort -n | head -20; echo ""; echo "Task count: $(ls qsp_derive_{job_id}_*.out 2>/dev/null | wc -l)"',
        ],
        cwd=log_dir,
    )
    status, output = job_manager.transport.exec(summary_cmd)

    if status == 0:
        print(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Show logs from latest HPC derivation job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show task 0 from latest job
  python -m qsp_hpc.batch.logs_show

  # Show task 3 from latest job
  python -m qsp_hpc.batch.logs_show 3

  # Show last 100 lines instead of 50
  python -m qsp_hpc.batch.logs_show 3 --lines 100
        """,
    )

    parser.add_argument(
        "task", type=int, nargs="?", default=0, help="Array task ID to show logs for (default: 0)"
    )

    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of lines to show from end of log (default: 50)",
    )

    args = parser.parse_args()

    try:
        show_logs(array_task_id=args.task, lines=args.lines)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
