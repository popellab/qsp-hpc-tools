#!/usr/bin/env python3
"""
Batch Processing Utilities

Provides utility functions for splitting simulations across batch jobs.
"""


def calculate_batch_split(num_simulations: int, max_tasks: int) -> tuple[int, int]:
    """
    Calculate optimal split of simulations across batch tasks.

    Given a total number of simulations and a maximum number of parallel tasks,
    computes how many simulations should be assigned per task (jobs_per_chunk)
    and how many tasks are actually needed.

    Args:
        num_simulations: Total number of simulations to run
        max_tasks: Maximum number of parallel tasks allowed

    Returns:
        Tuple of (jobs_per_chunk, n_tasks):
        - jobs_per_chunk: Number of simulations per task (at least 1)
        - n_tasks: Actual number of tasks needed

    Examples:
        >>> calculate_batch_split(100, 10)
        (10, 10)  # 100 sims / 10 tasks = 10 sims/task, 10 tasks

        >>> calculate_batch_split(25, 10)
        (3, 9)    # 25 sims / 10 tasks = 3 sims/task (rounded up), 9 tasks

        >>> calculate_batch_split(5, 10)
        (1, 5)    # 5 sims / 10 tasks = 1 sim/task (minimum), 5 tasks
    """
    # Ensure at least 1 simulation per task
    # Use ceiling division: (a + b - 1) // b
    jobs_per_chunk = max(1, (num_simulations + max_tasks - 1) // max_tasks)

    # Calculate actual number of tasks needed with this chunk size
    n_tasks = (num_simulations + jobs_per_chunk - 1) // jobs_per_chunk

    return jobs_per_chunk, n_tasks


def calculate_num_tasks(num_simulations: int, jobs_per_chunk: int) -> int:
    """
    Calculate number of tasks needed for given simulations and chunk size.

    Args:
        num_simulations: Total number of simulations to run
        jobs_per_chunk: Number of simulations per task

    Returns:
        Number of tasks needed (rounded up)

    Examples:
        >>> calculate_num_tasks(100, 10)
        10  # 100 sims / 10 per task = 10 tasks

        >>> calculate_num_tasks(105, 10)
        11  # 105 sims / 10 per task = 11 tasks (rounded up)
    """
    return (num_simulations + jobs_per_chunk - 1) // jobs_per_chunk
