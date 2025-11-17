"""
Constants used across the qsp-hpc-tools package.

This module defines shared constants to avoid magic numbers and ensure
consistency across the codebase.
"""

# Hash prefix length for identifiers (pool names, file names, etc.)
# Using 8 hex characters = 32 bits = 1 in 4 billion collision probability
HASH_PREFIX_LENGTH = 8

# Job monitoring timeouts and delays
SLURM_REGISTRATION_DELAY = 5  # seconds to wait for SLURM to register jobs
JOB_QUEUE_TIMEOUT = 120  # seconds to wait before assuming queue monitoring failed
