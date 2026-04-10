"""
QSP HPC Tools

A Python package for running quantitative systems pharmacology (QSP) simulations
on HPC clusters with intelligent caching and pooling.

Main components:
- simulation: QSP simulator and simulation pool management
- batch: HPC job submission and management via SLURM
- utils: Utility functions for hashing and caching
"""

__version__ = "0.1.0"

from qsp_hpc.simulation.qsp_simulator import QSPSimulator
from qsp_hpc.simulation.simulation_pool import SimulationPoolManager

__all__ = [
    "SimulationPoolManager",
    "QSPSimulator",
]
