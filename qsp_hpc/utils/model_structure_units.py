"""Unit lookup from model_structure.json.

Single source of truth for name→units mapping used when building
Pint Quantities in test statistic derivation. Merges species, compartments,
and parameters so calibration targets can reference any model entity
(e.g., V_T compartment volume, phi_collagen rule output, rho_collagen constant)
without needing a separate species_units.json.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_units_from_model_structure(path: str | Path) -> dict[str, str]:
    """Build a unified name→units dict from model_structure.json.

    Merges species, compartments (using ``volume_units``), and parameters.
    On name collisions, species win over parameters win over compartments.
    """
    with open(path, "r") as f:
        data = json.load(f)

    units: dict[str, str] = {}
    for c in data.get("compartments", []):
        name = c["name"]
        unit = c.get("volume_units") or c.get("units") or "dimensionless"
        units[name] = unit
    for p in data.get("parameters", []):
        units[p["name"]] = p.get("units") or "dimensionless"
    for s in data.get("species", []):
        units[s["name"]] = s.get("units") or "dimensionless"
    return units
