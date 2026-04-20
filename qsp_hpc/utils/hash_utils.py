#!/usr/bin/env python3
"""
Hash computation utilities for parameter and species definitions.

This module provides stable content hashing for parameter and species definitions,
ensuring that semantic changes create new hashes while preserving hashes during
pure syntactic changes (like renaming).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def compute_pool_id_hash(
    priors_csv: Union[str, Path],
    model_script: str,
    model_version: str,
    submodel_priors_yaml: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> str:
    """Hash inputs that determine raw simulation outputs (full sha256 hex).

    Used by :class:`SimulationPool` and :class:`QSPResultLoader` to derive
    the pool directory's 8-hex prefix. Includes only inputs that change the
    parameter draws or sim trajectories — NOT test_stats (those live in a
    subdir) and NOT scenario (it's the dir-name suffix).

    ``seed`` is part of the hash because ``theta_pool`` draws are
    seed-dependent: row ``sample_index=i`` points at a different theta
    under a different seed, so pools from different seeds cannot be
    mixed. Leaving seed out silently served stale draws when users
    reran with a new seed (see #20). ``seed=None`` reproduces the
    pre-fix hash for backward compatibility with legacy pools; new
    callers should always pass the seed they use for theta sampling.
    """
    h = hashlib.sha256()
    h.update(Path(priors_csv).read_text().encode("utf-8"))
    if submodel_priors_yaml is not None:
        smp = Path(submodel_priors_yaml)
        if smp.exists():
            h.update(smp.read_text().encode("utf-8"))
    h.update(model_script.encode("utf-8"))
    h.update(model_version.encode("utf-8"))
    if seed is not None:
        h.update(f"seed={int(seed)}".encode("utf-8"))
    return h.hexdigest()


def compute_test_stats_hash(test_stats_csv: Union[str, Path]) -> str:
    """Hash the test_stats CSV content (full sha256 hex).

    Used as the subdirectory name under ``{pool_dir}/test_stats/`` so multiple
    test_stats variants can share the same raw sim pool.
    """
    return hashlib.sha256(Path(test_stats_csv).read_text().encode("utf-8")).hexdigest()


def _safe_sort_key(entry: Dict[str, Any]) -> tuple:
    """
    Generate a safe sort key for model context entries.

    Args:
        entry: Model context entry dict

    Returns:
        Tuple of (reaction, rule, reaction_rate) for sorting
    """
    reaction = str(entry.get("reaction", "")) if entry.get("reaction") is not None else ""
    rule = str(entry.get("rule", "")) if entry.get("rule") is not None else ""
    reaction_rate = (
        str(entry.get("reaction_rate", "")) if entry.get("reaction_rate") is not None else ""
    )
    return (reaction, rule, reaction_rate)


def compute_definition_hash(definition: Dict[str, Any], definition_type: str = "parameter") -> str:
    """
    Compute stable content hash for a parameter or species definition.

    Hash reflects structural/mechanistic changes only:
    - For parameters: units, model_context, canonical_scale, tags
    - For species: units, compartment, model_context, tags

    Excludes from hash (won't trigger re-extraction):
    - name (to allow renaming without hash change)
    - description/notes (editorial improvements shouldn't trigger re-extraction)
    - value (values are estimates, not definitions)
    - created_at, created_by (metadata)

    Args:
        definition: Parameter or species definition dict
        definition_type: "parameter" or "species"

    Returns:
        8-character hex hash string
    """
    # Create a stable representation containing only semantic content
    semantic_content = {}

    if definition_type == "parameter":
        # Include structural fields for parameters
        if "units" in definition:
            semantic_content["units"] = definition["units"]
        if "canonical_scale" in definition:
            semantic_content["canonical_scale"] = definition["canonical_scale"]

    elif definition_type == "species":
        # Include structural fields for species
        if "compartment" in definition:
            semantic_content["compartment"] = definition["compartment"]
        if "units" in definition:
            semantic_content["units"] = definition["units"]

    # Always include model_context and tags for both types
    if "model_context" in definition:
        # Normalize model context for stable hashing
        semantic_content["model_context"] = normalize_model_context(definition["model_context"])

    if "tags" in definition:
        # Sort tags for stable hashing
        semantic_content["tags"] = (
            sorted(definition["tags"])
            if isinstance(definition["tags"], list)
            else definition["tags"]
        )

    # Convert to stable JSON string
    semantic_json = json.dumps(semantic_content, sort_keys=True, separators=(",", ":"))

    # Compute hash
    hash_bytes = hashlib.sha256(semantic_json.encode("utf-8")).digest()

    # Return first HASH_PREFIX_LENGTH characters (32 bits) as hex
    from qsp_hpc.constants import HASH_PREFIX_LENGTH

    return hash_bytes.hex()[:HASH_PREFIX_LENGTH]


def normalize_model_context(model_context):
    """
    Normalize model context entries for stable hashing.

    Ensures consistent ordering and format of model context entries
    so that semantically identical contexts produce the same hash.

    Args:
        model_context: Dict with 'derived_from_context' and 'reactions_and_rules'

    Returns:
        Normalized model context dict
    """
    if not isinstance(model_context, dict):
        raise ValueError(
            "model_context must be a dict with 'derived_from_context' and 'reactions_and_rules'"
        )

    normalized = {}

    # Normalize derived_from_context (only names, not descriptions)
    if "derived_from_context" in model_context:
        derived_from = []
        for item in model_context["derived_from_context"]:
            if isinstance(item, dict):
                derived_from.append(item.get("name", ""))
            else:
                derived_from.append(str(item))
        normalized["derived_from_context"] = sorted(derived_from)

    # Normalize reactions_and_rules
    if "reactions_and_rules" in model_context:
        reactions = []
        for entry in model_context["reactions_and_rules"]:
            normalized_entry = {}

            # Add fields in consistent order (excluding descriptions)
            for field in ["reaction", "reaction_rate", "rule", "rule_type"]:
                if field in entry:
                    normalized_entry[field] = entry[field]

            # For other_parameters and other_species, extract just names (not descriptions)
            if "other_parameters" in entry:
                if isinstance(entry["other_parameters"], list):
                    names = [
                        item.get("name", item) if isinstance(item, dict) else str(item)
                        for item in entry["other_parameters"]
                    ]
                    normalized_entry["other_parameters"] = sorted(names)

            if "other_species" in entry:
                if isinstance(entry["other_species"], list):
                    names = [
                        item.get("name", item) if isinstance(item, dict) else str(item)
                        for item in entry["other_species"]
                    ]
                    normalized_entry["other_species"] = sorted(names)

            reactions.append(normalized_entry)

        # Sort entries using module-level sort key
        reactions.sort(key=_safe_sort_key)
        normalized["reactions_and_rules"] = reactions

    return normalized


def generate_filename(name: str, content_hash: str, extension: str = "yaml") -> str:
    """
    Generate hash-based filename.

    Args:
        name: Parameter or species name
        content_hash: Content hash
        extension: File extension (without dot)

    Returns:
        Filename like "k_cell_clear_xyz789ab.yaml"
    """
    return f"{name}_{content_hash}.{extension}"
