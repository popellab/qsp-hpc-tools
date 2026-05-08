"""SimulationBatch — Layer 2 contract returned by HPCSession.run_scenario.

Per ``notes/architecture/local_observable_eval_plan.md`` Layer 2: a
single dataclass shape that both training (run_scenario kind="training")
and PPC (run_scenario kind="ppc" or simulate_with_parameters) hand back
to the runner. The runner-side observable evaluator
(``evaluate_targets_to_x`` in qsp-inference) consumes this directly.

Long-form is the on-disk and in-memory storage layout; the long->wide
pivot lives inside ``evaluate_targets_to_x`` on a per-target projection
basis (so we never materialize all 252 species in wide form). See plan
Open Q1 + D8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SimulationBatch:
    """Long-form trajectory batch keyed by ``sample_index``.

    Attributes
    ----------
    theta : (n_sims, n_params) ndarray
        Parameter draws. Row order matches ``sample_index``.
    sample_index : (n_sims,) int64 ndarray
        Global theta-pool position. Same ``sample_index`` across scenarios
        with shared (prior, seed) refers to the same draw — that's what
        makes cross-scenario alignment a sample_index intersection (D1).
    traj_df : pd.DataFrame
        Long-form trajectory rows: columns ``sample_index``, ``time``,
        ``species`` (categorical, includes species/compartments/rules),
        ``value``. Per-sim time axes are heterogeneous under D4
        solver-native cadence; the schema absorbs that natively.
    species_units : dict[str, str]
        Canonical unit string per species (informational, D3). Plain
        strings, not Pint registry references.
    param_names : list[str]
        Column names for ``theta``, in priors-CSV order.
    pool_id : str
        ``sha256(binary_bytes | scenario_yaml_content)``. Provenance and
        cache key.
    burn_in_df : pd.DataFrame | None
        Optional pre-diagnosis trajectory rows (PPC plotting only). Same
        long-form schema as ``traj_df``. Default off; opt-in via
        ``run_scenario(return_burn_in=True)``.
    """

    theta: np.ndarray
    sample_index: np.ndarray
    traj_df: pd.DataFrame
    species_units: dict[str, str]
    param_names: list[str]
    pool_id: str
    burn_in_df: Optional[pd.DataFrame] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.theta.ndim != 2:
            raise ValueError(
                f"SimulationBatch.theta must be 2-D (n_sims, n_params); got shape {self.theta.shape}"
            )
        if self.sample_index.ndim != 1:
            raise ValueError(
                f"SimulationBatch.sample_index must be 1-D (n_sims,); got shape {self.sample_index.shape}"
            )
        if len(self.sample_index) != self.theta.shape[0]:
            raise ValueError(
                f"sample_index length {len(self.sample_index)} != theta rows {self.theta.shape[0]}"
            )
        if len(self.param_names) != self.theta.shape[1]:
            raise ValueError(
                f"param_names length {len(self.param_names)} != theta cols {self.theta.shape[1]}"
            )

    @property
    def n_sims(self) -> int:
        return int(self.theta.shape[0])

    # ------------------------------------------------------------------
    # Cross-scenario alignment helper
    # ------------------------------------------------------------------

    def slice_to_indices(self, indices: Sequence[int]) -> "SimulationBatch":
        """Return a new SimulationBatch restricted to the given ``sample_index`` values.

        Used by sbi_runner step 5: the cross-scenario intersection is
        computed in sample_index space (D1), and each scenario's
        SimulationBatch is then projected onto that shared set before
        observable evaluation. Order of ``indices`` determines the output
        row order.

        Indices not present in this batch's ``sample_index`` raise
        ``KeyError`` — silent dropping is a bug surface (a missed sim in
        one scenario shouldn't quietly shift cross-scenario joins).
        """
        wanted = np.asarray(indices, dtype=np.int64)
        # Build a position map sample_index -> row index in this batch.
        my_idx = np.asarray(self.sample_index, dtype=np.int64)
        pos = pd.Series(np.arange(my_idx.size, dtype=np.int64), index=my_idx)
        try:
            row_positions = pos.reindex(wanted).to_numpy()
        except Exception as e:
            raise KeyError(f"sample_index reindex failed: {e}") from e
        if np.any(np.isnan(row_positions.astype(np.float64))):
            missing = wanted[np.isnan(row_positions.astype(np.float64))]
            raise KeyError(
                f"slice_to_indices: {len(missing)} requested sample_index not in batch "
                f"(first 5: {missing[:5].tolist()})"
            )
        row_positions = row_positions.astype(np.int64)

        sliced_theta = self.theta[row_positions]
        sliced_si = my_idx[row_positions]

        # traj_df is sample_index-keyed; filter rows by membership in `wanted`.
        wanted_set = set(int(x) for x in wanted)
        if "sample_index" in self.traj_df.columns:
            mask = self.traj_df["sample_index"].isin(wanted_set)
            sliced_traj = self.traj_df.loc[mask].copy()
        else:
            sliced_traj = self.traj_df.iloc[0:0].copy()

        sliced_burn = None
        if self.burn_in_df is not None and "sample_index" in self.burn_in_df.columns:
            mask = self.burn_in_df["sample_index"].isin(wanted_set)
            sliced_burn = self.burn_in_df.loc[mask].copy()

        return SimulationBatch(
            theta=sliced_theta,
            sample_index=sliced_si,
            traj_df=sliced_traj,
            species_units=dict(self.species_units),
            param_names=list(self.param_names),
            pool_id=self.pool_id,
            burn_in_df=sliced_burn,
            metadata=dict(self.metadata),
        )
