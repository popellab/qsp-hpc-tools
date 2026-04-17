"""Read already-combined QSP pool results directly from the filesystem.

``QSPResultLoader`` is a lean counterpart to :class:`QSPSimulator` for the
case when the combined test stats + params CSVs already exist at a filesystem
path the current process can read directly (e.g., workflows running on the
HPC that originally generated them, or any host with the pool mounted).

It does not SSH, does not SCP, does not combine chunks, does not submit
derivation jobs. If the combined files are missing or short, it raises
:class:`FileNotFoundError` / :class:`ValueError` so the caller can fall back
to a full :class:`QSPSimulator` invocation.

The hashing identity (pool id + test_stats hash) is computed the same way
as :class:`QSPSimulator` so the two stay in lockstep:

    pool_id         = f"{model_version}_{sha256(priors+script+version)[:8]}_{scenario}"
    test_stats_hash = sha256(test_stats_csv content)

and the expected on-disk layout is::

    {pool_root}/{pool_id}/test_stats/{test_stats_hash}/
        combined_params.csv       (header + N rows)
        combined_test_stats.csv   (no header, N rows)

Usage::

    loader = QSPResultLoader(
        pool_root="/home/me/data/qsp_simulations",
        priors_csv="parameters/priors.csv",
        calibration_targets="calibration_targets/baseline",
        model_version="multi_scenario_v4_all_scenarios",
        model_script="immune_oncology_model_PDAC",
        scenario="baseline_no_treatment",
    )
    params, observables, sample_index = loader.load(n_simulations=20000)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

HASH_PREFIX_LENGTH = 8


class QSPResultLoader:
    """Filesystem-only loader for a QSP scenario's pre-combined pool results."""

    def __init__(
        self,
        pool_root: Union[str, Path],
        priors_csv: Union[str, Path],
        model_version: str,
        model_script: str,
        scenario: str,
        *,
        test_stats_csv: Optional[Union[str, Path]] = None,
        calibration_targets: Optional[Union[str, Path]] = None,
        submodel_priors_yaml: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if test_stats_csv is not None and calibration_targets is not None:
            raise ValueError("Provide test_stats_csv OR calibration_targets, not both")
        if test_stats_csv is None and calibration_targets is None:
            raise ValueError("Must provide test_stats_csv or calibration_targets")

        self.pool_root = Path(pool_root)
        self.priors_csv = Path(priors_csv)
        self.model_version = model_version
        self.model_script = model_script
        self.scenario = scenario
        self.submodel_priors_yaml = (
            Path(submodel_priors_yaml) if submodel_priors_yaml is not None else None
        )
        self.seed = seed

        self._temp_csv: Optional[Path] = None
        self._test_stats_df: Optional[pd.DataFrame] = None
        if calibration_targets is not None:
            # Mirror QSPSimulator's YAML → DataFrame → temp-CSV flow so the
            # hash matches exactly.
            from qsp_hpc.calibration import load_calibration_targets

            self._test_stats_df = load_calibration_targets(Path(calibration_targets))
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            tmp.close()
            self._temp_csv = Path(tmp.name)
            self._test_stats_df.to_csv(self._temp_csv, index=False)
            test_stats_csv = self._temp_csv

        self.test_stats_csv = Path(test_stats_csv)
        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")
        if not self.test_stats_csv.exists():
            raise FileNotFoundError(f"Test stats CSV not found: {self.test_stats_csv}")

    # ------------------------------------------------------------------
    # Identity (delegates to shared helpers, matches SimulationPool)
    # ------------------------------------------------------------------
    def priors_hash(self) -> str:
        from qsp_hpc.utils.hash_utils import compute_pool_id_hash

        return compute_pool_id_hash(
            priors_csv=self.priors_csv,
            model_script=self.model_script,
            model_version=self.model_version,
            submodel_priors_yaml=self.submodel_priors_yaml,
            seed=self.seed,
        )

    def test_stats_hash(self) -> str:
        from qsp_hpc.utils.hash_utils import compute_test_stats_hash

        return compute_test_stats_hash(self.test_stats_csv)

    def pool_id(self) -> str:
        return (
            f"{self.model_version}_"
            f"{self.priors_hash()[:HASH_PREFIX_LENGTH]}_"
            f"{self.scenario}"
        )

    def pool_dir(self) -> Path:
        return self.pool_root / self.pool_id()

    def test_stats_dir(self) -> Path:
        return self.pool_dir() / "test_stats" / self.test_stats_hash()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, n_simulations: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(params, observables, sample_index)`` for ``n_simulations``.

        ``sample_index`` is the global theta-pool position for each row,
        used downstream to align scenarios in multi-scenario inference.
        Combined CSVs predating the sample_index schema fall back to
        ``np.arange(n_simulations)``.

        Raises
        ------
        FileNotFoundError
            If either ``combined_params.csv`` or ``combined_test_stats.csv``
            is missing at the expected path.
        ValueError
            If the files are present but have fewer rows than
            ``n_simulations``.
        """
        ts_dir = self.test_stats_dir()
        params_file = ts_dir / "combined_params.csv"
        stats_file = ts_dir / "combined_test_stats.csv"

        if not params_file.exists():
            raise FileNotFoundError(
                f"combined_params.csv not found at {params_file}. "
                f"Populate the pool via QSPSimulator first, or check pool_root "
                f"and hashes."
            )
        if not stats_file.exists():
            raise FileNotFoundError(f"combined_test_stats.csv not found at {stats_file}.")

        params_df = pd.read_csv(params_file)
        if "sample_index" in params_df.columns:
            sample_index = params_df["sample_index"].astype(np.int64).to_numpy()
            params = params_df.drop(columns=["sample_index"]).to_numpy()
        else:
            sample_index = np.arange(len(params_df), dtype=np.int64)
            params = params_df.to_numpy()

        # combined_test_stats.csv has NO header.
        stats_df = pd.read_csv(stats_file, header=None)
        observables = stats_df.to_numpy()

        if len(params) < n_simulations:
            raise ValueError(f"combined_params.csv has {len(params)} rows; need {n_simulations}")
        if len(observables) < n_simulations:
            raise ValueError(
                f"combined_test_stats.csv has {len(observables)} rows; " f"need {n_simulations}"
            )

        return params[:n_simulations], observables[:n_simulations], sample_index[:n_simulations]

    @staticmethod
    def load_aligned(
        loaders: "list[QSPResultLoader]", n_simulations: int
    ) -> Tuple[np.ndarray, "dict[str, np.ndarray]", np.ndarray]:
        """Load multiple scenarios and align rows by ``sample_index``.

        Returns ``(theta, observables_by_scenario, sample_index)`` where
        ``theta`` is the parameter matrix for rows present in *all*
        scenarios, ``observables_by_scenario[scen]`` is the corresponding
        observable matrix for that scenario, and ``sample_index`` is the
        shared index column (sorted ascending).

        Replaces the old positional join in multi-scenario workflows that
        silently drifted when scenarios' theta orderings diverged.
        """
        per_scen = {}
        for ld in loaders:
            theta, x, idx = ld.load(n_simulations)
            per_scen[ld.scenario] = (theta, x, idx)

        # Intersect indices across scenarios.
        common = None
        for theta, x, idx in per_scen.values():
            s = set(idx.tolist())
            common = s if common is None else (common & s)
        common_arr = np.array(sorted(common), dtype=np.int64)

        # Slice each scenario down to the common indices, in shared order.
        theta_ref = None
        obs_by_scen: "dict[str, np.ndarray]" = {}
        for scen, (theta, x, idx) in per_scen.items():
            order = pd.Series(np.arange(len(idx)), index=idx).reindex(common_arr).to_numpy()
            if np.isnan(order).any():
                raise RuntimeError(f"Index alignment failed for scenario {scen}")
            order = order.astype(np.int64)
            if theta_ref is None:
                theta_ref = theta[order]
            else:
                # Sanity check: theta values must match across scenarios for shared indices.
                if not np.allclose(theta_ref, theta[order], equal_nan=True):
                    raise RuntimeError(
                        f"Theta mismatch between scenarios at shared sample_indices "
                        f"(scenario={scen}). Pools disagree on what theta a given "
                        f"sample_index represents — likely a stale parquet from before "
                        f"the deterministic theta pool was introduced."
                    )
            obs_by_scen[scen] = x[order]

        return theta_ref, obs_by_scen, common_arr

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"QSPResultLoader(pool_id={self.pool_id()!r}, "
            f"test_stats_hash={self.test_stats_hash()[:HASH_PREFIX_LENGTH]}..., "
            f"scenario={self.scenario!r})"
        )
