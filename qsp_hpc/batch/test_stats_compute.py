"""Shared test-statistics computation primitives.

Two pure functions extracted from ``derive_test_stats_worker`` so that
both the standalone derive job and other callers (in-process derivation
in ``cpp_simulator``/``qsp_simulator``, and an upcoming inline-derive
path in the C++ batch worker) can share the same hot-path code without
pulling in the CLI/file-I/O scaffolding around them:

  - ``build_test_stat_registry``: compile per-test-stat Python functions
    from the ``model_output_code`` column of a test-stats CSV.
  - ``compute_test_statistics_batch``: evaluate those functions across
    a batch of simulations, returning an ``(n_sims, n_test_stats)``
    matrix.

``derive_test_stats_worker`` re-exports both names, so existing imports
and ``mock.patch`` targets pointing at that module continue to work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qsp_hpc.utils.logging_config import setup_logger

logger = setup_logger(__name__, verbose=True)


def build_test_stat_registry(test_stats_df: pd.DataFrame) -> dict:
    """
    Build test statistic function registry from CSV model_output_code column.

    Each row in the CSV should have:
    - test_statistic_id: Unique identifier
    - model_output_code: Python function code as string

    The function code should define a function named 'compute_test_statistic' with signature:
        def compute_test_statistic(time: np.ndarray, species_dict: dict) -> float

    Where:
        - time: numpy array of time points (days)
        - species_dict: maps species names (e.g., 'V_T.CD8') to raw floats or
          numpy arrays in their canonical model_structure.json units

    Args:
        test_stats_df: DataFrame with test statistics configuration

    Returns:
        Dictionary mapping test_statistic_id -> compiled function
    """
    registry = {}

    # Check for model_output_code column
    if "model_output_code" not in test_stats_df.columns:
        raise ValueError(
            "Test statistics CSV missing required 'model_output_code' column. "
            "This column should contain Python function code to compute test statistics."
        )

    function_col = "model_output_code"

    for _, row in test_stats_df.iterrows():
        test_stat_id = row["test_statistic_id"]

        # Check if function is provided
        if pd.isna(row[function_col]):
            raise ValueError(
                f"Test statistic '{test_stat_id}' has empty {function_col}. "
                "All test statistics must define a Python function."
            )

        function_code = row[function_col]

        try:
            namespace = {"np": np, "numpy": np}

            # Compile and execute the function code in an isolated namespace.
            # Security note: function_code comes from user-authored calibration target
            # definitions (YAML/CSV) and is trusted project input, not external user input.
            exec(function_code, namespace)

            # Extract the 'compute_test_statistic' function
            if "compute_test_statistic" not in namespace:
                raise ValueError(
                    f"Test statistic '{test_stat_id}': {function_col} must define "
                    "a function named 'compute_test_statistic'"
                )

            registry[test_stat_id] = namespace["compute_test_statistic"]
            logger.debug(f"Loaded function for '{test_stat_id}'")

        except Exception as e:
            logger.error(f"Failed to compile function for '{test_stat_id}': {e}")
            logger.error(f"Function code:\n{function_code}")
            raise

    logger.info(f"Built registry with {len(registry)} test statistic functions")
    return registry


def compute_test_statistics_batch(
    sim_df: pd.DataFrame,
    test_stats_df: pd.DataFrame,
    test_stat_registry: dict,
    species_units: dict,
    template_defaults: dict[str, float] | None = None,
    aux_by_sample_index: dict[int, dict[str, float]] | None = None,
    auxiliary_units: dict[str, str] | None = None,
) -> np.ndarray:
    """
    Compute test statistics for a batch of simulations.

    Args:
        sim_df: DataFrame with full simulation data (from Parquet)
                Columns: simulation_id, status, time, species_1, species_2, ...
        test_stats_df: DataFrame with test statistics configuration
                       Columns: test_statistic_id, required_species, model_output_code
        test_stat_registry: Dict mapping test_statistic_id -> compiled function
                           Functions have signature: compute_test_statistic(time, species_dict)
        species_units: Dict mapping species names to unit strings (e.g., {'V_T.CD8': 'cell'}).
            Retained for documentation / future reintroduction; no longer used
            to wrap values, since species_dict carries raw floats in canonical
            model_structure.json units.
        template_defaults: Optional ``{name: default}`` map loaded from the
            pool's ``pool_manifest.json``. When a calibration-target
            function's ``required_species`` lists a parameter that
            isn't in ``sim_df`` (thin-parquet pools post-#23), we fall
            back to ``template_defaults[name]`` as a scalar. ``None``
            preserves the pre-#23 behavior: every parameter has to be
            a parquet column or raise.

    Returns:
        test_stats_matrix: Array of shape (n_sims, n_test_stats)
    """
    template_defaults = template_defaults or {}
    n_sims = len(sim_df)
    n_test_stats = len(test_stats_df)

    test_stats_matrix = np.full((n_sims, n_test_stats), np.nan, dtype=float)

    logger.info(f"Computing {n_test_stats} test statistics for {n_sims} simulations...")

    # ── Plan phase (once per batch) ──────────────────────────────────────────
    #
    # Pintless derive: species_dict carries raw floats / numpy arrays in their
    # canonical model_structure.json units. Observable code is responsible for
    # any inline numerical conversions (mirroring SubmodelTarget forward_model
    # convention). The plan phase still hoists registry / template / column
    # lookups out of the inner loop for the 110k-sim hot path.

    def aux_by_sample_index_first_keys(m):
        """Union of aux names across all sample_indices in ``m``.

        Aux names should be the same across all sample_indices in a
        well-formed sidecar, but unioning is robust to partial fills.
        """
        names: set[str] = set()
        for rec in (m or {}).values():
            names.update(rec.keys())
        return names

    # Auxiliary parameter sidecar: per-sim aux draws keyed by sample_index.
    # ``auxiliary_units`` is retained on the API surface but no longer used
    # for wrapping — observable code knows the declared units per aux name.
    aux_by_sample_index = aux_by_sample_index or {}
    auxiliary_units = auxiliary_units or {}
    sample_index_col = (
        sim_df["sample_index"].to_numpy() if "sample_index" in sim_df.columns else None
    )

    # Per-test-stat metadata: (col_j, tsid, func, required_species, missing_required)
    # ``func is None`` → registry miss (logged once); ``missing_required`` is
    # the subset of required species that can't be resolved from sim_df or
    # template_defaults for this batch. Both cases leave the output column
    # as NaN without entering the hot path.
    tests_meta: list[tuple[int, str, object, list[str], list[str]]] = []
    all_required: set[str] = set()
    for j, row in test_stats_df.iterrows():
        tsid = row["test_statistic_id"]
        required = [s.strip() for s in row["required_species"].split(",")]
        func = test_stat_registry.get(tsid)
        if func is None:
            logger.warning(
                f"Test statistic '{tsid}' not found in registry. "
                "Skipping (function may have failed to compile)."
            )
        tests_meta.append((j, tsid, func, required, []))
        all_required.update(required)

    # Per-species resolution plan. Strategies (Pintless: values are raw
    # floats / numpy arrays, no Pint Quantities):
    #   ('series',)             — time-series column (or scalar compartment)
    #   ('param', col)          — param:<name> column (always scalar per sim)
    #   ('template', float)     — pre-resolved template_defaults scalar reused
    #                             across every sim in the batch
    #   ('aux',)                — populated per-sim from aux_by_sample_index
    #   ('missing',)            — unresolvable; every test stat that requires
    #                             this species fails fast with NaN
    species_plan: dict[str, tuple] = {}
    sim_cols = set(sim_df.columns)
    aux_names_set = set(aux_by_sample_index_first_keys(aux_by_sample_index))
    for s in all_required:
        if s in sim_cols:
            species_plan[s] = ("series",)
        elif f"param:{s}" in sim_cols:
            species_plan[s] = ("param", f"param:{s}")
        elif s in template_defaults:
            species_plan[s] = ("template", float(template_defaults[s]))
        elif s in aux_names_set:
            # Aux names are populated per-sim from the aux samples sidecar
            # (see the inner loop). Mark them with a noop strategy so the
            # 'missing' guard below doesn't short-circuit aux-bearing test
            # stats.
            species_plan[s] = ("aux",)
        else:
            species_plan[s] = ("missing",)

    # Back-fill each test stat's list of unresolvable species so the inner
    # loop can short-circuit without catching ValueError.
    for meta_idx, (j, tsid, func, required, _) in enumerate(tests_meta):
        missing = [s for s in required if species_plan[s][0] == "missing"]
        tests_meta[meta_idx] = (j, tsid, func, required, missing)

    # Pre-materialize column arrays we index per sim. `.to_numpy()` on
    # list-typed columns yields an object array; indexing into it is
    # ~10x cheaper than `sim_df.iloc[i][col]` or `sim_row[col]`.
    series_cols_np = {
        s: sim_df[s].to_numpy() for s, plan in species_plan.items() if plan[0] == "series"
    }
    param_cols_np = {
        s: sim_df[plan[1]].to_numpy() for s, plan in species_plan.items() if plan[0] == "param"
    }
    time_col_np = sim_df["time"].to_numpy() if "time" in sim_cols else None
    status_np = sim_df["status"].to_numpy() if "status" in sim_cols else np.zeros(n_sims, dtype=int)

    # ── Execute phase (sim outer, test-stat inner) ──────────────────────────
    #
    # Per sim we build ``time`` and ``species_dict`` once and hand them to
    # every test stat. With Pint removed the per-sim work is just float /
    # array materialization; observable code does any inline conversions.

    for i in range(n_sims):
        if status_np[i] != 0:
            # status==0 = success; anything else = qsp_sim/MATLAB failure,
            # leave the whole row NaN.
            continue

        try:
            if time_col_np is not None:
                time_arr = np.asarray(time_col_np[i], dtype=float)
            else:
                time_arr = None  # pragma: no cover — guarded by upstream
        except Exception as e:
            logger.warning(f"Error extracting time for simulation {i}: {e}")
            continue

        # Build species_dict once per sim — the union over all test stats'
        # required species. Handing extra entries to a test stat is free.
        species_dict: dict[str, object] = {}
        for s, plan in species_plan.items():
            kind = plan[0]
            if kind == "series":
                val = series_cols_np[s][i]
                # Compartment columns (e.g. V_T) and non-time-series species
                # arrive as Python/numpy scalars; time-series species arrive
                # as list-of-floats.
                if isinstance(val, (int, float, np.integer, np.floating)):
                    species_dict[s] = float(val)
                else:
                    species_dict[s] = np.asarray(val, dtype=float)
            elif kind == "param":
                species_dict[s] = float(param_cols_np[s][i])
            elif kind == "template":
                species_dict[s] = plan[1]
            # 'missing' / 'aux' → not populated here; aux is filled below.

        # Attach auxiliary parameter draws (one record per sample_index) as
        # raw floats. The wrapper relocates them into _constants under the
        # YAML-declared units; observable code is authored knowing those
        # units. When the parquet carries no sample_index — e.g. legacy
        # single-batch test pools — aux merging is silently skipped.
        if aux_by_sample_index and sample_index_col is not None:
            sid = int(sample_index_col[i])
            aux_record = aux_by_sample_index.get(sid)
            if aux_record is not None:
                for aux_name, aux_value in aux_record.items():
                    species_dict[aux_name] = float(aux_value)

        for j, tsid, func, required, missing in tests_meta:
            if func is None:
                continue
            if missing:
                logger.warning(
                    f"Error computing {tsid} for simulation {i}: "
                    f"Species {missing} not found in simulation data or "
                    "pool manifest template_defaults"
                )
                continue
            try:
                result = func(time_arr, species_dict)
                test_stats_matrix[i, j] = float(result)
            except Exception as e:
                logger.warning(f"Error computing {tsid} for simulation {i}: {e}")
                # test_stats_matrix[i, j] already NaN from np.full

    n_computed: int = int(np.sum(~np.isnan(test_stats_matrix)))
    n_total = test_stats_matrix.size
    logger.info(
        f"Computed {n_computed}/{n_total} test statistic values ({100*n_computed/n_total:.1f}%)"
    )

    return test_stats_matrix  # type: ignore[no-any-return]


def derive_chunk_to_csv(
    chunk_parquet: Path,
    output_dir: Path,
    chunk_idx: int,
    test_stats_df: pd.DataFrame,
    test_stat_registry: dict,
    species_units: dict,
    template_defaults: dict[str, float] | None = None,
    aux_by_sample_index: dict[int, dict[str, float]] | None = None,
    auxiliary_units: dict[str, str] | None = None,
) -> int:
    """Derive test stats for a single chunk parquet, write CSV shards.

    Mirrors ``derive_test_stats_worker.process_single_batch`` but for one
    chunk parquet — the unit produced by a single SLURM array task. Used
    by the inline-derive path in ``cpp_batch_worker`` and reusable from
    the cold-path worker.

    Streams the parquet row-group-by-row-group to keep peak memory at
    one row group (list-typed time-series columns blow up if loaded with
    ``pd.read_parquet`` on wide scenarios — see the comment in
    ``process_single_batch``).

    Writes:
      - ``{output_dir}/chunk_{chunk_idx:03d}_test_stats.csv`` (no header,
        (n_sims, n_test_stats) matrix)
      - ``{output_dir}/chunk_{chunk_idx:03d}_params.csv`` (header,
        per-sim parameter values plus optional ``sample_index``)
        — only written when the parquet carries ``param:*`` columns.

    Returns the number of sims processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    params_output_file = output_dir / f"chunk_{chunk_idx:03d}_params.csv"
    test_stats_output_file = output_dir / f"chunk_{chunk_idx:03d}_test_stats.csv"

    pf = pq.ParquetFile(str(chunk_parquet))
    n_row_groups = pf.num_row_groups
    schema_names = pf.schema_arrow.names
    total_sims = pf.metadata.num_rows
    logger.info(
        f"Inline derive on {chunk_parquet.name}: "
        f"{total_sims} sims across {n_row_groups} row group(s)"
    )

    param_prefix = "param:"
    param_cols = [col for col in schema_names if col.startswith(param_prefix)]
    clean_names = [col[len(param_prefix) :] for col in param_cols]
    has_sample_index = "sample_index" in schema_names

    params_f = None
    params_header_written = False
    try:
        if param_cols:
            params_f = open(params_output_file, "w")
        with open(test_stats_output_file, "w") as ts_f:
            for rg_idx in range(n_row_groups):
                sim_df = pf.read_row_group(rg_idx).to_pandas().reset_index(drop=True)

                if params_f is not None:
                    rg_params_df = sim_df[param_cols].copy()
                    rg_params_df.columns = clean_names
                    if has_sample_index:
                        rg_params_df.insert(
                            0,
                            "sample_index",
                            sim_df["sample_index"].astype("int64").values,
                        )
                    rg_params_df.to_csv(
                        params_f,
                        index=False,
                        header=not params_header_written,
                        float_format="%.12e",
                    )
                    params_header_written = True

                test_stats_matrix_rg = compute_test_statistics_batch(
                    sim_df,
                    test_stats_df,
                    test_stat_registry,
                    species_units,
                    template_defaults=template_defaults,
                    aux_by_sample_index=aux_by_sample_index,
                    auxiliary_units=auxiliary_units,
                )
                np.savetxt(ts_f, test_stats_matrix_rg, delimiter=",", fmt="%.12e")
    finally:
        if params_f is not None:
            params_f.close()

    if param_cols:
        logger.debug(f"  ✓ Parameters saved: {params_output_file.name}")
    logger.info(f"  ✓ Saved: {test_stats_output_file.name} ({total_sims} sims)")
    return total_sims
