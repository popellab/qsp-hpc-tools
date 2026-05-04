"""Invoke the C++ `qsp_sim` binary for one parameter set and parse the result.

This is the lowest layer of the C++ backend: given a parameter dict,
render an XML, spawn `qsp_sim`, parse the raw-binary trajectory. Batch
parallelism and caching live one layer up (M4 onwards).
"""

from __future__ import annotations

import struct
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from qsp_hpc.cpp.param_xml import ParamXMLRenderer

# Must match qsp_sim.cpp's binary format. Header layout:
#   magic | version=2 | n_times | n_species | n_compartments | n_rules | dt | t_end
# Body: n_times × (n_species + n_compartments + n_rules) doubles, in
# row-major order with columns laid out as species → compartments → rules.
_MAGIC = 0x51535042  # "QSPB" little-endian
_HEADER_STRUCT = struct.Struct("<IIQQQQdd")
_HEADER_SIZE = _HEADER_STRUCT.size  # 56 bytes
_SUPPORTED_VERSION = 2


class QspSimError(RuntimeError):
    """`qsp_sim` exited nonzero or produced unreadable output."""


class BinaryFormatError(RuntimeError):
    """Raw-binary trajectory file doesn't match the expected format."""


@dataclass
class TrajectoryHeader:
    """Parsed binary header — knows the column layout for the body."""

    version: int
    n_times: int
    n_species: int
    n_compartments: int
    n_rules: int
    dt_days: float
    t_end_days: float

    @property
    def n_columns(self) -> int:
        return self.n_species + self.n_compartments + self.n_rules


@dataclass
class SimResult:
    """One simulation's output.

    ``trajectory`` columns are laid out as
    ``[species..., compartments..., rules...]`` (matching the binary
    body order).
    """

    trajectory: np.ndarray  # shape (n_times, n_columns), float64
    species_names: list[str]
    compartment_names: list[str]
    rule_names: list[str]
    dt_days: float
    t_end_days: float

    @property
    def time_days(self) -> np.ndarray:
        """Time axis reconstructed from dt and n_times (t=0 is first row)."""
        return np.arange(self.trajectory.shape[0]) * self.dt_days

    @property
    def column_names(self) -> list[str]:
        """All column names in trajectory order."""
        return list(self.species_names) + list(self.compartment_names) + list(self.rule_names)


def read_binary_trajectory(path: Path) -> tuple[np.ndarray, TrajectoryHeader]:
    """Parse a qsp_sim --binary-out file (format version 2).

    Returns ``(trajectory, header)``. Trajectory shape is
    ``(n_times, n_columns)`` with columns laid out as
    ``[species..., compartments..., rules...]``.
    """
    data = path.read_bytes()
    if len(data) < _HEADER_SIZE:
        raise BinaryFormatError(
            f"Binary file truncated: {len(data)} bytes < {_HEADER_SIZE}-byte header"
        )
    magic, version, n_t, n_sp, n_comp, n_rules, dt, t_end = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != _MAGIC:
        raise BinaryFormatError(
            f"Bad magic: got 0x{magic:08x}, expected 0x{_MAGIC:08x} "
            f"(is this a qsp_sim --binary-out file?)"
        )
    if version != _SUPPORTED_VERSION:
        raise BinaryFormatError(
            f"Unsupported binary version {version}; this code handles "
            f"{_SUPPORTED_VERSION} (rebuild qsp_sim if you have an older binary)"
        )
    n_cols = n_sp + n_comp + n_rules
    expected = _HEADER_SIZE + n_t * n_cols * 8
    if len(data) != expected:
        raise BinaryFormatError(
            f"Binary file size mismatch: got {len(data)} bytes, "
            f"expected {expected} ({n_t} times × {n_cols} cols × 8 + {_HEADER_SIZE})"
        )
    arr = np.frombuffer(data, dtype="<f8", offset=_HEADER_SIZE).reshape(n_t, n_cols)
    header = TrajectoryHeader(
        version=version,
        n_times=n_t,
        n_species=n_sp,
        n_compartments=n_comp,
        n_rules=n_rules,
        dt_days=float(dt),
        t_end_days=float(t_end),
    )
    return arr, header


class CppRunner:
    """Renders parameter XMLs, invokes qsp_sim, parses raw-binary output.

    Thread-unsafe (inherits ParamXMLRenderer's non-thread-safety). For
    parallel batches, create one runner per worker process.
    """

    def __init__(
        self,
        binary_path: str | Path,
        template_path: str | Path,
        subtree: str | None = "QSP",
        default_timeout_s: float = 120.0,
        scenario_yaml: str | Path | None = None,
        drug_metadata_yaml: str | Path | None = None,
        healthy_state_yaml: str | Path | None = None,
    ):
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.exists():
            raise FileNotFoundError(f"qsp_sim binary not found: {self.binary_path}")
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"Not a file: {self.binary_path}")

        # Batch-constant simulation config. These flags don't vary per-sim,
        # so we resolve + validate them once here and append to every
        # `qsp_sim` invocation in run_one.
        def _resolve(p: str | Path | None) -> Path | None:
            if p is None:
                return None
            rp = Path(p).resolve()
            if not rp.exists():
                raise FileNotFoundError(f"YAML not found: {rp}")
            return rp

        self.scenario_yaml = _resolve(scenario_yaml)
        self.drug_metadata_yaml = _resolve(drug_metadata_yaml)
        self.healthy_state_yaml = _resolve(healthy_state_yaml)
        if self.scenario_yaml is not None and self.drug_metadata_yaml is None:
            raise ValueError("scenario_yaml requires drug_metadata_yaml")

        self._renderer = ParamXMLRenderer(template_path, subtree=subtree)
        self.default_timeout_s = default_timeout_s
        self._species_names: list[str] | None = None
        self._compartment_names: list[str] | None = None
        self._rule_names: list[str] | None = None

    @property
    def parameter_names(self) -> frozenset[str]:
        """All parameters the template allows overriding."""
        return self._renderer.parameter_names

    @property
    def template_defaults(self) -> dict[str, float]:
        """Every model parameter's template value as ``{name: float}``.

        Forwarded from the underlying ParamXMLRenderer; see
        :meth:`ParamXMLRenderer.template_defaults` for use cases.
        """
        return self._renderer.template_defaults

    @property
    def species_names(self) -> list[str] | None:
        """Species column names from the last qsp_sim invocation (None
        until the first run_one call)."""
        return self._species_names

    @property
    def compartment_names(self) -> list[str] | None:
        """Compartment column names from the last v2 qsp_sim invocation
        (None until the first run_one call; empty list for v1 binaries)."""
        return self._compartment_names

    @property
    def rule_names(self) -> list[str] | None:
        """Assignment-rule column names from the last v2 qsp_sim
        invocation (None until the first run_one call; empty list for
        v1 binaries)."""
        return self._rule_names

    def run_one(
        self,
        params: Mapping[str, float],
        t_end_days: float,
        dt_days: float,
        workdir: str | Path,
        timeout_s: float | None = None,
        keep_files: bool = False,
        evolve_state_path: str | Path | None = None,
        params_hash: str | None = None,
        evolve_trajectory_path: str | Path | None = None,
        evolve_trajectory_dt_days: float | None = None,
    ) -> SimResult:
        """Run one simulation end-to-end.

        Args:
            params: Subset of `self.parameter_names` to override in the
                XML template. Unknown names raise ParamNotFoundError.
            t_end_days: Simulation end time (days).
            dt_days: Output step (days). qsp_sim writes one row every dt.
            workdir: Directory for the rendered XML and binary output.
                Created if missing. Files named with a short UUID so
                concurrent calls into the same directory don't collide.
            timeout_s: Override for default_timeout_s.
            keep_files: If True, leave XML + binary on disk after parsing
                (useful for debugging). If False, delete on success;
                always keep on failure. Note: ``evolve_trajectory_path``
                output is preserved regardless — the caller owns that
                file and decides when to clean it up.
            evolve_state_path: If provided, pass ``--initial-state`` to
                qsp_sim instead of ``--evolve-to-diagnosis`` — skips the
                ~857-day healthy-state integration by loading a previously
                dumped ODE state. Mutually exclusive with any runner-level
                ``healthy_state_yaml`` for this call: the cached state
                supersedes it. Typically supplied by
                :class:`CppEvolveCache`.
            params_hash: Optional SHA-256 hex of the rendered param-XML
                bytes, passed to qsp_sim as ``--params-hash`` for
                cache-consistency checks. Required when
                ``evolve_state_path`` is set (without it, qsp_sim cannot
                verify the cached state matches the current theta).
            evolve_trajectory_path: If provided, pass
                ``--evolve-trajectory-out`` to qsp_sim — dumps a binary
                v2 trajectory of the burn-in phase (healthy → diagnosis)
                to this path. Same magic / column layout as the
                post-scenario binary, so :func:`read_binary_trajectory`
                handles both. Requires that the runner be configured
                with ``healthy_state_yaml`` (the upstream
                ``--evolve-to-diagnosis`` flag); silently no-ops when
                ``evolve_state_path`` is set since burn-in is skipped
                in cached-state mode.
            evolve_trajectory_dt_days: Sampling interval (model-time
                days) for the burn-in dump. ``None`` lets qsp_sim use
                the evolve spec's ``step_days``.

        Returns:
            SimResult with trajectory, species_names, dt, t_end.
        """
        if evolve_state_path is not None and params_hash is None:
            raise ValueError(
                "run_one: params_hash is required when evolve_state_path "
                "is set — it guards against theta/cache drift"
            )
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        sim_id = uuid.uuid4().hex[:12]
        xml_path = work / f"{sim_id}.xml"
        bin_path = work / f"{sim_id}.bin"
        species_path = work / f"{sim_id}.species.txt"
        compartments_path = work / f"{sim_id}.comps.txt"
        rules_path = work / f"{sim_id}.rules.txt"

        self._renderer.render_to_file(params, xml_path)

        cmd = [
            str(self.binary_path),
            "--param",
            str(xml_path),
            "--binary-out",
            str(bin_path),
            "--species-out",
            str(species_path),
            "--compartments-out",
            str(compartments_path),
            "--rules-out",
            str(rules_path),
            "--t-end-days",
            repr(float(t_end_days)),
            "--dt-days",
            repr(float(dt_days)),
        ]
        if self.scenario_yaml is not None:
            cmd += [
                "--scenario",
                str(self.scenario_yaml),
                "--drug-metadata",
                str(self.drug_metadata_yaml),
            ]
        if evolve_state_path is not None:
            # qsp_sim's QSTH header stores a 32-char params_hash. Truncate
            # here so callers can pass the full 64-char SHA-256 digest
            # (which is what the cache keeps for filenames / diagnostics)
            # without having to know the on-wire length.
            from qsp_hpc.cpp.evolve_cache import wire_hash

            cmd += [
                "--initial-state",
                str(evolve_state_path),
                "--params-hash",
                wire_hash(params_hash),
            ]
        elif self.healthy_state_yaml is not None:
            cmd += ["--evolve-to-diagnosis", str(self.healthy_state_yaml)]
            if evolve_trajectory_path is not None:
                cmd += ["--evolve-trajectory-out", str(evolve_trajectory_path)]
                if evolve_trajectory_dt_days is not None:
                    cmd += [
                        "--evolve-trajectory-dt-days",
                        repr(float(evolve_trajectory_dt_days)),
                    ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s if timeout_s is not None else self.default_timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            self._stash_failure(work, sim_id, xml_path, reason="timeout")
            raise QspSimError(
                f"qsp_sim timed out after {e.timeout}s for sim {sim_id}; "
                f"XML preserved at {work / 'failed' / f'{sim_id}.xml'}"
            ) from e

        if proc.returncode != 0:
            stash = self._stash_failure(work, sim_id, xml_path, reason="nonzero-exit")
            raise QspSimError(
                f"qsp_sim exited {proc.returncode} for sim {sim_id}.\n"
                f"  XML: {stash}\n"
                f"  stderr:\n{_indent(proc.stderr)}"
            )

        try:
            traj, header = read_binary_trajectory(bin_path)
        except BinaryFormatError as e:
            stash = self._stash_failure(work, sim_id, xml_path, reason="bad-binary")
            raise QspSimError(
                f"qsp_sim produced unreadable binary for sim {sim_id}.\n"
                f"  XML: {stash}\n"
                f"  error: {e}"
            ) from e

        species = species_path.read_text().splitlines()
        if len(species) != header.n_species:
            raise QspSimError(
                f"species-out line count ({len(species)}) != binary header "
                f"n_species ({header.n_species}) for sim {sim_id}"
            )

        if not compartments_path.exists() or not rules_path.exists():
            raise QspSimError(
                f"--compartments-out / --rules-out files missing for sim "
                f"{sim_id} (rebuild qsp_sim if it was built before binary v2)"
            )
        compartments = compartments_path.read_text().splitlines()
        rules = rules_path.read_text().splitlines()
        if len(compartments) != header.n_compartments:
            raise QspSimError(
                f"compartments-out line count ({len(compartments)}) != "
                f"header n_compartments ({header.n_compartments}) for "
                f"sim {sim_id}"
            )
        if len(rules) != header.n_rules:
            raise QspSimError(
                f"rules-out line count ({len(rules)}) != header n_rules "
                f"({header.n_rules}) for sim {sim_id}"
            )

        if traj.shape[1] != header.n_columns:
            raise QspSimError(
                f"trajectory width ({traj.shape[1]}) != header n_columns "
                f"({header.n_columns}) for sim {sim_id}"
            )
        self._species_names = species
        self._compartment_names = compartments
        self._rule_names = rules

        if not keep_files:
            xml_path.unlink(missing_ok=True)
            bin_path.unlink(missing_ok=True)
            species_path.unlink(missing_ok=True)
            compartments_path.unlink(missing_ok=True)
            rules_path.unlink(missing_ok=True)

        return SimResult(
            trajectory=traj,
            species_names=species,
            compartment_names=compartments,
            rule_names=rules,
            dt_days=header.dt_days,
            t_end_days=header.t_end_days,
        )

    def dump_evolve_state(
        self,
        params: Mapping[str, float],
        params_hash: str,
        state_out: str | Path,
        workdir: str | Path,
        *,
        timeout_s: float | None = None,
    ) -> Path:
        """Run qsp_sim's evolve-to-diagnosis phase only, serializing the
        post-evolve ODE state to ``state_out`` (QSTH binary blob).

        This is the cache-population path used by
        :class:`CppEvolveCache`. No trajectory output is produced — qsp_sim
        exits 0 as soon as it has written the blob. ``params_hash`` is
        stamped into the blob header so the corresponding load
        (``--initial-state``) can verify it against the current theta.

        Returns the resolved ``state_out`` path on success.
        """
        if self.healthy_state_yaml is None:
            raise ValueError(
                "dump_evolve_state requires runner.healthy_state_yaml — "
                "the dump is the result of evolving from that YAML's "
                "healthy IC"
            )
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        out = Path(state_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        sim_id = uuid.uuid4().hex[:12]
        xml_path = work / f"{sim_id}.dump.xml"
        self._renderer.render_to_file(params, xml_path)

        cmd = [
            str(self.binary_path),
            "--param",
            str(xml_path),
            "--evolve-to-diagnosis",
            str(self.healthy_state_yaml),
            "--dump-state",
            str(out),
            "--params-hash",
            params_hash,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s if timeout_s is not None else self.default_timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            self._stash_failure(work, sim_id, xml_path, reason="dump-timeout")
            raise QspSimError(
                f"qsp_sim --dump-state timed out after {e.timeout}s for "
                f"sim {sim_id}; XML preserved at "
                f"{work / 'failed' / f'{sim_id}.dump-timeout.xml'}"
            ) from e

        if proc.returncode != 0:
            stash = self._stash_failure(work, sim_id, xml_path, reason="dump-nonzero-exit")
            raise QspSimError(
                f"qsp_sim --dump-state exited {proc.returncode} for sim "
                f"{sim_id}.\n  XML: {stash}\n"
                f"  stderr:\n{_indent(proc.stderr)}"
            )
        if not out.exists():
            stash = self._stash_failure(work, sim_id, xml_path, reason="dump-missing-output")
            raise QspSimError(
                f"qsp_sim --dump-state exited 0 but {out} was not written."
                f"\n  XML: {stash}\n  stderr:\n{_indent(proc.stderr)}"
            )
        xml_path.unlink(missing_ok=True)
        return out

    @staticmethod
    def _stash_failure(work: Path, sim_id: str, xml_path: Path, reason: str) -> Path:
        """Move the XML into workdir/failed/ so it survives the exception
        for debugging. Returns the new path."""
        failed_dir = work / "failed"
        failed_dir.mkdir(exist_ok=True)
        dest = failed_dir / f"{sim_id}.{reason}.xml"
        if xml_path.exists():
            xml_path.replace(dest)
        return dest


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
