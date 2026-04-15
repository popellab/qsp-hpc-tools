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

# Must match qsp_sim.cpp's binary format.
_MAGIC = 0x51535042  # "QSPB" little-endian
_SUPPORTED_VERSIONS = {1}
_HEADER_STRUCT = struct.Struct("<IIQQdd")  # magic, version, n_t, n_sp, dt, t_end
_HEADER_SIZE = _HEADER_STRUCT.size  # 40 bytes


class QspSimError(RuntimeError):
    """`qsp_sim` exited nonzero or produced unreadable output."""


class BinaryFormatError(RuntimeError):
    """Raw-binary trajectory file doesn't match the expected format."""


@dataclass
class SimResult:
    """One simulation's output."""

    trajectory: np.ndarray  # shape (n_times, n_species), float64
    species_names: list[str]  # length n_species
    dt_days: float
    t_end_days: float

    @property
    def time_days(self) -> np.ndarray:
        """Time axis reconstructed from dt and n_times (t=0 is first row)."""
        return np.arange(self.trajectory.shape[0]) * self.dt_days


def read_binary_trajectory(path: Path) -> tuple[np.ndarray, float, float]:
    """Parse a qsp_sim --binary-out file.

    Returns (trajectory[n_t, n_sp], dt_days, t_end_days).
    """
    data = path.read_bytes()
    if len(data) < _HEADER_SIZE:
        raise BinaryFormatError(
            f"Binary file truncated: {len(data)} bytes < {_HEADER_SIZE}-byte header"
        )
    magic, version, n_t, n_sp, dt, t_end = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != _MAGIC:
        raise BinaryFormatError(
            f"Bad magic: got 0x{magic:08x}, expected 0x{_MAGIC:08x} "
            f"(is this a qsp_sim --binary-out file?)"
        )
    if version not in _SUPPORTED_VERSIONS:
        raise BinaryFormatError(
            f"Unsupported binary version {version}; this code handles "
            f"{sorted(_SUPPORTED_VERSIONS)}"
        )
    expected = _HEADER_SIZE + n_t * n_sp * 8
    if len(data) != expected:
        raise BinaryFormatError(
            f"Binary file size mismatch: got {len(data)} bytes, "
            f"expected {expected} ({n_t} times × {n_sp} species × 8 + {_HEADER_SIZE})"
        )
    arr = np.frombuffer(data, dtype="<f8", offset=_HEADER_SIZE).reshape(n_t, n_sp)
    return arr, float(dt), float(t_end)


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
    ):
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.exists():
            raise FileNotFoundError(f"qsp_sim binary not found: {self.binary_path}")
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"Not a file: {self.binary_path}")

        self._renderer = ParamXMLRenderer(template_path, subtree=subtree)
        self.default_timeout_s = default_timeout_s
        self._species_names: list[str] | None = None

    @property
    def parameter_names(self) -> frozenset[str]:
        """All parameters the template allows overriding."""
        return self._renderer.parameter_names

    @property
    def species_names(self) -> list[str] | None:
        """Species column names from the last qsp_sim invocation (None
        until the first run_one call)."""
        return self._species_names

    def run_one(
        self,
        params: Mapping[str, float],
        t_end_days: float,
        dt_days: float,
        workdir: str | Path,
        timeout_s: float | None = None,
        keep_files: bool = False,
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
                always keep on failure.

        Returns:
            SimResult with trajectory, species_names, dt, t_end.
        """
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        sim_id = uuid.uuid4().hex[:12]
        xml_path = work / f"{sim_id}.xml"
        bin_path = work / f"{sim_id}.bin"
        species_path = work / f"{sim_id}.species.txt"

        self._renderer.render_to_file(params, xml_path)

        cmd = [
            str(self.binary_path),
            "--param",
            str(xml_path),
            "--binary-out",
            str(bin_path),
            "--species-out",
            str(species_path),
            "--t-end-days",
            repr(float(t_end_days)),
            "--dt-days",
            repr(float(dt_days)),
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
            traj, dt, t_end = read_binary_trajectory(bin_path)
        except BinaryFormatError as e:
            stash = self._stash_failure(work, sim_id, xml_path, reason="bad-binary")
            raise QspSimError(
                f"qsp_sim produced unreadable binary for sim {sim_id}.\n"
                f"  XML: {stash}\n"
                f"  error: {e}"
            ) from e

        species = species_path.read_text().splitlines()
        if len(species) != traj.shape[1]:
            raise QspSimError(
                f"species-out line count ({len(species)}) != trajectory "
                f"column count ({traj.shape[1]}) for sim {sim_id}"
            )
        self._species_names = species

        if not keep_files:
            xml_path.unlink(missing_ok=True)
            bin_path.unlink(missing_ok=True)
            species_path.unlink(missing_ok=True)

        return SimResult(
            trajectory=traj,
            species_names=species,
            dt_days=dt,
            t_end_days=t_end,
        )

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
