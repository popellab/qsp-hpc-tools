"""Render per-simulation parameter XMLs for the C++ QSP simulator.

The C++ driver (`qsp_sim`, built in a consumer repo such as SPQSP_PDAC
or pdac-build) reads a Boost-property-tree XML where every numeric leaf
element holds one model parameter or initial value. A priors CSV gives
the subset of those parameters we want to vary; everything else inherits
the value from the template.

Why the mapping is simple
-------------------------
The qsp-codegen package (`qsp-codegen` → QSPParam.cpp → param_all.xml,
merged via `qsp-refresh-param-xml`) assigns each QSP parameter a *leaf
tag that is unique within the
`Param/QSP/...` subtree* — e.g. `k_C1_growth` appears exactly once under
QSP, at `Param/QSP/init_value/Parameter/k_C1_growth`. So a priors-CSV
column name maps directly to the QSP-subtree leaf with the same tag,
with no disambiguation needed. We verify uniqueness at load time within
that subtree; a codegen change that breaks the invariant surfaces
immediately instead of producing silently-wrong sims.

Tags outside the QSP subtree (ABM section in `Param/`, which reuses names
like `lifespanSD` across cell types) are ignored. The renderer writes
the full tree back to disk unmodified outside of the QSP subtree, so the
C++ driver reads the same file it would read from the template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
from xml.etree import ElementTree


class ParamNotFoundError(KeyError):
    """Raised when a parameter name has no matching leaf in the template."""


class DuplicateLeafTagError(RuntimeError):
    """Raised if the template has two leaves with the same tag (invariant
    broken by an upstream codegen change)."""


def _iter_leaves(root: ElementTree.Element):
    """Yield every leaf element (no children) in document order."""
    for elem in root.iter():
        if len(elem) == 0:
            yield elem


# Default subtree for the QSP simulator: matches what QSPParam reads.
# The full tree may include ABM / dosing sections that duplicate tag
# names across cell types and are not relevant to the QSP param sweep.
DEFAULT_SUBTREE = "QSP"


def _format_value(v: float) -> str:
    # repr() gives a round-trippable float literal and Boost property_tree
    # parses it back exactly. Scientific notation is fine — Boost's default
    # extractor uses stream >> double, which handles "1.23e-4" identically.
    return repr(float(v))


class ParamXMLRenderer:
    """Parse a template XML once; render per-sim XMLs cheaply.

    Thread-unsafe by design — `render()` mutates an internal working tree
    to avoid deep-copies on the hot path. Use one renderer per thread, or
    per process, rather than sharing across a thread pool.
    """

    def __init__(
        self,
        template_path: str | Path,
        subtree: str | None = DEFAULT_SUBTREE,
    ):
        """
        Args:
            template_path: Path to the reference XML (e.g. param_all.xml).
            subtree: Direct-child tag of the root element to scope the
                substitutable leaf map to. The full tree is still
                serialized; only leaves inside this subtree are
                substitution targets. Pass None to scope to the whole
                tree (then all leaves must have globally unique tags).
        """
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template XML not found: {self.template_path}")

        self.subtree = subtree

        # Keep the original bytes so render() can reset the working tree
        # without re-reading disk.
        self._template_bytes = self.template_path.read_bytes()
        self._tree = ElementTree.ElementTree(ElementTree.fromstring(self._template_bytes))

        # Build {leaf_tag: element} map over the working tree. Re-built on
        # each render to point at the current tree's elements, not stale
        # references from before a reset.
        self._leaf_map: dict[str, ElementTree.Element] = {}
        self._rebuild_leaf_map()

        # Snapshot names at construction — the set is stable because the
        # template doesn't change between renders.
        self._parameter_names = frozenset(self._leaf_map.keys())

    def _subtree_root(self) -> ElementTree.Element:
        if self.subtree is None:
            return self._tree.getroot()
        node = self._tree.getroot().find(self.subtree)
        if node is None:
            raise ValueError(
                f"Subtree {self.subtree!r} not found as a direct child of "
                f"root element {self._tree.getroot().tag!r} in "
                f"{self.template_path}"
            )
        return node

    def _rebuild_leaf_map(self) -> None:
        self._leaf_map.clear()
        for leaf in _iter_leaves(self._subtree_root()):
            tag = leaf.tag
            if tag in self._leaf_map:
                raise DuplicateLeafTagError(
                    f"Template has multiple leaves with tag {tag!r} "
                    f"inside subtree {self.subtree!r}; priors-column-name "
                    f"→ leaf mapping is ambiguous. This violates an "
                    f"invariant normally enforced by qsp-codegen — "
                    f"investigate the generator (qsp-codegen package)."
                )
            self._leaf_map[tag] = leaf

    @property
    def parameter_names(self) -> frozenset[str]:
        """All substitutable parameter names in the template."""
        return self._parameter_names

    @property
    def template_defaults(self) -> dict[str, float]:
        """Snapshot of every parameter's template value as ``{name: float}``.

        Pulled from the freshly-reset working tree so this is always the
        unmodified template default, not any per-render override. Useful
        for enriching downstream artefacts (e.g. the per-sim Parquet) with
        every model parameter so calibration-target functions can read any
        of them via ``species_dict[name]`` even when the workflow only
        varies a handful via the priors.
        """
        self._reset_tree()
        return {name: float(elem.text) for name, elem in self._leaf_map.items()}

    def _reset_tree(self) -> None:
        self._tree = ElementTree.ElementTree(ElementTree.fromstring(self._template_bytes))
        self._rebuild_leaf_map()

    def render(self, params: Mapping[str, float]) -> bytes:
        """Return serialized XML bytes with `params` substituted in.

        Parameters not in `params` keep the template default. Unknown
        parameter names raise ParamNotFoundError — silently dropping them
        would produce mis-parameterized simulations with no loud signal.
        """
        unknown = set(params) - self._parameter_names
        if unknown:
            raise ParamNotFoundError(
                f"{len(unknown)} parameter(s) not in template: "
                f"{sorted(unknown)[:10]}"
                + (f" ... (+{len(unknown) - 10} more)" if len(unknown) > 10 else "")
            )

        self._reset_tree()
        for name, value in params.items():
            self._leaf_map[name].text = _format_value(value)

        # Writing without XML declaration matches the template's own style
        # (no <?xml ... ?> prolog on param_all.xml), which keeps Boost's
        # property_tree parser happy.
        return ElementTree.tostring(self._tree.getroot(), encoding="utf-8")

    def render_to_file(self, params: Mapping[str, float], out_path: str | Path) -> Path:
        """Render and write to disk; return the written path."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.render(params))
        return out
