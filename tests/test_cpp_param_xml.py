"""Unit tests for qsp_hpc.cpp.param_xml.ParamXMLRenderer."""

from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree

import pytest

from qsp_hpc.cpp.param_xml import (
    DuplicateLeafTagError,
    ParamNotFoundError,
    ParamXMLRenderer,
)

# A minimal template that mirrors the real param_all.xml shape: a Param
# root with a QSP subtree (substitutable) and an ABM subtree where tags
# may repeat across cell types. Kept local so the suite doesn't depend
# on SPQSP_PDAC being checked out alongside this repo.
MINI_TEMPLATE = b"""<Param>
  <QSP>
    <init_value>
      <Compartment>
        <V_C>5.0</V_C>
        <V_P>60.0</V_P>
      </Compartment>
      <Parameter>
        <k_C1_growth>0.002</k_C1_growth>
        <k_C1_death>1e-4</k_C1_death>
      </Parameter>
    </init_value>
  </QSP>
  <ABM>
    <CellA><lifespanSD>1.0</lifespanSD></CellA>
    <CellB><lifespanSD>2.0</lifespanSD></CellB>
  </ABM>
</Param>
"""


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    p = tmp_path / "template.xml"
    p.write_bytes(MINI_TEMPLATE)
    return p


def _get_leaf_text(xml_bytes: bytes, tag: str) -> str:
    root = ElementTree.fromstring(xml_bytes)
    for elem in root.iter(tag):
        return elem.text or ""
    raise AssertionError(f"tag {tag!r} not found in rendered XML")


def test_parameter_names_exposed(template_path: Path):
    r = ParamXMLRenderer(template_path)
    assert r.parameter_names == frozenset({"V_C", "V_P", "k_C1_growth", "k_C1_death"})


def test_render_substitutes_named_leaves(template_path: Path):
    r = ParamXMLRenderer(template_path)
    out = r.render({"k_C1_growth": 0.005, "V_C": 7.5})
    assert float(_get_leaf_text(out, "k_C1_growth")) == 0.005
    assert float(_get_leaf_text(out, "V_C")) == 7.5


def test_unspecified_params_keep_template_defaults(template_path: Path):
    r = ParamXMLRenderer(template_path)
    out = r.render({"V_C": 7.5})
    # V_P and the Parameter leaves inherit template values.
    assert float(_get_leaf_text(out, "V_P")) == 60.0
    assert float(_get_leaf_text(out, "k_C1_growth")) == 0.002
    assert float(_get_leaf_text(out, "k_C1_death")) == 1e-4


def test_unknown_param_raises(template_path: Path):
    r = ParamXMLRenderer(template_path)
    with pytest.raises(ParamNotFoundError) as excinfo:
        r.render({"k_DOES_NOT_EXIST": 1.0})
    assert "k_DOES_NOT_EXIST" in str(excinfo.value)


def test_unknown_param_error_lists_up_to_ten(template_path: Path):
    r = ParamXMLRenderer(template_path)
    bad = {f"bad_{i}": 1.0 for i in range(15)}
    with pytest.raises(ParamNotFoundError) as excinfo:
        r.render(bad)
    msg = str(excinfo.value)
    # Truncation marker ensures we don't spam the user with 1000 names.
    assert "+5 more" in msg


def test_render_is_idempotent_across_calls(template_path: Path):
    r = ParamXMLRenderer(template_path)
    a = r.render({"V_C": 1.0})
    b = r.render({"V_C": 1.0})
    assert a == b
    # Successive renders with different keys must not leak state across calls
    # — i.e. the renderer must reset the working tree between renders.
    c = r.render({"V_P": 99.0})
    assert float(_get_leaf_text(c, "V_C")) == 5.0  # V_C must go back to template


def test_render_to_file_writes_bytes(template_path: Path, tmp_path: Path):
    r = ParamXMLRenderer(template_path)
    out = tmp_path / "sub" / "rendered.xml"
    written = r.render_to_file({"V_C": 3.0}, out)
    assert written == out
    assert out.exists()
    assert float(_get_leaf_text(out.read_bytes(), "V_C")) == 3.0


def test_missing_template_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ParamXMLRenderer(tmp_path / "does_not_exist.xml")


def test_duplicate_leaf_tag_detected(tmp_path: Path):
    # Synthesize an XML that violates the unique-tag invariant *inside*
    # the QSP subtree — this is the case the renderer must reject.
    bad = tmp_path / "bad.xml"
    bad.write_bytes(b"<Param><QSP><A><X>1</X></A><B><X>2</X></B></QSP></Param>")
    with pytest.raises(DuplicateLeafTagError) as excinfo:
        ParamXMLRenderer(bad)
    assert "'X'" in str(excinfo.value)


def test_duplicates_outside_subtree_ignored(template_path: Path):
    # The ABM subtree in MINI_TEMPLATE has two <lifespanSD> leaves. Those
    # must not raise, nor appear in parameter_names — they belong to a
    # different subsystem (the ABM) than the QSP sim reads.
    r = ParamXMLRenderer(template_path)  # subtree defaults to "QSP"
    assert "lifespanSD" not in r.parameter_names


def test_subtree_none_scopes_to_whole_tree(template_path: Path):
    # With subtree=None and duplicates present anywhere, construction raises.
    with pytest.raises(DuplicateLeafTagError):
        ParamXMLRenderer(template_path, subtree=None)


def test_missing_subtree_raises(template_path: Path):
    with pytest.raises(ValueError, match="Subtree 'nope' not found"):
        ParamXMLRenderer(template_path, subtree="nope")


def test_scientific_notation_round_trips(template_path: Path):
    r = ParamXMLRenderer(template_path)
    out = r.render({"k_C1_death": 1.234e-15})
    # Boost property_tree parses via stream >> double, which reads repr()
    # output exactly. We verify Python's own parse is also exact here.
    assert float(_get_leaf_text(out, "k_C1_death")) == 1.234e-15


# --- Integration with real SPQSP_PDAC template (opt-in) ---------------------
# These run only if SPQSP_PDAC is checked out alongside this repo (or the
# path is given via env var). They catch real template-structure drift.


def _real_template_path() -> Path | None:
    env = os.environ.get("SPQSP_PDAC_ROOT")
    if env:
        candidate = Path(env) / "PDAC" / "sim" / "resource" / "param_all.xml"
        return candidate if candidate.exists() else None
    # Try sibling directories of this repo.
    here = Path(__file__).resolve().parent.parent
    for sibling in ("SPQSP_PDAC", "SPQSP_PDAC-cpp-sweep"):
        candidate = here.parent / sibling / "PDAC" / "sim" / "resource" / "param_all.xml"
        if candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(
    _real_template_path() is None,
    reason="SPQSP_PDAC/PDAC/sim/resource/param_all.xml not found",
)
def test_real_template_loads_and_has_expected_params():
    path = _real_template_path()
    assert path is not None  # mypy
    r = ParamXMLRenderer(path)
    # Spot-check a handful of names that appear in the codegen's
    # _xml_paths[] list — if any are missing, codegen/template drift.
    expected = {"V_C", "V_P", "V_T", "k_C1_growth", "k_C1_death", "C_max"}
    missing = expected - r.parameter_names
    assert not missing, f"missing expected params in real template: {missing}"
    # The real template has ~576 substitutable leaves per QSPParam.cpp.
    assert len(r.parameter_names) > 500
