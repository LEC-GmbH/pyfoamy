import subprocess
import warnings

import pytest

from pyfoamy.case import Case
from pyfoamy.fv_constraints import (
    TemplateFvConstraint, CONSTRAINT_KIND, fv_constraint_dict,
)
from pyfoamy.include_entries import find_template, validate_call
from pyfoamy.parsing.grammar import Dictionary, Directive


# --- template-based fvConstraints ------------------------------------------


@pytest.mark.openfoam
def test_find_constraint_template_locates_a_real_template():
    path = find_template(CONSTRAINT_KIND, "limitTemperature")
    assert path.name == "limitTemperature"
    assert path.is_file()


@pytest.mark.openfoam
def test_find_constraint_template_raises_for_unknown_type():
    with pytest.raises(ValueError, match="bogusConstraintType"):
        find_template(CONSTRAINT_KIND, "bogusConstraintType")


@pytest.mark.openfoam
def test_validate_constraint_call_accepts_a_correct_call():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_call(
            CONSTRAINT_KIND, "limitTemperature",
            {"name": "limitT", "min": 200, "max": 6000},
        )


@pytest.mark.openfoam
def test_validate_constraint_call_raises_for_missing_required_arg():
    with pytest.raises(ValueError, match="min"):
        validate_call(
            CONSTRAINT_KIND, "limitTemperature",
            {"name": "limitT", "max": 6000},
        )


@pytest.mark.openfoam
def test_validate_constraint_call_accepts_the_real_phase_keyword():
    # "phase" is read by limitTemperature.C but not in its template;
    # it must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_call(
            CONSTRAINT_KIND, "limitTemperature",
            {"name": "limitT", "min": 200, "max": 6000, "phase": "u"},
        )


@pytest.mark.openfoam
def test_validate_constraint_call_warns_for_unrecognized_kwarg():
    with pytest.warns(UserWarning, match="maxx"):
        validate_call(
            CONSTRAINT_KIND, "limitTemperature",
            {"name": "limitT", "min": 200, "max": 6000, "maxx": 6000},
        )


def test_template_fv_constraint_call_text():
    tfc = TemplateFvConstraint(
        "limitTemperature",
        kwargs={"name": "limitT", "min": 200, "max": 6000},
    )
    assert tfc.call_text() == (
        "limitTemperature(name=limitT, min=200, max=6000)"
    )


def test_template_fv_constraint_renders_raw_directive_by_default():
    tfc = TemplateFvConstraint(
        "limitTemperature",
        kwargs={"name": "limitT", "min": 200, "max": 6000},
    )
    node = tfc.render(expand=False)
    assert isinstance(node, Directive)
    assert node.name == "#includeConstraint"
    reparsed = Dictionary(str(node))
    assert list(reparsed)[0].name == "#includeConstraint"


@pytest.mark.openfoam
def test_template_fv_constraint_renders_resolved_entry_via_expand():
    tfc = TemplateFvConstraint(
        "limitTemperature",
        kwargs={"name": "limitT", "min": 200, "max": 6000},
    )
    entry = tfc.render(expand=True)
    assert entry.key.data == "limitT"
    resolved = entry.value
    assert str(resolved["type"]) == "limitTemperature"
    assert int(resolved["min"]) == 200
    assert int(resolved["max"]) == 6000


def test_fv_constraint_dict_merges_type_and_settings():
    result = fv_constraint_dict(
        "limitTemperature",
        settings={"cellZone": "all", "min": 200, "max": 6000},
    )
    assert result == {
        "type": "limitTemperature", "cellZone": "all",
        "min": 200, "max": 6000,
    }


# --- Case integration ------------------------------------------------------


@pytest.mark.openfoam
def test_case_add_constraint_rejects_duplicate_name_unless_overwrite(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint("limitT", "limitTemperature", min=200, max=6000)
    with pytest.raises(ValueError, match="limitT"):
        case.add_constraint("limitT", "limitTemperature", min=250, max=6000)
    case.add_constraint(
        "limitT", "limitTemperature", min=250, max=6000, overwrite=True,
    )


def test_case_add_constraint_dict_rejects_duplicate_name_unless_overwrite(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint_dict(
        "limitP", "limitPressure", settings={"cellZone": "all", "min": 1e4},
    )
    with pytest.raises(ValueError, match="limitP"):
        case.add_constraint_dict(
            "limitP", "limitPressure",
            settings={"cellZone": "all", "min": 2e4},
        )
    case.add_constraint_dict(
        "limitP", "limitPressure",
        settings={"cellZone": "all", "min": 2e4}, overwrite=True,
    )


@pytest.mark.openfoam
def test_case_write_renders_raw_includeConstraint_by_default(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint("limitT", "limitTemperature", min=200, max=6000)
    case.write()
    fv_constraints = case["system/fvConstraints"]
    assert isinstance(fv_constraints[0], Directive)
    assert fv_constraints[0].name == "#includeConstraint"
    assert "limitT" not in fv_constraints


@pytest.mark.openfoam
def test_case_write_renders_resolved_entry_when_expand_true(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint("limitT", "limitTemperature", min=200, max=6000)
    case.write(expand_constraints=True)
    fv_constraints = case["system/fvConstraints"]
    assert not any(isinstance(n, Directive) for n in fv_constraints)
    assert int(fv_constraints["limitT"]["min"]) == 200


@pytest.mark.openfoam
def test_case_default_expand_constraints_controls_write_without_override(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.default_expand_constraints = True
    case.add_constraint("limitT", "limitTemperature", min=200, max=6000)
    case.write()
    fv_constraints = case["system/fvConstraints"]
    assert int(fv_constraints["limitT"]["min"]) == 200


def test_case_add_constraint_dict_writes_a_plain_entry(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint_dict(
        "limitP", "limitPressure", settings={"cellZone": "all", "min": 1e4},
    )
    case.write()
    fv_constraints = case["system/fvConstraints"]
    assert str(fv_constraints["limitP"]["type"]) == "limitPressure"


@pytest.mark.openfoam
def test_case_write_constraints_round_trips_through_foamDictionary(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint("limitT", "limitTemperature", min=200, max=6000)
    case.add_constraint_dict(
        "limitP", "limitPressure", settings={"cellZone": "all", "min": 1e4},
    )
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-expand", "system/fvConstraints"],
        capture_output=True, text=True, cwd=tmp_path / "myCase",
    )
    assert result.returncode == 0, result.stderr
    assert "limitT" in result.stdout
    assert "limitP" in result.stdout


# --- region-aware ------------------------------------------------------


def _multiregion_case(tmp_path):
    case = Case.create(
        tmp_path / "myCase", start_time=0,
        regions={"fluid": "fluid", "metal": "solid"},
    )
    return Case(tmp_path / "myCase")


def test_add_constraint_requires_region_when_case_has_regions(tmp_path):
    case = _multiregion_case(tmp_path)
    with pytest.raises(ValueError, match="region"):
        case.add_constraint_dict(
            "limitP", "limitPressure", settings={"min": 1e4},
        )


def test_add_constraint_rejects_unknown_region(tmp_path):
    case = _multiregion_case(tmp_path)
    with pytest.raises(KeyError, match="bogusRegion"):
        case.add_constraint_dict(
            "limitP", "limitPressure", settings={"min": 1e4},
            region="bogusRegion",
        )


def test_add_constraint_dict_allows_no_region_without_declared_regions(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_constraint_dict(
        "limitP", "limitPressure", settings={"min": 1e4},
    )


def test_add_constraint_writes_to_the_regions_own_fvConstraints(tmp_path):
    case = _multiregion_case(tmp_path)
    case.add_constraint_dict(
        "limitP", "limitPressure",
        settings={"cellZone": "all", "min": 1e4}, region="fluid",
    )
    case.write()
    fv_constraints = case["system/fluid/fvConstraints"]
    assert float(fv_constraints["limitP"]["min"]) == 1e4


def test_add_constraint_same_name_in_two_regions_does_not_conflict(
    tmp_path,
):
    case = _multiregion_case(tmp_path)
    case.add_constraint_dict(
        "limitP", "limitPressure", settings={"min": 1e4}, region="fluid",
    )
    case.add_constraint_dict(
        "limitP", "limitPressure", settings={"min": 2e4}, region="metal",
    )
    case.write()
    fluid = case["system/fluid/fvConstraints"]
    metal = case["system/metal/fvConstraints"]
    assert float(fluid["limitP"]["min"]) == 1e4
    assert float(metal["limitP"]["min"]) == 2e4


def test_region_add_constraint_sugar_matches_the_explicit_region_kwarg(
    tmp_path,
):
    case = _multiregion_case(tmp_path)
    case.regions["fluid"].add_constraint_dict(
        "limitP", "limitPressure",
        settings={"cellZone": "all", "min": 1e4},
    )
    case.write()
    fv_constraints = case["system/fluid/fvConstraints"]
    assert float(fv_constraints["limitP"]["min"]) == 1e4


@pytest.mark.openfoam
def test_case_write_region_constraints_round_trip_through_foamDictionary(
    tmp_path,
):
    case = _multiregion_case(tmp_path)
    case.add_constraint(
        "limitT", "limitTemperature", region="fluid", min=200, max=6000,
    )
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-expand", "system/fluid/fvConstraints"],
        capture_output=True, text=True, cwd=case.root,
    )
    assert result.returncode == 0, result.stderr
    assert "limitT" in result.stdout
