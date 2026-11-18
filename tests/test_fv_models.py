import subprocess
import warnings

import pytest

from pyfoamy.case import Case
from pyfoamy.fv_models import (
    TemplateFvModel, MODEL_KIND, fv_model_dict,
)
from pyfoamy.include_entries import find_template, validate_call
from pyfoamy.parsing.grammar import Dictionary, Directive


# --- template-based fvModels ---------------------------------------------


@pytest.mark.openfoam
def test_find_model_template_locates_a_real_template():
    path = find_template(MODEL_KIND, "porousZone")
    assert path.name == "porousZone"
    assert path.is_file()


@pytest.mark.openfoam
def test_find_model_template_raises_for_unknown_type():
    with pytest.raises(ValueError, match="bogusModelType"):
        find_template(MODEL_KIND, "bogusModelType")


@pytest.mark.openfoam
def test_validate_model_call_accepts_a_correct_call():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_call(
            MODEL_KIND, "porousZone",
            {"name": "zone", "porosityForce/cellZone": "myZone"},
        )


@pytest.mark.openfoam
def test_validate_model_call_raises_for_missing_required_arg():
    with pytest.raises(ValueError, match="porosityForce/cellZone"):
        validate_call(MODEL_KIND, "porousZone", {"name": "zone"})


@pytest.mark.openfoam
def test_validate_model_call_accepts_the_real_phase_keyword():
    # "phase" is looked up by several fvModel classes but not in the
    # templates; it must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_call(
            MODEL_KIND, "porousZone",
            {
                "name": "zone", "porosityForce/cellZone": "myZone",
                "phase": "u",
            },
        )


@pytest.mark.openfoam
def test_validate_model_call_warns_for_unrecognized_kwarg():
    with pytest.warns(UserWarning, match="cellZoneNaem"):
        validate_call(
            MODEL_KIND, "porousZone",
            {
                "name": "zone", "porosityForce/cellZone": "myZone",
                "cellZoneNaem": "myZone",
            },
        )


def test_template_fv_model_call_text():
    tfm = TemplateFvModel(
        "porousZone",
        kwargs={
            "name": "zone", "porosityForce/cellZone": "myZone",
        },
    )
    assert tfm.call_text() == (
        "porousZone(name=zone, porosityForce/cellZone=myZone)"
    )


def test_template_fv_model_renders_raw_directive_by_default():
    tfm = TemplateFvModel(
        "porousZone",
        kwargs={"name": "zone", "porosityForce/cellZone": "myZone"},
    )
    node = tfm.render(expand=False)
    assert isinstance(node, Directive)
    assert node.name == "#includeModel"
    reparsed = Dictionary(str(node))
    assert list(reparsed)[0].name == "#includeModel"


@pytest.mark.openfoam
def test_template_fv_model_renders_resolved_entry_via_expand():
    tfm = TemplateFvModel(
        "porousZone",
        kwargs={"name": "zone", "porosityForce/cellZone": "myZone"},
    )
    entry = tfm.render(expand=True)
    assert entry.key.data == "zone"
    resolved = entry.value
    assert str(resolved["type"]) == "porosityForce"
    sub = resolved["porosityForce"]
    assert str(sub["cellZone"]) == "myZone"


def test_fv_model_dict_merges_type_libs_and_settings():
    result = fv_model_dict(
        "heatSource", settings={"cellZone": "heater", "Q": 1000},
    )
    assert result == {
        "type": "heatSource", "cellZone": "heater", "Q": 1000,
    }


def test_fv_model_dict_omits_libs_when_not_given():
    result = fv_model_dict("heatSource", settings={"cellZone": "heater"})
    assert "libs" not in result


# --- Case integration ------------------------------------------------------


@pytest.mark.openfoam
def test_case_add_model_rejects_duplicate_name_unless_overwrite(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model("zone", "porousZone", **{"porosityForce/cellZone": "a"})
    with pytest.raises(ValueError, match="zone"):
        case.add_model(
            "zone", "porousZone", **{"porosityForce/cellZone": "b"},
        )
    case.add_model(
        "zone", "porousZone", overwrite=True,
        **{"porosityForce/cellZone": "b"},
    )


def test_case_add_model_dict_rejects_duplicate_name_unless_overwrite(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model_dict(
        "heater", "heatSource", settings={"cellZone": "heater", "Q": 1000},
    )
    with pytest.raises(ValueError, match="heater"):
        case.add_model_dict(
            "heater", "heatSource",
            settings={"cellZone": "heater", "Q": 2000},
        )
    case.add_model_dict(
        "heater", "heatSource",
        settings={"cellZone": "heater", "Q": 2000}, overwrite=True,
    )


@pytest.mark.openfoam
def test_case_write_renders_raw_includeModel_by_default(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model("zone", "porousZone", **{"porosityForce/cellZone": "a"})
    case.write()
    fv_models = case["constant/fvModels"]
    assert isinstance(fv_models[0], Directive)
    assert fv_models[0].name == "#includeModel"
    assert "zone" not in fv_models


@pytest.mark.openfoam
def test_case_write_renders_resolved_entry_when_expand_true(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model("zone", "porousZone", **{"porosityForce/cellZone": "a"})
    case.write(expand_models=True)
    fv_models = case["constant/fvModels"]
    assert not any(isinstance(n, Directive) for n in fv_models)
    assert str(fv_models["zone"]["type"]) == "porosityForce"


@pytest.mark.openfoam
def test_case_default_expand_models_controls_write_without_override(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.default_expand_models = True
    case.add_model("zone", "porousZone", **{"porosityForce/cellZone": "a"})
    case.write()
    fv_models = case["constant/fvModels"]
    assert str(fv_models["zone"]["type"]) == "porosityForce"


def test_case_add_model_dict_writes_a_plain_entry(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model_dict(
        "heater", "heatSource", settings={"cellZone": "heater", "Q": 1000},
    )
    case.write()
    fv_models = case["constant/fvModels"]
    assert str(fv_models["heater"]["type"]) == "heatSource"
    assert int(fv_models["heater"]["Q"]) == 1000


@pytest.mark.openfoam
def test_case_add_model_prefers_a_case_relative_template(tmp_path):
    # As in the roomHeating tutorial: "#includeModel heatSource" ->
    # constant/heatSource, no etc/ template.
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    (case.root / "constant" / "heatSource").write_text(
        "type heatSource;\ncellZone heater;\nQ 1000;\n"
    )
    case.add_model("heater", "heatSource")
    case.write(expand_models=True)
    fv_models = case["constant/fvModels"]
    assert int(fv_models["heater"]["Q"]) == 1000


@pytest.mark.openfoam
def test_case_write_models_round_trips_through_foamDictionary(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model("zone", "porousZone", **{"porosityForce/cellZone": "a"})
    case.add_model_dict(
        "heater", "heatSource", settings={"cellZone": "heater", "Q": 1000},
    )
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-expand", "constant/fvModels"],
        capture_output=True, text=True, cwd=tmp_path / "myCase",
    )
    assert result.returncode == 0, result.stderr
    assert "zone" in result.stdout
    assert "heater" in result.stdout


# --- region-aware ------------------------------------------------------


def _multiregion_case(tmp_path):
    case = Case.create(
        tmp_path / "myCase", start_time=0,
        regions={"fluid": "fluid", "metal": "solid"},
    )
    return Case(tmp_path / "myCase")


def test_add_model_requires_region_when_case_has_regions(tmp_path):
    case = _multiregion_case(tmp_path)
    with pytest.raises(ValueError, match="region"):
        case.add_model_dict("heater", "heatSource", settings={"Q": 1000})


def test_add_model_rejects_unknown_region(tmp_path):
    case = _multiregion_case(tmp_path)
    with pytest.raises(KeyError, match="bogusRegion"):
        case.add_model_dict(
            "heater", "heatSource", settings={"Q": 1000},
            region="bogusRegion",
        )


def test_add_model_dict_allows_no_region_without_declared_regions(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_model_dict("heater", "heatSource", settings={"Q": 1000})


def test_add_model_writes_to_the_regions_own_fvModels(tmp_path):
    case = _multiregion_case(tmp_path)
    case.add_model_dict(
        "heater", "heatSource", settings={"cellZone": "heater", "Q": 1000},
        region="fluid",
    )
    case.write()
    fv_models = case["constant/fluid/fvModels"]
    assert int(fv_models["heater"]["Q"]) == 1000


def test_add_model_same_name_in_two_regions_does_not_conflict(tmp_path):
    case = _multiregion_case(tmp_path)
    case.add_model_dict(
        "heater", "heatSource", settings={"Q": 1000}, region="fluid",
    )
    case.add_model_dict(
        "heater", "heatSource", settings={"Q": 2000}, region="metal",
    )
    case.write()
    assert int(case["constant/fluid/fvModels"]["heater"]["Q"]) == 1000
    assert int(case["constant/metal/fvModels"]["heater"]["Q"]) == 2000


def test_region_add_model_sugar_matches_the_explicit_region_kwarg(
    tmp_path,
):
    case = _multiregion_case(tmp_path)
    case.regions["fluid"].add_model_dict(
        "heater", "heatSource", settings={"cellZone": "heater", "Q": 1000},
    )
    case.write()
    fv_models = case["constant/fluid/fvModels"]
    assert int(fv_models["heater"]["Q"]) == 1000


def test_region_add_model_rejects_unknown_region(tmp_path):
    case = _multiregion_case(tmp_path)
    with pytest.raises(KeyError, match="bogusRegion"):
        case.regions["bogusRegion"]


@pytest.mark.openfoam
def test_case_write_region_models_round_trip_through_foamDictionary(
    tmp_path,
):
    case = _multiregion_case(tmp_path)
    case.add_model(
        "zone", "porousZone", region="fluid",
        **{"porosityForce/cellZone": "a"},
    )
    case.write()

    result = subprocess.run(
        ["foamDictionary", "-expand", "constant/fluid/fvModels"],
        capture_output=True, text=True, cwd=case.root,
    )
    assert result.returncode == 0, result.stderr
    assert "zone" in result.stdout
