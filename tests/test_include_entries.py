
import pytest

from pyfoamy.fv_constraints import CONSTRAINT_KIND
from pyfoamy.fv_models import MODEL_KIND
from pyfoamy.function_objects import FUNCTION_KIND
from pyfoamy.include_entries import _known_keys, _required_args, find_template


@pytest.mark.openfoam
def test_required_args_excludes_fields_regardless_of_placeholder_name():
    # "fields (<fieldNames>);" is filled by a generic mechanism, so
    # "fields" is not required despite its placeholder.
    path = find_template(FUNCTION_KIND, "probes")
    assert _required_args(path) == {"points"}


@pytest.mark.openfoam
def test_required_args_uses_the_full_nested_path_not_the_hint_text():
    # The required argument is "porosityForce/cellZone" (as in the
    # FOAM FATAL IO ERROR), not the hint "cellZoneName".
    path = find_template(MODEL_KIND, "porousZone")
    assert _required_args(path) == {"porosityForce/cellZone"}


@pytest.mark.openfoam
def test_known_keys_includes_both_the_path_and_the_hint_text():
    path = find_template(MODEL_KIND, "porousZone")
    known = _known_keys(path)
    assert "porosityForce/cellZone" in known
    assert "cellZoneName" in known


@pytest.mark.openfoam
def test_known_keys_follows_the_includeEtc_chain():
    path = find_template(FUNCTION_KIND, "probes")
    known = _known_keys(path)
    # Keys other than "points"/"fields" come from #includeEtc'd
    # probes.cfg, so the include chain is followed.
    assert known >= {
        "points", "fields", "type", "libs",
        "writeControl", "writeInterval", "probeLocations",
    }


@pytest.mark.openfoam
def test_known_keys_follows_multi_level_includeEtc_chains():
    # forceCoeffsCompressible -> forceCoeffs.cfg -> forces.cfg: 2 levels.
    path = find_template(FUNCTION_KIND, "forceCoeffsCompressible")
    known = _known_keys(path)
    assert "magUInf" in known  # from the main template
    assert "type" in known  # from forceCoeffs.cfg
    assert "writeControl" in known  # from forces.cfg (2 levels deep)


def test_find_template_prefers_a_case_relative_file(tmp_path):
    # readConfigFile checks $FOAM_CASE/<case_relative_dir>/<name> before
    # etc/, e.g. "#includeModel heatSource" -> constant/heatSource.
    (tmp_path / "constant").mkdir()
    local = tmp_path / "constant" / "myLocalModel"
    local.write_text("type myLocalModel;\ncellZone all;\nQ 1000;\n")
    found = find_template(MODEL_KIND, "myLocalModel", case_root=tmp_path)
    assert found == local


@pytest.mark.openfoam
def test_find_template_falls_back_to_etc_when_no_case_relative_file():
    found = find_template(MODEL_KIND, "porousZone", case_root="/nonexistent")
    assert found.is_file()


@pytest.mark.openfoam
def test_find_template_raises_for_unknown_type():
    with pytest.raises(ValueError, match="bogusType"):
        find_template(CONSTRAINT_KIND, "bogusType")
