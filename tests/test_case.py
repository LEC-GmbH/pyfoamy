import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyfoamy import Case, FixedValue, PatchRole, native
from pyfoamy.custom_lib import CustomLibError
from pyfoamy.parsing.grammar import (
    DimensionSet, Dictionary, KeywordEntry, ParseError, String,
)
from pyfoamy.parsing.regions import Region

FIXTURES = Path(__file__).parent / "fixtures"

_ROLES = {
    PatchRole.WALL: ("wall",),
    PatchRole.MOVING_WALL: ("movingWall",),
    PatchRole.SLIP: ("slipNCC",),
    PatchRole.MOVING_SLIP: ("movingSlipNCC",),
    PatchRole.INLET: ("inlet",),
}


def test_create_makes_expected_directories(tmp_path):
    case = Case.create(tmp_path / "myCase")
    root = tmp_path / "myCase"
    assert (root / "constant").is_dir()
    assert (root / "system").is_dir()
    assert (root / "constant" / "geometry").is_dir()
    assert case.root == root


def test_create_with_start_time_creates_that_directory(tmp_path):
    Case.create(tmp_path / "myCase", start_time=0)
    assert (tmp_path / "myCase" / "0").is_dir()


def test_create_formats_a_non_integer_start_time(tmp_path):
    Case.create(tmp_path / "myCase", start_time=0.5)
    assert (tmp_path / "myCase" / "0.5").is_dir()


def test_create_without_start_time_makes_no_time_directory(tmp_path):
    Case.create(tmp_path / "myCase")
    entries = {p.name for p in (tmp_path / "myCase").iterdir()}
    assert entries == {"constant", "system"}


def test_create_writes_a_minimal_control_dict(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0.5)
    d = case.read_dict("system/controlDict")
    assert str(d["startFrom"]) == "startTime"
    assert float(d["startTime"]) == 0.5
    assert int(d["endTime"]) == 1
    assert int(d["deltaT"]) == 1
    assert str(d["writeControl"]) == "timeStep"
    assert int(d["writeInterval"]) == 1
    assert "regionSolvers" not in d


def test_create_defaults_start_time_to_zero_in_control_dict(tmp_path):
    case = Case.create(tmp_path / "myCase")
    d = case.read_dict("system/controlDict")
    assert int(d["startTime"]) == 0


def test_create_does_not_overwrite_an_existing_control_dict(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict(
        "system/controlDict",
        {
            "startFrom": "startTime",
            "startTime": 0,
            "endTime": 99,
            "deltaT": 1,
            "writeControl": "timeStep",
            "writeInterval": 1,
        },
    )
    # Calling create() again on the same case must not clobber the
    # customization - only fill in a controlDict when none exists yet.
    Case.create(tmp_path / "myCase")
    d = case.read_dict("system/controlDict")
    assert int(d["endTime"]) == 99


def test_create_with_regions_makes_per_region_subdirectories(tmp_path):
    Case.create(
        tmp_path / "myCase",
        start_time=0,
        regions={"fluid": "fluid", "metal": "solid"},
    )
    root = tmp_path / "myCase"
    assert (root / "constant" / "fluid").is_dir()
    assert (root / "constant" / "metal").is_dir()
    assert (root / "system" / "fluid").is_dir()
    assert (root / "system" / "metal").is_dir()
    assert (root / "0" / "fluid").is_dir()
    assert (root / "0" / "metal").is_dir()
    # Geometry stays shared at the case root, not per-region.
    assert (root / "constant" / "geometry").is_dir()
    assert not (root / "constant" / "fluid" / "geometry").exists()


def test_create_with_regions_adds_region_solvers_to_control_dict(
    tmp_path,
):
    case = Case.create(
        tmp_path / "myCase", regions={"fluid": "fluid", "metal": "solid"}
    )
    d = case.read_dict("system/controlDict")
    region_solvers = d["regionSolvers"]
    assert [
        (e.key.data, str(e.value)) for e in region_solvers
        if isinstance(e, KeywordEntry)
    ] == [("fluid", "fluid"), ("metal", "solid")]


def test_create_with_regions_but_no_start_time_makes_no_time_dirs(
    tmp_path,
):
    Case.create(tmp_path / "myCase", regions={"fluid": "fluid"})
    assert not (tmp_path / "myCase" / "0").exists()


@pytest.mark.openfoam
def test_created_single_region_case_is_openable_by_native_time(tmp_path):
    Case.create(tmp_path / "myCase", start_time=0)
    # Time must construct; only the absent mesh ("points"/"polyMesh")
    # may fail.
    with pytest.raises(native.NativeError, match="points|polyMesh"):
        with native.open_case(tmp_path, "myCase"):
            pass


def test_write_dict_and_read_dict_roundtrip(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict(
        "system/controlDict",
        {
            "application": "foamRun",
            "startFrom": "startTime",
            "startTime": 0,
            "endTime": 1,
        },
    )
    d = case.read_dict("system/controlDict")
    assert str(d["application"]) == "foamRun"
    assert int(d["endTime"]) == 1


def test_write_dict_adds_a_foam_file_header_when_missing(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict(
        "constant/physicalProperties",
        {"foo": "bar"},
        class_name="dictionary",
    )
    d = case.read_dict("constant/physicalProperties")
    header = d["FoamFile"]
    assert str(header["class"]) == "dictionary"
    assert str(header["object"]) == "physicalProperties"
    assert header["location"].data == "constant"


def test_write_dict_does_not_duplicate_an_existing_foam_file_header(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    dictionary = Dictionary({
        "FoamFile": {
            "version": 2.0,
            "format": "ascii",
            "class": "polyBoundaryMesh",
            "location": String("constant/polyMesh"),
            "object": "boundary",
        },
        "foo": "bar",
    })
    case.write_dict("constant/polyMesh/boundary", dictionary)
    d = case.read_dict("constant/polyMesh/boundary")
    assert str(d["FoamFile"]["class"]) == "polyBoundaryMesh"
    # write_dict must not prepend a second header.
    foam_file_entries = [
        n for n in d
        if isinstance(n, KeywordEntry) and n.key.data == "FoamFile"
    ]
    assert len(foam_file_entries) == 1


def test_write_dict_does_not_mutate_the_caller_dictionary(tmp_path):
    case = Case.create(tmp_path / "myCase")
    dictionary = Dictionary({"foo": "bar"})
    nodes_before = list(dictionary)
    case.write_dict("system/someDict", dictionary)
    assert list(dictionary) == nodes_before
    assert "FoamFile" not in dictionary


def test_write_dict_rejects_an_unsupported_value_type(tmp_path):
    case = Case.create(tmp_path / "myCase")
    with pytest.raises(TypeError):
        case.write_dict("system/someDict", ["not", "a", "dict"])


def test_write_dict_header_false_writes_no_foam_file_header(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict(
        "constant/foam/species.foam", {"foo": "bar"}, header=False,
    )
    text = (tmp_path / "myCase/constant/foam/species.foam").read_text()
    assert "FoamFile" not in text
    assert text == Dictionary({"foo": "bar"}).as_text + "\n"


def test_read_mesh_at_and_read_field_at_default_to_python_backend():
    # No OpenFOAM env needed - the default backend is pure Python.
    case = Case(FIXTURES / "cavity_ascii")
    mesh = case.read_mesh_at(0.1)
    assert len(mesh.points) == 882
    field = case.read_field_at(0.1, "U")
    assert field.dimensions.length == 1


def test_read_mesh_at_and_read_field_at_with_region_python_backend():
    case = Case(FIXTURES / "multiregion_cylinder")
    fluid_mesh = case.read_mesh_at(0.0, region="fluid")
    film_mesh = case.read_mesh_at(0.0, region="film")
    assert len(fluid_mesh.points) == 156
    assert len(film_mesh.points) == 76
    fluid_t = case.read_field_at(0, "T", region="fluid")
    assert fluid_t.internal.shape == (fluid_mesh.n_cells,)
    assert (fluid_t.internal == 300).all()


def test_decomposed_case_reads_as_the_complete_case():
    complete = Case(FIXTURES / "cavity_binary")
    decomposed = Case(FIXTURES / "cavity_decomposed")
    mesh = decomposed.read_mesh_at(0.1)
    assert np.array_equal(mesh.points, complete.read_mesh_at(0.1).points)
    for name in ("U", "p", "phi"):
        expected = complete.read_field_at(0.1, name)
        field = decomposed.read_field_at(0.1, name)
        assert np.array_equal(field.internal, expected.internal)
        for patch, values in expected.patches.items():
            if values.value is None:
                assert field.patches[patch].value is None
            else:
                assert np.array_equal(field.patches[patch].value, values.value)


def test_decomposed_read_needs_the_python_backend():
    case = Case(FIXTURES / "cavity_decomposed")
    with pytest.raises(ValueError, match="decomposed"):
        case.read_mesh_at(0.1, backend="openfoam")


def test_list_regions_passthrough():
    case = Case(FIXTURES / "multiregion_cylinder")
    assert case.list_regions() == [
        Region("fluid", "multicomponentFluid"),
        Region("film", "film"),
    ]


def test_unknown_backend_raises_value_error():
    case = Case(FIXTURES / "cavity_ascii")
    with pytest.raises(ValueError, match="unknown backend"):
        case.read_mesh_at(0.1, backend="matlab")


@pytest.mark.openfoam
def test_default_backend_can_be_set_at_construction():
    with Case(
        FIXTURES / "multiregion_cylinder", default_backend="openfoam"
    ) as case:
        mesh = case.read_mesh_at(0.0, region="fluid")
    assert len(mesh.points) == 156


@pytest.mark.openfoam
def test_native_backend_matches_python_backend_on_real_fixture():
    with Case(FIXTURES / "multiregion_cylinder") as case:
        python_mesh = case.read_mesh_at(0.0, region="fluid")
        native_mesh = case.read_mesh_at(
            0.0, region="fluid", backend="openfoam"
        )
        python_u = case.read_field_at(0, "U", region="fluid")
        native_u = case.read_field_at(
            0.0, "U", region="fluid", backend="openfoam"
        )
    assert np.array_equal(python_mesh.points, native_mesh.points)
    assert np.array_equal(python_mesh.faces.values, native_mesh.faces.values)
    assert python_mesh.patches == native_mesh.patches
    assert python_mesh.n_cells == native_mesh.n_cells
    assert python_u.dimensions == native_u.dimensions
    assert np.array_equal(python_u.internal, native_u.internal)


@pytest.mark.openfoam
def test_context_manager_closes_all_cached_native_handles():
    with Case(FIXTURES / "multiregion_cylinder") as case:
        case.read_mesh_at(0.0, region="fluid", backend="openfoam")
        handle = case.reader._native_cases["fluid"]
    assert not handle._handle


@pytest.mark.openfoam
def test_repeated_native_reads_on_the_same_region_reuse_one_handle():
    with Case(FIXTURES / "multiregion_cylinder") as case:
        case.read_mesh_at(0.0, region="fluid", backend="openfoam")
        first = case.reader._native_cases["fluid"]
        case.read_field_at(0.0, "U", region="fluid", backend="openfoam")
        second = case.reader._native_cases["fluid"]
        assert first is second


def test_getitem_reads_an_existing_dict_and_write_persists_edits(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    case["system/controlDict"]["endTime"] = 42
    case.write()
    d = case.read_dict("system/controlDict")
    assert int(d["endTime"]) == 42
    # Untouched fields survive - write() rewrites the whole cached
    # dict, not just the mutated key.
    assert str(d["startFrom"]) == "startTime"


def test_getitem_raises_for_an_unregistered_path(tmp_path):
    case = Case.create(tmp_path / "myCase")
    with pytest.raises(KeyError, match="myDict"):
        case["system/myDict"]


def test_getitem_returns_the_same_cached_object_on_repeated_access(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    first = case["system/controlDict"]
    second = case["system/controlDict"]
    assert first is second


def test_getitem_accepts_a_path_like_key_matching_the_string_form(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    by_string = case["system/controlDict"]
    by_path = case[Path("system", "controlDict")]
    assert by_string is by_path


def test_construction_eagerly_registers_the_standard_dicts_as_empty(
    tmp_path,
):
    case = Case(tmp_path / "myCase")
    assert list(case["system/controlDict"]) == []
    assert list(case["system/fvSchemes"]) == []
    assert list(case["system/fvSolution"]) == []


def test_construction_eagerly_loads_existing_standard_dicts_from_disk(
    tmp_path,
):
    Case.create(tmp_path / "myCase")
    case = Case(tmp_path / "myCase")
    assert str(case["system/controlDict"]["startFrom"]) == (
        "startTime"
    )


def test_add_dict_registers_an_empty_dict_by_default(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_dict("system/myDict")
    assert list(d) == []
    assert case["system/myDict"] is d


def test_add_dict_registers_given_content(tmp_path):
    case = Case(tmp_path / "myCase")
    case.add_dict("system/myDict", {"foo": "bar"})
    assert str(case["system/myDict"]["foo"]) == "bar"


def test_add_dict_then_mutate_and_write(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.add_dict("system/myDict")
    case["system/myDict"]["foo"] = "bar"
    case.write()
    d = case.read_dict("system/myDict")
    assert str(d["foo"]) == "bar"


def test_add_dict_rejects_a_path_already_registered(tmp_path):
    case = Case(tmp_path / "myCase")
    case.add_dict("system/myDict")
    with pytest.raises(ValueError, match="myDict"):
        case.add_dict("system/myDict")


def test_add_dict_never_reads_from_disk_even_if_present(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict("system/someDict", {"foo": "bar"})
    other_case = Case(tmp_path / "myCase")
    d = other_case.add_dict("system/someDict")
    assert "foo" not in d


def test_add_dict_header_false_is_written_without_a_header(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.add_dict(
        "constant/foam/species.foam", {"foo": "bar"}, header=False,
    )
    case.write()
    text = (tmp_path / "myCase/constant/foam/species.foam").read_text()
    assert "FoamFile" not in text
    assert text == Dictionary({"foo": "bar"}).as_text + "\n"


def test_add_dict_header_defaults_to_true(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.add_dict("constant/myDict", {"foo": "bar"})
    case.write()
    d = case.read_dict("constant/myDict")
    assert "FoamFile" in d


def test_path_prefix_view_add_dict_is_sugar_for_case_add_dict(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.constant.add_dict("dynamicMeshDict")
    assert case["constant/dynamicMeshDict"] is d


def test_add_field_uses_the_default_template(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("p", time=0)
    assert case["0/p"] is d
    assert str(d["dimensions"]) == "[pressure]"
    assert str(d["internalField"]) == "uniform 0"


def test_add_field_overrides_internal_value(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("k", time=0, internal_value=0.05)
    assert str(d["internalField"]) == "uniform 0.05"


def test_add_field_supports_a_phase_suffixed_path(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("T", time=0, path="T.u")
    assert case["0/T.u"] is d


def test_add_field_unregistered_name_requires_type_and_dimensions(
    tmp_path,
):
    case = Case(tmp_path / "myCase")
    with pytest.raises(ValueError, match="CH4"):
        case.add_field("CH4", time=0)


def test_add_field_unregistered_name_with_explicit_type(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field(
        "CH4", time=0, type="scalar", dimensions="[]",
        internal_value=0.23,
    )
    assert str(d["internalField"]) == "uniform 0.23"


def test_add_field_unregistered_name_gets_a_default_boundary_field(
    tmp_path,
):
    case = Case(tmp_path / "myCase", patch_roles=_ROLES)
    d = case.add_field(
        "CH4", time=0, type="scalar", dimensions="[]", internal_value=0,
    )
    assert str(d["boundaryField"]["wall"]["type"]) == "zeroGradient"


def test_add_field_registered_name_gets_its_own_default_boundary_field(
    tmp_path,
):
    case = Case(tmp_path / "myCase", patch_roles=_ROLES)
    d = case.add_field("T", time=0)
    assert str(d["boundaryField"]["wall"]["type"]) == "zeroGradient"


def test_fields_exact_name_returns_the_registered_field(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("U", time=0)
    assert case.fields["U"] is d


def test_fields_multiplexes_across_phase_suffixed_variants(tmp_path):
    case = Case(tmp_path / "myCase")
    t_u = case.add_field("T", time=0, path="T.u")
    t_b = case.add_field("T", time=0, path="T.b")
    case.fields["T"].set_bc("inlet1", FixedValue(310))
    assert str(t_u["boundaryField"]["inlet1"]["type"]) == (
        "fixedValue"
    )
    assert str(t_b["boundaryField"]["inlet1"]["type"]) == (
        "fixedValue"
    )


def test_fields_prefers_exact_name_over_phase_variants(tmp_path):
    case = Case(tmp_path / "myCase")
    plain = case.add_field("T", time=0)
    case.add_field("T", time=0, path="T.u")
    assert case.fields["T"] is plain


def test_fields_unknown_name_raises(tmp_path):
    case = Case(tmp_path / "myCase")
    with pytest.raises(KeyError, match="nonexistent"):
        case.fields["nonexistent"]


def test_add_field_rejects_an_unknown_type(tmp_path):
    case = Case(tmp_path / "myCase")
    with pytest.raises(ValueError, match="unknown field type"):
        case.add_field(
            "CH4", time=0, type="quaternion", dimensions="[]"
        )


def test_add_field_unregistered_name_defaults_to_zero(tmp_path):
    case = Case(tmp_path / "myCase")
    s = case.add_field("CH4", time=0, type="scalar", dimensions="[]")
    v = case.add_field("Uc", time=0, type="vector", dimensions="[velocity]")
    assert str(s["internalField"]) == "uniform 0"
    assert str(v["internalField"]) == "uniform (0 0 0)"


def test_add_field_accepts_a_dimension_set(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field(
        "CH4", time=0, type="scalar",
        dimensions=DimensionSet(1, -3, 0, 0, 0, 0, 0),
    )
    assert str(d["dimensions"]) == "[1 -3 0 0 0 0 0]"


def test_add_field_registered_name_with_matching_type(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("U", time=0, type="vector")
    assert str(d["dimensions"]) == "[velocity]"


def test_add_field_registered_name_with_contradicting_type(tmp_path):
    case = Case(tmp_path / "myCase")
    with pytest.raises(ValueError, match="contradicts"):
        case.add_field("U", time=0, type="scalar")


def test_add_field_formats_string_and_float_times(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.add_field("p", time="0.5")
    case.add_field("p", time=2.0)
    case.write()
    assert (tmp_path / "myCase/0.5/p").is_file()
    assert (tmp_path / "myCase/2/p").is_file()
    with pytest.raises(ValueError, match="finite"):
        case.add_field("T", time=float("inf"))


def test_add_field_rejects_a_path_already_registered(tmp_path):
    case = Case(tmp_path / "myCase")
    case.add_field("p", time=0)
    with pytest.raises(ValueError, match="0/p"):
        case.add_field("p", time=0)


@pytest.mark.openfoam
def test_write_uses_the_registered_class_name_for_a_field(tmp_path):
    case = Case(tmp_path / "myCase")
    case.add_field("k", time=0)
    case.write()
    result = subprocess.run(
        [
            "foamDictionary", "-entry", "FoamFile/class", "-value",
            str(tmp_path / "myCase" / "0" / "k"),
        ],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "volScalarField"


def test_setitem_replaces_the_pending_dict_wholesale(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case["system/someDict"] = {"foo": "bar"}
    case.write()
    d = case.read_dict("system/someDict")
    assert str(d["foo"]) == "bar"


def test_setitem_rejects_an_unsupported_value_type(tmp_path):
    case = Case.create(tmp_path / "myCase")
    with pytest.raises(TypeError):
        case["system/someDict"] = ["not", "a", "dict"]


def test_write_only_flushes_dicts_reached_through_getitem(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict("system/otherDict", {"untouched": True})
    before = (tmp_path / "myCase" / "system" / "otherDict").read_text()
    case["system/controlDict"]["endTime"] = 42
    case.write()
    after = (tmp_path / "myCase" / "system" / "otherDict").read_text()
    assert before == after


def test_write_orig_none_writes_the_exact_registered_path(tmp_path):
    case = Case(tmp_path / "myCase")
    case["system/someDict"] = {"foo": "bar"}
    case.write()
    assert (tmp_path / "myCase" / "system" / "someDict").is_file()
    assert not (
        tmp_path / "myCase" / "system" / "someDict.orig"
    ).is_file()


def test_write_orig_true_appends_orig_to_every_file(tmp_path):
    case = Case(tmp_path / "myCase")
    case["system/someDict"] = {"foo": "bar"}
    case.write(orig=True)
    assert (tmp_path / "myCase" / "system" / "controlDict.orig").is_file()
    assert (tmp_path / "myCase" / "system" / "someDict.orig").is_file()
    assert not (tmp_path / "myCase" / "system" / "controlDict").is_file()
    assert not (tmp_path / "myCase" / "system" / "someDict").is_file()


def test_write_orig_names_only_affects_matching_files(tmp_path):
    case = Case(tmp_path / "myCase")
    case["system/someDict"] = {"foo": "bar"}
    case.write(orig={"someDict"})
    assert (tmp_path / "myCase" / "system" / "someDict.orig").is_file()
    assert not (tmp_path / "myCase" / "system" / "someDict").is_file()
    assert (tmp_path / "myCase" / "system" / "controlDict").is_file()
    assert not (
        tmp_path / "myCase" / "system" / "controlDict.orig"
    ).is_file()


def test_write_orig_empty_collection_writes_no_orig_files(tmp_path):
    case = Case(tmp_path / "myCase")
    case.write(orig=set())
    assert (tmp_path / "myCase" / "system" / "controlDict").is_file()
    assert not (
        tmp_path / "myCase" / "system" / "controlDict.orig"
    ).is_file()


def test_system_property_is_sugar_for_the_system_prefixed_path(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    assert case.system["controlDict"] is case["system/controlDict"]


def test_constant_property_is_sugar_for_the_constant_prefixed_path(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    case.constant.add_dict("dynamicMeshDict")
    case.constant["dynamicMeshDict"]["foo"] = "bar"
    assert str(case["constant/dynamicMeshDict"]["foo"]) == "bar"


def test_system_setitem_writes_through_to_the_prefixed_path(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.system["myDict"] = {"foo": "bar"}
    assert str(case["system/myDict"]["foo"]) == "bar"


def test_regions_accessor_scopes_system_and_constant_by_region(
    tmp_path,
):
    case = Case.create(
        tmp_path / "myCase", regions={"fluid": "fluid", "metal": "solid"}
    )
    fluid = case.regions["fluid"]
    fluid.system.add_dict("blockMeshDict")
    fluid.constant.add_dict("physicalProperties")
    fluid.system["blockMeshDict"]["convertToMeters"] = 1
    fluid.constant["physicalProperties"]["foo"] = "bar"
    assert (
        int(case["system/fluid/blockMeshDict"]["convertToMeters"])
        == 1
    )
    assert (
        str(case["constant/fluid/physicalProperties"]["foo"])
        == "bar"
    )


def test_regions_accessor_rejects_an_unknown_region_name(tmp_path):
    case = Case.create(tmp_path / "myCase", regions={"fluid": "fluid"})
    with pytest.raises(KeyError, match="fluid"):
        case.regions["metal"]


def test_regions_accessor_rejects_any_region_on_a_single_region_case(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase")
    with pytest.raises(KeyError):
        case.regions["fluid"]


def test_regions_accessor_validates_freshly_on_every_access(tmp_path):
    case = Case.create(tmp_path / "myCase", regions={"fluid": "fluid"})
    case.regions["fluid"]
    case["system/controlDict"]["regionSolvers"] = {"metal": "solid"}
    case.write()
    with pytest.raises(KeyError):
        case.regions["fluid"]
    case.regions["metal"]


def test_regions_accessor_sees_unwritten_region_solvers(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case["system/controlDict"]["regionSolvers"] = {"fluid": "fluid"}
    assert case.list_regions() == [Region("fluid", "fluid")]
    case.regions["fluid"].system.add_dict("blockMeshDict")


def test_case_with_an_unparseable_standard_dict_still_opens(tmp_path):
    case = Case.create(tmp_path / "myCase")
    schemes = tmp_path / "myCase/system/fvSchemes"
    schemes.write_text("// keep me\nddtSchemes { default 0x1F; }\n")
    case = Case(tmp_path / "myCase")
    case["system/controlDict"]["endTime"] = 2
    with pytest.raises(ParseError):
        case["system/fvSchemes"]
    case.write()
    assert schemes.read_text().startswith("// keep me")
    assert int(case.read_dict("system/controlDict")["endTime"]) == 2


def test_write_orig_copies_an_unaccessed_standard_dict_verbatim(tmp_path):
    Case.create(tmp_path / "myCase")
    control = tmp_path / "myCase/system/controlDict"
    control.write_text(control.read_text() + "// trailing comment\n")
    Case(tmp_path / "myCase").write(orig=True)
    orig = tmp_path / "myCase/system/controlDict.orig"
    assert orig.read_text() == control.read_text()


def test_add_libs_adds_a_quoted_entry(tmp_path):
    case = Case(tmp_path / "myCase")
    case.add_libs("libpyfoamy.so")
    control_dict = case["system/controlDict"]
    assert [v.data for v in control_dict["libs"]] == ["libpyfoamy.so"]
    rendered = str(control_dict["libs"])
    assert rendered == '("libpyfoamy.so")'


def test_add_libs_merges_with_an_existing_entry(tmp_path):
    case = Case(tmp_path / "myCase")
    case["system/controlDict"]["libs"] = ["libexisting.so"]
    case.add_libs("libpyfoamy.so")
    assert [v.data for v in case["system/controlDict"]["libs"]] == [
        "libexisting.so", "libpyfoamy.so",
    ]


def test_add_libs_does_not_duplicate_an_already_present_name(tmp_path):
    case = Case(tmp_path / "myCase")
    case.add_libs("libpyfoamy.so")
    case.add_libs("libpyfoamy.so")
    assert [v.data for v in case["system/controlDict"]["libs"]] == [
        "libpyfoamy.so",
    ]


def test_require_pyfoamy_lib_raises_for_unsupported_version(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    case = Case(tmp_path / "myCase")
    with pytest.raises(CustomLibError, match="13"):
        case.require_pyfoamy_lib()
    # The build failed before any mutation - add_libs() is never
    # reached, matching the project's fail-fast convention.
    assert "libs" not in case["system/controlDict"]


@pytest.mark.slow
@pytest.mark.openfoam
def test_require_pyfoamy_lib_builds_and_adds_the_entry(tmp_path):
    case = Case(tmp_path / "myCase")
    path = case.require_pyfoamy_lib()
    assert path.is_file()
    assert [v.data for v in case["system/controlDict"]["libs"]] == [
        "libpyfoamy.so",
    ]


def test_add_field_without_patch_roles_sets_no_patch_condition(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("T", time=0)
    assert [type(node).__name__ for node in d["boundaryField"]] == [
        "Directive",
    ]


def test_set_bc_keeps_the_condition_class(tmp_path):
    case = Case(tmp_path / "myCase")
    d = case.add_field("T", time=0)
    d.set_bc("inlet", FixedValue(300))
    assert isinstance(d["boundaryField"]["inlet"], FixedValue)
