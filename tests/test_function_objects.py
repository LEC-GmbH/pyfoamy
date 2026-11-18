import warnings

import pytest

from pyfoamy.case import Case
from pyfoamy import function_objects as fo
from pyfoamy.enums import SurfaceFieldValueOperation, TimeControlMode
from pyfoamy.function_objects import (
    TemplateFunctionObject, TimeControl, cut_plane, FUNCTION_KIND,
    function_object_dict, iso_surface, periodic_times,
)
from pyfoamy.include_entries import find_template, validate_call
from pyfoamy.parsing.grammar import Dictionary, Directive


def test_time_step_interval_is_optional():
    assert TimeControl("timeStep").to_dict("execute") == {
        "executeControl": "timeStep",
    }
    assert TimeControl("timeStep", interval=10).to_dict("execute") == {
        "executeControl": "timeStep",
        "executeInterval": 10,
    }


def test_write_time_interval_is_optional():
    assert TimeControl("writeTime").to_dict("write") == {
        "writeControl": "writeTime",
    }


def test_adjustable_run_time_requires_interval():
    with pytest.raises(ValueError, match="requires interval"):
        TimeControl("adjustableRunTime")
    assert TimeControl(
        "adjustableRunTime", interval=0.1
    ).to_dict("write") == {
        "writeControl": "adjustableRunTime",
        "writeInterval": 0.1,
    }


@pytest.mark.parametrize("mode", ["runTime", "clockTime", "cpuTime"])
def test_other_interval_required_modes(mode):
    with pytest.raises(ValueError, match="requires interval"):
        TimeControl(mode)


def test_none_mode_rejects_interval():
    with pytest.raises(ValueError, match="doesn't use interval"):
        TimeControl("none", interval=1)


def test_run_times_requires_times_or_frequencies():
    with pytest.raises(ValueError, match="exactly one"):
        TimeControl("runTimes")
    with pytest.raises(ValueError, match="exactly one"):
        TimeControl("runTimes", times=[1, 2], frequencies=[(0, 1)])


def test_run_times_with_times():
    tc = TimeControl("runTimes", times=[1, 2, 3])
    assert tc.to_dict("write") == {
        "writeControl": "runTimes",
        "writeTimes": [1, 2, 3],
    }


def test_run_times_with_frequencies_and_repeat():
    tc = TimeControl(
        "runTimes", frequencies=[(0, 1), (10, 5)], repeat=True
    )
    assert tc.to_dict("write") == {
        "writeControl": "runTimes",
        "writeFrequencies": [[0, 1], [10, 5]],
        "writeRepeat": True,
    }


def test_times_disallowed_outside_run_times():
    with pytest.raises(ValueError, match="doesn't use times"):
        TimeControl("timeStep", times=[1, 2])


def test_repeat_requires_frequencies():
    with pytest.raises(ValueError, match="requires frequencies"):
        TimeControl("runTimes", times=[1, 2], repeat=True)


def test_mode_coerces_from_a_plain_string():
    assert TimeControl("timeStep").mode is TimeControlMode.TIME_STEP


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="TimeControlMode"):
        TimeControl("bogus")


def test_renders_correctly_through_the_real_writer():
    tc = TimeControl("adjustableRunTime", interval=5)
    d = Dictionary(tc.to_dict("write"))
    reparsed = Dictionary(d.as_text)
    assert str(reparsed["writeControl"]) == "adjustableRunTime"
    assert float(reparsed["writeInterval"]) == 5


# --- template-based function objects ------------------------------------


@pytest.mark.openfoam
def test_find_function_template_locates_a_real_template():
    path = find_template(FUNCTION_KIND, "probes")
    assert path.name == "probes"
    assert path.is_file()


@pytest.mark.openfoam
def test_find_function_template_raises_for_unknown_type():
    with pytest.raises(ValueError, match="bogusFunctionObjectType"):
        find_template(FUNCTION_KIND, "bogusFunctionObjectType")


@pytest.mark.openfoam
def test_validate_function_call_accepts_a_correct_call():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_call(
            FUNCTION_KIND, "probes",
            {"name": "probe", "points": [(0, 0, 0)], "fields": ("T", "p")},
        )


@pytest.mark.openfoam
def test_validate_function_call_raises_for_missing_required_arg():
    with pytest.raises(ValueError, match="points"):
        validate_call(FUNCTION_KIND, "probes", {"name": "probe"})


@pytest.mark.openfoam
def test_validate_function_call_warns_for_unrecognized_kwarg():
    with pytest.warns(UserWarning, match="writeIntervall"):
        validate_call(
            FUNCTION_KIND, "probes",
            {
                "name": "probe", "points": [(0, 0, 0)],
                "writeIntervall": 5,
            },
        )


def test_template_function_object_call_text():
    tfo = TemplateFunctionObject(
        "probes",
        kwargs={"name": "probe", "points": [(0, 0, 0)], "fields": ("T",)},
    )
    assert tfo.call_text() == (
        "probes(name=probe, points=((0 0 0)), fields=(T))"
    )


def test_template_function_object_renders_raw_directive_by_default():
    tfo = TemplateFunctionObject(
        "probes", kwargs={"name": "probe", "points": [(0, 0, 0)]},
    )
    node = tfo.render(expand=False)
    assert isinstance(node, Directive)
    assert node.name == "#includeFunc"
    reparsed = Dictionary(str(node))
    assert list(reparsed)[0].name == "#includeFunc"


@pytest.mark.openfoam
def test_template_function_object_renders_resolved_entry_via_expand():
    tfo = TemplateFunctionObject(
        "probes",
        kwargs={
            "name": "probe", "points": [(0, 0, 0)], "fields": ("T", "p"),
        },
    )
    entry = tfo.render(expand=True)
    assert entry.key.data == "probe"
    resolved = entry.value
    assert str(resolved["type"]) == "probes"
    # "2 (T p)": a Compound of count and List.
    fields_list = resolved["fields"][-1]
    assert [str(w) for w in fields_list] == ["T", "p"]


@pytest.mark.openfoam
def test_template_function_object_per_entry_expand_overrides_default():
    tfo = TemplateFunctionObject(
        "probes", kwargs={"name": "probe", "points": [(0, 0, 0)]},
        expand=True,
    )
    # write()'s own expand=False default is overridden by self.expand
    entry = tfo.render(expand=False)
    assert entry.key.data == "probe"
    assert str(entry.value["type"]) == "probes"


def test_function_object_dict_merges_type_libs_settings_and_controls():
    result = function_object_dict(
        "residuals", ["libutilityFunctionObjects.so"],
        settings={"fields": ("p", "U")},
        write_control=TimeControl("writeTime"),
        execute_at_start=False,
    )
    assert result == {
        "type": "residuals",
        "libs": ["libutilityFunctionObjects.so"],
        "fields": ("p", "U"),
        "writeControl": TimeControlMode.WRITE_TIME,
        "executeAtStart": False,
    }


def test_function_object_dict_without_libs_writes_none():
    assert function_object_dict("bXiFields", None, {"fields": ["b"]}) == {
        "type": "bXiFields", "fields": ["b"],
    }


def test_time_window_is_written_without_prefix():
    control = TimeControl("timeStep", interval=2, start=1, begin=0, end=5)
    assert control.to_dict("write") == {
        "writeControl": "timeStep", "writeInterval": 2,
        "startTime": 1, "beginTime": 0, "endTime": 5,
    }


def test_periodic_times_cover_the_end():
    assert periodic_times([20, 10], 100, 250) == [
        10, 20, 110, 120, 210, 220, 310, 320,
    ]
    assert periodic_times([(10, 1), (0, 5)], 100, 150) == [
        (0, 5), (10, 1), (100, 5), (110, 1), (200, 5), (210, 1),
    ]


def test_periodic_times_without_period_or_times():
    assert periodic_times([20, 10], None, 250) == [10, 20]
    assert periodic_times([], 100, 250) == []


# --- sampled sets and typed function objects ------------------------------

@pytest.mark.parametrize("kwargs, type_name", [
    ({"cell": True}, "lineCell"),
    ({"face": True}, "lineFace"),
    ({"cell": True, "face": True}, "lineCellFace"),
    ({"n_points": 5}, "lineUniform"),
])
def test_line_set_type_follows_the_sampling(kwargs, type_name):
    result = fo.line_set((0, 0, 0), (1, 0, 0), axis="x", **kwargs)
    assert result["type"] == type_name
    assert result["axis"] == "x"
    assert ("nPoints" in result) == ("n_points" in kwargs)


def test_line_set_needs_a_sampling():
    with pytest.raises(ValueError, match="n_points, cell or face"):
        fo.line_set((0, 0, 0), (1, 0, 0))


def test_box_uniform_set_writes_both_corners_in_one_entry():
    result = fo.box_uniform_set((0, 0, 0), (1, 1, 1), (2, 3, 4))
    assert str(result["box"]) == "(0 0 0) (1 1 1)"


def test_surfaces_takes_time_control():
    result = fo.surfaces(
        {"mid": cut_plane((0, 0, 0), (0, 0, 1))}, ["p"],
        write_control=TimeControl("writeTime"), execute_at_start=False,
    )
    assert result["type"] == "surfaces"
    assert [str(lib) for lib in result["libs"]] == ['"libsampling.so"']
    assert result["surfaces"]["mid"]["type"] == "cutPlane"
    assert result["writeControl"] == "writeTime"
    assert result["executeAtStart"] is False


def test_wall_field_without_field_writes_no_objects():
    assert "objects" not in fo.wall_field("yPlus")
    result = fo.wall_field("yPlus", write_field=False, phase="u")
    assert result["objects"] == []
    assert result["phase"] == "u"


def test_vol_field_value_rejects_a_surface_operation():
    with pytest.raises(ValueError, match="areaAverage"):
        fo.vol_field_value(["T"], "areaAverage")
    result = fo.vol_field_value(
        ["T"], "volAverage", cell_zone="cylinder", weight_fields=["rho"],
    )
    assert result["cellZone"] == "cylinder"
    assert result["weightFields"] == ["rho"]


def test_surface_field_value_takes_exactly_one_selection():
    with pytest.raises(ValueError, match="exactly one"):
        fo.surface_field_value(["phi"], "sum")
    with pytest.raises(ValueError, match="exactly one"):
        fo.surface_field_value(["phi"], "sum", patch="inlet", face_zone="f")
    result = fo.surface_field_value(["phi"], "sum", patch="inlet")
    assert result["patch"] == "inlet"
    assert "weightFields" not in result


def _all_typed_function_objects():
    seeds = {
        "arc": fo.arc_uniform_set((0, 0, 0), (0, 0, 1), (1, 0, 0), 0, 90, 4),
        "box": fo.box_uniform_set((0, 0, 0), (1, 1, 1), (2, 2, 2)),
        "circle": fo.circle_random_set((0, 0, 0), (0, 0, 1), 0.1, 4),
        "sphere": fo.sphere_random_set((0, 0, 0), 0.1, 4),
        "boundary": fo.boundary_random_set(["inlet"], 4),
        "points": fo.points_set([(0, 0, 0), (1, 0, 0)]),
        "line": fo.line_set((0, 0, 0), (1, 0, 0), n_points=4),
    }
    plane = cut_plane((0, 0, 0), (0, 0, 1))
    result = {
        "surfaces": fo.surfaces(
            {"plane": plane, "walls": fo.patch_surface(["wall"])}, ["p"],
        ),
        "sets": fo.sets(
            {"l": fo.line_set((0, 0, 0), (1, 0, 0), axis="x", cell=True)},
            ["p"],
        ),
        "writeObjects": fo.write_objects(["points"]),
        "writeNow": fo.write_now(
            execute_control=TimeControl("runTimes", times=[1, 2]),
        ),
        "writeInterval": fo.set_write_interval(
            {"type": "table", "values": [(0, 1), (1, 2)]},
        ),
        "timeStep": fo.time_step(),
        "runTime": fo.run_time(),
        "minTimeStep": fo.stop_at_time_step(1e-9),
        "checkMesh": fo.check_mesh(write_surfaces=True),
        "yPlus": fo.wall_field("yPlus", write_field=False),
        "Tf": fo.surface_interpolate("T", "TSurfaceInterpolate"),
        "volume": fo.vol_field_value(["T"], "volAverage"),
        "surface": fo.surface_field_value(
            ["T"], "areaAverage", sampled_surface=plane,
        ),
    }
    for name, seed in seeds.items():
        result[f"streamlines_{name}"] = fo.streamlines(seed, ["U"])
    return result


def test_typed_function_objects_render():
    text = str(Dictionary(_all_typed_function_objects()))
    assert '("libsampling.so")' in text
    assert "writeNow();" in text


@pytest.mark.openfoam
def test_typed_function_objects_are_read_by_foamDictionary(tmp_path):
    import subprocess
    case = Case.create(tmp_path / "myCase", start_time=0)
    case.add_dict("system/functions", _all_typed_function_objects())
    case.write()
    result = subprocess.run(
        ["foamDictionary", "-expand", "-keywords", "system/functions"],
        capture_output=True, text=True, cwd=tmp_path / "myCase",
    )
    assert result.returncode == 0, result.stderr
    assert "streamlines_box" in result.stdout


# --- Case integration -----------------------------------------------------


@pytest.mark.openfoam
def test_case_add_function_rejects_duplicate_name_unless_overwrite(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function("probe", "probes", points=[(0, 0, 0)])
    with pytest.raises(ValueError, match="probe"):
        case.add_function("probe", "probes", points=[(1, 0, 0)])
    case.add_function(
        "probe", "probes", points=[(1, 0, 0)], overwrite=True,
    )


def test_case_add_function_dict_rejects_duplicate_name_unless_overwrite(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function_dict(
        "res", "residuals", ["libutilityFunctionObjects.so"],
    )
    with pytest.raises(ValueError, match="res"):
        case.add_function_dict(
            "res", "residuals", ["libutilityFunctionObjects.so"],
        )
    case.add_function_dict(
        "res", "residuals", ["libutilityFunctionObjects.so"],
        overwrite=True,
    )


@pytest.mark.openfoam
def test_case_write_renders_raw_includeFunc_by_default(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function("probe", "probes", points=[(0, 0, 0)], fields=("T",))
    case.write()
    functions = case["system/functions"]
    assert isinstance(functions[0], Directive)
    assert functions[0].name == "#includeFunc"
    assert "probe" not in functions


@pytest.mark.openfoam
def test_case_write_renders_resolved_entry_when_expand_true(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function("probe", "probes", points=[(0, 0, 0)], fields=("T",))
    case.write(expand_functions=True)
    functions = case["system/functions"]
    assert not any(isinstance(n, Directive) for n in functions)
    assert str(functions["probe"]["type"]) == "probes"


@pytest.mark.openfoam
def test_case_default_expand_functions_controls_write_without_override(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.default_expand_functions = True
    case.add_function("probe", "probes", points=[(0, 0, 0)], fields=("T",))
    case.write()
    functions = case["system/functions"]
    assert str(functions["probe"]["type"]) == "probes"


def test_case_add_function_dict_writes_a_plain_entry(tmp_path):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function_dict(
        "res", "residuals", ["libutilityFunctionObjects.so"],
        settings={"fields": ("p", "U")},
        write_control=TimeControl("writeTime"),
    )
    case.write()
    functions = case["system/functions"]
    assert str(functions["res"]["type"]) == "residuals"
    assert str(functions["res"]["writeControl"]) == "writeTime"


@pytest.mark.openfoam
def test_case_write_functions_round_trips_through_foamDictionary(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function(
        "probe", "probes", points=[(0, 0, 0)], fields=("T", "p"),
    )
    case.add_function_dict(
        "res", "residuals", ["libutilityFunctionObjects.so"],
    )
    case.write()

    import subprocess
    result = subprocess.run(
        ["foamDictionary", "-expand", "system/functions"],
        capture_output=True, text=True, cwd=tmp_path / "myCase",
    )
    assert result.returncode == 0, result.stderr
    assert "probe" in result.stdout
    assert "res" in result.stdout


def test_add_function_dict_needs_no_region_even_with_declared_regions(
    tmp_path,
):
    # system/functions is case-global, never per region.
    case = Case.create(
        tmp_path / "myCase", start_time=0,
        regions={"fluid": "fluid", "metal": "solid"},
    )
    case = Case(tmp_path / "myCase")
    case.add_function_dict(
        "res", "residuals", ["libutilityFunctionObjects.so"],
    )
    case.write()
    functions = case["system/functions"]
    assert str(functions["res"]["type"]) == "residuals"


def test_cut_plane_has_required_fields_only_by_default():
    assert cut_plane((0, 0, 0.05), (0, 0, 1)) == {
        "type": "cutPlane", "planeType": "pointAndNormal",
        "point": (0, 0, 0.05), "normal": (0, 0, 1),
    }


def test_cut_plane_through_three_points():
    points = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert cut_plane(points=points) == {
        "type": "cutPlane", "planeType": "embeddedPoints",
        "point1": (0, 0, 0), "point2": (1, 0, 0), "point3": (0, 1, 0),
    }


@pytest.mark.parametrize("args, kwargs", [
    ((), {}),
    (((0, 0, 0),), {}),
    (((0, 0, 0), (0, 0, 1)), {"points": ((0, 0, 0),) * 3}),
])
def test_cut_plane_takes_exactly_one_form(args, kwargs):
    with pytest.raises(ValueError, match="point and normal, or points"):
        cut_plane(*args, **kwargs)


def test_cut_plane_includes_optional_fields_when_given():
    result = cut_plane(
        (0, 0, 0.05), (0, 0, 1), interpolate=True, cell_zone="core",
    )
    assert result == {
        "type": "cutPlane", "planeType": "pointAndNormal",
        "point": (0, 0, 0.05), "normal": (0, 0, 1),
        "interpolate": True, "cellZone": "core",
    }


def test_iso_surface_has_required_fields_only_by_default():
    assert iso_surface("alpha", 0.5) == {
        "type": "isoSurface", "isoField": "alpha", "isoValue": 0.5,
    }


def test_iso_surface_includes_optional_fields_when_given():
    result = iso_surface("alpha", 0.5, interpolate=True, cell_zone="core")
    assert result == {
        "type": "isoSurface", "isoField": "alpha", "isoValue": 0.5,
        "interpolate": True, "cellZone": "core",
    }


def test_surface_field_value_operation_accepts_area_variants():
    # The operations that only exist for surfaceFieldValue, not
    # volFieldValue (orientedSum/area*/areaNormal*).
    for value in (
        "orientedSum", "areaAverage", "areaIntegrate",
        "areaNormalAverage", "areaNormalIntegrate",
    ):
        assert SurfaceFieldValueOperation.coerce(value) == value


def test_surface_field_value_operation_rejects_unknown_value():
    # "volAverage" is a real operation, just volFieldValue's, not
    # surfaceFieldValue's - the two lists are different.
    with pytest.raises(ValueError, match="unknown"):
        SurfaceFieldValueOperation.coerce("volAverage")


def test_surface_field_value_with_cut_plane_writes_via_add_function_dict(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function_dict(
        "flameArea", "surfaceFieldValue", ["libfieldFunctionObjects.so"],
        settings={
            "sampledSurface": cut_plane((0, 0, 0.05), (0, 0, 1)),
            "operation": SurfaceFieldValueOperation.coerce("areaIntegrate"),
            "fields": ("alpha",),
            "writeFields": False,
        },
    )
    case.write()
    entry = case["system/functions"]["flameArea"]
    assert str(entry["operation"]) == "areaIntegrate"
    assert str(entry["sampledSurface"]["type"]) == "cutPlane"


@pytest.mark.openfoam
def test_surface_field_value_with_cut_plane_round_trips_through_foamDictionary(
    tmp_path,
):
    case = Case.create(tmp_path / "myCase", start_time=0)
    case = Case(tmp_path / "myCase")
    case.add_function_dict(
        "flameArea", "surfaceFieldValue", ["libfieldFunctionObjects.so"],
        settings={
            "sampledSurface": cut_plane((0, 0, 0.05), (0, 0, 1)),
            "operation": SurfaceFieldValueOperation.coerce("areaIntegrate"),
            "fields": ("alpha",),
            "writeFields": False,
        },
    )
    case.write()

    import subprocess
    result = subprocess.run(
        ["foamDictionary", "-expand", "system/functions"],
        capture_output=True, text=True, cwd=tmp_path / "myCase",
    )
    assert result.returncode == 0, result.stderr
    assert "cutPlane" in result.stdout
    assert "areaIntegrate" in result.stdout


def test_field_and_solver_function_objects():
    assert fo.derived_field("MachNo", phase="u", velocity="U") == {
        "type": "MachNo", "libs": list(fo._FIELD), "phase": "u", "U": "U",
    }
    assert fo.derived_field("mag", field="U", result="magU")["result"] == (
        "magU"
    )
    assert fo.divide(["Cp", "Cv"], "gamma")["fields"] == ["Cp", "Cv"]
    assert fo.turbulence_fields(("k", "nut"))["fields"] == ["k", "nut"]
    tracer = fo.scalar_transport("tracer0", schemes_field="Yi")
    assert tracer["schemesField"] == "Yi"
    assert "solverField" not in tracer
    assert tracer["diffusivity"] == "viscosity"
    limit = fo.fluid_max_delta_t(0.5, 1)
    assert limit == {"type": "fluidMaxDeltaT", "maxCo": 0.5, "maxDeltaT": 1}


def test_coded_field_function_objects_render():
    text = Dictionary({
        "a": fo.auto_write_field("Xi"), "b": fo.read_field("Ydefault"),
    }).as_text
    assert 'lookupObjectRef<volScalarField>("Xi")' in text
    assert 'fields.append("Xi")' in text
    assert "mesh().store(Ydefault)" in text
