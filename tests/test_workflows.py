import pytest

from pyfoamy import Runner, workflows


def _recorder(tmp_path, **kwargs):
    return Runner(
        tmp_path, mode="record", script_path=tmp_path / "Allrun",
        header="", **kwargs,
    )


def _lines(tmp_path):
    return (tmp_path / "Allrun").read_text().splitlines()


@pytest.mark.parametrize("arguments, expected", [
    (dict(), "1:0"),
    (dict(times=[5]), "5:5"),
    (dict(times=[1, 2.5]), "1,2.5"),
    (dict(start=3), "3:"),
    (dict(stop=4), ":4"),
    (dict(times=[1], start=3, stop=4), "1:1,3:4"),
])
def test_time_selection(arguments, expected):
    assert workflows.time_selection(**arguments) == expected


def test_remove_times_in_the_processor_directories(tmp_path):
    workflows.remove_times(
        _recorder(tmp_path), start=2, processor=True, constant=True,
        suffix="clean",
    )
    assert _lines(tmp_path) == [
        "runApplication -suffix clean foamListTimes -rm -time 2: "
        "-processor -constant",
    ]


def test_decompose_the_mesh_of_constant(tmp_path):
    workflows.decompose(
        _recorder(tmp_path), fields=False, constant=True, force=True,
        remove=True, suffix="s",
    )
    assert _lines(tmp_path) == [
        "runApplication -suffix decomposePar.s foamListTimes -rm -time 1:0 "
        "-processor -constant",
        "runApplication -suffix s decomposePar -constant -noFields -force",
    ]


def test_decompose_without_a_selection_runs_nothing(tmp_path):
    workflows.decompose(_recorder(tmp_path))
    assert not (tmp_path / "Allrun").exists()


def test_reconstruct_selected_fields_of_one_time(tmp_path):
    workflows.reconstruct(
        _recorder(tmp_path), times=[7], fields=["p", "U"], remove=True,
    )
    assert _lines(tmp_path) == [
        "runApplication -suffix reconstructPar foamListTimes -rm -time 7:7",
        "runApplication reconstructPar -time 7:7 -fields '(p U)'",
    ]


def test_processor_directories_are_ordered_by_number(tmp_path):
    for name in ("processor10", "processor2", "processors", "constant"):
        (tmp_path / name).mkdir()
    (tmp_path / "processor3").write_text("")
    assert [
        path.name for path in workflows.processor_directories(tmp_path)
    ] == ["processor2", "processor10"]


def _write_owner(directory):
    directory.mkdir(parents=True)
    (directory / "owner").write_bytes(
        b'FoamFile\n{\n    format      binary;\n    class       labelList;\n'
        b'    note        "nPoints:1115  nCells:888  nFaces:2864  '
        b'nInternalFaces:2376";\n    object      owner;\n}\n\n2864\n(\xff\xfe'
    )


def test_mesh_statistics_reads_the_header_note(tmp_path):
    _write_owner(tmp_path / "constant" / "polyMesh")
    _write_owner(tmp_path / "2.5" / "fluid" / "polyMesh")
    expected = {"points": 1115, "cells": 888, "faces": 2864}
    assert workflows.mesh_statistics(tmp_path) == expected
    assert workflows.mesh_statistics(tmp_path, 2.5, "fluid") == expected


def test_mesh_statistics_without_a_note_raises(tmp_path):
    (tmp_path / "constant" / "polyMesh").mkdir(parents=True)
    (tmp_path / "constant" / "polyMesh" / "owner").write_text("FoamFile{}")
    with pytest.raises(ValueError, match="no mesh sizes"):
        workflows.mesh_statistics(tmp_path)


def test_move_mesh_records_edits_run_and_clean_up(tmp_path):
    valid = workflows.move_mesh(
        _recorder(tmp_path, rerun="$REDO"), 10, 20, 0.5,
        write_times=[20], write_objects=['"motionScale_.*"'],
        fatal_sets=["zeroVolumeCells"], suffix="move",
    )
    assert valid is True
    lines = _lines(tmp_path)
    edit = "foamDictionary -writePrecision 15 -entry"
    assert lines[:5] == [
        f"{edit} startTime -set 10 system/controlDict",
        f"{edit} endTime -set 20 system/controlDict",
        f"{edit} deltaT -set 0.5 system/controlDict",
        f"{edit} maxDeltaT -set 0.5 system/controlDict",
        f"{edit} writeInterval -set 99999 system/controlDict",
    ]
    text = "\n".join(lines)
    for name in ("writeMovedMesh", "checkMesh", "checkMeshWrite"):
        assert f"{edit} {name} -set '{{" in text
        assert f"{edit} {name} -remove system/functions" in lines
    assert '(points "motionScale_.*")' in text
    run = lines.index(
        "runParallel $REDO -suffix move foamRun -solver movingMesh"
    )
    assert lines[run - 1] == (
        "runApplication $REDO -suffix move rm -rvf postProcessing/checkMesh/20"
    )
    assert lines[run + 1].endswith("-remove system/functions")


def test_move_mesh_without_checks_and_function_objects(tmp_path):
    workflows.move_mesh(
        _recorder(tmp_path), 0, 1, 0.1, check_mesh=False,
        function_objects=False,
    )
    assert _lines(tmp_path)[5:] == [
        "runParallel foamRun -solver movingMesh -noFunctionObjects",
    ]


class _Live:
    """Stands in for a live runner: nothing runs, the sets are there."""
    mode = "live"

    def __init__(self, root):
        self.root = root

    def run(self, *args, **kwargs):
        pass

    set_entry = remove_entry = run


@pytest.mark.parametrize("written, valid", [
    ("skewFaces.vtk", True), ("zeroVolumeCells.vtk", False),
])
def test_move_mesh_fails_on_a_fatal_set(tmp_path, written, valid):
    sets = tmp_path / "postProcessing" / "checkMesh" / "20"
    sets.mkdir(parents=True)
    (sets / written).write_text("")
    assert workflows.move_mesh(
        _Live(tmp_path), 10, 20, 1, write_times=[20],
        fatal_sets=["zeroVolumeCells"],
    ) is valid
