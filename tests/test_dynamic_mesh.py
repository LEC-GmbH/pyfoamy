import pytest

from pyfoamy import dynamic_mesh
from pyfoamy.parsing.grammar import Dictionary


def test_rotating_motion_takes_one_speed():
    assert dynamic_mesh.rotating_motion((0, 0, 0), (0, 0, 1), rpm=60) == {
        "type": "rotatingMotion", "axis": (0, 0, 1), "origin": (0, 0, 0),
        "rpm": 60,
    }
    assert dynamic_mesh.rotating_motion(
        (0, 0, 0), (0, 0, 1), omega=6.28,
    )["omega"] == 6.28
    for speeds in ({}, {"omega": 1, "rpm": 1}):
        with pytest.raises(ValueError, match="either omega or rpm"):
            dynamic_mesh.rotating_motion((0, 0, 0), (0, 0, 1), **speeds)


def test_mover_list_of_solid_bodies_is_written_as_openfoam_reads_it():
    motion = dynamic_mesh.rotating_motion((0, 0, 0), (0, 0, 1), rpm=60)
    mover = dynamic_mesh.mover_list(
        {"rotor": dynamic_mesh.solid_body("rotorCells", motion)},
    )
    assert list(mover) == ["type", "libs", "rotor"]
    assert mover["rotor"]["solidBodyMotionFunction"] == motion
    text = str(Dictionary({"mover": mover}))
    assert 'libs ("libfvMotionSolvers.so");' in " ".join(text.split())
    assert "cellZone rotorCells;" in " ".join(text.split())


def test_mover_list_needs_a_mover():
    with pytest.raises(ValueError, match="at least one mover"):
        dynamic_mesh.mover_list({})


def test_mesh_to_mesh_leaves_out_what_is_not_given():
    assert dynamic_mesh.mesh_to_mesh([0, 90], time_delta=1e-9) == {
        "type": "meshToMesh", "libs": dynamic_mesh._MESH_TO_MESH,
        "times": [0, 90], "timeDelta": 1e-9,
    }
    entry = dynamic_mesh.mesh_to_mesh(
        (0, 90), time_delta=1e-9, begin=0, repeat=720,
    )
    assert list(entry) == [
        "type", "libs", "times", "begin", "timeDelta", "repeat",
    ]


@pytest.mark.parametrize("times", [[], [0, 0], [90, 0]])
def test_mesh_to_mesh_needs_rising_times(times):
    with pytest.raises(ValueError, match="rising times"):
        dynamic_mesh.mesh_to_mesh(times, time_delta=1e-9)
