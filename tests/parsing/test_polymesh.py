from pathlib import Path

import numpy as np
import pytest

from pyfoamy.parsing import MeshPatch, PolyMesh

FIXTURES = Path(__file__).parent.parent / "fixtures"


def mesh_dir(variant: str) -> Path:
    return FIXTURES / f"cavity_{variant}" / "constant" / "polyMesh"


@pytest.mark.parametrize("variant", ["ascii", "binary"])
def test_polymesh_read(variant):
    mesh = PolyMesh.read(mesh_dir(variant))
    assert len(mesh.points) == 882
    assert len(mesh.faces) == 1640
    assert len(mesh.owner) == 1640
    assert len(mesh.neighbour) == 760
    assert mesh.n_cells == 400
    assert mesh.points.shape == (882, 3)

    assert list(mesh.patches) == ["movingWall", "fixedWalls", "frontAndBack"]
    assert mesh.patches["movingWall"] == MeshPatch("wall", 760, 20, ("wall",))
    assert len(mesh.patch_points("movingWall")) == 42


def test_polymesh_ascii_and_binary_agree():
    ascii_mesh = PolyMesh.read(mesh_dir("ascii"))
    binary_mesh = PolyMesh.read(mesh_dir("binary"))
    assert np.array_equal(ascii_mesh.owner, binary_mesh.owner)
    assert np.array_equal(ascii_mesh.neighbour, binary_mesh.neighbour)
    assert np.array_equal(ascii_mesh.faces.values, binary_mesh.faces.values)
    assert ascii_mesh.patches == binary_mesh.patches
    assert np.allclose(ascii_mesh.points, binary_mesh.points, atol=1e-6)


def _patch_names(mesh: PolyMesh) -> list:
    return list(mesh.patches)


def test_read_at_falls_back_to_constant_when_nothing_is_overridden():
    # "0.1/" has no polyMesh, so everything comes from constant.
    case_dir = FIXTURES / "cavity_ascii"
    at_time = PolyMesh.read_at(case_dir, 0.1)
    plain = PolyMesh.read(case_dir / "constant" / "polyMesh")
    assert np.array_equal(at_time.points, plain.points)
    assert np.array_equal(at_time.faces.values, plain.faces.values)
    assert np.array_equal(at_time.owner, plain.owner)
    assert np.array_equal(at_time.neighbour, plain.neighbour)
    assert at_time.patches == plain.patches


_HEADER = (
    "FoamFile {{ version 2.0; format ascii; class {cls}; object x; }}\n"
)


def _write_mesh(directory, points, faces, owner, neighbour, patch_name):
    directory.mkdir(parents=True)
    (directory / "points").write_text(
        _HEADER.format(cls="vectorField")
        + f"{len(points)}\n("
        + " ".join(f"({x} {y} {z})" for x, y, z in points)
        + ")\n"
    )
    (directory / "faces").write_text(
        _HEADER.format(cls="faceList")
        + f"{len(faces)}\n("
        + "".join(
            f"{len(f)}(" + " ".join(str(i) for i in f) + ")"
            for f in faces
        )
        + ")\n"
    )
    (directory / "owner").write_text(
        _HEADER.format(cls="labelList")
        + f"{len(owner)}\n("
        + " ".join(str(i) for i in owner)
        + ")\n"
    )
    (directory / "neighbour").write_text(
        _HEADER.format(cls="labelList")
        + f"{len(neighbour)}\n("
        + " ".join(str(i) for i in neighbour)
        + ")\n"
    )
    (directory / "boundary").write_text(
        _HEADER.format(cls="polyBoundaryMesh")
        + "1\n"
        + f"({patch_name} {{type patch; nFaces 1; startFace 0;}})\n"
    )


def test_read_at_moving_mesh_only_overrides_points(tmp_path):
    _write_mesh(
        tmp_path / "constant" / "polyMesh",
        points=[(0, 0, 0), (1, 0, 0)],
        faces=[[0, 1, 2, 3]],
        owner=[0],
        neighbour=[],
        patch_name="patch0",
    )
    (tmp_path / "0.1" / "polyMesh").mkdir(parents=True)
    (tmp_path / "0.1" / "polyMesh" / "points").write_text(
        _HEADER.format(cls="vectorField") + "2\n((0.1 0 0) (1.1 0 0))\n"
    )

    mesh = PolyMesh.read_at(tmp_path, 0.1)
    assert mesh.points.tolist() == [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]
    # faces/owner/neighbour/boundary weren't overridden - must fall
    # back to constant, not error or return empty data.
    assert mesh.faces[0].tolist() == [0, 1, 2, 3]
    assert mesh.owner.tolist() == [0]


def test_read_at_topology_change_overrides_everything_together(
    tmp_path,
):
    _write_mesh(
        tmp_path / "constant" / "polyMesh",
        points=[(0, 0, 0), (1, 0, 0)],
        faces=[[0, 1, 2, 3]],
        owner=[0],
        neighbour=[],
        patch_name="patch0",
    )
    _write_mesh(
        tmp_path / "0.2" / "polyMesh",
        points=[(0.2, 0, 0), (1.2, 0, 0)],
        faces=[[0, 1, 2]],
        owner=[5],
        neighbour=[],
        patch_name="newPatch",
    )

    mesh = PolyMesh.read_at(tmp_path, 0.2)
    assert mesh.points.tolist() == [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]]
    assert mesh.faces[0].tolist() == [0, 1, 2]
    assert mesh.owner.tolist() == [5]
    assert _patch_names(mesh) == ["newPatch"]


def test_read_at_with_region_on_a_synthetic_two_region_case(tmp_path):
    # constant/<region>/polyMesh; two regions must not collide.
    _write_mesh(
        tmp_path / "constant" / "fluid" / "polyMesh",
        points=[(0, 0, 0), (1, 0, 0)],
        faces=[[0, 1, 2, 3]],
        owner=[0],
        neighbour=[],
        patch_name="fluidPatch",
    )
    _write_mesh(
        tmp_path / "constant" / "solid" / "polyMesh",
        points=[(5, 5, 5)],
        faces=[[0, 1, 2]],
        owner=[3],
        neighbour=[],
        patch_name="solidPatch",
    )

    fluid_mesh = PolyMesh.read_at(tmp_path, 0.0, region="fluid")
    solid_mesh = PolyMesh.read_at(tmp_path, 0.0, region="solid")
    assert fluid_mesh.points.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert solid_mesh.points.tolist() == [[5.0, 5.0, 5.0]]
    assert _patch_names(fluid_mesh) == ["fluidPatch"]
    assert _patch_names(solid_mesh) == ["solidPatch"]


def test_read_at_with_region_on_the_real_multiregion_fixture():
    case_dir = FIXTURES / "multiregion_cylinder"
    fluid_mesh = PolyMesh.read_at(case_dir, 0.0, region="fluid")
    film_mesh = PolyMesh.read_at(case_dir, 0.0, region="film")
    assert len(fluid_mesh.points) == 156
    assert len(fluid_mesh.faces) == 251
    assert len(fluid_mesh.neighbour) == 97
    assert len(film_mesh.points) == 76
    assert len(film_mesh.faces) == 91
    assert len(film_mesh.neighbour) == 17
