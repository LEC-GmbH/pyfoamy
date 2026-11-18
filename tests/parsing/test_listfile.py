from pathlib import Path

import numpy as np
import pytest

from pyfoamy.parsing import ParseError
from pyfoamy.parsing.listfile import (
    read_face_list,
    read_header,
    read_label_list,
    read_vector_list,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def mesh_dir(variant: str) -> Path:
    return FIXTURES / f"cavity_{variant}" / "constant" / "polyMesh"


@pytest.mark.parametrize("variant", ["ascii", "binary"])
def test_read_header(variant):
    data = (mesh_dir(variant) / "points").read_bytes()
    header, fmt, offset = read_header(data)
    assert fmt == variant
    assert str(header["FoamFile"]["class"]) == "vectorField"
    # offset should land right after the header, well before any
    # binary payload.
    assert offset < 600


@pytest.mark.parametrize("variant", ["ascii", "binary"])
def test_read_vector_list_points(variant):
    points = read_vector_list(mesh_dir(variant) / "points")
    assert len(points) == 882
    assert points[0] == pytest.approx((0.0, 0.0, 0.0))
    assert points[1] == pytest.approx((0.005, 0.0, 0.0))
    assert points[-1] == pytest.approx((0.1, 0.1, 0.01))


@pytest.mark.parametrize("variant", ["ascii", "binary"])
def test_read_label_list_owner_and_neighbour(variant):
    owner = read_label_list(mesh_dir(variant) / "owner")
    neighbour = read_label_list(mesh_dir(variant) / "neighbour")
    assert len(owner) == 1640
    assert owner[:6].tolist() == [0, 0, 1, 1, 2, 2]
    assert len(neighbour) == 760
    assert neighbour[:6].tolist() == [1, 20, 2, 21, 3, 22]


@pytest.mark.parametrize("variant", ["ascii", "binary"])
def test_read_face_list(variant):
    faces = read_face_list(mesh_dir(variant) / "faces")
    assert len(faces) == 1640
    assert faces[0].tolist() == [1, 22, 463, 442]
    assert faces[1].tolist() == [21, 462, 463, 22]
    assert (faces.sizes() == 4).all()


def test_ascii_and_binary_agree_exactly_on_integers():
    ascii_dir = mesh_dir("ascii")
    binary_dir = mesh_dir("binary")
    for name in ("owner", "neighbour"):
        assert np.array_equal(
            read_label_list(ascii_dir / name),
            read_label_list(binary_dir / name),
        )
    ascii_faces = read_face_list(ascii_dir / "faces")
    binary_faces = read_face_list(binary_dir / "faces")
    assert np.array_equal(ascii_faces.offsets, binary_faces.offsets)
    assert np.array_equal(ascii_faces.values, binary_faces.values)


def test_ascii_and_binary_agree_approximately_on_points():
    # Ascii has writePrecision digits, binary the exact double.
    ascii_points = read_vector_list(mesh_dir("ascii") / "points")
    binary_points = read_vector_list(mesh_dir("binary") / "points")
    assert ascii_points.shape == binary_points.shape == (882, 3)
    assert np.allclose(ascii_points, binary_points, rtol=0, atol=1e-6)


def test_boundary_stays_text_and_parses_as_a_dictionary():
    # boundary is never binary.
    from pyfoamy.parsing.grammar import AnonymousListEntry, Dictionary

    for variant in ("ascii", "binary"):
        text = (mesh_dir(variant) / "boundary").read_text()
        d = Dictionary(text)
        entry = next(n for n in d if isinstance(n, AnonymousListEntry))
        assert entry.count == 3
        names = [str(e.key) for e in entry.body]
        assert names == ["movingWall", "fixedWalls", "frontAndBack"]
        moving = entry.body["movingWall"]
        assert str(moving["type"]) == "wall"
        assert moving["nFaces"] == 20
        assert moving["startFace"] == 760


def test_read_label_list_rejects_garbage(tmp_path):
    garbage = tmp_path / "not_a_mesh_file"
    garbage.write_bytes(b"this is not an OpenFOAM list file at all\n")
    with pytest.raises((ValueError, ParseError)):
        read_label_list(garbage)


def test_read_label_list_missing_file_error_names_both_candidates(tmp_path):
    # The error names the requested path, not only the .gz fallback.
    missing = tmp_path / "owner"
    with pytest.raises(FileNotFoundError) as excinfo:
        read_label_list(missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert str(missing) + ".gz" in message


_ASCII_LABEL_HEADER = b"""\
// a leading line comment, unlike the real fixture's block comment
FoamFile
{
    format      ascii;
    class       labelList;
    object      tinyLabels;
}
// another comment right before the count
3
(
1
2
3
)
"""


def test_ascii_comment_variations_before_count(tmp_path):
    # Comment skipping must not depend on the fixture's banner shape.
    path = tmp_path / "tinyLabels"
    path.write_bytes(_ASCII_LABEL_HEADER)
    assert read_label_list(path).tolist() == [1, 2, 3]


def test_synthetic_binary_label_list(tmp_path):
    import struct

    header = (
        b"FoamFile\n{\n"
        b"    format      binary;\n"
        b"    class       labelList;\n"
        b"    object      tinyLabels;\n"
        b"}\n"
        b"4\n("
    )
    payload = struct.pack("<4i", 10, -5, 0, 999)
    path = tmp_path / "tinyBinaryLabels"
    path.write_bytes(header + payload + b")\n")
    assert read_label_list(path).tolist() == [10, -5, 0, 999]


def test_ascii_face_list_with_mixed_sizes(tmp_path):
    path = tmp_path / "faces"
    path.write_bytes(
        b"FoamFile { format ascii; class faceList; object faces; }\n"
        b"3\n(3(0 1 2) 4(3 4 5 6) 0())\n"
    )
    faces = read_face_list(path)
    assert faces.offsets.tolist() == [0, 3, 7, 7]
    assert faces[1].tolist() == [3, 4, 5, 6]
    assert len(faces[2]) == 0
