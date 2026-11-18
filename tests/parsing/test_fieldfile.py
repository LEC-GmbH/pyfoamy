import math
import struct
import time
from pathlib import Path

import pytest
from conftest import TUTORIALS_ROOT

from pyfoamy.parsing.fieldfile import _PLACEHOLDER_PREFIX, read_field
from pyfoamy.parsing.grammar import Compound, Dictionary, List

FIXTURES = Path(__file__).parent.parent / "fixtures"
TIME_DIR = "0.1"
FIELD_NAMES = [
    "alphat", "k", "nut", "omega", "p", "phi", "rho", "T", "U",
]



def field_path(variant: str, name: str) -> Path:
    return FIXTURES / f"cavity_{variant}" / TIME_DIR / name


def _to_plain(value):
    if isinstance(value, List):
        return [_to_plain(v) for v in value]
    return float(value)


def field_value(value):
    assert isinstance(value, Compound)
    kind = str(value[0])
    assert kind in ("uniform", "nonuniform")
    return _to_plain(value[1])


def assert_values_close(a, b, tol=1e-3):
    # rel_tol: ascii writePrecision on large values. abs_tol: ~1e-10 vs
    # ascii 0. 1e-3 since the ascii/binary fixtures are separate solves;
    # turbulent fields differ by ~5e-4 (p/rho/T agree to ~1e-9).
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_values_close(x, y, tol)
    else:
        assert math.isclose(
            float(a), float(b), rel_tol=tol, abs_tol=tol
        )


@pytest.mark.parametrize("variant", ["ascii", "binary"])
@pytest.mark.parametrize("name", FIELD_NAMES)
def test_read_field_has_dimensions_and_fields(variant, name):
    d = read_field(field_path(variant, name))
    assert len(d["dimensions"].data.split()) == 7
    assert "internalField" in d
    assert "boundaryField" in d


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_binary_field_has_no_leftover_placeholder_tokens(name):
    d = read_field(field_path("binary", name))
    assert _PLACEHOLDER_PREFIX not in d.as_text


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_ascii_and_binary_dimensions_agree(name):
    d_ascii = read_field(field_path("ascii", name))
    d_binary = read_field(field_path("binary", name))
    assert d_ascii["dimensions"].data.split() == d_binary["dimensions"].data.split()


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_ascii_and_binary_internal_field_values_agree(name):
    d_ascii = read_field(field_path("ascii", name))
    d_binary = read_field(field_path("binary", name))
    assert_values_close(
        field_value(d_ascii["internalField"]),
        field_value(d_binary["internalField"]),
    )


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_ascii_and_binary_boundary_field_patches_agree(name):
    # Some patches (wall-function values) carry their own binary blob.
    d_ascii = read_field(field_path("ascii", name))
    d_binary = read_field(field_path("binary", name))
    patches_ascii = list(d_ascii["boundaryField"])
    patches_binary = list(d_binary["boundaryField"])
    assert [str(p.key) for p in patches_ascii] == [
        str(p.key) for p in patches_binary
    ]
    for patch_ascii, patch_binary in zip(patches_ascii, patches_binary):
        sub_ascii = patch_ascii.value
        sub_binary = patch_binary.value
        if "value" not in sub_ascii:
            continue
        assert_values_close(
            field_value(sub_ascii["value"]),
            field_value(sub_binary["value"]),
        )


@pytest.mark.tutorials
@pytest.mark.parametrize("name", ["p", "U"])
def test_read_field_on_real_tutorial_files(name):
    d = read_field(
        TUTORIALS_ROOT / "legacy/incompressible/icoFoam/cavity/cavity/0"
        / name
    )
    assert "dimensions" in d
    assert "internalField" in d
    assert "boundaryField" in d


# --- Small hand-built fields: ascii vs binary and malformed counts ----

_SYNTHETIC_VALUES = [
    1.5, 2.25, 3.0, -4.5, 5.75, 0.0, -1.0, 12.5, 6.25, 7.0, 8.5, 9.75,
]


def _synthetic_header(fmt: str) -> bytes:
    return (
        b"FoamFile\n{\n"
        b"    version     2.0;\n"
        b"    format      " + fmt.encode("ascii") + b";\n"
        b"    class       volScalarField;\n"
        b"    object      p;\n"
        b"}\n"
    )


def _ascii_scalar_field_text(declared_count, values) -> str:
    items = " ".join(str(v) for v in values)
    body = (
        f"internalField nonuniform List<scalar> {declared_count} "
        f"({items});\n"
        "boundaryField\n{\n}\n"
    )
    return _synthetic_header("ascii").decode("ascii") + body


def _binary_scalar_field_bytes(declared_count, values) -> bytes:
    body_head = (
        f"internalField nonuniform List<scalar> {declared_count}("
    ).encode("ascii")
    payload = struct.pack(f"<{len(values)}d", *values)
    body_tail = b");\nboundaryField\n{\n}\n"
    return _synthetic_header("binary") + body_head + payload + body_tail


def test_synthetic_small_field_ascii_and_binary_agree_exactly(tmp_path):
    values = _SYNTHETIC_VALUES
    ascii_path = tmp_path / "p_ascii"
    ascii_path.write_text(_ascii_scalar_field_text(len(values), values))
    binary_path = tmp_path / "p_binary"
    binary_path.write_bytes(
        _binary_scalar_field_bytes(len(values), values)
    )

    ascii_value = field_value(read_field(ascii_path)["internalField"])
    binary_value = field_value(read_field(binary_path)["internalField"])
    assert [float(v) for v in ascii_value] == values
    assert [float(v) for v in binary_value] == values


def test_ascii_header_with_actual_binary_bytes_fails_cleanly(tmp_path):
    # The header's `format` governs the whole file, no per-list
    # detection: binary data labelled ascii fails loudly as UTF-8.
    binary_bytes = _binary_scalar_field_bytes(
        len(_SYNTHETIC_VALUES), _SYNTHETIC_VALUES
    )
    mislabeled = binary_bytes.replace(
        b"format      binary;", b"format      ascii;", 1
    )
    path = tmp_path / "p"
    path.write_bytes(mislabeled)
    with pytest.raises(UnicodeDecodeError):
        read_field(path)


def test_binary_nonuniform_declared_count_too_large_raises(tmp_path):
    # Declares 5 scalars, has 3: the byte after the blob is not ')'.
    path = tmp_path / "p"
    path.write_bytes(_binary_scalar_field_bytes(5, [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match=r"expected '\)'"):
        read_field(path)


def test_binary_nonuniform_declared_count_too_small_raises(tmp_path):
    # Declares 2 scalars, has 3: the range ends inside the data.
    path = tmp_path / "p"
    path.write_bytes(_binary_scalar_field_bytes(2, [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match=r"expected '\)'"):
        read_field(path)


def test_binary_nonuniform_declared_count_past_eof_raises(tmp_path):
    # A count past EOF slices to b"" and must still raise ValueError.
    path = tmp_path / "p"
    path.write_bytes(_binary_scalar_field_bytes(1_000_000, [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match=r"expected '\)'"):
        read_field(path)


_ASCII_LISTS = """FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector> 3((0 1 2.5) (1e5 0 -0)
(-inf nan 1.));
scalars nonuniform List<scalar> 5(0 1.5 -2e-3 3 7E2);
empty nonuniform List<scalar> 0();
uniformList List<scalar> 2{1.5};
withVariable nonuniform List<scalar> 2(1 $x);
tensors nonuniform List<symmTensor> 1((1 2 3 4 5 6));
boundaryField
{
    wall { type fixedValue; value nonuniform List<scalar> 2(1 2); }
}
"""


def test_ascii_lists_read_as_the_grammar_reads_them(tmp_path):
    path = tmp_path / "U"
    path.write_text(_ASCII_LISTS)
    assert repr(read_field(path)) == repr(Dictionary(_ASCII_LISTS))


def test_ascii_list_with_a_wrong_count_is_left_to_the_grammar(tmp_path):
    text = _ASCII_LISTS + "bad nonuniform List<scalar> 3(1 2);\n"
    path = tmp_path / "p"
    path.write_text(text)
    assert repr(read_field(path)) == repr(Dictionary(text))


def test_large_ascii_scalar_list_reads_fast(tmp_path):
    n = 200000
    path = tmp_path / "T"
    path.write_text(
        _ASCII_LISTS
        + f"big nonuniform List<scalar> {n}(\n"
        + "\n".join(f"{300 + i * 1e-3!r}" for i in range(n))
        + "\n);\n"
    )
    start = time.perf_counter()
    values = read_field(path)["big"][1]
    assert len(values) == n and str(values[-1]) == "499.999"
    # The grammar alone needs over a minute for this list.
    assert time.perf_counter() - start < 20
