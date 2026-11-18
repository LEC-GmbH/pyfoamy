import pytest

from pyfoamy import function1
from pyfoamy.parsing.grammar import Dictionary

numpy = pytest.importorskip("numpy")


def test_table_writes_only_what_is_given():
    assert function1.table([(0, 1), (1, 2)]) == {
        "type": "table", "values": [(0, 1), (1, 2)],
    }
    assert function1.table(
        [[0, 1]], out_of_bounds="zero", interpolation_scheme="step",
    ) == {
        "type": "table", "outOfBounds": "zero",
        "interpolationScheme": "step", "values": [(0, 1)],
    }


def test_table_file_reads_two_columns_of_a_csv():
    result = function1.table_file("$FOAM_CASE/constant/p.dat")
    assert result["file"] == '"$FOAM_CASE/constant/p.dat"'
    assert result["format"] == "csv"
    assert result["separator"] == '" "'
    assert result["columns"] == [0, 1]


def test_wrappers_nest():
    inner = function1.table([(0, 1), (1, 2)])
    result = function1.repeat(function1.shift(inner, -5), 720, start=5)
    assert result == {
        "type": "repeat", "period": 720, "start": 5,
        "value": {"type": "shift", "start": -5, "value": inner},
    }
    assert function1.repeat(inner, 720) == {
        "type": "repeat", "period": 720, "value": inner,
    }
    assert function1.scale(inner, 2, x_scale=6) == {
        "type": "scale", "scale": 2, "xScale": 6, "value": inner,
    }
    assert "repeat" in str(Dictionary({"f": result}))


def test_periodic_samples_wrap_and_close_both_ends():
    x, y = function1.periodic_samples([-10, 100, 700], [1, 2, 3], 720, 9)
    numpy.testing.assert_allclose(x, [0, 100, 700, 710, 720])
    numpy.testing.assert_allclose(y, [9, 2, 3, 1, 9])


def test_periodic_samples_past_the_period_move_to_the_front():
    x, y = function1.periodic_samples([700, 730], [1, 2], 720, 9)
    numpy.testing.assert_allclose(x, [0, 10, 700, 720])
    numpy.testing.assert_allclose(y, [9, 2, 1, 9])


def test_periodic_samples_of_a_full_period_keep_one_end():
    x, y = function1.periodic_samples([0, 360, 720], [1, 2, 1], 720, 1)
    numpy.testing.assert_allclose(x, [0, 360, 720])
    numpy.testing.assert_allclose(y, [1, 2, 1])


def test_periodic_samples_leave_a_single_sample():
    x, y = function1.periodic_samples([-5], [3], 720, 3)
    numpy.testing.assert_allclose(x, [715])
    numpy.testing.assert_allclose(y, [3])


def test_periodic_samples_reject_a_non_positive_period():
    with pytest.raises(ValueError, match="positive"):
        function1.periodic_samples([0, 1], [0, 1], 0, 0)


def test_write_samples_is_what_table_file_reads(tmp_path):
    path = tmp_path / "p.dat"
    function1.write_samples(path, [0, 0.5], [1e5, 123456789])
    assert path.read_text() == "0 100000\n0.5 1.23457e+08\n"
