from decimal import Decimal

import pytest

from pyfoamy.parsing.grammar import (
    Dictionary,
    KeywordEntry,
    Label,
    List,
    Number,
    String,
    Uniform,
    Word,
    to_grammar_value,
)


def _keys(d):
    return [
        entry.key.data for entry in d if isinstance(entry, KeywordEntry)
    ]


def test_set_creates_new_entry_appended_at_end():
    d = Dictionary("a 1;")
    d.set("b", 2)
    assert _keys(d) == ["a", "b"]
    assert int(d["b"]) == 2


def test_set_updates_existing_entry_in_place():
    d = Dictionary("a 1;\nb 2;\nc 3;")
    d.set("b", 99)
    assert _keys(d) == ["a", "b", "c"]
    assert int(d["b"]) == 99


def test_set_updates_last_entry_when_key_duplicated():
    d = Dictionary("a 1;\na 2;")
    d.set("a", 99)
    values = [
        int(entry.value) for entry in d if isinstance(entry, KeywordEntry)
    ]
    assert values == [1, 99]


def test_set_string_value():
    d = Dictionary()
    d.set("solver", "GAMG")
    assert str(d["solver"]) == "GAMG"


def test_set_string_needing_quotes_raises():
    # '/' isn't valid in a bare WORD (see Word's own docstring) - no
    # auto-quoting any more, the caller must be explicit via String(...).
    d = Dictionary()
    with pytest.raises(ValueError, match="Expected end of text"):
        d.set("location", "constant/polyMesh")


def test_set_string_wrapped_explicitly_is_quoted():
    d = Dictionary()
    d.set("location", String("constant/polyMesh"))
    assert isinstance(d["location"], String)
    assert d["location"].data == "constant/polyMesh"


def test_set_bool():
    d = Dictionary()
    d.set("flag", True)
    assert d["flag"].as_switch().value is True
    d.set("flag", False)
    assert d["flag"].as_switch().value is False


def test_set_float_and_int():
    d = Dictionary()
    d.set("n", 5)
    d.set("x", 2.5)
    assert isinstance(d["n"], Label)
    assert int(d["n"]) == 5
    assert isinstance(d["x"], Number)
    assert float(d["x"]) == pytest.approx(2.5)


def test_set_list_nested():
    d = Dictionary()
    d.set("m", [[1, 2], [3, 4]])
    assert d["m"] == [[1, 2], [3, 4]]


def test_set_plain_dict_becomes_subdict():
    d = Dictionary()
    d.set("inlet", {"type": "fixedValue", "value": "uniform"})
    assert isinstance(d["inlet"], Dictionary)
    assert str(d["inlet"]["type"]) == "fixedValue"


def test_dictionary_constructed_from_plain_dict_recursive():
    d = Dictionary(
        {
            "solver": "GAMG",
            "tolerance": 1e-6,
            "smoother": {"type": "DIC", "iterations": 2},
        }
    )
    assert str(d["solver"]) == "GAMG"
    assert float(d["tolerance"]) == pytest.approx(1e-6)
    assert str(d["smoother"]["type"]) == "DIC"
    assert int(d["smoother"]["iterations"]) == 2


def test_set_subdict():
    d = Dictionary()
    sub = Dictionary()
    sub.set("type", "fixedValue")
    d.set("inlet", sub)
    assert isinstance(d["inlet"], Dictionary)
    assert str(d["inlet"]["type"]) == "fixedValue"


def test_set_overwrites_dict_with_scalar_and_vice_versa():
    d = Dictionary("a { x 1; }")
    d.set("a", 5)
    assert not isinstance(d["a"], Dictionary)
    assert int(d["a"]) == 5

    sub = Dictionary()
    sub.set("y", 2)
    d.set("a", sub)
    assert isinstance(d["a"], Dictionary)


def test_remove_existing():
    d = Dictionary("a 1;\nb 2;")
    d.remove("a")
    assert "a" not in d
    assert _keys(d) == ["b"]


def test_remove_missing_raises():
    d = Dictionary("a 1;")
    with pytest.raises(KeyError):
        d.remove("missing")


def test_setitem_is_sugar_for_set():
    d = Dictionary("a 1;")
    d["a"] = 2
    d["b"] = 3
    assert int(d["a"]) == 2
    assert int(d["b"]) == 3


def test_delitem_is_sugar_for_remove():
    d = Dictionary("a 1;\nb 2;")
    del d["a"]
    assert "a" not in d
    with pytest.raises(KeyError):
        del d["missing"]


def test_nested_setitem_chains_through_subdict():
    d = Dictionary(
        """
        boundaryField
        {
            inlet { type patch; }
        }
        """
    )
    d["boundaryField"]["inlet"]["type"] = "fixedValue"
    d["boundaryField"]["outlet"] = {"type": "patch"}
    assert str(d["boundaryField"]["inlet"]["type"]) == "fixedValue"


def test_nested_delitem_through_subdict():
    d = Dictionary("outer { a 1; b 2; }")
    del d["outer"]["a"]
    assert "a" not in d["outer"]
    assert "b" in d["outer"]


def test_to_grammar_value_scalars():
    assert isinstance(to_grammar_value(5), Label)
    assert isinstance(to_grammar_value(5.0), Number)
    assert str(to_grammar_value(True)) == "true"
    assert str(to_grammar_value(False)) == "false"
    assert isinstance(to_grammar_value("abc"), Word)


@pytest.mark.parametrize("value", [
    0, -7, 10**30, 0.0, -0.0, 0.1, 160.0, -2.5e-300, 1e20, 1e21,
    5e-324, float("nan"), float("-inf"), Decimal("5"), Decimal("-0"),
    Decimal("1.50"), Decimal("1E+2"),
])
def test_numbers_convert_as_their_text(value):
    # The int/float fast paths must give what parsing str(value) gives.
    expected = to_grammar_value(str(value))
    got = to_grammar_value(value)
    assert (type(got), repr(got)) == (type(expected), repr(expected))
    for cls in (Number, Label):
        try:
            expected = cls(str(value))
        except ValueError:
            with pytest.raises(ValueError):
                cls(value)
        else:
            assert repr(cls(value)) == repr(expected)


def test_to_grammar_value_unsafe_string_raises():
    with pytest.raises(ValueError, match="Expected end of text"):
        to_grammar_value("a b")
    with pytest.raises(ValueError):
        to_grammar_value("")


def test_to_grammar_value_string_wrapped_explicitly_is_quoted():
    assert isinstance(to_grammar_value(String("a b")), String)
    assert to_grammar_value(String("a b")).data == "a b"


def test_to_grammar_value_list_shape():
    value = to_grammar_value([1, 2, 3])
    assert str(value) == "(1 2 3)"


@pytest.mark.parametrize("value", [object(), None, {1, 2}])
def test_to_grammar_value_rejects_unknown_type(value):
    with pytest.raises(TypeError):
        to_grammar_value(value)


def test_to_grammar_value_numpy():
    np = pytest.importorskip("numpy")
    assert to_grammar_value(np.array([1.5, 2.0, 3.0])) == List(1.5, 2, 3)
    assert str(to_grammar_value(np.zeros((2, 3), dtype=int))) == (
        "((0 0 0) (0 0 0))"
    )
    assert to_grammar_value(np.bool_(True)) == Word("true")
    assert to_grammar_value(np.int64(3)) == Label(3)
    assert str(Uniform(np.array([0.0, 0.0, 1.0]))) == "uniform (0 0 1)"


def test_to_grammar_value_path_range_enum():
    import enum
    from pathlib import Path

    class Kind(enum.Enum):
        A = "laminar"

    assert to_grammar_value(Path("a/b c")) == String("a/b c")
    assert to_grammar_value(range(3)) == List(0, 1, 2)
    assert to_grammar_value(Kind.A) == Word("laminar")


def test_uniform_scalar():
    assert str(Uniform(0)) == "uniform 0"


def test_uniform_vector():
    assert str(Uniform((0, 0, 0))) == "uniform (0 0 0)"


def test_uniform_round_trips_through_set():
    d = Dictionary()
    d.set("internalField", Uniform((1, 0, 0)))
    text = str(d["internalField"])
    assert text == "uniform (1 0 0)"

    reparsed = Dictionary(f"internalField {text};")
    assert str(reparsed["internalField"]) == str(d["internalField"])
