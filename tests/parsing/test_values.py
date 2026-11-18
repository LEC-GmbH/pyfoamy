import subprocess
from decimal import Decimal

import pytest

from pyfoamy.parsing.grammar import Compound, Dictionary, String, Word

# Cases test_grammar.py does not cover: a Compound with an embedded
# Dictionary, and key validation at dictionary level.


# -- Word/String -----------------------------------------------------------


def test_word_rejects_non_str():
    # Coerced via str() and parsed, so ValueError, not TypeError.
    with pytest.raises(ValueError, match="not a valid bare WORD"):
        Word(1)


def test_word_rejects_unsafe_characters():
    with pytest.raises(ValueError, match="not a valid bare WORD"):
        Word("constant/polyMesh")


def test_word_rejects_number_leading_content():
    with pytest.raises(ValueError, match="not a valid bare WORD"):
        Word("123")


def test_word_rejects_empty_string():
    with pytest.raises(ValueError):
        Word("")


def test_string_coerces_non_str_via_str():
    assert str(String(1)) == '"1"'
    assert str(String(Decimal("1.5"))) == '"1.5"'


def test_string_of_a_safe_word_still_quotes():
    # String() always quotes.
    d = Dictionary()
    d.set("solver", String("GAMG"))
    assert isinstance(d["solver"], String)
    assert str(d["solver"]) == '"GAMG"'


# -- Compound with an embedded Dictionary -----------------------------


def test_compound_with_embedded_dictionary_matches_real_multivariateselection(
    tmp_path,
):
    # From tutorials/multicomponentFluid/smallPoolFire3D fvSchemes:
    #   div(phi,Yi_h)  Gauss multivariateSelection { O2 ... ; };
    species = Dictionary({
        "O2": Compound("limitedLinear01", 1),
        "CH4": Compound("limitedLinear01", 1),
    })
    entry = Compound("Gauss", "multivariateSelection", species)
    d = Dictionary()
    d.set("div(phi,Yi_h)", entry)

    reparsed = Dictionary(d.as_text)
    value = reparsed["div(phi,Yi_h)"]
    assert str(value[0]) == "Gauss"
    assert str(value[1]) == "multivariateSelection"


@pytest.mark.openfoam
def test_compound_with_embedded_dictionary_is_valid_per_foam_dictionary(
    tmp_path,
):
    # -expand, not -entry: OpenFOAM keeps the whole value flat, so
    # "-entry" on the embedded block fails with "Cannot find entry".
    species = Dictionary({
        "O2": Compound("limitedLinear01", 1),
        "CH4": Compound("limitedLinear01", 1),
    })
    entry = Compound("Gauss", "multivariateSelection", species)
    d = Dictionary()
    d.set("divSchemes", {"div(phi,Yi_h)": entry})
    path = tmp_path / "testDict"
    path.write_text(d.as_text)

    result = subprocess.run(
        ["foamDictionary", "-expand", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "multivariateSelection" in result.stdout
    assert "O2" in result.stdout and "CH4" in result.stdout


# -- Dictionary: recursive key validation through plain dicts ---------


def test_dictionary_rejects_int_key():
    with pytest.raises(TypeError, match="str, Word, String, or Variable"):
        Dictionary({123: -1})


def test_dictionary_rejects_bad_key_in_nested_subdict():
    with pytest.raises(TypeError):
        Dictionary({"a": {123: -1}})


def test_dictionary_recursive_validation_reaches_deep_nesting():
    with pytest.raises((TypeError, ValueError)):
        Dictionary({"a": {"b": {"c": {"1": -1}}}})


def test_dictionary_accepts_string_wrapped_numeric_key():
    d = Dictionary({String("1"): -1})
    assert d["1"] == -1


# -- Dictionary.set() key handling -------------------------------------


def test_set_rejects_int_key():
    d = Dictionary()
    with pytest.raises(TypeError):
        d.set(123, "value")


def test_set_rejects_unsafe_string_key():
    d = Dictionary()
    with pytest.raises(ValueError, match="not a valid bare WORD"):
        d.set("123", "value")


def test_set_accepts_string_wrapped_key():
    d = Dictionary()
    d.set(String("123"), "value")
    assert str(d["123"]) == "value"
    entry = next(iter(d))
    assert isinstance(entry.key, String)


def test_set_accepts_word_wrapped_key():
    d = Dictionary()
    d.set(Word("solver"), "GAMG")
    assert str(d["solver"]) == "GAMG"
    entry = next(iter(d))
    assert isinstance(entry.key, Word)
