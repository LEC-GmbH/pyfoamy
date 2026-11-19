import shutil
import subprocess

import pytest

ct = pytest.importorskip("cantera")

from pyfoamy.cantera.__main__ import main  # noqa: E402
from pyfoamy.parsing.grammar import Dictionary  # noqa: E402


def _run(capsys, *argv):
    status = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    assert "Traceback" not in err
    return status, out, err


def _error(capsys, *argv) -> str:
    status, out, err = _run(capsys, *argv)
    assert status == 1
    assert out == ""
    assert err.startswith("python -m pyfoamy.cantera: error: ")
    return err


def _usage_error(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc_info:
        main([str(a) for a in argv])
    assert exc_info.value.code == 2
    return capsys.readouterr().err


def _thermo(out_dir):
    return Dictionary.from_file(out_dir / "thermo")


def _reactions(out_dir):
    return Dictionary.from_file(out_dir / "reactions")["reactions"]


@pytest.mark.slow
def test_gri30_writes_every_reaction_and_specie(tmp_path, capsys):
    status, out, _ = _run(capsys, "gri30.yaml", "-o", tmp_path)
    assert status == 0
    assert "325 reactions" in out and "53 species" in out
    assert len(_reactions(tmp_path)) == 325
    assert len(_thermo(tmp_path)["species"]) == 53


def test_yaml_defaults_to_shared_tcommon_and_sutherland(tmp_path, capsys):
    status, out, err = _run(capsys, "h2o2.yaml", "-o", tmp_path)
    assert status == 0
    gas = ct.Solution("h2o2.yaml")
    assert len(_reactions(tmp_path)) == gas.n_reactions

    thermo = _thermo(tmp_path)
    names = [str(name) for name in thermo["species"]]
    assert names == gas.species_names
    assert len({
        thermo[n]["thermodynamics"]["Tcommon"] for n in names
    }) == 1
    assert "As" in thermo["H2"]["transport"]
    assert thermo["H2O"]["elements"]["O"] == 1
    assert list(tmp_path.glob("*.yaml")) == []
    assert "warning: H2: sutherland" in err


def test_per_species_keeps_the_source_polynomials(tmp_path, capsys):
    status, out, _ = _run(
        capsys, "h2o2.yaml", "-o", tmp_path, "--per-species",
        "--transport-model", "const",
    )
    assert status == 0
    thermo = _thermo(tmp_path)
    source = ct.Solution("h2o2.yaml").species("H2").thermo.coeffs
    assert float(thermo["H2"]["thermodynamics"]["Tcommon"]) == source[0]
    assert "mu" in thermo["H2"]["transport"]
    assert "janaf/const" in out


def test_openfoam_version_selects_the_reaction_type_names(tmp_path, capsys):
    _run(capsys, "h2o2.yaml", "-o", tmp_path, "--openfoam-version", "dev")
    assert str(_reactions(tmp_path)["reaction0"]["type"]).startswith(
        "Reversible<"
    )


def test_chemkin_also_writes_the_converted_yaml(h2, tmp_path, capsys):
    status, out, _ = _run(
        capsys, h2.mechanism, "--thermo", h2.thermo,
        "--transport", h2.valid_transport, "-o", tmp_path,
    )
    assert status == 0
    assert ct.Solution(str(tmp_path / "mech.yaml")).n_species == 10
    assert len(_thermo(tmp_path)["species"]) == 10


def test_cti_also_writes_the_converted_yaml(h2_cti, tmp_path, capsys):
    out_dir = tmp_path / "out"
    status, _, _ = _run(capsys, h2_cti, "-o", out_dir)
    assert status == 0
    assert (out_dir / "h2.yaml").is_file()
    assert len(_reactions(out_dir)) == 1


def test_existing_outputs_need_force(tmp_path, capsys):
    _run(capsys, "h2o2.yaml", "-o", tmp_path, "--per-species")
    err = _error(capsys, "h2o2.yaml", "-o", tmp_path, "--per-species")
    assert "--force" in err
    status, _, _ = _run(
        capsys, "h2o2.yaml", "-o", tmp_path, "--per-species", "--force",
    )
    assert status == 0


def test_an_output_never_overwrites_an_input(h2, tmp_path, capsys):
    thermo = tmp_path / "thermo"
    shutil.copy(h2.thermo, thermo)
    err = _error(
        capsys, h2.mechanism, "--thermo", thermo, "-o", tmp_path, "--force",
    )
    assert "would overwrite an input" in err


def test_missing_chemkin_file_is_one_error(h2, tmp_path, capsys):
    err = _error(capsys, h2.missing, "-o", tmp_path)
    assert "not found" in err


def test_ck2yaml_error_points_to_the_permissive_flag(h2, tmp_path, capsys):
    redundant = tmp_path / "redundant.ck"
    redundant.write_text(h2.mechanism.read_text().replace(
        "SPECIES\n", "SPECIES\nAR\n", 1,
    ))
    err = _error(capsys, redundant, "--thermo", h2.thermo, "-o", tmp_path)
    assert "multiple declarations for species 'AR'" in err
    assert "--permissive" in err
    assert "from_chemkin" not in err


def test_chemkin_without_transport_data_is_one_error(h2, tmp_path, capsys):
    err = _error(capsys, h2.mechanism, "--thermo", h2.thermo, "-o", tmp_path)
    assert "no transport data" in err


def test_cti_syntax_error_shows_the_line(tmp_path, capsys):
    path = tmp_path / "syntax.cti"
    path.write_text("ideal_gas(name='gas', elements='H O' species='H2')\n")
    err = _error(capsys, path, "-o", tmp_path)
    assert "line 1" in err


def test_cantera_errors_lose_their_asterisk_frame(tmp_path, capsys):
    err = _error(capsys, tmp_path / "nope.yaml", "-o", tmp_path)
    assert "nope.yaml" in err and "not found" in err
    assert "*****" not in err


@pytest.mark.parametrize("argv", [
    ("h2o2.yaml", "--thermo", "x.dat"),
    ("h2o2.yaml", "--permissive"),
    ("h2o2.yaml", "--thermo-model", "foo"),
])
def test_usage_errors_exit_with_2(capsys, argv):
    assert "usage:" in _usage_error(capsys, *argv)


@pytest.mark.openfoam
def test_openfoam_reads_both_files(tmp_path, capsys):
    if shutil.which("foamDictionary") is None:
        pytest.skip("foamDictionary not found (OpenFOAM not sourced)")
    _run(capsys, "h2o2.yaml", "-o", tmp_path)
    species = subprocess.run(
        ["foamDictionary", "-entry", "species", "-value",
         str(tmp_path / "thermo")],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "H2O2" in species
    reactions = subprocess.run(
        ["foamDictionary", "-entry", "reactions", "-keywords",
         str(tmp_path / "reactions")],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert len(reactions) == ct.Solution("h2o2.yaml").n_reactions
