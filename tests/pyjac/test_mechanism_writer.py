from pyfoamy.pyjac.records import Reaction, Species
from pyfoamy.pyjac import mechanism_writer

SPECIES = [
    Species(name="H2", mw=2.016, low=(0.0,) * 7, high=(0.0,) * 7),
    Species(name="O2", mw=31.998, low=(0.0,) * 7, high=(0.0,) * 7),
    Species(name="N2", mw=28.014, low=(0.0,) * 7, high=(0.0,) * 7),
]
REACTIONS = [
    Reaction(
        reactants=[0, 1], reactant_nu=[2.0, 1.0], products=[],
        product_nu=[], A=1e13, b=0.0, E=5000.0, reversible=True,
    ),
    Reaction(
        reactants=[0], reactant_nu=[1.0], products=[], product_nu=[],
        A=1e10, b=0.0, E=0.0, reversible=False, third_body=True,
    ),
    Reaction(
        reactants=[1], reactant_nu=[1.0], products=[], product_nu=[],
        A=1e10, b=0.0, E=0.0, reversible=False,
        pressure_dependent=True, low_rate_params=(1e9, 0.0, 0.0),
    ),
]


def test_write_mechanism_header_defines(tmp_path):
    mechanism_writer.write_mechanism_header(tmp_path, SPECIES, REACTIONS)
    result = (tmp_path / "mechanism.h").read_text()
    assert "#define NSP 3\n" in result
    assert "#define NN 4\n" in result
    assert "#define FWD_RATES 3\n" in result
    assert "#define REV_RATES 1\n" in result
    assert "#define PRES_MOD_RATES 2\n" in result


def test_write_mechanism_header_guard_uses_lowercase_h(tmp_path):
    # Upstream quirk, kept: "MECHANISM_h" from header_ext['c'][1:].
    mechanism_writer.write_mechanism_header(tmp_path, SPECIES, [])
    result = (tmp_path / "mechanism.h").read_text()
    assert result.startswith("#ifndef MECHANISM_h\n#define MECHANISM_h\n")


def test_write_mechanism_header_species_index_comment(tmp_path):
    mechanism_writer.write_mechanism_header(tmp_path, SPECIES, REACTIONS)
    result = (tmp_path / "mechanism.h").read_text()
    assert "/* Species Indexes\n0  H2\n1  O2\n2  N2\n*/\n\n" in result


def test_write_mechanism_header_last_spec_comment(tmp_path):
    mechanism_writer.write_mechanism_header(tmp_path, SPECIES, REACTIONS)
    result = (tmp_path / "mechanism.h").read_text()
    assert "//last_spec N2\n" in result


def test_write_mechanism_header_omits_dead_mechanism_c_declarations(
    tmp_path,
):
    mechanism_writer.write_mechanism_header(tmp_path, SPECIES, REACTIONS)
    result = (tmp_path / "mechanism.h").read_text()
    assert "set_same_initial_conditions" not in result
    assert "apply_mask" not in result
    assert "apply_reverse_mask" not in result
    assert "RATES_TEST" not in result
    assert "string.h" not in result


def test_write_header_defines_conp_not_conv(tmp_path):
    mechanism_writer.write_header(tmp_path)
    result = (tmp_path / "header.h").read_text()
    assert "#define CONP\n" in result
    assert "//#define CONV\n" in result
    assert result == (
        "#ifndef HEAD\n"
        "#define HEAD\n"
        "#include <stdlib.h>\n"
        "#include <math.h>\n"
        "/** Constant pressure or volume. */\n"
        "#define CONP\n"
        "//#define CONV\n"
        "\n"
        "/** Include mechanism header to get NSP and NN **/\n"
        '#include "mechanism.h"\n'
        "// OpenMP\n"
        "#ifdef _OPENMP\n"
        " #include <omp.h>\n"
        "#else\n"
        " #define omp_get_max_threads() 1\n"
        " #define omp_get_num_threads() 1\n"
        "#endif\n"
        "#endif\n"
    )
