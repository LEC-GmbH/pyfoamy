import math
import shutil
import subprocess

import pytest

from pyfoamy.pyjac.records import GAS_CONSTANT_SI, Reaction, Species
from pyfoamy.pyjac import rates

# Toy mechanism with only a1 nonzero: h/u/cv/cp have closed forms that
# the tests derive independently of rates.py.
SPECIES = [
    Species(name="A", mw=2.0, low=(1.0, 0, 0, 0, 0, 0, 0),
            high=(2.0, 0, 0, 0, 0, 0, 0), t_mid=1000.0),
    Species(name="B", mw=32.0, low=(3.0, 0, 0, 0, 0, 0, 0),
            high=(4.0, 0, 0, 0, 0, 0, 0), t_mid=1000.0),
]


def _skip_unless_gcc_available():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")


def _run_c_harness(tmp_path, main_body):
    _skip_unless_gcc_available()
    (tmp_path / "header.h").write_text(
        "#ifndef HEAD\n#define HEAD\n#include <math.h>\n"
        f"#define NSP {len(SPECIES)}\n#endif\n"
    )
    includes = "".join(
        f'#include "{p.name}"\n' for p in sorted(tmp_path.glob("*.h"))
        if p.name != "header.h"
    )
    (tmp_path / "harness.c").write_text(
        "#include <stdio.h>\n" + includes
        + f"int main() {{\n{main_body}\n  return 0;\n}}\n"
    )
    sources = [
        str(p) for p in tmp_path.glob("*.c") if p.name != "harness.c"
    ]
    binary = tmp_path / "run"
    subprocess.run(
        ["gcc", "-O2", "-o", str(binary), str(tmp_path / "harness.c"),
         *sources, "-lm", "-I", str(tmp_path)],
        check=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if line
    )


def test_write_mass_mole_round_trip(tmp_path):
    rates.write_mass_mole(tmp_path, SPECIES)
    values = _run_c_harness(
        tmp_path,
        '  double X[1] = {0.5}, Y[1], X2[1];\n'
        '  mole2mass(X, Y);\n'
        '  printf("Y0=%.17g\\n", Y[0]);\n'
        '  mass2mole(Y, X2);\n'
        '  printf("X2_0=%.17g\\n", X2[0]);\n'
        '  printf("rho=%.17g\\n", getDensity(300.0, 101325.0, X));\n',
    )
    mw_avg = 0.5 * 2.0 + 0.5 * 32.0  # 17.0
    expected_y0 = 0.5 * 2.0 / mw_avg
    expected_rho = 101325.0 * mw_avg / (GAS_CONSTANT_SI * 300.0)
    assert float(values["Y0"]) == pytest.approx(expected_y0)
    assert float(values["X2_0"]) == pytest.approx(0.5)
    assert float(values["rho"]) == pytest.approx(expected_rho)


def test_write_chem_utils_thermo_properties(tmp_path):
    rates.write_chem_utils(tmp_path, SPECIES)
    values = _run_c_harness(
        tmp_path,
        '  double h[2], u[2], cv[2], cp[2];\n'
        '  eval_h(500.0, h); eval_u(500.0, u);\n'
        '  eval_cv(500.0, cv); eval_cp(500.0, cp);\n'
        '  printf("h_lo=%.17g\\n", h[0]);\n'
        '  printf("u_lo=%.17g\\n", u[0]);\n'
        '  printf("cv_lo=%.17g\\n", cv[0]);\n'
        '  printf("cp_lo=%.17g\\n", cp[0]);\n'
        '  eval_h(1500.0, h); eval_u(1500.0, u);\n'
        '  eval_cv(1500.0, cv); eval_cp(1500.0, cp);\n'
        '  printf("h_hi=%.17g\\n", h[0]);\n'
        '  printf("u_hi=%.17g\\n", u[0]);\n'
        '  printf("cv_hi=%.17g\\n", cv[0]);\n'
        '  printf("cp_hi=%.17g\\n", cp[0]);\n',
    )
    # Every NASA-7 coefficient except a1 is zero, so:
    #   h(T)  = RU/mw * a1 * T      u(T)  = RU/mw * (a1 - 1) * T
    #   cp(T) = RU/mw * a1          cv(T) = RU/mw * (a1 - 1)
    ru_mw = GAS_CONSTANT_SI / 2.0
    a1_lo, a1_hi = 1.0, 2.0  # species A's low/high a1 (T=500 <= t_mid < 1500)
    assert float(values["h_lo"]) == pytest.approx(ru_mw * a1_lo * 500.0)
    assert float(values["u_lo"]) == pytest.approx(
        ru_mw * (a1_lo - 1.0) * 500.0
    )
    assert float(values["cp_lo"]) == pytest.approx(ru_mw * a1_lo)
    assert float(values["cv_lo"]) == pytest.approx(ru_mw * (a1_lo - 1.0))
    assert float(values["h_hi"]) == pytest.approx(ru_mw * a1_hi * 1500.0)
    assert float(values["u_hi"]) == pytest.approx(
        ru_mw * (a1_hi - 1.0) * 1500.0
    )
    assert float(values["cp_hi"]) == pytest.approx(ru_mw * a1_hi)
    assert float(values["cv_hi"]) == pytest.approx(ru_mw * (a1_hi - 1.0))


def test_write_chem_utils_eval_conc_matches_manual_density(tmp_path):
    rates.write_chem_utils(tmp_path, SPECIES)
    rates.write_mass_mole(tmp_path, SPECIES)
    values = _run_c_harness(
        tmp_path,
        '  double X[1] = {0.5}, Y[1];\n'
        '  mole2mass(X, Y);\n'
        '  double y_N, mw_avg, rho, conc[2];\n'
        '  eval_conc(300.0, 101325.0, Y, &y_N, &mw_avg, &rho, conc);\n'
        '  printf("rho=%.17g\\n", rho);\n'
        '  printf("conc0=%.17g\\n", conc[0]);\n'
        '  printf("conc1=%.17g\\n", conc[1]);\n',
    )
    mw_avg = 0.5 * 2.0 + 0.5 * 32.0
    expected_rho = 101325.0 * mw_avg / (GAS_CONSTANT_SI * 300.0)
    assert float(values["rho"]) == pytest.approx(expected_rho)
    # sum(rho * Y_i / mw_i) = rho / mw_avg
    total = float(values["conc0"]) + float(values["conc1"])
    assert total == pytest.approx(expected_rho / mw_avg)


def _arrhenius_value(A, b, E, T):
    return A * T ** b * math.exp(-E / T)


# Second toy mechanism for write_rxn_rates; a1-only keeps Kc tractable.
RXN_SPECIES = [
    Species(name="A", mw=2.0, low=(1.0, 0, 0, 0, 0, 0, 0),
            high=(2.0, 0, 0, 0, 0, 0, 0), t_mid=1000.0),
    Species(name="B", mw=32.0, low=(3.0, 0, 0, 0, 0, 0, 0),
            high=(4.0, 0, 0, 0, 0, 0, 0), t_mid=1000.0),
]


def test_write_rxn_rates_irreversible(tmp_path):
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=False,
    )
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    values = _run_c_harness(
        tmp_path,
        '  double C[2] = {2.0, 0.0};\n'
        '  double fwd[1], rev[1];\n'
        '  eval_rxn_rates(500.0, 101325.0, C, fwd, rev);\n'
        '  printf("fwd0=%.17g\\n", fwd[0]);\n',
    )
    expected_kf = _arrhenius_value(1e10, 0.5, 2000.0, 500.0)
    assert float(values["fwd0"]) == pytest.approx(2.0 * expected_kf)


def test_write_rxn_rates_reversible_equilibrium(tmp_path):
    # A <=> B without reverse Arrhenius: kr = kf/Kc from NASA-7.
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=True,
    )
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    T = 500.0
    values = _run_c_harness(
        tmp_path,
        '  double C[2] = {2.0, 3.0};\n'
        '  double fwd[1], rev[1];\n'
        f'  eval_rxn_rates({T}, 101325.0, C, fwd, rev);\n'
        '  printf("fwd0=%.17g\\n", fwd[0]);\n'
        '  printf("rev0=%.17g\\n", rev[0]);\n',
    )
    # With only a1 nonzero each species adds nu * (-a1 + (a1-1) logT);
    # sum_nu = 0, so the (P_std/R)**sum_nu prefactor is 1.
    a1_a, a1_b = 1.0, 3.0  # species A/B's low-range a1 (T=500 <= t_mid)
    log_t = math.log(T)
    kc_arg = (a1_a - a1_b) + (a1_b - a1_a) * log_t
    kc = math.exp(kc_arg)
    kf = _arrhenius_value(1e10, 0.5, 2000.0, T)
    assert float(values["fwd0"]) == pytest.approx(2.0 * kf)
    assert float(values["rev0"]) == pytest.approx(3.0 * kf / kc)


def test_write_rxn_rates_pressure_dependent_header_declares_pres_mod(
    tmp_path,
):
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, reversible=False, third_body=True,
    )
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    header = (tmp_path / "rates.h").read_text()
    assert "get_rxn_pres_mod" in header


def test_write_rxn_rates_no_pressure_dependence_omits_pres_mod(tmp_path):
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, reversible=False,
    )
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    header = (tmp_path / "rates.h").read_text()
    assert "get_rxn_pres_mod" not in header


def test_write_rxn_pressure_mod_plain_third_body(tmp_path):
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, third_body=True,
        third_body_efficiencies=[(1, 2.0), (0, 0.5)],
    )
    # rates.h is write_rxn_rates()'s job.
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    rates.write_rxn_pressure_mod(tmp_path, [reaction])
    T, pres = 500.0, 101325.0
    values = _run_c_harness(
        tmp_path,
        '  double C[2] = {3.0, 4.0};\n'
        '  double pres_mod[1];\n'
        f'  get_rxn_pres_mod({T}, {pres}, C, pres_mod);\n'
        '  printf("pm0=%.17g\\n", pres_mod[0]);\n',
    )
    m = pres / (GAS_CONSTANT_SI * T)
    # m + (eff - 1) * C[isp] per non-default efficiency.
    expected = m + (2.0 - 1.0) * 4.0 + (0.5 - 1.0) * 3.0
    assert float(values["pm0"]) == pytest.approx(expected)


def test_write_rxn_pressure_mod_plain_third_body_explicit_partner(
    tmp_path,
):
    # Third-body reaction with one explicit collision partner instead
    # of "M": pres_mod is that species' concentration, not m.
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, third_body=True, pdep_species=1,
    )
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    rates.write_rxn_pressure_mod(tmp_path, [reaction])
    T, pres = 500.0, 101325.0
    values = _run_c_harness(
        tmp_path,
        '  double C[2] = {3.0, 4.0};\n'
        '  double pres_mod[1];\n'
        f'  get_rxn_pres_mod({T}, {pres}, C, pres_mod);\n'
        '  printf("pm0=%.17g\\n", pres_mod[0]);\n',
    )
    assert float(values["pm0"]) == pytest.approx(4.0)


def test_write_rxn_pressure_mod_lindemann_falloff(tmp_path):
    # No troe_params/sri_params - the simple (F = 1) Lindemann blend.
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=6e11, b=0.0, E=0.0, pressure_dependent=True,
        third_body_efficiencies=[(1, 3.0)],
        low_rate_params=(1e10, 0.5, 1000.0),
    )
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    rates.write_rxn_pressure_mod(tmp_path, [reaction])
    T, pres = 500.0, 101325.0
    values = _run_c_harness(
        tmp_path,
        '  double C[2] = {3.0, 4.0};\n'
        '  double pres_mod[1];\n'
        f'  get_rxn_pres_mod({T}, {pres}, C, pres_mod);\n'
        '  printf("pm0=%.17g\\n", pres_mod[0]);\n',
    )
    m = pres / (GAS_CONSTANT_SI * T)
    thd = m + (3.0 - 1.0) * 4.0
    k0 = _arrhenius_value(1e10, 0.5, 1000.0, T)
    kinf = _arrhenius_value(6e11, 0.0, 0.0, T)  # falls back to A/b/E
    pr = k0 * thd / kinf
    assert float(values["pm0"]) == pytest.approx(pr / (1.0 + pr))


def test_write_spec_rates_routes_last_species_through_dy_n(tmp_path):
    # C is untouched and last: exercises the zero-fill and the dy_N
    # path besides the ordinary sp_rates[] assignment.
    species3 = RXN_SPECIES + [
        Species(name="C", mw=12.0, low=(4.0, 0, 0, 0, 0, 0, 0),
                high=(4.5, 0, 0, 0, 0, 0, 0), t_mid=1000.0),
    ]
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, reversible=True,
    )
    rates.write_rxn_rates(tmp_path, species3, [reaction])
    touched = rates.write_spec_rates(tmp_path, species3, [reaction])
    assert touched == [True, True, False]

    values = _run_c_harness(
        tmp_path,
        '  double fwd[1] = {2.0}, rev[1] = {0.5};\n'
        '  double sp_rates[2], dy_N;\n'
        '  eval_spec_rates(fwd, rev, fwd, sp_rates, &dy_N);\n'
        '  printf("spA=%.17g\\n", sp_rates[0]);\n'
        '  printf("spB=%.17g\\n", sp_rates[1]);\n'
        '  printf("dyN=%.17g\\n", dy_N);\n',
    )
    net = 2.0 - 0.5
    assert float(values["spA"]) == pytest.approx(-net)  # A: net reactant
    assert float(values["spB"]) == pytest.approx(net)   # B: net product
    assert float(values["dyN"]) == pytest.approx(0.0)   # C: never touched


def test_write_derivs_debug_writes_the_state_into_foam_case(
    tmp_path, monkeypatch,
):
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=False,
    )
    rates.write_mass_mole(tmp_path, RXN_SPECIES)
    rates.write_chem_utils(tmp_path, RXN_SPECIES)
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    touched = rates.write_spec_rates(tmp_path, RXN_SPECIES, [reaction])
    rates.write_derivs(tmp_path, RXN_SPECIES, [reaction], touched, debug=True)
    case = tmp_path / "case"
    case.mkdir()
    monkeypatch.setenv("FOAM_CASE", str(case))
    _run_c_harness(
        tmp_path,
        "  double y[2] = {500.0, 0.4};\n"
        "  double dy[2];\n"
        "  dydt(0.0, 101325.0, y, dy);\n",
    )
    (log,) = case.glob("pyjac_debug_state_*.log")
    assert log.read_text().startswith("SOURCE=dydt T=500 ")


def test_write_derivs_irreversible_reaction(tmp_path):
    # A -> B irreversible, re-derived with the closed forms checked by
    # the tests above.
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=False,
    )
    rates.write_mass_mole(tmp_path, RXN_SPECIES)
    rates.write_chem_utils(tmp_path, RXN_SPECIES)
    rates.write_rxn_rates(tmp_path, RXN_SPECIES, [reaction])
    touched = rates.write_spec_rates(tmp_path, RXN_SPECIES, [reaction])
    rates.write_derivs(tmp_path, RXN_SPECIES, [reaction], touched)

    T, pres, Y_A = 500.0, 101325.0, 0.4
    values = _run_c_harness(
        tmp_path,
        f'  double y[2] = {{{T}, {Y_A}}};\n'
        '  double dy[2];\n'
        f'  dydt(0.0, {pres}, y, dy);\n'
        '  printf("dy0=%.17g\\n", dy[0]);\n'
        '  printf("dy1=%.17g\\n", dy[1]);\n',
    )

    mw_a, mw_b = 2.0, 32.0
    a1_a, a1_b = 1.0, 3.0  # species A/B's low-range a1 (T=500 <= t_mid)
    y_n = 1.0 - Y_A
    mw_avg = 1.0 / (Y_A / mw_a + y_n / mw_b)
    rho = pres * mw_avg / (GAS_CONSTANT_SI * T)
    conc_a = rho * Y_A / mw_a
    kf = _arrhenius_value(1e10, 0.5, 2000.0, T)
    fwd0 = conc_a * kf
    dy1_molar = -fwd0  # A: net reactant
    dyN_molar = fwd0  # B: net product
    cp_a = GAS_CONSTANT_SI / mw_a * a1_a
    cp_b = GAS_CONSTANT_SI / mw_b * a1_b
    cp_avg = cp_a * Y_A + cp_b * y_n
    h_a = GAS_CONSTANT_SI / mw_a * a1_a * T
    h_b = GAS_CONSTANT_SI / mw_b * a1_b * T
    expected_dy0 = (-1.0 / (rho * cp_avg)) * (
        dy1_molar * h_a * mw_a + dyN_molar * h_b * mw_b
    )
    expected_dy1 = dy1_molar * (mw_a / rho)

    assert float(values["dy0"]) == pytest.approx(expected_dy0)
    assert float(values["dy1"]) == pytest.approx(expected_dy1)
