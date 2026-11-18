import shutil
import subprocess

import pytest

from pyfoamy.pyjac.records import (
    ChebyshevRate, GAS_CONSTANT_SI, Reaction, Species, STANDARD_PRESSURE,
)
from pyfoamy.pyjac import jacobian, rates


def _skip_unless_gcc_available():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")


def _dbdt_species_expr(index, coeffs):
    a0, a1, a2, a3, a4, a5, _a6 = coeffs
    return (
        f"dBdT[{index}] = ({a0 - 1.0:.16e} + {a5:.16e} / T) / T"
        f" + {a1 / 2.0:.16e} + T * ({a2 / 3.0:.16e}"
        f" + T * ({a3 / 4.0:.16e} + {a4 / 5.0:.16e} * T))"
    )


def test_write_rates_irreversible():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=False,
    )
    result = jacobian.write_rates(reaction)
    assert "kf = " in result
    assert "kr" not in result


def test_write_rates_reversible_uses_equilibrium_constant():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=True,
    )
    result = jacobian.write_rates(reaction)
    assert "kf = " in result
    assert result.strip().endswith("kr = kf / Kc;")


def test_write_rates_explicit_reverse_params():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.5, E=2000.0, reversible=True,
        reverse_params=(5e8, 0.2, 500.0),
    )
    result = jacobian.write_rates(reaction)
    lines = result.strip().splitlines()
    assert lines[0].strip().startswith("kf = ")
    assert lines[1].strip().startswith("kr = ")
    assert "Kc" not in result


def test_falloff_low_over_high_with_low_rate_params():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=6e11, b=0.0, E=0.0, pressure_dependent=True,
        low_rate_params=(1e10, 0.0, 0.0),
    )
    beta, e, k0_over_kinf = jacobian.falloff_low_over_high(reaction)
    assert beta == pytest.approx(0.0)
    assert e == pytest.approx(0.0)
    # b=E=0: forward_rate_expr() takes the plain-A branch.
    assert k0_over_kinf == str(1e10 / 6e11)


def test_falloff_low_over_high_with_high_rate_params():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, pressure_dependent=True,
        high_rate_params=(6e11, 0.0, 0.0),
    )
    beta, e, k0_over_kinf = jacobian.falloff_low_over_high(reaction)
    assert beta == pytest.approx(0.0)
    assert e == pytest.approx(0.0)
    assert k0_over_kinf == str(1e10 / 6e11)


def test_falloff_low_over_high_requires_a_rate():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, pressure_dependent=True,
    )
    with pytest.raises(ValueError, match="low_rate_params"):
        jacobian.falloff_low_over_high(reaction)


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            (1e10, 1.5, 500.0),
            "1.5000000000000000e+00 + (5.0000000000000000e+02 / T)",
        ),
        ((1e10, 1.5, 0.0), "1.5000000000000000e+00"),
        ((1e10, 0.0, 500.0), "(5.0000000000000000e+02 / T)"),
        ((1e10, 0.0, 0.0), ""),
    ],
)
def test_rate_constant_temperature_expr(params, expected):
    assert jacobian.rate_constant_temperature_expr(params) == expected


def test_write_pr_uses_pdep_species_concentration():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pdep_species=3,
    )
    body, conc_temp_log = jacobian.write_pr(reaction)
    assert body == "  conc_temp = conc[3];\n"
    assert conc_temp_log is None


def test_write_pr_plain_third_body_uses_m():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True,
    )
    body, conc_temp_log = jacobian.write_pr(reaction)
    assert body == "  conc_temp = m;\n"
    assert conc_temp_log is None


def test_write_pr_falloff_writes_pr_line():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=6e11, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        third_body_efficiencies=[(2, 2.0), (3, 0.5)],
        low_rate_params=(1e10, 0.0, 0.0),
    )
    body, conc_temp_log = jacobian.write_pr(reaction)
    assert body == (
        "  conc_temp = (m + 1.0 * conc[2] - 0.5 * conc[3]);\n"
        f"  Pr = conc_temp * ({1e10 / 6e11});\n"
    )
    assert conc_temp_log == [(2, 1.0), (3, -0.5)]


def test_write_pr_delta_optimization_shorter_than_recompute():
    reaction_a = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True,
        third_body_efficiencies=[(2, 2.0), (3, 0.5), (4, 3.0)],
    )
    reaction_b = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True,
        third_body_efficiencies=[(2, 2.0), (3, 0.5)],
    )
    body_a, log_a = jacobian.write_pr(reaction_a)
    assert body_a == (
        "  conc_temp = (m + 1.0 * conc[2] - 0.5 * conc[3]"
        " + 2.0 * conc[4]);\n"
    )
    body_b, _log_b = jacobian.write_pr(reaction_b, log_a)
    assert body_b == "  conc_temp += (-2.0 * conc[4]);\n"


def test_write_pr_identical_efficiencies_skip_recompute():
    reaction_a = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True, third_body_efficiencies=[(2, 2.0)],
    )
    reaction_b = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True, third_body_efficiencies=[(2, 2.0)],
    )
    _body_a, log_a = jacobian.write_pr(reaction_a)
    body_b, _log_b = jacobian.write_pr(reaction_b, log_a)
    assert body_b == ""


def test_write_pr_identical_efficiencies_still_writes_pr_line():
    reaction_a = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=6e11, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        third_body_efficiencies=[(2, 2.0)],
        low_rate_params=(1e10, 0.0, 0.0),
    )
    reaction_b = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=6e11, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        third_body_efficiencies=[(2, 2.0)],
        low_rate_params=(1e10, 0.0, 0.0),
    )
    _body_a, log_a = jacobian.write_pr(reaction_a)
    body_b, _log_b = jacobian.write_pr(reaction_b, log_a)
    assert body_b == f"  Pr = conc_temp * ({1e10 / 6e11});\n"


def test_write_troe_three_params():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True, troe_params=(0.5, 100.0, 1000.0),
    )
    result = jacobian.write_troe(reaction)
    assert result == (
        "  Fcent = 5.0000000000000000e-01 * exp(T / "
        "-1.0000000000000000e+02) + 5.0000000000000000e-01 * "
        "exp(T / -1.0000000000000000e+03);\n"
        "  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * "
        "log10(fmax(Fcent, 1.0e-300)) - 0.4;\n"
        "  B = 0.806 - 1.1762 * log10(fmax(Fcent, 1.0e-300)) - "
        "0.14 * log10(fmax(Pr, 1.0e-300));\n"
        "  lnF_AB = 2.0 * log(fmax(Fcent, 1.0e-300)) * "
        "A / (B * B * B * (1.0 + A * A / (B * B)) * "
        "(1.0 + A * A / (B * B)));\n"
    )


def test_write_troe_four_params_nonzero_adds_term():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True,
        troe_params=(0.5, 100.0, 1000.0, 50.0),
    )
    result = jacobian.write_troe(reaction)
    assert "+ exp(-5.0000000000000000e+01 / T);\n" in result
    assert result.splitlines()[0].endswith(
        "+ exp(-5.0000000000000000e+01 / T);"
    )


def test_write_troe_four_params_zero_matches_three_param():
    reaction3 = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True, troe_params=(0.5, 100.0, 1000.0),
    )
    reaction4 = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True,
        troe_params=(0.5, 100.0, 1000.0, 0.0),
    )
    assert jacobian.write_troe(reaction3) == jacobian.write_troe(
        reaction4,
    )


def test_write_sri_is_a_fixed_statement():
    assert jacobian.write_sri() == (
        "  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * "
        "log10(fmax(Pr, 1.0e-300)));\n"
    )


def test_write_sri_dt_three_param():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True, sri_params=(1.1, 700.0, 1200.0),
    )
    result = jacobian.write_sri_dt(reaction, 0.5, 300.0)
    assert result == (
        " + X * (((770.0000000000001 / (T * T)) * exp(-700.0 / T)"
        " - 8.3333333333333339e-04 * exp(T / -1200.0)) / "
        "(1.1 * exp(-700.0 / T) + exp(T / -1200.0)) - "
        "X * 0.8685889638065035 * log10(fmax(Pr, 1.0e-300)) * "
        "(5.0000000000000000e-01 + (3.0000000000000000e+02 / T)"
        " - 1.0) * log(1.1 * exp(-700.0 / T) + exp(T / -1200.0))"
        " / T)"
    )


def test_write_sri_dt_five_param_adds_trailing_term():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True,
        sri_params=(1.1, 700.0, 1200.0, 1.5, 0.5),
    )
    result = jacobian.write_sri_dt(reaction, 0.5, 300.0)
    assert result.endswith(" + (0.5 / T)")


def test_write_sri_dt_five_param_zero_e_matches_three_param():
    reaction3 = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True, sri_params=(1.1, 700.0, 1200.0),
    )
    reaction5 = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True,
        sri_params=(1.1, 700.0, 1200.0, 1.5, 0.0),
    )
    assert jacobian.write_sri_dt(
        reaction3, 0.5, 300.0,
    ) == jacobian.write_sri_dt(reaction5, 0.5, 300.0)


def test_write_troe_dt_four_param():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        pressure_dependent=True,
        troe_params=(0.7, 200.0, 2000.0, 100.0),
    )
    result = jacobian.write_troe_dt(reaction, 0.8, 500.0)
    assert result == (
        " + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - "
        "lnF_AB * (-2.9097730287517870e-01 * B + "
        "5.1081716961460466e-01 * A) / Fcent) * "
        "(-1.5000000000000002e-03 * exp(T / -2.0000000000000000e+02)"
        " - 3.5000000000000000e-04 * exp(T / "
        "-2.0000000000000000e+03) + (1.0000000000000000e+02 / "
        "(T * T)) * exp(-1.0000000000000000e+02 / T))) - "
        "lnF_AB * (4.3429448190325176e-01 * B + "
        "6.0801227466455256e-02 * A) * "
        "(8.0000000000000004e-01 + (5.0000000000000000e+02 / T)"
        " - 1.0) / T"
    )


def test_get_pdep_dt_low_rate_reversible():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=6e11, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        low_rate_params=(1e10, 2.0, 500.0),
    )
    result = jacobian.get_pdep_dt(
        reaction, rxn_ind=5, rev_idx=3, pres_rxn_ind=2,
    )
    assert result == (
        "  j_temp = (pres_mod[2] * (("
        "(2.0000e+00 + (5.0000000000000000e+02 / T) - 1.0)"
        " / (T * (1.0 + Pr))))"
        " * (fwd_rates[5] - rev_rates[3])"
        " + (pres_mod[2]"
    )


def test_get_pdep_dt_high_rate_irreversible():
    reaction = Reaction(
        reactants=[0, 1], reactant_nu=[1.0, 1.0], products=[2],
        product_nu=[1.0], A=1e10, b=0.5, E=1000.0, reversible=False,
        third_body=True, pressure_dependent=True,
        high_rate_params=(2e13, -0.3, 500.0),
    )
    result = jacobian.get_pdep_dt(
        reaction, rxn_ind=7, rev_idx=-1, pres_rxn_ind=0,
    )
    assert result == (
        "  j_temp = (pres_mod[0] * ((-Pr * "
        "(8.0000e-01 + (5.0000000000000000e+02 / T) - 1.0)"
        " / (T * (1.0 + Pr))))"
        " * fwd_rates[7]"
        " + (pres_mod[0]"
    )


def test_get_pdep_dt_dispatches_to_sri():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        low_rate_params=(5e9, 0.0, 0.0),
        sri_params=(1.1, 700.0, 1200.0),
    )
    beta_0minf, e_0minf, _k = jacobian.falloff_low_over_high(reaction)
    expected_addendum = jacobian.write_sri_dt(
        reaction, beta_0minf, e_0minf,
    )
    result = jacobian.get_pdep_dt(
        reaction, rxn_ind=0, rev_idx=0, pres_rxn_ind=0,
    )
    assert expected_addendum in result
    assert "X *" in result


def test_get_pdep_dt_dispatches_to_troe():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        low_rate_params=(5e9, 0.0, 0.0),
        troe_params=(0.7, 200.0, 2000.0),
    )
    beta_0minf, e_0minf, _k = jacobian.falloff_low_over_high(reaction)
    expected_addendum = jacobian.write_troe_dt(
        reaction, beta_0minf, e_0minf,
    )
    result = jacobian.get_pdep_dt(
        reaction, rxn_ind=0, rev_idx=0, pres_rxn_ind=0,
    )
    assert expected_addendum in result


def test_get_pdep_dt_plain_lindemann_has_no_addendum():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        low_rate_params=(5e9, 0.0, 0.0),
    )
    result = jacobian.get_pdep_dt(
        reaction, rxn_ind=0, rev_idx=0, pres_rxn_ind=0,
    )
    assert "X" not in result
    assert "Fcent" not in result


def _dcp_term(y_str, mw, a1, a2, a3, a4):
    return (
        f"({y_str} * {8314.4621 / mw:.16e} * "
        f"({a1:.16e} + T * ({2.0 * a2:.16e}"
        f" + T * ({3.0 * a3:.16e} + {4.0 * a4:.16e} * T))))"
    )


def test_write_dcp_dt_single_bucket_two_species():
    species = [
        Species(
            "A", 2.0, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            (11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0), t_mid=1000.0,
        ),
        Species(
            "B", 4.0, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7), t_mid=1000.0,
        ),
    ]
    result = jacobian.write_dcp_dt(species)
    low_a = _dcp_term("y[1]", 2.0, 2.0, 3.0, 4.0, 5.0)
    low_b = _dcp_term("y_N", 4.0, 0.2, 0.3, 0.4, 0.5)
    high_a = _dcp_term("y[1]", 2.0, 12.0, 13.0, 14.0, 15.0)
    high_b = _dcp_term("y_N", 4.0, 1.2, 1.3, 1.4, 1.5)
    assert result == (
        "  if (T <= 1000.0) {\n"
        f"    working_temp = {low_a}\n    + {low_b};\n"
        "  } else {\n"
        f"    working_temp = {high_a}\n    + {high_b};\n"
        "  }\n\n"
    )


def test_write_dcp_dt_second_bucket_uses_plus_equals():
    species = [
        Species("A", 1.0, (0.0,) * 7, (0.0,) * 7, t_mid=1000.0),
        Species("B", 1.0, (0.0,) * 7, (0.0,) * 7, t_mid=2000.0),
    ]
    result = jacobian.write_dcp_dt(species)
    first_block, second_block = result.split("if (T <= 2000.0)")
    assert "working_temp = " in first_block
    assert "+=" not in first_block
    assert "working_temp += " in second_block


def test_get_elementary_rxn_dt_explicit_reverse_params():
    reaction = Reaction(
        reactants=[0], reactant_nu=[2.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.5, E=200.0, reversible=True,
        reverse_params=(5e8, 0.2, 500.0),
    )
    result = jacobian.get_elementary_rxn_dt(reaction, 3, 1)
    # product_nu sums to 1, so only the forward side gets a
    # nu-correction term.
    assert result == (
        "fwd_rates[3] * (5.0000000000000000e-01"
        " + (2.0000000000000000e+02 / T) + -1.0)"
        " - rev_rates[1] * (2.0000000000000001e-01"
        " + (5.0000000000000000e+02 / T))"
        ")) * rho_inv;\n"
    )


def test_get_elementary_rxn_dt_explicit_reverse_params_trivial():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        reverse_params=(5e8, 0.0, 0.0),
    )
    assert jacobian.get_elementary_rxn_dt(reaction, 0, 0) == ""


def test_get_elementary_rxn_dt_equilibrium_reversible():
    reaction = Reaction(
        reactants=[0], reactant_nu=[2.0], products=[1, 2],
        product_nu=[1.0, 3.0], A=1e10, b=0.3, E=150.0,
        reversible=True,
    )
    result = jacobian.get_elementary_rxn_dt(reaction, 5, 2)
    assert result == (
        "(fwd_rates[5] - rev_rates[2]) * "
        "(2.9999999999999999e-01 + (1.5000000000000000e+02 / T))"
        " + fwd_rates[5] * -1.0"
        " - rev_rates[2] * (-3.0 + -T * "
        "(dBdT[1] + 3.0 * dBdT[2] - 2.0 * dBdT[0])"
        "))) * rho_inv;\n"
    )


def test_get_elementary_rxn_dt_irreversible():
    reaction = Reaction(
        reactants=[0], reactant_nu=[2.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.5, E=200.0, reversible=False,
    )
    result = jacobian.get_elementary_rxn_dt(reaction, 8, -1)
    assert result == (
        "fwd_rates[8] * (5.0000000000000000e-01"
        " + (2.0000000000000000e+02 / T) + -1.0)"
        ")) * rho_inv;\n"
    )


def test_get_elementary_rxn_dt_irreversible_trivial():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=False,
    )
    assert jacobian.get_elementary_rxn_dt(reaction, 0, -1) == ""


def test_write_cheb_ut_smallest_fit_has_no_finishing_loops():
    cheb = ChebyshevRate(
        t_range=(300.0, 2500.0), p_range=(1000.0, 1.0e7),
        coefficients=[[0.0, 0.0], [2.0, 3.0]],
    )
    result = jacobian.write_cheb_ut(cheb)
    assert result == (
        "  cheb_temp_0 = 1;\n"
        "  cheb_temp_1 = Pred;\n"
        "  dot_prod[1]= 2.0000000000000000e+00"
        " + Pred * 3.0000000000000000e+00;\n"
        "  cheb_temp_0 = 1.0;\n"
        "  cheb_temp_1 = 2.0 * Tred;\n"
        "  kf = dot_prod[1] + 2.0 * Tred * dot_prod[2];\n"
    )


def test_write_cheb_ut_exercises_both_finishing_loops():
    cheb = ChebyshevRate(
        t_range=(300.0, 2500.0), p_range=(1000.0, 1.0e7),
        coefficients=[
            [0.0, 0.0, 0.0], [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
        ],
    )
    result = jacobian.write_cheb_ut(cheb)
    assert result == (
        "  cheb_temp_0 = 1;\n"
        "  cheb_temp_1 = Pred;\n"
        "  dot_prod[1]= 1.0000000000000000e+00"
        " + Pred * 2.0000000000000000e+00;\n"
        "  dot_prod[2]= 8.0000000000000000e+00"
        " + Pred * 1.0000000000000000e+01;\n"
        "  dot_prod[3]= 2.1000000000000000e+01"
        " + Pred * 2.4000000000000000e+01;\n"
        "  cheb_temp_0 = 2 * Pred * cheb_temp_1 - cheb_temp_0;\n"
        "  dot_prod[1] += 3.0000000000000000e+00 * cheb_temp_0;\n"
        "  dot_prod[2] += 1.2000000000000000e+01 * cheb_temp_0;\n"
        "  dot_prod[3] += 2.7000000000000000e+01 * cheb_temp_0;\n"
        "  cheb_temp_0 = 1.0;\n"
        "  cheb_temp_1 = 2.0 * Tred;\n"
        "  kf = dot_prod[1] + 2.0 * Tred * dot_prod[2];\n"
        "  cheb_temp_0 = 2.0 * Tred * cheb_temp_1 - cheb_temp_0;\n"
        "  kf += dot_prod[3] * cheb_temp_0;\n"
    )


def test_write_cheb_rxn_dt_reversible():
    cheb = ChebyshevRate(
        t_range=(300.0, 2500.0), p_range=(1000.0, 1.0e6),
        coefficients=[[0.0, 0.0], [1.0, 2.0]],
    )
    reaction = Reaction(
        reactants=[0], reactant_nu=[2.0], products=[1, 2],
        product_nu=[1.0, 1.0], A=1.0, b=0.0, E=0.0, reversible=True,
        chebyshev=cheb,
    )
    result = jacobian.write_cheb_rxn_dt(
        "  jac[7] += (", reaction, 3, 1,
    )
    assert result == (
        "  Tred = ((2.0 / T) - 3.7333333333333337e-03)"
        " / -2.9333333333333334e-03;\n"
        "  Pred = (2.0 * log10(pres) - 9.0000000000000000e+00)"
        " / 3.0000000000000000e+00;\n"
        "  cheb_temp_0 = 1;\n"
        "  cheb_temp_1 = Pred;\n"
        "  dot_prod[1]= 1.0000000000000000e+00"
        " + Pred * 2.0000000000000000e+00;\n"
        "  cheb_temp_0 = 1.0;\n"
        "  cheb_temp_1 = 2.0 * Tred;\n"
        "  kf = dot_prod[1] + 2.0 * Tred * dot_prod[2];\n"
        "  jac[7] += (kf * (1.5699443815868494e+03 / T)"
        " * (fwd_rates[3] - rev_rates[1])"
        " + fwd_rates[3] * -1.0"
        " - rev_rates[1] * (-1.0 + -T * "
        "(dBdT[1] + dBdT[2] - 2.0 * dBdT[0])"
        "))) * rho_inv;\n"
    )


def test_write_cheb_rxn_dt_irreversible_omits_rev_terms():
    cheb = ChebyshevRate(
        t_range=(300.0, 2500.0), p_range=(1000.0, 1.0e6),
        coefficients=[[0.0, 0.0], [1.0, 2.0]],
    )
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1.0, b=0.0, E=0.0, reversible=False,
        chebyshev=cheb,
    )
    result = jacobian.write_cheb_rxn_dt("  jac[0] += (", reaction, 0, -1)
    assert "rev_rates" not in result
    assert "dBdT" not in result
    assert result.endswith(
        " * (fwd_rates[0])"
        ")) * rho_inv;\n"
    )


def test_write_plog_rxn_dt_irreversible_two_points():
    reaction = Reaction(
        reactants=[0], reactant_nu=[2.0], products=[1],
        product_nu=[1.0], A=-1.0, b=-1.0, E=-1.0, reversible=False,
        plog=[
            (1.0e3, 1.0e9, 0.0, 100.0),
            (1.0e6, 5.0e9, 0.4, 250.0),
        ],
    )
    result = jacobian.write_plog_rxn_dt(
        "  jac[1] += (", reaction, 1, -1,
    )
    assert result == (
        "  if (pres <= 1.0000e+03) {\n"
        "    jac[1] += (fwd_rates[1] * "
        "((1.0000000000000000e+02 / T) + -1.0))) * rho_inv;\n"
        "  } else if ((pres > 1.0000e+03)"
        " && (pres <= 1.0000e+06)) {\n"
        "    jac[1] += ((1.0000000000000000e+02 / T + "
        "(4.0000000000000002e-01 + 1.5000000000000000e+02 / T)"
        " * (log(pres) - 6.9077552789821368e+00)"
        " / 6.9077552789821368e+00) * (fwd_rates[1])"
        " + fwd_rates[1] * -1.0)) * rho_inv;\n"
        "  } else if (pres > 1.0000e+06) {\n"
        "    jac[1] += (fwd_rates[1] * "
        "(4.0000000000000002e-01 + (2.5000000000000000e+02 / T)"
        " + -1.0))) * rho_inv;\n"
        "  }\n"
    )


def test_write_plog_rxn_dt_boundaries_ignore_reverse_params():
    # Boundary-point reactions never get `reverse_params` (as in
    # ReacInfo()), so outside the PLOG range the dBdT path is used;
    # p1 == 1.0 exercises the "skip log(p1)" branch.
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=-1.0, b=-1.0, E=-1.0, reversible=True,
        reverse_params=(5.0e8, 0.1, 150.0),
        plog=[
            (1.0, 1.0e10, 0.5, 200.0),
            (1.0e6, 2.0e10, 0.2, 350.0),
        ],
    )
    result = jacobian.write_plog_rxn_dt(
        "  jac[3] += (", reaction, 3, 2,
    )
    assert "dBdT" in result
    assert result.count("dBdT") == 6  # once per branch (3 branches)
    assert "log(pres) -" not in result
    assert result.startswith("  if (pres <= 1.0000e+00) {\n")


def test_write_plog_rxn_dt_negative_a_ratio_not_implemented():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=-1.0, b=-1.0, E=-1.0, reversible=False,
        plog=[
            (1.0e3, 1.0e10, 0.5, 200.0),
            (1.0e6, -2.0e10, 0.2, 350.0),
        ],
    )
    with pytest.raises(NotImplementedError):
        jacobian.write_plog_rxn_dt("  jac[0] += (", reaction, 0, -1)


def test_write_plog_rxn_dt_undefined_derivative_raises():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=-1.0, b=-1.0, E=-1.0, reversible=False,
        plog=[
            (1.0e3, 1.0e10, 0.0, 0.0),
            (1.0e6, 1.0e10, 0.0, 0.0),
        ],
    )
    with pytest.raises(ValueError, match="undefined"):
        jacobian.write_plog_rxn_dt("  jac[0] += (", reaction, 0, -1)


def test_write_dt_completion_j_nplusone_touched():
    species = [
        Species("A", 2.0, (0.0,) * 7, (0.0,) * 7),
        Species("B", 4.0, (0.0,) * 7, (0.0,) * 7),
    ]
    result = jacobian.write_dt_completion(species, True)
    assert result == (
        "  //Complete dT wrt T calculations\n"
        "  jac[0] = -("
        "spec_rates[0] * 2.00000000e+00 * "
        "(-working_temp * h[0] / cp_avg + cp[0])"
        " + jac[1] * h[0] * rho"
        "\n    + spec_rates[1] * 4.00000000e+00 * "
        "(-working_temp * h[1] / cp_avg + cp[1])"
        " + J_nplusone[0] * h[1] * rho"
        ") / (rho * cp_avg);\n"
    )


def test_write_dt_completion_j_nplusone_not_touched():
    species = [
        Species("A", 2.0, (0.0,) * 7, (0.0,) * 7),
        Species("B", 4.0, (0.0,) * 7, (0.0,) * 7),
    ]
    result = jacobian.write_dt_completion(species, False)
    assert result == (
        "  //Complete dT wrt T calculations\n"
        "  jac[0] = -("
        "spec_rates[0] * 2.00000000e+00 * "
        "(-working_temp * h[0] / cp_avg + cp[0])"
        " + jac[1] * h[0] * rho"
        "\n    + spec_rates[1] * 4.00000000e+00 * "
        "(-working_temp * h[1] / cp_avg + cp[1])"
        ") / (rho * cp_avg);\n"
    )
    # Only the last species' J_nplusone addend depends on
    # j_nplusone_touched.
    assert "jac[1] * h[0] * rho" in result
    assert "J_nplusone" not in result


_DR_DY_SPECIES = [
    Species("A", 2.0, (0.0,) * 7, (0.0,) * 7),
    Species("B", 4.0, (0.0,) * 7, (0.0,) * 7),
    Species("C", 6.0, (0.0,) * 7, (0.0,) * 7),
]


def test_write_dr_dy_plain_elementary_has_no_pres_mod_temp():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    result = jacobian.write_dr_dy(reaction, 4, 2, -1)
    assert result == (
        "  j_temp = -mw_avg * rho_inv * "
        "(fwd_rates[4] - rev_rates[2]);\n"
    )
    assert "pres_mod_temp" not in result


def test_write_dr_dy_third_body_only():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        third_body=True, third_body_efficiencies=[(2, 2.0)],
    )
    result = jacobian.write_dr_dy(reaction, 4, 2, 1)
    assert result == (
        "  pres_mod_temp = (fwd_rates[4] - rev_rates[2]);\n"
        "  j_temp = -mw_avg * rho_inv * pres_mod[1] * "
        "(2.0 * fwd_rates[4] - 2.0 * rev_rates[2]);\n"
    )
    assert "pres_mod_temp *=" not in result


def test_write_dr_dy_troe_falloff_writes_all_three_lines():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=6e11, b=0.0, E=0.0, reversible=True,
        third_body=True, pressure_dependent=True,
        third_body_efficiencies=[(2, 2.0)],
        low_rate_params=(1e10, 0.0, 0.0),
        troe_params=(0.7, 200.0, 2000.0),
    )
    result = jacobian.write_dr_dy(reaction, 4, 2, 1)
    lines = result.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].strip().startswith("pres_mod_temp = ((1.0 / (1.0 + Pr))")
    assert "Fcent" in lines[0]
    assert lines[1].strip().startswith("j_temp = ")
    assert lines[2].strip().startswith("pres_mod_temp *=")
    assert "pow(Fcent, 1.0 / (1 + A * A / (B * B)))" in lines[2]


def test_write_dr_dy_species_uninvolved_species_has_no_s_term():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    result = jacobian.write_dr_dy_species(
        _DR_DY_SPECIES, reaction, -1, 2, _DR_DY_SPECIES[2],
    )
    # Species 2 is last (mw_frac = 1) and not in the reaction.
    assert result == "j_temp * 0.0000000000000000e+00"


def test_write_dr_dy_species_simple_reactant_appends_kf():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    result = jacobian.write_dr_dy_species(
        _DR_DY_SPECIES, reaction, -1, 0, _DR_DY_SPECIES[0],
    )
    assert result == "j_temp * 6.6666666666666674e-01 + kf"


def _kc_terms(nu, coeffs):
    a0, a1, a2, a3, a4, a5, a6 = coeffs
    return [
        nu * (a6 - a0), nu * (a0 - 1.0), nu * a1 / 2.0, nu * a2 / 6.0,
        nu * a3 / 12.0, nu * a4 / 20.0, nu * a5,
    ]


def _kc_bucket_expr(terms):
    c0, c1, c2, c3, c4, c5, c6 = terms
    return (
        f"({c0:.16e} + {c1:.16e} * logT + T * ({c2:.16e}"
        f" + T * ({c3:.16e} + T * ({c4:.16e} + {c5:.16e} * T))) - "
        f"{c6:.16e} / T)"
    )


def test_write_kc_single_bucket_net_zero_nu():
    low_a = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    high_a = (11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0)
    low_b = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    high_b = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
    species = [
        Species("A", 2.0, low_a, high_a, t_mid=1000.0),
        Species("B", 4.0, low_b, high_b, t_mid=1000.0),
    ]
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    result = jacobian.write_kc(species, reaction)

    lo_combined = [
        a + b for a, b in zip(_kc_terms(-1.0, low_a), _kc_terms(1.0, low_b))
    ]
    hi_combined = [
        a + b
        for a, b in zip(_kc_terms(-1.0, high_a), _kc_terms(1.0, high_b))
    ]
    assert result == (
        "  if (T <= 1000.0) {\n"
        f"    Kc = {_kc_bucket_expr(lo_combined)};\n"
        "  } else {\n"
        f"    Kc = {_kc_bucket_expr(hi_combined)};\n"
        "  }\n\n"
        "  Kc = exp(Kc);\n"
    )


def test_write_kc_nonzero_sum_nu_adds_pressure_scale():
    species = [
        Species("A", 2.0, (1.0,) * 7, (1.0,) * 7, t_mid=1000.0),
        Species("B", 4.0, (1.0,) * 7, (1.0,) * 7, t_mid=1000.0),
        Species("C", 6.0, (1.0,) * 7, (1.0,) * 7, t_mid=1000.0),
    ]
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1, 2],
        product_nu=[1.0, 1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    result = jacobian.write_kc(species, reaction)
    # sum_nu = -1 (reactant) + 1 + 1 (products) = 1
    num = STANDARD_PRESSURE / GAS_CONSTANT_SI
    assert result.endswith(f"  Kc = {num:.16e} * exp(Kc);\n")


_SMALL_MECHANISM_SPECIES = [
    Species(
        "H2", 2.016,
        (2.34433112, 0.00798052075, -1.9478151e-05, 2.01572094e-08,
         -7.37611761e-12, -917.935173, 0.683010238),
        (3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
         2.00255376e-14, -950.158922, -3.20502331),
        t_mid=1000.0,
    ),
    Species(
        "O2", 31.998,
        (3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09,
         3.24372837e-12, -1063.94356, 3.65767573),
        (3.66096065, 6.56365811e-04, -1.41149627e-07, 2.05797935e-11,
         -1.29913436e-15, -1215.97718, 3.41536279),
        t_mid=1000.0,
    ),
    Species(
        "H2O", 18.015,
        (4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09,
         1.77197817e-12, -30293.7267, -0.849032208),
        (2.67703787, 2.97318329e-03, -7.7376969e-07, 9.44336689e-11,
         -4.26900959e-15, -29885.8938, 6.88255571),
        t_mid=1000.0,
    ),
]
_SMALL_MECHANISM_REACTIONS = [
    Reaction(
        reactants=[0, 1], reactant_nu=[2.0, 1.0], products=[2],
        product_nu=[2.0], A=1e13, b=0.0, E=5000.0, reversible=True,
    ),
]


def test_write_jacobian_writes_files(tmp_path):
    seen_sp = [True] * len(_SMALL_MECHANISM_SPECIES)
    touched = jacobian.write_jacobian(
        tmp_path, _SMALL_MECHANISM_SPECIES, _SMALL_MECHANISM_REACTIONS,
        seen_sp,
    )
    assert (tmp_path / "jacob.h").exists()
    assert (tmp_path / "jacob.c").exists()
    assert len(touched) == len(_SMALL_MECHANISM_SPECIES) ** 2

    header = (tmp_path / "jacob.h").read_text()
    assert "void eval_jacob" in header

    body = (tmp_path / "jacob.c").read_text()
    assert "void eval_jacob (" in body
    assert body.rstrip().endswith("} // end eval_jacob")
    assert "eval_jacob_0" not in body
    assert "jacobs/" not in body


def test_write_jacobian_no_reactions_still_writes_valid_file(tmp_path):
    seen_sp = [True] * len(_SMALL_MECHANISM_SPECIES)
    touched = jacobian.write_jacobian(
        tmp_path, _SMALL_MECHANISM_SPECIES, [], seen_sp,
    )
    n = len(_SMALL_MECHANISM_SPECIES)
    # jac[0] (dT/dT) and every dT/dY_j are always touched, even
    # without reactions.
    assert touched[0] is True
    for j_sp in range(n - 1):
        assert touched[n * (j_sp + 1)] is True
    assert not any(
        touched[k_sp + 1 + n * (j_sp + 1)]
        for k_sp in range(n - 1) for j_sp in range(n - 1)
    )
    body = (tmp_path / "jacob.c").read_text()
    assert "eval_h(T, h)" in body
    assert "} // end eval_jacob" in body


def test_write_jacobian_compiles_and_runs(tmp_path):
    _skip_unless_gcc_available()
    species = _SMALL_MECHANISM_SPECIES
    reactions = _SMALL_MECHANISM_REACTIONS
    seen_sp = [True] * len(species)

    rates.write_mass_mole(tmp_path, species)
    rates.write_chem_utils(tmp_path, species)
    rates.write_rxn_rates(tmp_path, species, reactions)
    rates.write_spec_rates(tmp_path, species, reactions)
    jacobian.write_jacobian(tmp_path, species, reactions, seen_sp)

    (tmp_path / "header.h").write_text(
        f"#ifndef HEAD\n#define HEAD\n#include <math.h>\n"
        f"#define NSP {len(species)}\n#endif\n"
    )
    (tmp_path / "harness.c").write_text(
        "#include <stdio.h>\n"
        '#include "header.h"\n'
        '#include "mass_mole.h"\n#include "chem_utils.h"\n'
        '#include "rates.h"\n#include "jacob.h"\n'
        "int main() {\n"
        "  double y[NSP] = {1500.0, 0.1, 0.2};\n"
        "  double jac[NSP * NSP];\n"
        "  int i;\n"
        "  for (i = 0; i < NSP * NSP; i++) jac[i] = 0.0;\n"
        "  eval_jacob(1.0, 101325.0, y, jac);\n"
        "  int finite = 1, any_nonzero = 0;\n"
        "  for (i = 0; i < NSP * NSP; i++) {\n"
        "    if (jac[i] != jac[i]) finite = 0;\n"
        "    if (jac[i] != 0.0) any_nonzero = 1;\n"
        "  }\n"
        '  printf("finite=%d\\n", finite);\n'
        '  printf("any_nonzero=%d\\n", any_nonzero);\n'
        "  return 0;\n"
        "}\n"
    )
    sources = [
        str(p) for p in tmp_path.glob("*.c") if p.name != "harness.c"
    ]
    binary = tmp_path / "run"
    subprocess.run(
        ["gcc", "-O1", "-o", str(binary), str(tmp_path / "harness.c"),
         *sources, "-lm", "-I", str(tmp_path)],
        check=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if line
    )
    assert values["finite"] == "1"
    assert values["any_nonzero"] == "1"


def test_write_sparse_multiplier_skips_untouched_entries(tmp_path):
    nvars = 2
    touched = [False, True, True, False]
    jacobian.write_sparse_multiplier(tmp_path, touched, nvars)

    header = (tmp_path / "sparse_multiplier.h").read_text()
    assert "#define N_A 2" in header

    body = (tmp_path / "sparse_multiplier.c").read_text()
    assert "w[0] = A[2] * Vm[1];" in body
    assert "w[1] = A[1] * Vm[0];" in body


def test_write_sparse_multiplier_zeroes_all_untouched_row(tmp_path):
    nvars = 2
    touched = [False, False, True, False]
    jacobian.write_sparse_multiplier(tmp_path, touched, nvars)

    body = (tmp_path / "sparse_multiplier.c").read_text()
    assert "w[0] = A[2] * Vm[1];" in body
    assert "w[1] = 0;" in body


def test_dbdt_declaration_lines_no_reversible_reactions():
    species = [Species("A", 1.0, (0,) * 7, (0,) * 7)]
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[0],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=False,
    )
    assert jacobian.dbdt_declaration_lines(species, [reaction]) == ""


def test_dbdt_declaration_lines_buckets_by_t_mid_and_dedups():
    low_a = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    high_a = (11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0)
    low_b = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    high_b = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
    species = [
        Species("A", 1.0, low_a, high_a, t_mid=1000.0),
        Species("B", 1.0, low_b, high_b, t_mid=1000.0),
        Species("C", 1.0, (0,) * 7, (0,) * 7, t_mid=2000.0),
    ]
    # A is touched by both reactions; its dBdT line must not repeat.
    reaction1 = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    reaction2 = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[0],
        product_nu=[2.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    result = jacobian.dbdt_declaration_lines(
        species, [reaction1, reaction2],
    )
    expected = (
        "  if (T <= 1000.0) {\n"
        f"    {_dbdt_species_expr(0, low_a)};\n"
        f"    {_dbdt_species_expr(1, low_b)};\n"
        "  } else {\n"
        f"    {_dbdt_species_expr(0, high_a)};\n"
        f"    {_dbdt_species_expr(1, high_b)};\n"
        "  }\n\n"
    )
    assert result == expected


def test_dbdt_declaration_lines_skips_explicit_reverse_params():
    species = [
        Species("A", 1.0, (1.0,) * 7, (2.0,) * 7, t_mid=1000.0),
        Species("B", 1.0, (1.0,) * 7, (2.0,) * 7, t_mid=1000.0),
    ]
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[1],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
        reverse_params=(5e8, 0.2, 500.0),
    )
    result = jacobian.dbdt_declaration_lines(species, [reaction])
    # dBdT[] is declared by write_jacobian(); none is computed here.
    assert result == ""


def test_dbdt_expr_single_product_unit_nu():
    reaction = Reaction(
        reactants=[], reactant_nu=[], products=[0], product_nu=[1.0],
        A=1e10, b=0.0, E=0.0, reversible=True,
    )
    assert jacobian.dbdt_expr(reaction) == "dBdT[0]"


def test_dbdt_expr_single_product_negative_unit_nu():
    reaction = Reaction(
        reactants=[], reactant_nu=[], products=[0], product_nu=[-1.0],
        A=1e10, b=0.0, E=0.0, reversible=True,
    )
    assert jacobian.dbdt_expr(reaction) == "-dBdT[0]"


def test_dbdt_expr_two_products_and_a_coefficient():
    reaction = Reaction(
        reactants=[], reactant_nu=[], products=[0, 1],
        product_nu=[1.0, 2.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    assert (
        jacobian.dbdt_expr(reaction) == "dBdT[0] + 2.0 * dBdT[1]"
    )


def test_dbdt_expr_reactant_only_species():
    reaction = Reaction(
        reactants=[1], reactant_nu=[1.0], products=[0],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    assert jacobian.dbdt_expr(reaction) == "dBdT[0] - dBdT[1]"


def test_dbdt_expr_reactant_only_species_as_first_term():
    # Upstream quirk, kept: a lone reactant-only term still gets its
    # leading connector.
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[],
        product_nu=[], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    assert jacobian.dbdt_expr(reaction) == " - dBdT[0]"


def test_dbdt_expr_cancelling_species_is_skipped():
    reaction = Reaction(
        reactants=[0], reactant_nu=[1.0], products=[0],
        product_nu=[1.0], A=1e10, b=0.0, E=0.0, reversible=True,
    )
    assert jacobian.dbdt_expr(reaction) == ""


def test_write_sparse_multiplier_sums_multiple_terms(tmp_path):
    nvars = 2
    touched = [True, True, True, True]
    jacobian.write_sparse_multiplier(tmp_path, touched, nvars)

    header = (tmp_path / "sparse_multiplier.h").read_text()
    assert "#define N_A 4" in header

    body = (tmp_path / "sparse_multiplier.c").read_text()
    assert "w[0] = A[0] * Vm[0] + A[2] * Vm[1];" in body
    assert "w[1] = A[1] * Vm[0] + A[3] * Vm[1];" in body
