# Derived from pyJac 1.0.6, Copyright (c) 2016 Kyle E Niemeyer, Nicholas J
# Curtis; MIT licence, see LICENSE-pyJac.

"""C source for species thermodynamics and rates (pyJac port)."""

import math
from pathlib import Path
from typing import List, Tuple

from pyfoamy.pyjac.records import (
    ChebyshevRate, GAS_CONSTANT_SI, PlogPoint, Reaction, Species,
    STANDARD_PRESSURE,
)

_HEADER_GUARD = "#ifndef {0}\n#define {0}\n"


_STATE_CLAMP_T_MIN = 300.0


def state_clamp_lines(n: int) -> str:
    lines = [
        "  // Defensive clamp: an ODE integrator's internal trial\n"
        "  // steps aren't guaranteed physical - a real diesel-case\n"
        "  // sigFpe was traced to seulex handing this function\n"
        "  // T=-910K and negative concentrations mid-step, with no\n"
        "  // protection anywhere in real pyJac's own generated code\n"
        "  // either (tasks/dlb-foam.md's 2026-09-08 section). Clamp\n"
        "  // before any rate evaluation: T to a floor within every\n"
        "  // NASA-7 fit's near-universal low-temperature validity\n"
        "  // bound (a 1K floor was tried first and still crashed -\n"
        "  // it's inside the fit's extrapolation region, where Kc\n"
        "  // itself can underflow to exactly 0 and a later division\n"
        "  // by it traps; 300K is not), mass fractions to non-\n"
        "  // negative.\n"
        f"  double y[{n}];\n"
        f"  y[0] = fmax(y_in[0], {_STATE_CLAMP_T_MIN});\n"
    ]
    lines.extend(f"  y[{i}] = fmax(y_in[{i}], 0.0);\n" for i in range(1, n))
    lines.append("\n")
    return "".join(lines)


def _last_species_complement(array: str, n: int) -> str:
    terms = " + ".join(f"{array}[{i}]" for i in range(n - 1))
    return f"1.0 - ({terms})"


def _weighted_sum(array: str, species: List[Species], weight) -> str:
    terms = " + ".join(
        f"({array}[{i}] * {weight(sp):.16e})"
        for i, sp in enumerate(species[:-1])
    )
    return terms


def write_mass_mole(path: Path, species: List[Species]) -> None:
    """Writes `mass_mole.h`/`mass_mole.c`: mole/mass fraction and
    density conversions.

    Port of pyJac's `rate_subs.write_mass_mole()`.

    Args:
        species: In the order the generated code expects (dependent
            species last).
    """
    n = len(species)
    last_mw = species[-1].mw

    (path / "mass_mole.h").write_text(
        _HEADER_GUARD.format("MASS_MOLE_HEAD") + "\n"
        '#include "header.h"\n\n'
        "void mole2mass (const double*, double*);\n"
        "void mass2mole (const double*, double*);\n"
        "double getDensity (const double, const double, const double*);\n"
        "\n#endif\n"
    )

    mole2mass = (
        "/** Function converting species mole fractions to mass "
        "fractions.\n"
        " *\n"
        " * \\param[in]  X  array of species mole fractions\n"
        " * \\param[out] Y  array of species mass fractions\n"
        " */\n"
        "void mole2mass (const double * X, double * Y) {\n\n"
        "  // mole fraction of final species\n"
        f"  double X_N;\n  X_N = {_last_species_complement('X', n)};\n"
        "  // average molecular weight\n"
        "  double mw_avg = 0.0;\n"
        f"  mw_avg += {_weighted_sum('X', species, lambda sp: sp.mw)};\n"
        f"  mw_avg += X_N * {last_mw:.16e};\n\n"
        "  // calculate mass fractions\n"
        + "".join(
            f"  Y[{i}] = X[{i}] * {sp.mw:.16e} / mw_avg;\n"
            for i, sp in enumerate(species[:-1])
        )
        + "\n} // end mole2mass\n\n"
    )

    mass2mole = (
        "/** Function converting species mass fractions to mole "
        "fractions.\n"
        " *\n"
        " * \\param[in]  Y  array of species mass fractions\n"
        " * \\param[out] X  array of species mole fractions\n"
        " */\n"
        "void mass2mole (const double * Y, double * X) {\n\n"
        "  // mass fraction of final species\n"
        f"  double Y_N;\n  Y_N = {_last_species_complement('Y', n)};\n"
        "  // average molecular weight\n"
        "  double mw_avg = 0.0;\n"
        + "".join(
            f"  mw_avg += Y[{i}] / {sp.mw:.16e};\n"
            for i, sp in enumerate(species[:-1])
        )
        + f"  mw_avg += Y_N / {last_mw:.16e};\n"
        "  mw_avg = 1.0 / mw_avg;\n\n"
        "  // calculate mole fractions\n"
        + "".join(
            f"  X[{i}] = Y[{i}] * mw_avg / {sp.mw:.16e};\n"
            for i, sp in enumerate(species[:-1])
        )
        + "\n} // end mass2mole\n\n"
    )

    get_density = (
        "/** Function calculating density from mole fractions.\n"
        " *\n"
        " * \\param[in]  temp  temperature\n"
        " * \\param[in]  pres  pressure\n"
        " * \\param[in]  X     array of species mole fractions\n"
        r" * \return     rho  mixture mass density" + "\n"
        " */\n"
        "double getDensity (const double temp, const double pres, "
        "const double * X) {\n\n"
        "  // mole fraction of final species\n"
        f"  double X_N;\n  X_N = {_last_species_complement('X', n)};\n"
        "  // average molecular weight\n"
        "  double mw_avg = 0.0;\n"
        f"  mw_avg += {_weighted_sum('X', species, lambda sp: sp.mw)};\n"
        f"  mw_avg += X_N * {last_mw:.16e};\n\n"
        f"  return pres * mw_avg / ({GAS_CONSTANT_SI:.8e} * temp);\n"
        "} // end getDensity\n\n"
    )

    (path / "mass_mole.c").write_text(
        '#include "mass_mole.h"\n\n' + mole2mass + mass2mole + get_density
    )


def _enthalpy_expr(coeffs, mw: float, offset: float) -> str:
    a1, a2, a3, a4, a5, a6, _a7 = coeffs
    ru_mw = GAS_CONSTANT_SI / mw
    return (
        f"{ru_mw:.16e} * ({a6:.16e} + T * ({a1 + offset:.16e} + T * ("
        f"{a2 / 2.0:.16e} + T * ({a3 / 3.0:.16e} + T * ("
        f"{a4 / 4.0:.16e} + {a5 / 5.0:.16e} * T)))))"
    )


def _specific_heat_expr(coeffs, mw: float, offset: float) -> str:
    a1, a2, a3, a4, a5, _a6, _a7 = coeffs
    ru_mw = GAS_CONSTANT_SI / mw
    return (
        f"{ru_mw:.16e} * ({a1 + offset:.16e} + T * ({a2:.16e} + T * ("
        f"{a3:.16e} + T * ({a4:.16e} + {a5:.16e} * T))))"
    )


def _piecewise_species_loop(array: str, species: List[Species], expr) -> str:
    return "".join(
        f"  if (T <= {sp.t_mid:}) {{\n"
        f"    {array}[{isp}] = {expr(sp.low, sp.mw)};\n"
        "  } else {\n"
        f"    {array}[{isp}] = {expr(sp.high, sp.mw)};\n"
        "  }\n\n"
        for isp, sp in enumerate(species)
    )


def write_chem_utils(path: Path, species: List[Species]) -> None:
    """Writes `chem_utils.h`/`chem_utils.c`: concentrations and
    thermodynamic properties (enthalpy, internal energy, `cv`, `cp`).

    Port of pyJac's `rate_subs.write_chem_utils()`. `eval_h()` is what
    DLBFoam's `loadBalanced_pyJac::Qdot()` calls.
    """
    n = len(species)
    last_mw = species[-1].mw

    (path / "chem_utils.h").write_text(
        _HEADER_GUARD.format("CHEM_UTILS_HEAD") + "\n"
        '#include "header.h"\n\n'
        "void eval_conc (const double, const double, "
        "const double * __restrict__, double * __restrict__, "
        "double * __restrict__, double * __restrict__, "
        "double * __restrict__);\n"
        "void eval_conc_rho (const double, const double, "
        "const double * __restrict__, double * __restrict__, "
        "double * __restrict__, double * __restrict__, "
        "double * __restrict__);\n"
        "void eval_h (const double, double * __restrict__);\n"
        "void eval_u (const double, double * __restrict__);\n"
        "void eval_cv (const double, double * __restrict__);\n"
        "void eval_cp (const double, double * __restrict__);\n"
        "\n#endif\n"
    )

    eval_conc = (
        "void eval_conc (const double T, const double pres, "
        "const double * __restrict__ y, double * __restrict__ y_N, "
        "double * __restrict__ mw_avg, double * __restrict__ rho, "
        "double * __restrict__ conc) {\n\n"
        "  // mass fraction of final species\n"
        f"  *y_N = {_last_species_complement('y', n)};\n"
        f"  *mw_avg = {_weighted_sum('y', species, lambda sp: 1.0 / sp.mw)}"
        f" + ((*y_N) * {1.0 / last_mw:.16e});\n"
        "  *mw_avg = 1.0 / *mw_avg;\n"
        "  // mass-averaged density\n"
        f"  *rho = pres * (*mw_avg) / ({GAS_CONSTANT_SI:.8e} * T);\n"
        + "".join(
            f"  conc[{i}] = (*rho) * y[{i}] * {1.0 / sp.mw:.16e};\n"
            for i, sp in enumerate(species[:-1])
        )
        + f"  conc[{n - 1}] = (*rho) * (*y_N) * {1.0 / last_mw:.16e};\n\n"
        "} // end eval_conc\n\n"
    )

    eval_conc_rho = (
        "void eval_conc_rho (const double T, const double rho, "
        "const double * __restrict__ y, double * __restrict__ y_N, "
        "double * __restrict__ mw_avg, double * __restrict__ pres, "
        "double * __restrict__ conc) {\n\n"
        "  // mass fraction of final species\n"
        f"  *y_N = {_last_species_complement('y', n)};\n"
        f"  *mw_avg = {_weighted_sum('y', species, lambda sp: 1.0 / sp.mw)}"
        f" + ((*y_N) * {1.0 / last_mw:.16e});\n"
        "  *mw_avg = 1.0 / *mw_avg;\n"
        "  // pressure\n"
        f"  *pres = rho * {GAS_CONSTANT_SI:.8e} * T / (*mw_avg);\n"
        + "".join(
            f"  conc[{i}] = rho * y[{i}] * {1.0 / sp.mw:.16e};\n"
            for i, sp in enumerate(species[:-1])
        )
        + f"  conc[{n - 1}] = rho * (*y_N) * {1.0 / last_mw:.16e};\n\n"
        "} // end eval_conc\n\n"
    )

    eval_h = (
        "void eval_h (const double T, double * __restrict__ h) {\n\n"
        + _piecewise_species_loop(
            "h", species, lambda c, mw: _enthalpy_expr(c, mw, 0.0)
        )
        + "} // end eval_h\n\n"
    )
    eval_u = (
        "void eval_u (const double T, double * __restrict__ u) {\n\n"
        + _piecewise_species_loop(
            "u", species, lambda c, mw: _enthalpy_expr(c, mw, -1.0)
        )
        + "} // end eval_u\n\n"
    )
    eval_cv = (
        "void eval_cv (const double T, double * __restrict__ cv) {\n\n"
        + _piecewise_species_loop(
            "cv", species, lambda c, mw: _specific_heat_expr(c, mw, -1.0)
        )
        + "} // end eval_cv\n\n"
    )
    eval_cp = (
        "void eval_cp (const double T, double * __restrict__ cp) {\n\n"
        + _piecewise_species_loop(
            "cp", species, lambda c, mw: _specific_heat_expr(c, mw, 0.0)
        )
        + "} // end eval_cp\n\n"
    )

    (path / "chem_utils.c").write_text(
        '#include "header.h"\n#include "chem_utils.h"\n\n'
        + eval_conc + eval_conc_rho + eval_h + eval_u + eval_cv + eval_cp
    )


def forward_rate_expr(A: float, b: float, E: float) -> str:
    """Builds `A * T^b * exp(-E/T)` in the cheapest form for `b`/`E`.

    Port of pyJac's `rate_subs.rxn_rate_const()`.
    """
    if A == 0.0:
        raise ValueError("a reaction's forward A must be nonzero")
    if A > 0:
        log_a = math.log(A)
        if not E:
            if not b:
                return str(A)
            sign = "+" if b > 0 else "-"
            return f"exp({log_a:.16e} {sign} {abs(b)} * logT)"
        if not b:
            return f"exp({log_a:.16e} - ({E:.16e} / T))"
        sign = "+" if b > 0 else "-"
        return f"exp({log_a:.16e} {sign} {abs(b)} * logT - ({E:.16e} / T))"
    if not E:
        if not b:
            return str(A)
        if b.is_integer():
            return str(A) + " * T" * int(b)
        sign = "" if b > 0 else "-"
        return f"{A:.16e} * exp({sign}{abs(b)} * logT)"
    if not b:
        return f"{A:.16e} * exp(-({E:.16e} / T))"
    sign = "" if b > 0 else "-"
    return f"{A:.16e} * exp({sign}{abs(b)} * logT - ({E:.16e} / T))"


def chebyshev_rate_lines(chebyshev: ChebyshevRate) -> str:
    """Builds the Chebyshev forward-rate evaluation.

    `T`/`pres`/`dot_prod[]`/`Tred`/`Pred`/`cheb_temp_0`/`cheb_temp_1`/
    `kf` must already be declared. Port of `rate_subs.get_cheb_rate()`.
    """
    t_low, t_high = chebyshev.t_range
    p_low, p_high = chebyshev.p_range
    n_temp = len(chebyshev.coefficients)
    n_pres = len(chebyshev.coefficients[0])
    t_inv_sum = 1.0 / t_low + 1.0 / t_high
    t_inv_sub = 1.0 / t_high - 1.0 / t_low
    p_log_sum = math.log10(p_low) + math.log10(p_high)
    p_log_sub = math.log10(p_high) - math.log10(p_low)

    lines = [
        f"Tred = ((2.0 / T) - {t_inv_sum:.8e}) / {t_inv_sub:.8e}",
        f"Pred = (2.0 * log10(pres) - {p_log_sum:.8e}) / {p_log_sub:.8e}",
        "cheb_temp_0 = 1", "cheb_temp_1 = Pred",
    ]
    for i in range(n_temp):
        lines.append(
            f"dot_prod[{i}] = {chebyshev.coefficients[i][0]:.8e} + "
            f"Pred * {chebyshev.coefficients[i][1]:.8e}"
        )
    new_is_one = True
    for j in range(2, n_pres):
        new, old = (1, 0) if new_is_one else (0, 1)
        lines.append(
            f"cheb_temp_{old} = 2 * Pred * cheb_temp_{new} - "
            f"cheb_temp_{old}"
        )
        for i in range(n_temp):
            lines.append(
                f"dot_prod[{i}] += "
                f"{chebyshev.coefficients[i][j]:.8e} * cheb_temp_{old}"
            )
        new_is_one = not new_is_one

    lines += ["cheb_temp_0 = 1", "cheb_temp_1 = Tred",
              "kf = dot_prod[0] + Tred * dot_prod[1]"]
    new_is_one = True
    for i in range(2, n_temp):
        new, old = (1, 0) if new_is_one else (0, 1)
        lines.append(
            f"cheb_temp_{old} = 2 * Tred * cheb_temp_{new} - "
            f"cheb_temp_{old}"
        )
        lines.append(f"kf += dot_prod[{i}] * cheb_temp_{old}")
        new_is_one = not new_is_one
    lines.append("kf = pow(10.0, kf)")

    return "".join(f"  {line};\n" for line in lines)


def plog_rate_lines(points: List[PlogPoint]) -> str:
    """Builds the PLOG forward-rate evaluation; `pres`/`kf`/`kf2` must
    already be declared.

    Port of the inline PLOG handling of `rate_subs.write_rxn_rates()`.
    """
    p0, a0, b0, e0 = points[0]
    lines = [
        f"  if (pres <= {p0:.4e}) {{\n",
        f"    kf = {forward_rate_expr(a0, b0, e0)};\n",
    ]
    for idx in range(len(points) - 1):
        p, a, b, e = points[idx]
        p2, a2, b2, e2 = points[idx + 1]
        lines.append(
            f"  }} else if ((pres > {p:.4e}) && (pres <= {p2:.4e})) {{\n"
        )
        lines.append(f"    kf = log({forward_rate_expr(a, b, e)});\n")
        lines.append(f"    kf2 = log({forward_rate_expr(a2, b2, e2)});\n")
        pres_log_diff = math.log(p2) - math.log(p)
        lines.append(
            "    kf = exp(kf + (kf2 - kf) * (log(pres) - "
            f"{math.log(p):.16e}) / {pres_log_diff:.16e});\n"
        )
    p_last, a_last, b_last, e_last = points[-1]
    lines.append(f"  }} else if (pres > {p_last:.4e}) {{\n")
    lines.append(f"    kf = {forward_rate_expr(a_last, b_last, e_last)};\n")
    lines.append("  }\n")
    return "".join(lines)


def _stoichiometry_terms(indices: List[int], nus: List[float]) -> str:
    terms = ""
    for isp, nu in zip(indices, nus):
        if float(nu).is_integer():
            terms += f"C[{isp}] * " * int(nu)
        else:
            terms += f"pow(C[{isp}], {nu}) *"
    return terms


def equilibrium_coeffs(
    coeffs: Tuple[float, ...], scaled_nu: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """One species' term of the piecewise `Kc` sum.

    Its NASA-7 coefficients times `scaled_nu`, recombined for
    `equilibrium_constant_expr()`. Shared with `jacobian.write_kc()`.

    Args:
        coeffs: NASA-7 coefficients (`low` or `high`).
        scaled_nu: Net stoichiometric coefficient, already multiplied
            by the caller's sign/scale (e.g. `-1` for a reactant).
    """
    a1, a2, a3, a4, a5, a6, a7 = coeffs
    terms = (a7 - a1, a1 - 1.0, a2 / 2.0, a3 / 6.0, a4 / 12.0, a5 / 20.0, a6)
    return tuple(scaled_nu * term for term in terms)


def equilibrium_constant_expr(coeffs7: Tuple[float, ...]) -> str:
    """The `Kc` expression for one temperature range.

    Args:
        coeffs7: The 7 coefficients accumulated with
            `equilibrium_coeffs()` over the reaction's species.
    """
    c0, c1, c2, c3, c4, c5, c6 = coeffs7
    return (
        f"({c0:.16e} + {c1:.16e} * logT + T * ({c2:.16e} + T * "
        f"({c3:.16e} + T * ({c4:.16e} + {c5:.16e} * T))) - "
        f"{c6:.16e} / T)"
    )


def equilibrium_constant_lines(
    reaction: Reaction, species: List[Species],
) -> str:
    """Equilibrium-constant (`Kc`) reverse-rate setup for a reaction
    without explicit reverse parameters.

    A temperature-piecewise sum of the NASA-7 fits of the net-nonzero
    species, scaled by the net change in moles. Port of the `Kc` part
    of pyJac's `rate_subs.write_rxn_rates()` (nested `__get_arrays()`).
    """
    sum_nu = 0.0
    coeffs_by_t_mid = {}

    def accumulate(sp_index: int, nu: float) -> None:
        sp = species[sp_index]
        lo = equilibrium_coeffs(sp.low, nu)
        hi = equilibrium_coeffs(sp.high, nu)
        if sp.t_mid not in coeffs_by_t_mid:
            coeffs_by_t_mid[sp.t_mid] = (lo, hi)
            return
        old_lo, old_hi = coeffs_by_t_mid[sp.t_mid]
        coeffs_by_t_mid[sp.t_mid] = (
            tuple(a + b for a, b in zip(lo, old_lo)),
            tuple(a + b for a, b in zip(hi, old_hi)),
        )

    for i, prod_sp in enumerate(reaction.products):
        if prod_sp in reaction.reactants:
            j = reaction.reactants.index(prod_sp)
            nu = reaction.product_nu[i] - reaction.reactant_nu[j]
        else:
            nu = reaction.product_nu[i]
        if nu == 0:
            continue
        sum_nu += nu
        accumulate(prod_sp, nu)

    for i, reac_sp in enumerate(reaction.reactants):
        if reac_sp in reaction.products:
            continue
        nu = reaction.reactant_nu[i]
        sum_nu -= nu
        accumulate(reac_sp, -nu)

    lines = []
    for group_index, (t_mid, (lo, hi)) in enumerate(coeffs_by_t_mid.items()):
        op = "Kc = " if group_index == 0 else "Kc += "
        lines.append(
            f"  if (T <= {t_mid:}) {{\n"
            f"    {op}{equilibrium_constant_expr(lo)};\n"
            "  } else {\n"
            f"    {op}{equilibrium_constant_expr(hi)};\n"
            "  }\n\n"
        )
    scale = (STANDARD_PRESSURE / GAS_CONSTANT_SI) ** sum_nu
    lines.append(f"  Kc = {scale:.16e} * exp(Kc);\n")
    return "".join(lines)


def write_rxn_rates(
    path: Path, species: List[Species], reactions: List[Reaction],
) -> None:
    """Writes `rates.h`/`rxn_rates.c`, forward/reverse reaction rates.

    Port of pyJac's `rate_subs.write_rxn_rates()` without
    `fwd_rxn_mapping` (reactions are never reordered).
    """
    has_pressure_dependence = any(
        r.third_body or r.pressure_dependent for r in reactions
    )
    header_body = (
        "void eval_rxn_rates (const double, const double, "
        "const double * __restrict__, double * __restrict__, "
        "double * __restrict__);\n"
        "void eval_spec_rates (const double * __restrict__, "
        "const double * __restrict__, const double * __restrict__, "
        "double * __restrict__, double * __restrict__);\n"
    )
    if has_pressure_dependence:
        header_body += (
            "void get_rxn_pres_mod (const double, const double, "
            "const double * __restrict__, double * __restrict__);\n"
        )
    (path / "rates.h").write_text(
        _HEADER_GUARD.format("RATES_HEAD") + "\n"
        '#include "header.h"\n\n' + header_body + "\n#endif\n"
    )

    reversible_index = {}
    for i, r in enumerate(reactions):
        if r.reversible:
            reversible_index[i] = len(reversible_index)

    needs_kf_kc = any(
        r.reversible and r.reverse_params is None for r in reactions
    )
    chebyshevs = [r.chebyshev for r in reactions if r.chebyshev is not None]
    needs_cheb = bool(chebyshevs)
    needs_plog = any(r.plog is not None for r in reactions)

    lines = [
        "void eval_rxn_rates (const double T, const double pres, "
        "const double * __restrict__ C, double * __restrict__ "
        "fwd_rxn_rates, double * __restrict__ rev_rxn_rates) {\n\n",
        "  double logT = log(T);\n\n",
    ]
    if needs_kf_kc:
        lines.append("  double kf;\n  double Kc;\n")
    if needs_cheb:
        if not needs_kf_kc:
            lines.append("  double kf;\n")
        max_n_temp = max(len(c.coefficients) for c in chebyshevs)
        lines.append(
            "  double Tred;\n  double Pred;\n"
            "  double cheb_temp_0, cheb_temp_1;\n"
            f"  double dot_prod[{max_n_temp}];\n"
        )
    if needs_plog:
        if not needs_kf_kc and not needs_cheb:
            lines.append("  double kf;\n")
        lines.append("  double kf2;\n")
    lines.append("\n")

    for i_rxn, rxn in enumerate(reactions):
        lines.append(f"  // rxn {i_rxn}\n")

        if (
            rxn.reversible and rxn.reverse_params is None
            and rxn.chebyshev is None and rxn.plog is None
        ):
            kf_expr = forward_rate_expr(rxn.A, rxn.b, rxn.E)
            lines.append(f"  kf = {kf_expr};\n")
        elif rxn.chebyshev is not None:
            lines.append(chebyshev_rate_lines(rxn.chebyshev))
        elif rxn.plog is not None:
            lines.append(plog_rate_lines(rxn.plog))

        reactant_terms = _stoichiometry_terms(rxn.reactants, rxn.reactant_nu)
        if (
            (rxn.reversible and rxn.reverse_params is None)
            or rxn.plog is not None or rxn.chebyshev is not None
        ):
            forward_term = "kf"
        else:
            forward_term = forward_rate_expr(rxn.A, rxn.b, rxn.E)
        lines.append(
            f"  fwd_rxn_rates[{i_rxn}] = {reactant_terms}{forward_term};\n"
        )

        if rxn.reversible:
            if rxn.reverse_params is None:
                lines.append(equilibrium_constant_lines(rxn, species))
                reverse_term = "kf / Kc"
            else:
                reverse_term = forward_rate_expr(*rxn.reverse_params)
            product_terms = _stoichiometry_terms(rxn.products, rxn.product_nu)
            lines.append(
                f"  rev_rxn_rates[{reversible_index[i_rxn]}] = "
                f"{product_terms}{reverse_term};\n\n"
            )

    lines.append("} // end eval_rxn_rates\n\n")
    (path / "rxn_rates.c").write_text('#include "rates.h"\n' + "".join(lines))


def _third_body_term(base: str, efficiencies) -> str:
    line = base
    for isp, eff in efficiencies or []:
        if eff == 1.0:
            continue
        if eff > 1.0:
            line += f" + {eff - 1.0}"
        else:
            line += f" - {1.0 - eff}"
        line += f" * C[{isp}]"
    return line


def _falloff_limit_expr(
    override: Tuple[float, float, float], fallback: Reaction,
) -> str:
    if override is not None:
        return forward_rate_expr(*override)
    return forward_rate_expr(fallback.A, fallback.b, fallback.E)


def _troe_logfcent_expr(troe_params: Tuple[float, ...]) -> str:
    alpha, t3, t1 = troe_params[0], troe_params[1], troe_params[2]
    term1 = (
        f"exp(-T / {t3:.8e})" if t3 > 0.0 else f"exp(T / {abs(t3):.8e})"
    )
    term2 = (
        f"exp(-T / {t1:.8e})" if t1 > 0.0 else f"exp(T / {abs(t1):.8e})"
    )
    expr = f"{1.0 - alpha:.8e} * {term1} + {alpha:.8e} * {term2}"
    if len(troe_params) == 4 and troe_params[3] != 0.0:
        t2 = troe_params[3]
        term3 = (
            f"exp(-{t2:.8e} / T)" if t2 > 0.0 else f"exp({abs(t2):.8e} / T)"
        )
        expr += f" + {term3}"
    return f"log10( fmax({expr}, 1.0e-300))"


def _sri_blend_expr(sri_params: Tuple[float, ...]) -> str:
    a, b, c = sri_params[0], sri_params[1], sri_params[2]
    term_b = f"exp(-{b:.6} / T)" if b > 0.0 else f"exp({abs(b):.6} / T)"
    term_c = (
        f" + exp(-T / {c:.6}), X) " if c > 0.0
        else f" + exp(T / {abs(c):.6}), X) "
    )
    expr = f"pow({a:.6} * {term_b}{term_c}"
    if (
        len(sri_params) == 5 and sri_params[3] != 1.0
        and sri_params[4] != 0.0
    ):
        d, e = sri_params[3], sri_params[4]
        expr += f"* {d:.8e} * pow(T, {e:.6}) "
    return expr


def _falloff_pr_factor(reaction: Reaction, simple: bool) -> str:
    if reaction.low_rate_params is not None:
        return "Pr / (1.0 + Pr)" if simple else "* Pr / (1.0 + Pr)"
    if reaction.high_rate_params is not None:
        return "1.0 / (1.0 + Pr)" if simple else "/ (1.0 + Pr)"
    raise ValueError(
        "a pressure-dependent Reaction needs low_rate_params or "
        "high_rate_params set"
    )


def write_rxn_pressure_mod(path: Path, reactions: List[Reaction]) -> None:
    """Writes `rxn_rates_pres_mod.c`, the third-body/falloff
    `pres_mod[]` factors, one per such reaction in `reactions` order.

    Port of pyJac's `rate_subs.write_rxn_pressure_mod()`.
    """
    thd_flag = any(r.third_body for r in reactions)
    pdep_flag = any(r.pressure_dependent for r in reactions)
    troe_flag = any(
        r.pressure_dependent and r.troe_params is not None
        for r in reactions
    )
    sri_flag = any(
        r.pressure_dependent and r.sri_params is not None
        for r in reactions
    )

    body = [
        "void get_rxn_pres_mod (const double T, const double pres, "
        "const double * __restrict__ C, double * __restrict__ "
        "pres_mod) {\n"
    ]
    if thd_flag or pdep_flag:
        body.append(
            "  // third body variable declaration\n  double thd;\n\n"
        )
    if pdep_flag:
        body.append(
            "  // pressure dependence variable declarations\n"
            "  double k0;\n  double kinf;\n  double Pr;\n\n"
        )
        if troe_flag:
            body.append(
                "  // troe variable declarations\n"
                "  double logFcent;\n  double A;\n  double B;\n\n"
            )
        if sri_flag:
            body.append("  // sri variable declarations\n  double X;\n\n")

    body.append(
        "  double logT = log(T);\n"
        f"  double m = pres / ({GAS_CONSTANT_SI:.8e} * T);\n\n"
    )

    pind = 0
    for i_rxn, rxn in enumerate(reactions):
        if not (rxn.third_body or rxn.pressure_dependent):
            continue
        body.append(f"  // reaction {i_rxn}\n")

        if rxn.third_body:
            if rxn.pdep_species is not None:
                # A third-body reaction with one explicit collision
                # partner (`third_body_efficiencies` is empty, see
                # adapter.py): only that species counts, not `m`.
                body.append(
                    f"  pres_mod[{pind}] = C[{rxn.pdep_species}];\n"
                )
            else:
                expr = _third_body_term(
                    f"pres_mod[{pind}] = m", rxn.third_body_efficiencies,
                )
                body.append(f"  {expr};\n")

        if rxn.pressure_dependent:
            if rxn.pdep_species is None:
                thd_expr = _third_body_term(
                    "thd = m", rxn.third_body_efficiencies,
                )
                body.append(f"  {thd_expr};\n")

            k0_expr = _falloff_limit_expr(rxn.low_rate_params, rxn)
            kinf_expr = _falloff_limit_expr(rxn.high_rate_params, rxn)
            body.append(f"  k0 = {k0_expr};\n")
            body.append(f"  kinf = {kinf_expr};\n")

            if rxn.pdep_species is not None:
                body.append(f"  Pr = k0 * C[{rxn.pdep_species}] / kinf;\n")
            else:
                body.append("  Pr = k0 * thd / kinf;\n")

            simple = rxn.troe_params is None and rxn.sri_params is None
            if rxn.troe_params is not None:
                body.append(
                    f"  logFcent = {_troe_logfcent_expr(rxn.troe_params)};\n"
                    "  A = log10(fmax(Pr, 1.0e-300)) - "
                    "0.67 * logFcent - 0.4;\n"
                    "  B = 0.806 - 1.1762 * logFcent - "
                    "0.14 * log10(fmax(Pr, 1.0e-300));\n"
                )
                line = (
                    f"  pres_mod[{pind}] = pow(10.0, "
                    "logFcent / (1.0 + A * A / (B * B))) "
                )
            elif rxn.sri_params is not None:
                body.append(
                    "  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * "
                    "log10(fmax(Pr, 1.0e-300)));\n"
                )
                line = (
                    f"  pres_mod[{pind}] = "
                    f"{_sri_blend_expr(rxn.sri_params)}"
                )
            else:
                line = f"  pres_mod[{pind}] = " + _falloff_pr_factor(
                    rxn, True,
                )

            if not simple:
                line += _falloff_pr_factor(rxn, False)
            body.append(f"{line};\n")

        body.append("\n")
        pind += 1

    body.append("} // end get_rxn_pres_mod\n\n")
    (path / "rxn_rates_pres_mod.c").write_text(
        "#include <math.h>\n"
        '#include "header.h"\n'
        '#include "rates.h"\n\n'
        + "".join(body)
    )


def net_nu(species_index: int, reaction: Reaction) -> float:
    """Net signed stoichiometric coefficient of `species_index`.

    Positive for a net product, negative for a net reactant, `0.0` if
    absent or cancelling. Port of pyJac's `utils.get_nu()`.
    """
    in_products = species_index in reaction.products
    in_reactants = species_index in reaction.reactants
    if in_products and in_reactants:
        nu = (
            reaction.product_nu[reaction.products.index(species_index)]
            - reaction.reactant_nu[
                reaction.reactants.index(species_index)
            ]
        )
        return 0.0 if nu == 0 else nu
    if in_products:
        return reaction.product_nu[reaction.products.index(species_index)]
    if in_reactants:
        return -reaction.reactant_nu[
            reaction.reactants.index(species_index)
        ]
    return 0.0


def write_spec_rates(
    path: Path, species: List[Species], reactions: List[Reaction],
) -> List[bool]:
    """Writes `spec_rates.c`, net species production rates.

    The last (dependent) species' rate goes through the `dy_N` output
    pointer, not `sp_rates[]`. Port of pyJac's
    `rate_subs.write_spec_rates()` without `fwd_spec_mapping`/
    `fwd_rxn_mapping` (no reordering here).

    Returns:
        One `bool` per species, `True` if its net rate is not
        identically zero; feeds `write_derivs()`.
    """
    n = len(species)
    reversible_index = {}
    for i, r in enumerate(reactions):
        if r.reversible:
            reversible_index[i] = len(reversible_index)
    pdep_index = {}
    for i, r in enumerate(reactions):
        if r.third_body or r.pressure_dependent:
            pdep_index[i] = len(pdep_index)

    lines = [
        '#include "header.h"\n#include "rates.h"\n\n'
        "void eval_spec_rates (const double * __restrict__ fwd_rates, "
        "const double * __restrict__ rev_rates, "
        "const double * __restrict__ pres_mod, "
        "double * __restrict__ sp_rates, "
        "double * __restrict__ dy_N) {\n"
    ]
    # One flag suffices for both "first use" (`=` vs `+=`) and "ever
    # seen" (zero-fill), unlike upstream: they coincide here.
    touched = [False] * n

    for rind, rxn in enumerate(reactions):
        lines.append(f"  // rxn {rind}\n")
        my_specs = sorted(
            {sp for sp in rxn.reactants + rxn.products}
        )
        for spind in my_specs:
            nu = net_nu(spind, rxn)
            if nu == 0.0:
                continue
            lines.append(f"  // sp {spind}\n")

            sign = "-" if nu < 0 else "+"
            target = "(*dy_N)" if spind + 1 == n else f"sp_rates[{spind}]"
            if touched[spind]:
                line = f"  {target} {sign}= "
            else:
                line = f"  {target} = " + (sign if nu < 0 else "")
            touched[spind] = True

            abs_nu = abs(nu)
            if abs_nu != 1.0:
                if float(abs_nu).is_integer():
                    line += f"{float(abs_nu)} * "
                else:
                    line += f"{abs_nu:3} * "

            if rxn.reversible:
                rxn_out = (
                    f"(fwd_rates[{rind}] - "
                    f"rev_rates[{reversible_index[rind]}])"
                )
            else:
                rxn_out = f"fwd_rates[{rind}]"
            if rxn.third_body or rxn.pressure_dependent:
                rxn_out += f" * pres_mod[{pdep_index[rind]}]"

            lines.append(f"{line}{rxn_out};\n")
        lines.append("\n")

    for i, was_touched in enumerate(touched):
        if not was_touched:
            target = "(*dy_N)" if i + 1 == n else f"sp_rates[{i}]"
            lines.append(f"  // sp {i}\n  {target} = 0.0;\n")

    lines.append("} // end eval_spec_rates\n\n")
    (path / "spec_rates.c").write_text("".join(lines))
    return touched


def write_derivs(
    path: Path, species: List[Species], reactions: List[Reaction],
    species_nonzero: List[bool], debug: bool = False,
    clamp_state: bool = False,
) -> None:
    """Writes `dydt.h`/`dydt.c`, the constant-pressure RHS driver.

    Port of the `CONP` branch of pyJac's `rate_subs.write_derivs()`;
    `CONV` is dropped as `header.h` always defines `CONP`.

    Layout: `y[0]`/`dy[0]` is temperature, `y[1..NSP-1]`/`dy[1..NSP-1]`
    the first `NSP - 1` mass fractions/rates. The last species is
    `1 - sum(y[1:])` with its rate in a local `dy_N`, as in
    `write_spec_rates()`.

    Args:
        species_nonzero: Return value of `write_spec_rates()`; only
            those species enter the temperature equation.
        debug: Overwrite `$FOAM_CASE/pyjac_debug_state_<pid>.log` (else
            in the working directory) on every call with the state fed to
            `eval_rxn_rates()` (overwritten rather than appended to bound
            disk use).
        clamp_state: Clamp the input state (temperature floor, mass
            fractions non-negative) in a local `y` shadowing the
            renamed parameter `y_in`, against non-physical trial
            states of the ODE integrator. Off by default: changes
            the generated math.

    Raises:
        ValueError: If `species_nonzero` doesn't have one entry per
            species.
    """
    n = len(species)
    if len(species_nonzero) != n:
        raise ValueError(
            "species_nonzero must have exactly one entry per species"
        )
    num_reversible = sum(1 for r in reactions if r.reversible)
    num_pdep = sum(
        1 for r in reactions if r.third_body or r.pressure_dependent
    )

    (path / "dydt.h").write_text(
        _HEADER_GUARD.format("DYDT_HEAD") + "\n"
        '#include "header.h"\n\n'
        "void dydt (const double, const double, "
        "const double * __restrict__, double * __restrict__);\n"
        "\n#endif\n"
    )

    debug_includes = (
        "#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n"
        if debug else ""
    )
    y_param = "y_in" if clamp_state else "y"
    lines = [
        f'{debug_includes}'
        '#include "header.h"\n'
        '#include "chem_utils.h"\n'
        '#include "rates.h"\n\n'
        "void dydt (const double t, const double pres, "
        f"const double * __restrict__ {y_param}, "
        "double * __restrict__ dy) {\n\n"
    ]
    if clamp_state:
        lines.append(state_clamp_lines(n))
    lines.append(
        "  // species molar concentrations\n"
        f"  double conc[{n}];\n"
        "  double y_N;\n  double mw_avg;\n  double rho;\n"
        "  eval_conc (y[0], pres, &y[1], &y_N, &mw_avg, &rho, conc);\n\n"
    )
    if debug:
        lines.append(
            "  // DEBUG: overwrite (not append) a small per-process\n"
            "  // file with the exact state fed to eval_rxn_rates() -\n"
            "  // unconditionally, not just when something \"looks\"\n"
            "  // out of range, so it catches whatever the real\n"
            "  // failure mode turns out to be. Overwriting keeps disk\n"
            "  // usage bounded across a real run (this runs on every\n"
            "  // RHS evaluation) - only the LAST state before a crash\n"
            "  // is ever needed, and that's exactly what survives.\n"
            "  {\n"
            '    const char *_dbg_dir = getenv("FOAM_CASE");\n'
            "    char _dbg_path[4096];\n"
            "    snprintf(_dbg_path, sizeof(_dbg_path), "
            '"%s/pyjac_debug_state_%d.log",\n'
            '             _dbg_dir ? _dbg_dir : ".", (int) getpid());\n'
            '    FILE *_dbgf = fopen(_dbg_path, "w");\n'
            "    if (_dbgf) {\n"
            '      fprintf(_dbgf, "SOURCE=dydt T=%.17g pres=%.17g '
            'rho=%.17g mw_avg=%.17g y_N=%.17g\\n",\n'
            "              y[0], pres, rho, mw_avg, y_N);\n"
            '      fprintf(_dbgf, "CONC");\n'
            f"      for (int _i = 0; _i < {n}; _i++) {{\n"
            '        fprintf(_dbgf, " %.17g", conc[_i]);\n'
            "      }\n"
            '      fprintf(_dbgf, "\\n");\n'
            "      fclose(_dbgf);\n"
            "    }\n"
            "  }\n\n"
        )
    lines.append(
        "  // local arrays holding reaction rates\n"
        f"  double fwd_rates[{len(reactions)}];\n"
    )
    lines.append(
        f"  double rev_rates[{num_reversible}];\n" if num_reversible
        else "  double* rev_rates = 0;\n"
    )
    lines.append(
        "  eval_rxn_rates (y[0], pres, conc, fwd_rates, rev_rates);\n\n"
    )
    if num_pdep:
        lines.append(
            "  // get pressure modifications to reaction rates\n"
            f"  double pres_mod[{num_pdep}];\n"
            "  get_rxn_pres_mod (y[0], pres, conc, pres_mod);\n"
        )
    else:
        lines.append("  double* pres_mod = 0;\n")
    lines.append("\n")

    lines.append(
        "  // evaluate species molar net production rates\n"
        "  double dy_N;\n"
        "  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[1], "
        "&dy_N);\n"
    )

    lines.append(
        "  // local array holding constant pressure specific heat\n"
        f"  double cp[{n}];\n"
        "  eval_cp (y[0], cp);\n\n"
    )

    cp_terms = [f"(cp[{isp}] * y[{isp + 1}])" for isp in range(n - 1)]
    cp_terms.append(f"(cp[{n - 1}] * y_N)")
    lines.append(
        "  // constant pressure mass-average specific heat\n"
        f"  double cp_avg = {' + '.join(cp_terms)};\n\n"
    )

    lines.append(
        "  // local array for species enthalpies\n"
        f"  double h[{n}];\n"
        "  eval_h (y[0], h);\n"
    )

    energy_terms = []
    for isp, sp in enumerate(species):
        if not species_nonzero[isp]:
            continue
        dy_ref = f"dy[{isp + 1}]" if isp < n - 1 else "dy_N"
        energy_terms.append(f"({dy_ref} * h[{isp}] * {sp.mw:.16e})")
    lines.append(
        "  // rate of change of temperature\n"
        "  dy[0] = (-1.0 / (rho * cp_avg)) * "
        f"({' + '.join(energy_terms)});\n\n"
    )

    lines.append(
        "  // calculate rate of change of species mass fractions\n"
    )
    for isp, sp in enumerate(species[:-1]):
        lines.append(f"  dy[{isp + 1}] *= ({sp.mw:.16e} / rho);\n")
    lines.append("\n} // end dydt\n\n")

    (path / "dydt.c").write_text("".join(lines))
