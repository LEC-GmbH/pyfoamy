# Derived from pyJac 1.0.6, Copyright (c) 2016 Kyle E Niemeyer, Nicholas J
# Curtis; MIT licence, see LICENSE-pyJac.

"""Analytical Jacobian C source (port of pyJac's `create_jacobian.py`)."""

import math
from pathlib import Path
from typing import List, Optional, Tuple

from pyfoamy.pyjac.records import (
    ArrheniusParams, ChebyshevRate, GAS_CONSTANT_SI, Reaction, Species,
    STANDARD_PRESSURE,
)
from pyfoamy.pyjac.rates import (
    chebyshev_rate_lines, forward_rate_expr, net_nu, plog_rate_lines,
    state_clamp_lines,
)

_HEADER_GUARD = "#ifndef {0}\n#define {0}\n"


def write_rates(reaction: Reaction) -> str:
    """Builds `kf`/`kr` for one reaction; `kf`/`kr` (and `kf2` for PLOG)
    must already be declared.

    Port of `create_jacobian.write_rates()`; forward-rate logic is shared
    with `rates.write_rxn_rates()`.
    """
    lines = []
    if reaction.chebyshev is None and reaction.plog is None:
        kf_expr = forward_rate_expr(reaction.A, reaction.b, reaction.E)
        lines.append(f"  kf = {kf_expr};\n")
    elif reaction.plog is not None:
        lines.append(plog_rate_lines(reaction.plog))
    else:
        lines.append(chebyshev_rate_lines(reaction.chebyshev))

    if reaction.reversible and reaction.reverse_params is None:
        lines.append("  kr = kf / Kc;\n")
    elif reaction.reverse_params is not None:
        kr_expr = forward_rate_expr(*reaction.reverse_params)
        lines.append(f"  kr = {kr_expr};\n")
    return "".join(lines)


def falloff_low_over_high(
    reaction: Reaction,
) -> Tuple[float, float, str]:
    """Falloff exponent/activation-temperature differences and `k0/kinf`.

    Port of `create_jacobian.get_infs()`.

    Args:
        reaction: A falloff reaction with exactly one of `low_rate_params`/
            `high_rate_params` set.

    Returns:
        `(beta_0minf, E_0minf, k0_over_kinf_expr)`: low-limit minus
        high-limit exponent and activation temperature (reversed for a
        chemically activated reaction), and the C expression of the ratio.

    Raises:
        ValueError: If neither `low_rate_params` nor `high_rate_params` is
            set.
    """
    if reaction.low_rate_params is not None:
        low_a, low_b, low_e = reaction.low_rate_params
        beta_0minf = low_b - reaction.b
        e_0minf = low_e - reaction.E
        k0_over_kinf = forward_rate_expr(
            low_a / reaction.A, beta_0minf, e_0minf,
        )
    elif reaction.high_rate_params is not None:
        high_a, high_b, high_e = reaction.high_rate_params
        beta_0minf = reaction.b - high_b
        e_0minf = reaction.E - high_e
        k0_over_kinf = forward_rate_expr(
            reaction.A / high_a, beta_0minf, e_0minf,
        )
    else:
        raise ValueError(
            "a pressure-dependent Reaction needs low_rate_params or "
            "high_rate_params set"
        )
    return beta_0minf, e_0minf, k0_over_kinf


def rate_constant_temperature_expr(
    params: ArrheniusParams, threshold: float = 1.0e-90,
) -> str:
    """Temperature-derivative factor of an Arrhenius rate: `b + E/T`, `b`,
    `E/T` or `""`, keeping the nonzero terms.

    Port of `create_jacobian.get_rxn_params_dt()`, taking the forward or
    explicit reverse parameters directly.

    Args:
        params: `(A, b, E)`; `A` is unused.
        threshold: Magnitude below which `b`/`E` count as zero (upstream's
            `1e-90`).
    """
    _a, b, e = params
    if abs(b) > threshold and abs(e) > threshold:
        return f"{b:.16e} + ({e:.16e} / T)"
    if abs(b) > threshold:
        return f"{b:.16e}"
    if abs(e) > threshold:
        return f"({e:.16e} / T)"
    return ""


def write_sparse_multiplier(
    path: Path, touched: List[bool], nvars: int,
) -> None:
    """Writes `sparse_multiplier.h`/`.c`, a `w = A * Vm` multiply skipping the
    Jacobian entries `touched` marks as always zero.

    Port of `create_jacobian.write_sparse_multiplier()`, fixing an upstream
    bug: it selects nonzero indices with `touched[nvars]` instead of
    `touched[x]`, which is `True` for every real mechanism, so its multiply
    is never sparse.

    Args:
        touched: `write_jacobian()` output; `True` at each flattened
            `(row, col)` index that can be nonzero.
        nvars: Number of species (the Jacobian is `nvars` x `nvars`).
    """
    sparse_indices = [x for x in range(nvars * nvars) if touched[x]]

    (path / "sparse_multiplier.h").write_text(
        _HEADER_GUARD.format("SPARSE_HEAD")
        + f"\n#define N_A {len(sparse_indices)}"
        '\n#include "header.h"\n\n'
        "void sparse_multiplier (const double *, const double *, "
        "double*);\n\n"
        "#ifdef COMPILE_TESTING_METHODS\n"
        "  int test_sparse_multiplier();\n"
        "#endif\n\n#endif\n"
    )

    lines = [
        '#include "sparse_multiplier.h"\n\n'
        "void sparse_multiplier(const double * A, const double * Vm, "
        "double* w) {\n"
    ]
    for i in range(nvars):
        row = [x for x in sparse_indices if x % nvars == i]
        if not row:
            lines.append(f"  w[{i}] = 0;\n")
            continue
        terms = " + ".join(
            f"A[{index}] * Vm[{index // nvars}]" for index in row
        )
        lines.append(f"  w[{i}] = {terms};\n")
    lines.append("}\n")

    (path / "sparse_multiplier.c").write_text("".join(lines))


def _dbdt_species_expr(array_ref: str, coeffs) -> str:
    a0, a1, a2, a3, a4, a5, _a6 = coeffs
    return (
        f"{array_ref} = ({a0 - 1.0:.16e} + {a5:.16e} / T) / T"
        f" + {a1 / 2.0:.16e} + T * ({a2 / 3.0:.16e}"
        f" + T * ({a3 / 4.0:.16e} + {a4 / 5.0:.16e} * T))"
    )


def dbdt_declaration_lines(
    species: List[Species], reactions: List[Reaction],
) -> str:
    """Fills `dBdT[]` (temperature derivative of `B = S/R - H/(RT)`) for the
    species some reversible, `Kc`-derived reaction needs, bucketed by
    `t_mid` so each is computed once.

    `dBdT[]` must already be declared (`write_jacobian()` always does).
    Port of `create_jacobian.write_db_dt_def()` without the unused
    `do_unroll`.
    """
    reversible = [r for r in reactions if r.reversible]
    lines = []

    touched = [False] * len(species)
    t_mid_buckets = {}
    for rxn in reversible:
        if rxn.reverse_params is not None:
            continue
        for sp_ind in rxn.reactants + rxn.products:
            if touched[sp_ind]:
                continue
            touched[sp_ind] = True
            t_mid = species[sp_ind].t_mid
            t_mid_buckets.setdefault(t_mid, []).append(sp_ind)

    for t_mid, sp_indices in t_mid_buckets.items():
        lines.append(f"  if (T <= {t_mid}) {{\n")
        for sp_ind in sorted(sp_indices):
            expr = _dbdt_species_expr(
                f"dBdT[{sp_ind}]", species[sp_ind].low,
            )
            lines.append(f"    {expr};\n")
        lines.append("  } else {\n")
        for sp_ind in sorted(sp_indices):
            expr = _dbdt_species_expr(
                f"dBdT[{sp_ind}]", species[sp_ind].high,
            )
            lines.append(f"    {expr};\n")
        lines.append("  }\n\n")

    return "".join(lines)


def dbdt_expr(reaction: Reaction) -> str:
    """One reaction's dB/dT term of its reverse-rate temperature derivative;
    `dBdT[]` must already be computed.

    Port of `create_jacobian.get_db_dt()` without the unused `lang`/`specs`.
    """
    terms = []
    notfirst = False
    for sp_ind in reaction.products:
        if sp_ind in reaction.reactants:
            nu = (
                reaction.product_nu[reaction.products.index(sp_ind)]
                - reaction.reactant_nu[
                    reaction.reactants.index(sp_ind)
                ]
            )
        else:
            nu = reaction.product_nu[reaction.products.index(sp_ind)]

        if nu == 0:
            continue

        dbdt = f"dBdT[{sp_ind}]"

        if not notfirst:
            if nu == 1:
                terms.append(dbdt)
            elif nu == -1:
                terms.append("-" + dbdt)
            else:
                terms.append(f"{float(nu)} * {dbdt}")
        else:
            if nu == 1:
                terms.append(" + " + dbdt)
            elif nu == -1:
                terms.append(" - " + dbdt)
            else:
                sign = " + " if nu > 0 else " - "
                terms.append(f"{sign}{abs(float(nu))} * {dbdt}")
        notfirst = True

    for sp_ind in reaction.reactants:
        if sp_ind in reaction.products:
            continue
        nu = -reaction.reactant_nu[reaction.reactants.index(sp_ind)]
        dbdt = f"dBdT[{sp_ind}]"
        if nu == 1:
            terms.append(" + " + dbdt)
        elif nu == -1:
            terms.append(" - " + dbdt)
        else:
            sign = " + " if nu > 0 else " - "
            terms.append(f"{sign}{abs(float(nu))} * {dbdt}")

    return "".join(terms)


def write_pr(
    reaction: Reaction,
    last_conc_temp: Optional[List[Tuple[int, float]]] = None,
) -> Tuple[str, Optional[List[Tuple[int, float]]]]:
    """Builds `conc_temp` (the concentration scaling a falloff or third-body
    rate) and, for a falloff reaction, `Pr`; `conc_temp`/`Pr` must be
    declared and `conc[]` computed.

    Port of `create_jacobian.write_pr()` without the unused `specs`/
    `pdep_reacs`. As upstream, passing the previous reaction's returned
    `conc_temp_log` as `last_conc_temp` emits only the efficiency delta
    (or nothing if identical) when that is shorter than a recompute.

    Args:
        last_conc_temp: `conc_temp_log` of the preceding falloff/third-body
            reaction in a run; `None` recomputes from scratch.

    Returns:
        `(body, conc_temp_log)`; `conc_temp_log` is `None` unless the
        reaction uses the full efficiency-weighted mixture, else its
        `(species_index, efficiency - 1.0)` pairs.
    """
    line = "  conc_temp = "
    conc_temp_log = None
    if reaction.pdep_species is not None:
        line += f"conc[{reaction.pdep_species}]"
    elif not reaction.third_body_efficiencies:
        line += "m"
    else:
        conc_temp_log = []
        line += "(m"
        for isp, eff in reaction.third_body_efficiencies:
            if eff > 1.0:
                line += f" + {eff - 1.0} * "
            elif eff < 1.0:
                line += f" - {1.0 - eff} * "
            if eff != 1.0:
                line += f"conc[{isp}]"
                conc_temp_log.append((isp, eff - 1.0))
        line += ")"

        if last_conc_temp is not None:
            new_conc_temp = []
            for species, alpha in conc_temp_log:
                match = next(
                    (sp for sp in last_conc_temp if sp[0] == species),
                    None,
                )
                coeff = alpha - match[1] if match is not None else alpha
                if coeff != 0.0:
                    new_conc_temp.append((species, coeff))
            for species, alpha in last_conc_temp:
                match = next(
                    (sp for sp in conc_temp_log if sp[0] == species),
                    None,
                )
                if match is None:
                    new_conc_temp.append((species, -alpha))

            use_conc = (
                new_conc_temp
                if len(new_conc_temp) < len(conc_temp_log)
                else conc_temp_log
            )
            if use_conc:
                is_delta = use_conc == new_conc_temp
                line = "  conc_temp {}= ({}".format(
                    "+" if is_delta else "",
                    "" if is_delta else "m + ",
                )
                for i, (isp, alpha) in enumerate(use_conc):
                    if i > 0:
                        sign = "- " if alpha < 0 else "+ "
                        line += f" {sign}{abs(alpha)} * "
                    else:
                        line += f"{alpha} * "
                    line += f"conc[{isp}]"
                line += ")"
            else:
                line = ""

    body = f"{line};\n" if line else ""

    if reaction.pressure_dependent:
        _beta_0minf, _e_0minf, k0_over_kinf = falloff_low_over_high(
            reaction,
        )
        body += f"  Pr = conc_temp * ({k0_over_kinf});\n"

    return body, conc_temp_log


def write_troe(reaction: Reaction) -> str:
    """Builds the Troe intermediates `Fcent`, `A`, `B`, `lnF_AB`; they must be
    declared and `Pr` computed (`write_pr()`).

    Port of `create_jacobian.write_troe()`.
    """
    alpha, t3, t1 = reaction.troe_params[:3]
    line = (
        f"  Fcent = {1.0 - alpha:.16e} * exp(T / {-t3:.16e})"
        f" + {alpha:.16e} * exp(T / {-t1:.16e})"
    )
    if len(reaction.troe_params) == 4 and reaction.troe_params[3] != 0.0:
        t2 = reaction.troe_params[3]
        line += f" + exp({-t2:.16e} / T)"
    line += ";\n"

    line += (
        "  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * "
        "log10(fmax(Fcent, 1.0e-300)) - 0.4;\n"
    )
    line += (
        "  B = 0.806 - 1.1762 * log10(fmax(Fcent, 1.0e-300)) - "
        "0.14 * log10(fmax(Pr, 1.0e-300));\n"
    )
    line += (
        "  lnF_AB = 2.0 * log(fmax(Fcent, 1.0e-300)) * "
        "A / (B * B * B * (1.0 + A * A / (B * B)) * "
        "(1.0 + A * A / (B * B)));\n"
    )
    return line


def write_sri() -> str:
    """Builds the SRI intermediate `X`; it must be declared and `Pr` computed
    (`write_pr()`).

    Port of `create_jacobian.write_sri()`; the SRI parameters enter via
    `rates.write_rxn_pressure_mod()` and `write_sri_dt()` instead.
    """
    return (
        "  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * "
        "log10(fmax(Pr, 1.0e-300)));\n"
    )


def write_sri_dt(
    reaction: Reaction, beta_0minf: float, e_0minf: float,
) -> str:
    """SRI blend contribution appended to `get_pdep_dt()`'s still-open
    expression; `X`/`Pr` must be computed.

    Port of `create_jacobian.write_sri_dt()` without `lang`/`k0kinf`.

    Args:
        beta_0minf: First result of `falloff_low_over_high()`.
        e_0minf: Second result of `falloff_low_over_high()`.
    """
    a, b, c = reaction.sri_params[:3]
    jline = (
        f" + X * ((({a * b:.16} / (T * T)) * exp({-b:.16} / T)"
        f" - {1.0 / c:.16e} * exp(T / {-c:.16})) / "
        f"({a:.16} * exp({-b:.16} / T) + exp(T / {-c:.16})) - "
        f"X * {2.0 / math.log(10.0):.16} * "
        "log10(fmax(Pr, 1.0e-300)) * "
        f"({beta_0minf:.16e} + ({e_0minf:.16e} / T) - 1.0) * "
        f"log({a:.16} * exp({-b:.16} / T) + "
        f"exp(T / {-c:.16})) / T)"
    )
    if len(reaction.sri_params) == 5 and reaction.sri_params[4] != 0.0:
        jline += f" + ({reaction.sri_params[4]:.16} / T)"
    return jline


def write_troe_dt(
    reaction: Reaction, beta_0minf: float, e_0minf: float,
) -> str:
    """Troe blend contribution appended to `get_pdep_dt()`'s still-open
    expression; `Fcent`/`A`/`B`/`lnF_AB`/`Pr` must be computed.

    Port of `create_jacobian.write_troe_dt()` without `lang`/`k0kinf`.

    Args:
        beta_0minf: First result of `falloff_low_over_high()`.
        e_0minf: Second result of `falloff_low_over_high()`.
    """
    alpha, t3, t1 = reaction.troe_params[:3]
    log10 = math.log(10.0)
    jline = (
        " + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - "
        f"lnF_AB * ({-(0.67 / log10):.16e} * B + "
        f"{1.1762 / log10:.16e} * A) / Fcent)"
        f" * ({-(1.0 - alpha) / t3:.16e} * exp(T / {-t3:.16e})"
        f" - {alpha / t1:.16e} * exp(T / {-t1:.16e})"
    )
    if len(reaction.troe_params) == 4 and reaction.troe_params[3] != 0.0:
        t2 = reaction.troe_params[3]
        jline += f" + ({t2:.16e} / (T * T)) * exp({-t2:.16e} / T)"
    jline += "))"
    jline += (
        f" - lnF_AB * ({1.0 / log10:.16e} * B + "
        f"{0.14 / log10:.16e} * A) * "
        f"({beta_0minf:.16e} + ({e_0minf:.16e} / T) - 1.0) / T"
    )
    return jline


def get_pdep_dt(
    reaction: Reaction, rxn_ind: int, rev_idx: int, pres_rxn_ind: int,
) -> str:
    """Temperature-derivative term of a pressure-dependent reaction through
    its falloff-blend correction, as an intentionally unclosed C
    expression fragment that `write_jacobian()` completes.

    Port of `create_jacobian.get_pdep_dt()`, taking `rev_idx` instead of
    `rev_reacs` like its sibling functions.

    Args:
        rxn_ind: Index among all reactions (`fwd_rates[]`).
        rev_idx: Index among reversible reactions (`rev_rates[]`); only
            meaningful if `reversible`.
        pres_rxn_ind: Index among pressure-dependent/third-body reactions
            (`pres_mod[]`).
    """
    beta_0minf, e_0minf, _k0_over_kinf = falloff_low_over_high(reaction)

    jline = f"  j_temp = (pres_mod[{pres_rxn_ind}]"
    jline += " * (("
    if reaction.high_rate_params is not None:
        jline += "-Pr * "
    jline += (
        f"({beta_0minf:.4e} + ({e_0minf:.16e} / T) - 1.0)"
        " / (T * (1.0 + Pr)))"
    )

    if reaction.sri_params is not None:
        jline += write_sri_dt(reaction, beta_0minf, e_0minf)
    elif reaction.troe_params is not None:
        jline += write_troe_dt(reaction, beta_0minf, e_0minf)

    jline += ") * "

    if reaction.reversible:
        jline += f"(fwd_rates[{rxn_ind}] - rev_rates[{rev_idx}])"
    else:
        jline += f"fwd_rates[{rxn_ind}]"

    jline += f" + (pres_mod[{pres_rxn_ind}]"

    return jline


def _dcp_dt_bucket_line(
    species: List[Species], sp_indices: List[int], n: int,
    first: bool, low: bool,
) -> str:
    op = "=" if first else "+="
    line = f"    working_temp {op} "
    for i, isp in enumerate(sorted(sp_indices)):
        sp = species[isp]
        if i:
            line += "\n    + "
        y_str = f"y[{isp + 1}]" if isp + 1 != n else "y_N"
        _a0, a1, a2, a3, a4, _a5, _a6 = sp.low if low else sp.high
        line += (
            f"({y_str} * {GAS_CONSTANT_SI / sp.mw:.16e} * "
            f"({a1:.16e} + T * ({2.0 * a2:.16e}"
            f" + T * ({3.0 * a3:.16e} + {4.0 * a4:.16e} * T))))"
        )
    line += ";\n"
    return line


def write_dcp_dt(species: List[Species]) -> str:
    """Builds `working_temp`, the sum of each species' `dCp/dT` (NASA-7)
    weighted by mass fraction; `working_temp` must be declared and zeroed
    and `y[]`/`y_N` hold the state.

    Port of `create_jacobian.write_dcp_dt()`. Bucketed by `t_mid` like
    `dbdt_declaration_lines()`, but every species contributes.
    """
    t_mid_buckets = {}
    for isp, sp in enumerate(species):
        t_mid_buckets.setdefault(sp.t_mid, []).append(isp)

    n = len(species)
    lines = []
    first = True
    for t_mid in sorted(t_mid_buckets):
        sp_indices = t_mid_buckets[t_mid]
        lines.append(f"  if (T <= {t_mid}) {{\n")
        lines.append(
            _dcp_dt_bucket_line(species, sp_indices, n, first, True),
        )
        lines.append("  } else {\n")
        lines.append(
            _dcp_dt_bucket_line(species, sp_indices, n, first, False),
        )
        lines.append("  }\n\n")
        first = False

    return "".join(lines)


def get_elementary_rxn_dt(
    reaction: Reaction, rxn_ind: int, rev_idx: int,
) -> str:
    """Temperature-derivative term of a plain elementary reaction, as an
    intentionally unclosed C expression fragment (see `get_pdep_dt()`).

    Port of `create_jacobian.get_elementary_rxn_dt()` without `lang`/
    `get_array`/`do_unroll`/`specs` and three provably dead upstream
    conditions.

    Args:
        rxn_ind: Index among all reactions (`fwd_rates[]`).
        rev_idx: Index among reversible reactions (`rev_rates[]`); only
            meaningful if `reversible`.
    """
    jline = ""
    forward_params = (reaction.A, reaction.b, reaction.E)

    if reaction.reversible and reaction.reverse_params is not None:
        dk_dt = rate_constant_temperature_expr(forward_params)
        nu = sum(reaction.reactant_nu)
        if dk_dt or nu != 1.0:
            jline = f"fwd_rates[{rxn_ind}] * ("
            if dk_dt:
                jline += dk_dt
            if nu != 1.0:
                if dk_dt:
                    jline += " + "
                jline += f"{1.0 - float(nu)}"
            jline += ")"

        dk_dt = rate_constant_temperature_expr(reaction.reverse_params)
        nu = sum(reaction.product_nu)
        if dk_dt or nu != 1.0:
            jline += f" - rev_rates[{rev_idx}] * ("
            if dk_dt:
                jline += dk_dt
            if nu != 1.0:
                if dk_dt:
                    jline += " + "
                jline += f"{1.0 - float(nu)}"
            jline += ")"
    elif reaction.reversible:
        dk_dt = rate_constant_temperature_expr(forward_params)
        if dk_dt:
            jline += (
                f"(fwd_rates[{rxn_ind}] - rev_rates[{rev_idx}])"
                f" * ({dk_dt})"
            )

        nu = sum(reaction.reactant_nu)
        if nu != 1.0:
            if jline:
                jline += " + "
            jline += f"fwd_rates[{rxn_ind}] * {1.0 - float(nu)}"

        dbdt = dbdt_expr(reaction)
        nu = sum(reaction.product_nu)
        if dbdt or nu != 1.0:
            jline += " - " if jline else "-"
            jline += f"rev_rates[{rev_idx}] * ("
            if nu != 1.0:
                jline += f"{1.0 - float(nu)} + "
            if dbdt:
                jline += f"-T * ({dbdt}))"
    else:
        dk_dt = rate_constant_temperature_expr(forward_params)
        nu = sum(reaction.reactant_nu)
        if dk_dt or nu != 1.0:
            jline += f"fwd_rates[{rxn_ind}] * ("
            jline += dk_dt
            if nu != 1.0:
                if jline:
                    jline += " + "
                jline += f"{1.0 - float(nu)}"
            jline += ")"

    if jline:
        jline += ")) * rho_inv;\n"
    return jline


def write_cheb_ut(chebyshev: ChebyshevRate) -> str:
    """Builds `kf` through the Chebyshev temperature-derivative recurrence
    (`dU(Tred)/dTred`); `kf`/`cheb_temp_0`/`cheb_temp_1`/`dot_prod[]` must
    be declared and `Tred`/`Pred` computed.

    Port of `create_jacobian.write_cheb_ut()` without `lang`, indexing
    `coefficients[i][j]` as upstream's `cheb_par[i, j]`.
    """
    coeffs = chebyshev.coefficients
    n_temp = len(coeffs)
    n_pres = len(coeffs[0]) if n_temp else 0
    lines = ["cheb_temp_0 = 1", "cheb_temp_1 = Pred"]

    for i in range(1, n_temp):
        a_i0 = i * coeffs[i][0]
        a_i1 = i * coeffs[i][1]
        lines.append(f"dot_prod[{i}]= {a_i0:.16e} + Pred * {a_i1:.16e}")

    update_one = True
    for j in range(2, n_pres):
        new, old = (1, 0) if update_one else (0, 1)
        lines.append(
            f"cheb_temp_{old} = 2 * Pred * cheb_temp_{new}"
            f" - cheb_temp_{old}"
        )
        for i in range(1, n_temp):
            coeff = i * coeffs[i][j]
            lines.append(
                f"dot_prod[{i}] += {coeff:.16e} * cheb_temp_{old}"
            )
        update_one = not update_one

    lines.append("cheb_temp_0 = 1.0")
    lines.append("cheb_temp_1 = 2.0 * Tred")
    lines.append("kf = dot_prod[1] + 2.0 * Tred * dot_prod[2]")

    update_one = True
    for i in range(3, n_temp):
        new, old = (1, 0) if update_one else (0, 1)
        lines.append(
            f"cheb_temp_{old} = 2.0 * Tred * cheb_temp_{new}"
            f" - cheb_temp_{old}"
        )
        lines.append(f"kf += dot_prod[{i}] * cheb_temp_{old}")
        update_one = not update_one

    return "".join(f"  {line};\n" for line in lines)


def write_cheb_rxn_dt(
    jline: str, reaction: Reaction, rxn_ind: int, rev_idx: int,
) -> str:
    """Builds the complete temperature-derivative statement of a Chebyshev
    reaction (`Tred`/`Pred`, `U(Tred)` sum, Jacobian entry).

    Unlike `get_pdep_dt()` it closes the statement, as upstream. Port of
    `create_jacobian.write_cheb_rxn_dt()` without `lang`/`get_array`/
    `do_unroll`/`specs`.

    Args:
        jline: Caller's prefix of the statement (e.g. assignment target).
        rxn_ind: Index among all reactions (`fwd_rates[]`).
        rev_idx: Index among reversible reactions (`rev_rates[]`); only
            meaningful if `reversible`.
    """
    cheb = reaction.chebyshev
    t_low, t_high = cheb.t_range
    p_low, p_high = cheb.p_range
    tlim_inv_sum = 1.0 / t_low + 1.0 / t_high
    tlim_inv_sub = 1.0 / t_high - 1.0 / t_low
    plim_log_sum = math.log10(p_low) + math.log10(p_high)
    plim_log_sub = math.log10(p_high) - math.log10(p_low)

    body = (
        f"  Tred = ((2.0 / T) - {tlim_inv_sum:.16e})"
        f" / {tlim_inv_sub:.16e};\n"
        f"  Pred = (2.0 * log10(pres) - {plim_log_sum:.16e})"
        f" / {plim_log_sub:.16e};\n"
    )
    body += write_cheb_ut(cheb)

    jline += (
        f"kf * ({-2.0 * math.log(10.0) / tlim_inv_sub:.16e} / T)"
    )
    jline += f" * (fwd_rates[{rxn_ind}]"
    if reaction.reversible:
        jline += f" - rev_rates[{rev_idx}]"
    jline += ")"

    nu = sum(reaction.reactant_nu)
    if nu != 1.0:
        jline += f" + fwd_rates[{rxn_ind}] * {1.0 - float(nu)}"

    if reaction.reversible:
        jline += f" - rev_rates[{rev_idx}] * ("
        nu = sum(reaction.product_nu)
        if nu != 1.0:
            jline += f"{1.0 - float(nu)} + "
        jline += f"-T * ({dbdt_expr(reaction)}))"

    jline += ")) * rho_inv;\n"
    return body + jline


def write_plog_rxn_dt(
    jline: str, reaction: Reaction, rxn_ind: int, rev_idx: int,
) -> str:
    """Builds the complete pressure-bracketed temperature-derivative
    statements of a PLOG reaction (an `if`/`else if` chain over its
    pressure points).

    Port of `create_jacobian.write_plog_rxn_dt()` without `lang`/
    `get_array`/`do_unroll`/`specs`. Outside the first and last point the
    derivative is Arrhenius at that point, via a synthetic single-point
    `Reaction`; like upstream's `ReacInfo` rebuild this drops explicit
    `reverse_params`, so those two ranges always use the
    equilibrium-constant reverse rate. Between points it interpolates
    log-linearly in pressure; negative `A_p2/A_p1` ratios raise
    `NotImplementedError` as upstream, and its assertion is a `ValueError`.

    Args:
        jline: Caller's statement prefix, reused unclosed in every branch.
        rxn_ind: Index among all reactions (`fwd_rates[]`).
        rev_idx: Index among reversible reactions (`rev_rates[]`); only
            meaningful if `reversible`.
    """
    plog = reaction.plog

    def boundary_reaction(a, b, e):
        return Reaction(
            reactants=reaction.reactants,
            reactant_nu=reaction.reactant_nu,
            products=reaction.products,
            product_nu=reaction.product_nu,
            A=a, b=b, E=e, reversible=reaction.reversible,
        )

    p1, a_p1, b_p1, e_p1 = plog[0]
    dkdt = get_elementary_rxn_dt(
        boundary_reaction(a_p1, b_p1, e_p1), rxn_ind, rev_idx,
    )

    body = []
    have_prev = False
    if dkdt:
        body.append(f"  if (pres <= {p1:.4e}) {{\n")
        body.append(f"  {jline}{dkdt}")
        have_prev = True

    for idx in range(len(plog) - 1):
        p1, a_p1, b_p1, e_p1 = plog[idx]
        p2, a_p2, b_p2, e_p2 = plog[idx + 1]

        if a_p2 / a_p1 < 0:
            raise NotImplementedError(
                "PLOG reactions with A_p2 / A_p1 < 0 aren't supported"
            )
        if not (b_p1 or e_p1 or b_p2 or e_p2):
            raise ValueError("PLOG derivative undefined")

        jline_p = ""
        if b_p1 != 0.0:
            jline_p += f"{b_p1:.16e}"
        if e_p1 != 0.0:
            if jline_p:
                jline_p += " + "
            jline_p += f"{e_p1:.16e} / T"
        if b_p2 - b_p1 != 0.0 or e_p2 - e_p1 != 0.0:
            if jline_p:
                jline_p += " + "
            jline_p += "("
            if b_p2 - b_p1 != 0.0:
                jline_p += f"{b_p2 - b_p1:.16e} + "
            if e_p2 - e_p1 != 0.0:
                jline_p += f"{e_p2 - e_p1:.16e} / T) * (log(pres)"
                if p1 != 1.0:
                    jline_p += f" - {math.log(p1):.16e}"
                jline_p += ") / "
                if p1 == p2:
                    raise ValueError(
                        "Cannot have equal pressures in PLOG"
                    )
                jline_p += f"{math.log(p2) - math.log(p1):.16e})"
        else:
            jline_p += ")"

        jline_p += f" * (fwd_rates[{rxn_ind}]"
        if reaction.reversible:
            jline_p += f" - rev_rates[{rev_idx}]"
        jline_p += ")"

        nu = sum(reaction.reactant_nu)
        if nu != 1.0:
            if jline_p:
                jline_p += " + "
            jline_p += f"fwd_rates[{rxn_ind}] * {1.0 - float(nu)}"

        if reaction.reversible:
            nu = sum(reaction.product_nu)
            dbdt = dbdt_expr(reaction)
            if nu != 1.0 or dbdt:
                jline_p += " - " if jline_p else "-"
                jline_p += f"rev_rates[{rev_idx}] * ("
                if nu != 1.0:
                    jline_p += f"{1.0 - float(nu)} + "
                if dbdt:
                    jline_p += f"-T * ({dbdt})"
                jline_p += ")"

        if jline_p:
            jline_p = f"{jline}({jline_p})) * rho_inv"
        else:
            jline_p = f"{jline}0.0e0)"

        if have_prev:
            body.append(
                f"  }} else if ((pres > {p1:.4e})"
                f" && (pres <= {p2:.4e})) {{\n"
            )
        else:
            body.append(
                f"  if ((pres > {p1:.4e})"
                f" && (pres <= {p2:.4e})) {{\n"
            )
        have_prev = True
        body.append(f"  {jline_p};\n")

    pn, a_pn, b_pn, e_pn = plog[-1]
    dkdt = get_elementary_rxn_dt(
        boundary_reaction(a_pn, b_pn, e_pn), rxn_ind, rev_idx,
    )
    if dkdt:
        if have_prev:
            body.append(f"  }} else if (pres > {pn:.4e}) {{\n")
        else:
            body.append("  j_temp = 0;\n")
            body.append(f"  if (pres > {pn:.4e}) {{\n")
        body.append(f"  {jline}{dkdt}")
        body.append("  }\n")

    return "".join(body)


def write_dt_completion(
    species: List[Species], j_nplusone_touched: bool,
) -> str:
    """Finishes `jac[0]` (`d(dT/dt)/dT`) from every species' enthalpy and
    heat-capacity terms plus its `dY/dT` entry.

    `working_temp`/`cp_avg`/`rho`/`spec_rates[]`/`h[]`/`cp[]` must be
    computed and the `jac[k+1]`/`J_nplusone` entries written. Port of
    `create_jacobian.write_dt_completion()`.

    Args:
        j_nplusone_touched: Whether any nonzero `J_nplusone` (the last
            species' `dY/dT`, outside `jac[]`) was written; if not, its
            `+ J_nplusone * h * rho` term is omitted.
    """
    n = len(species)
    lines = ["  //Complete dT wrt T calculations\n", "  jac[0] = -("]

    for k_sp, sp_k in enumerate(species):
        if k_sp:
            lines.append("\n    + ")
        lines.append(
            f"spec_rates[{k_sp}] * {sp_k.mw:.8e} * "
            f"(-working_temp * h[{k_sp}] / cp_avg + cp[{k_sp}])"
        )
        if k_sp + 1 == n:
            j_str = "J_nplusone[0]"
        else:
            j_str = f"jac[{k_sp + 1}]"
        if k_sp + 1 < n or j_nplusone_touched:
            lines.append(f" + {j_str} * h[{k_sp}] * rho")

    lines.append(") / (rho * cp_avg);\n")
    return "".join(lines)


def write_dr_dy(
    reaction: Reaction, rxn_ind: int, rev_idx: int, pres_rxn_ind: int,
) -> str:
    """Builds the species-independent part of a reaction's rate `R`
    (`pres_mod_temp`, if needed) and of `dR/dYj` (`j_temp`);
    `write_dr_dy_species()` completes them per species.

    `mw_avg`/`rho_inv`/`fwd_rates[]`/`rev_rates[]`/`pres_mod[]` and, for
    falloff, `Pr`/`Fcent`/`A`/`B`/`X` must be computed. Port of
    `create_jacobian.write_dr_dy()` taking `rev_idx` instead of
    `rev_reacs`. `pdep_species` is tested with `is not None`, since index
    `0` is falsy (upstream's is `''` when unset).

    Args:
        rxn_ind: Index among all reactions (`fwd_rates[]`).
        rev_idx: Index among reversible reactions (`rev_rates[]`); only
            meaningful if `reversible`.
        pres_rxn_ind: Index among pressure-dependent/third-body reactions
            (`pres_mod[]`).
    """
    lines = []
    has_pdep_sp_or_thd = bool(reaction.third_body_efficiencies) or (
        reaction.pdep_species is not None
    )

    if (
        reaction.pressure_dependent or reaction.third_body
    ) and has_pdep_sp_or_thd:
        jline = "  pres_mod_temp = "
        if reaction.pressure_dependent:
            jline += "("
            if reaction.low_rate_params is not None:
                jline += "(1.0 / (1.0 + Pr))"
            elif reaction.high_rate_params is not None:
                jline += "(-Pr / (1.0 + Pr))"
            if reaction.troe_params is not None:
                jline += (
                    " - log(fmax(Fcent, 1.0e-300)) * 2.0 * A * (B * "
                    f"{1.0 / math.log(10.0):.16}"
                    " + A * "
                    f"{0.14 / math.log(10.0):.16}) / "
                    "(B * B * B * (1.0 + A * A / (B * B)) "
                    "* (1.0 + A * A / (B * B)))"
                )
            elif reaction.sri_params is not None:
                a, b, c = reaction.sri_params[:3]
                jline += (
                    "- X * X * "
                    f"{2.0 / math.log(10.0):.16} * "
                    "log10(fmax(Pr, 1.0e-300)) * "
                    f"log({a:.4} * exp({-b:.4} / T) + "
                    f"exp(T / {-c:.4}))"
                )
            jline += ") * "
        if reaction.reversible:
            jline += f"(fwd_rates[{rxn_ind}] - rev_rates[{rev_idx}])"
        else:
            jline += f"fwd_rates[{rxn_ind}]"
        lines.append(f"{jline};\n")

    jline = "  j_temp = -mw_avg * rho_inv * "
    if reaction.pressure_dependent or reaction.third_body:
        jline += f"pres_mod[{pres_rxn_ind}] * ("
    else:
        jline += "("

    reac_nu = 0.0
    prod_nu = 0.0
    if reaction.third_body_efficiencies and not reaction.pressure_dependent:
        reac_nu = 1.0
        if reaction.reversible:
            prod_nu = 1.0

    reac_nu += sum(reaction.reactant_nu)
    if reaction.reversible:
        prod_nu += sum(reaction.product_nu)

    if reac_nu != 0:
        if reac_nu != 1:
            jline += f"{float(reac_nu)} * "
        jline += f"fwd_rates[{rxn_ind}]"

    if prod_nu != 0:
        if prod_nu == 1:
            jline += " - "
        else:
            jline += f" - {float(prod_nu)} * "
        jline += f"rev_rates[{rev_idx}]"

    if reaction.pressure_dependent and has_pdep_sp_or_thd:
        jline += " + pres_mod_temp"
    jline += ")"
    lines.append(f"{jline};\n")

    if reaction.pressure_dependent and has_pdep_sp_or_thd:
        if reaction.low_rate_params is not None:
            k0 = reaction.low_rate_params
            kinf = (reaction.A, reaction.b, reaction.E)
        else:
            k0 = (reaction.A, reaction.b, reaction.E)
            kinf = reaction.high_rate_params
        jline = "  pres_mod_temp *= "
        jline += forward_rate_expr(
            k0[0] / kinf[0], k0[1] - kinf[1], k0[2] - kinf[2],
        )
        if reaction.troe_params is not None:
            jline += " * pow(Fcent, 1.0 / (1 + A * A / (B * B)))"
        elif reaction.sri_params is not None:
            a, b, c = reaction.sri_params[:3]
            jline += f"* pow({a:.6} * "
            if b > 0.0:
                jline += f"exp(-{b:.6} / T)"
            else:
                jline += f"exp({abs(b):.6} / T)"
            if c > 0.0:
                jline += f" + exp(-T / {c:.6}), X) "
            else:
                jline += f" + exp(T / {abs(c):.6}), X) "
            if (
                len(reaction.sri_params) == 5
                and reaction.sri_params[3] != 1.0
                and reaction.sri_params[4] != 0.0
            ):
                d, e = reaction.sri_params[3], reaction.sri_params[4]
                jline += f"* {d:.8e} * pow(T, {e:.6}) "
        jline += " / (1.0 + Pr)"
        lines.append(f"{jline};\n")

    return "".join(lines)


def _get_s_term(reaction: Reaction, sp_ind: int, reac: bool) -> str:
    jline = "kf" if reac else "kr"
    species_list = reaction.reactants if reac else reaction.products
    nu_list = reaction.reactant_nu if reac else reaction.product_nu
    nu = nu_list[species_list.index(sp_ind)]
    if nu != 1:
        jline += f" * {float(nu)}"

    if (nu - 1) > 0:
        if float(nu).is_integer():
            for _ in range(int(nu) - 1):
                jline += f" * conc[{sp_ind}]"
        else:
            jline += f" * pow(conc[{sp_ind}], {nu - 1})"

    for i, isp in enumerate(species_list):
        if isp == sp_ind:
            continue
        this_nu = nu_list[i]
        if float(this_nu).is_integer():
            for _ in range(int(this_nu)):
                jline += f" * conc[{isp}]"
        else:
            jline += f" * pow(conc[{isp}], {this_nu})"

    return jline


def write_dr_dy_species(
    species: List[Species], reaction: Reaction, pres_rxn_ind: int,
    j_sp: int, sp_j: Species,
) -> str:
    """Completes `dR/dYj` per species: `j_temp` plus the species'
    third-body/falloff contribution and the concentration-power-law terms
    of this and (via the dependent species) the last species.

    Port of `create_jacobian.write_dr_dy_species()` with its nested
    `__get_s_term()` as `_get_s_term()`, without `lang`/`get_array`/
    `rxn_ind`/`rev_reacs` (unused) and three dead `if jline:` guards.
    `pdep_species` is tested with `is not None` (see `write_dr_dy()`).

    Args:
        pres_rxn_ind: Index among pressure-dependent/third-body reactions
            (`pres_mod[]`).
        j_sp: Species index of this Jacobian column.
        sp_j: `species[j_sp]`.
    """
    jline = "j_temp"
    last_spec = len(species) - 1
    mw_frac = sp_j.mw / species[last_spec].mw
    jline += f" * {1.0 - mw_frac:.16e}"

    if (
        (reaction.pressure_dependent and reaction.pdep_species is None)
        or reaction.third_body
    ) and reaction.third_body_efficiencies:
        alphaij = next(
            (eff for isp, eff in reaction.third_body_efficiencies
             if isp == j_sp), 1.0,
        )
        alphai_nspec = next(
            (eff for isp, eff in reaction.third_body_efficiencies
             if isp == last_spec), 1.0,
        )
        if alphai_nspec != 0:
            alphaij -= alphai_nspec * mw_frac
        if alphaij != 0:
            if alphaij != 1:
                if alphaij == -1:
                    jline += " - pres_mod_temp"
                else:
                    jline += f" + {alphaij:.16e} * pres_mod_temp"
            else:
                jline += " + pres_mod_temp"
    elif (
        reaction.pdep_species == j_sp
        or reaction.pdep_species == last_spec
    ):
        if reaction.pdep_species == j_sp:
            jline += " + pres_mod_temp"
        else:
            ratio = sp_j.mw / species[reaction.pdep_species].mw
            jline += f" - pres_mod_temp * {ratio:.16e}"

    s_term = ""
    if (reaction.pressure_dependent or reaction.third_body) and (
        j_sp in reaction.reactants
        or last_spec in reaction.reactants
        or (
            reaction.reversible
            and (
                j_sp in reaction.products
                or last_spec in reaction.products
            )
        )
    ):
        s_term += f" + pres_mod[{pres_rxn_ind}] * ("

    j_sp_add = False
    if j_sp in reaction.reactants or (
        reaction.reversible and j_sp in reaction.products
    ):
        j_sp_add = True
        add = ""
        if j_sp in reaction.reactants:
            if not s_term:
                add += " + "
            add += _get_s_term(reaction, j_sp, True)
        if reaction.reversible and j_sp in reaction.products:
            if not s_term:
                add += " - "
            elif s_term[-1] == "(":
                add += "-"
            add += _get_s_term(reaction, j_sp, False)
        s_term += add

    if last_spec in reaction.reactants or (
        reaction.reversible and last_spec in reaction.products
    ):
        pre = f"{mw_frac:.16e}"
        add = ""
        if j_sp_add:
            s_term += " - "
        else:
            if s_term and s_term[-1] == "(":
                pre = "-" + pre
            else:
                pre = " - " + pre
        if last_spec in reaction.reactants:
            add += _get_s_term(reaction, last_spec, True)
        if reaction.reversible and last_spec in reaction.products:
            if add:
                add += " - "
            else:
                add += "-"
            add += _get_s_term(reaction, last_spec, False)
        s_term += f"{pre} * ({add})"

    if (reaction.pressure_dependent or reaction.third_body) and s_term:
        s_term += ")"

    return jline + s_term


def _kc_bucket_expr(terms) -> str:
    c0, c1, c2, c3, c4, c5, c6 = terms
    return (
        f"({c0:.16e} + {c1:.16e} * logT + T * ({c2:.16e}"
        f" + T * ({c3:.16e} + T * ({c4:.16e} + {c5:.16e} * T))) - "
        f"{c6:.16e} / T)"
    )


def write_kc(species: List[Species], reaction: Reaction) -> str:
    """Builds the equilibrium constant `Kc` of a reaction whose reverse rate
    derives from it; `Kc`/`logT` must be declared.

    Port of `create_jacobian.write_kc()`, iterating species in
    `sorted(set(reactants + products))` order. Not shared with
    `rates.equilibrium_constant_lines()`, whose term order differs
    (same value).
    """
    sum_nu = 0.0
    coeffs = {}
    for isp in sorted(set(reaction.reactants + reaction.products)):
        sp = species[isp]
        nu = net_nu(isp, reaction)
        if nu == 0:
            continue
        sum_nu += nu

        lo = sp.low
        lo_terms = [
            nu * (lo[6] - lo[0]), nu * (lo[0] - 1.0), nu * lo[1] / 2.0,
            nu * lo[2] / 6.0, nu * lo[3] / 12.0, nu * lo[4] / 20.0,
            nu * lo[5],
        ]
        hi = sp.high
        hi_terms = [
            nu * (hi[6] - hi[0]), nu * (hi[0] - 1.0), nu * hi[1] / 2.0,
            nu * hi[2] / 6.0, nu * hi[3] / 12.0, nu * hi[4] / 20.0,
            nu * hi[5],
        ]

        t_mid = sp.t_mid
        if t_mid not in coeffs:
            coeffs[t_mid] = (lo_terms, hi_terms)
        else:
            prev_lo, prev_hi = coeffs[t_mid]
            coeffs[t_mid] = (
                [a + b for a, b in zip(lo_terms, prev_lo)],
                [a + b for a, b in zip(hi_terms, prev_hi)],
            )

    lines = []
    is_first = True
    for t_mid in sorted(coeffs):
        lo_terms, hi_terms = coeffs[t_mid]
        op = "=" if is_first else "+="
        lines.append(f"  if (T <= {t_mid}) {{\n")
        lines.append(f"    Kc {op} {_kc_bucket_expr(lo_terms)};\n")
        lines.append("  } else {\n")
        lines.append(f"    Kc {op} {_kc_bucket_expr(hi_terms)};\n")
        lines.append("  }\n\n")
        is_first = False

    line = "  Kc = "
    if sum_nu != 0:
        num = (STANDARD_PRESSURE / GAS_CONSTANT_SI) ** sum_nu
        line += f"{num:.16e} * "
    line += "exp(Kc);\n"
    lines.append(line)
    return "".join(lines)


#: Reactions per `jacob_rxn_<i>.c` chunk and species rows per
#: `jacob_finish_<i>.c` chunk, small enough for `-O3` to compile a chunk
#: in about a second.
_JAC_RXN_CHUNK_SIZE = 10
_JAC_SPECIES_CHUNK_SIZE = 6

_RXN_CHUNK_PARAMS = (
    "const double T, const double pres, "
    "const double * __restrict__ conc, "
    "const double * __restrict__ fwd_rates, "
    "const double * __restrict__ rev_rates, "
    "const double * __restrict__ pres_mod, "
    "const double * __restrict__ dBdT, "
    "const double mw_avg, const double rho_inv, "
    "double * __restrict__ jac, double * __restrict__ J_nplusone, "
    "double * __restrict__ J_nplusjplus"
)
_FINISH_CHUNK_PARAMS = (
    "const double * __restrict__ h, const double * __restrict__ cp, "
    "const double * __restrict__ spec_rates, const double mw_avg, "
    "const double rho_inv, const double working_temp, "
    "const double energy_factor, double * __restrict__ jac, "
    "double * __restrict__ J_nplusjplus"
)


# Declared in every chunk function, even where unused; simpler than
# tracking which reaction types land in which chunk.
def _reaction_scratch_declarations(
    reactions: List[Reaction], num_s: int,
) -> str:
    lines = [
        f"  double m = pres / ({GAS_CONSTANT_SI:.8e} * T);\n",
        "  double conc_temp;\n",
        "  double logT = log(T);\n",
        "  double j_temp = 0.0;\n",
        "  double kf = 0.0;\n",
        "  double pres_mod_temp = 0.0;\n",
        "  double Kc = 0.0;\n",
        "  double kr = 0;\n",
        "  double Pr = 0.0;\n",
    ]
    lines.extend(f"  double {x} = 0.0;\n" for x in ["Fcent", "A", "B"])
    lines.append("  double lnF_AB = 0.0;\n")
    lines.append("  double X = 0.0;\n")
    lines.append("  double Tred, Pred;\n")
    lines.append("  double cheb_temp_0, cheb_temp_1;\n")
    cheb = [r for r in reactions if r.chebyshev is not None]
    dim = max((len(r.chebyshev.coefficients) for r in cheb), default=1)
    lines.append(f"  double dot_prod[{dim}];\n")
    lines.append("  double kf2;\n")
    return "".join(lines)


def write_jacobian(
    path: Path, species: List[Species], reactions: List[Reaction],
    seen_sp: List[bool], debug: bool = False,
    clamp_state: bool = False,
) -> List[bool]:
    """Writes `jacob.h`/`jacob.c` (an `eval_jacob()` dispatcher) and
    `jacob_rxn_<i>.c`/`jacob_finish_<i>.c` chunk files: the analytical
    Jacobian `d(dY_k/dt)/dY_j` and `d(dT/dt)/dY_j` (index `0` is
    temperature) with the dependent-species convention.

    Port of `create_jacobian.write_jacobian()` without CUDA `do_unroll`/
    `smm`/`lang`. Chunking replaces one huge `eval_jacob()`, which GCC
    `-O3` compiles superlinearly slowly, and lets `wmake -j` build chunks
    in parallel; `_JAC_RXN_CHUNK_SIZE` reactions and
    `_JAC_SPECIES_CHUNK_SIZE` species rows per function. `J_nplusjplus` is
    declared with a literal size, not the macro `NSP`.

    Args:
        seen_sp: Return value of `write_spec_rates()`; gates species'
            enthalpy contributions to `d(dT/dt)/dY_j` that would be zero.
        debug: `eval_jacob()` overwrites the debug state file like
            `write_derivs()`.
        clamp_state: `eval_jacob()` clamps its input state like
            `write_derivs()`.

    Returns:
        The flattened `(row, col)` "touched" list, `True` where an entry
        can be nonzero; pass it to `write_sparse_multiplier()`.
    """
    num_s = len(species)
    num_r = len(reactions)
    rev_indices = [i for i, r in enumerate(reactions) if r.reversible]
    num_rev = len(rev_indices)
    pdep_indices = [
        i for i, r in enumerate(reactions)
        if r.third_body or r.pressure_dependent
    ]
    num_pdep = len(pdep_indices)

    touched = [False] * (num_s * num_s)
    j_nplusone_touched = False
    j_nplusjplus_touched = [False] * num_s
    last_conc_temp = None

    rxn_chunk_lines: List[List[str]] = []
    for rxn_ind, reaction in enumerate(reactions):
        if rxn_ind % _JAC_RXN_CHUNK_SIZE == 0:
            # `write_pr()`'s delta optimization needs the previous
            # `conc_temp` in scope, which a chunk boundary breaks.
            last_conc_temp = None
        lines: List[str] = []
        rev_idx = (
            rev_indices.index(rxn_ind) if reaction.reversible else -1
        )

        # with respect to temperature
        lines.append(f"  //partial of rxn {rxn_ind} wrt T\n")

        jline = ""
        pres_rxn_ind = -1
        if reaction.pressure_dependent:
            pres_rxn_ind = pdep_indices.index(rxn_ind)
            pr_body, last_conc_temp = write_pr(reaction, last_conc_temp)
            lines.append(pr_body)
            if reaction.troe_params is not None:
                lines.append(write_troe(reaction))
            elif reaction.sri_params is not None:
                lines.append(write_sri())
            jline = get_pdep_dt(reaction, rxn_ind, rev_idx, pres_rxn_ind)
        elif reaction.third_body:
            pres_rxn_ind = pdep_indices.index(rxn_ind)
            jline = f"  j_temp = ((-pres_mod[{pres_rxn_ind}] * "
            if reaction.reversible:
                jline += (
                    f"(fwd_rates[{rxn_ind}] - rev_rates[{rev_idx}])"
                )
            else:
                jline += f"fwd_rates[{rxn_ind}]"
            jline += f" / T) + (pres_mod[{pres_rxn_ind}]"
        else:
            jline = "  j_temp = ((1.0"

        jline += " / T) * ("

        do_t = True
        if reaction.plog is not None:
            lines.append(
                write_plog_rxn_dt(jline, reaction, rxn_ind, rev_idx),
            )
        elif reaction.chebyshev is not None:
            lines.append(
                write_cheb_rxn_dt(jline, reaction, rxn_ind, rev_idx),
            )
        else:
            dkdt = get_elementary_rxn_dt(reaction, rxn_ind, rev_idx)
            if dkdt:
                lines.append(jline + dkdt)
            else:
                do_t = False

        if do_t:
            participating = sorted(
                set(reaction.reactants + reaction.products),
            )
            for k_sp in participating:
                sp_k = species[k_sp]
                nu = net_nu(k_sp, reaction)
                if nu == 0:
                    continue
                if k_sp + 1 == num_s:
                    j_str = "J_nplusone[0]"
                else:
                    j_str = f"jac[{k_sp + 1}]"
                sign = "" if nu == 1 else ("-" if nu == -1 else "")
                middle = "" if nu in (1, -1) else f" * {float(nu)}"
                op = "+" if touched[k_sp + 1] else ""
                lines.append(
                    f"  {j_str} {op}= {sign}j_temp{middle} * "
                    f"{sp_k.mw:.16e};\n"
                )
                if k_sp + 1 == num_s:
                    j_nplusone_touched = True
                else:
                    touched[k_sp + 1] = True
            lines.append("\n")

        # with respect to species
        lines.append(f"  //partial of rxn {rxn_ind} wrt species\n")

        if reaction.reversible and reaction.reverse_params is None:
            lines.append(write_kc(species, reaction))

        lines.append(
            write_dr_dy(reaction, rxn_ind, rev_idx, pres_rxn_ind),
        )
        lines.append(write_rates(reaction))

        for j_sp, sp_j in enumerate(species[:-1]):
            dr_dyj = write_dr_dy_species(
                species, reaction, pres_rxn_ind, j_sp, sp_j,
            )
            for k_sp in sorted(
                set(reaction.reactants + reaction.products),
            ):
                sp_k = species[k_sp]
                nu = net_nu(k_sp, reaction)
                if nu == 0:
                    continue

                jline = "  "
                if k_sp + 1 < num_s:
                    lin_index = k_sp + 1 + num_s * (j_sp + 1)
                    op = "+" if touched[lin_index] else ""
                    jline += f"jac[{lin_index}] {op}= "
                    touched[lin_index] = True
                else:
                    op = "+" if j_nplusjplus_touched[j_sp] else ""
                    jline += f"J_nplusjplus[{j_sp}] {op}= "
                    j_nplusjplus_touched[j_sp] = True

                mw_frac = (sp_k.mw / sp_j.mw) * float(nu)
                if mw_frac == -1.0:
                    working_temp = " -"
                elif mw_frac != 1.0:
                    working_temp = f" {mw_frac:.16e} * "
                else:
                    working_temp = " "
                working_temp += f"({dr_dyj})"

                jline += working_temp + ";\n"
                lines.append(jline)

        lines.append("\n")
        rxn_chunk_lines.append(lines)

    # Temperature row: the k_sp/j_sp "finish" loop, chunked by k_sp.
    finish_chunk_lines: List[List[str]] = []
    for k_sp, sp_k in enumerate(species):
        lines = []
        for j_sp, sp_j in enumerate(species):
            if j_sp + 1 == num_s:
                continue
            lin_index = k_sp + 1 + num_s * (j_sp + 1)

            factor = (sp_k.mw / sp_j.mw) * (
                1.0 - sp_j.mw / species[-1].mw
            )
            if k_sp + 1 < num_s and touched[lin_index]:
                lines.append(
                    f"  jac[{lin_index}] += (spec_rates[{k_sp}] "
                    f"* mw_avg * {factor:.16e} * rho_inv);\n"
                )
            elif k_sp + 1 == num_s and j_nplusjplus_touched[j_sp]:
                lines.append(
                    f"  J_nplusjplus[{j_sp}] += (spec_rates[{k_sp}] "
                    f"* mw_avg * {factor:.16e} * rho_inv);\n"
                )

            my_index = num_s * (j_sp + 1)
            sign1 = "-" if touched[my_index] else ""
            sign2 = "" if touched[my_index] else "-"
            line = f"  jac[{my_index}] {sign1}= {sign2}("
            touched[my_index] = True

            if k_sp + 1 < num_s:
                if touched[lin_index]:
                    jac_part = f"working_temp * jac[{lin_index}] - "
                else:
                    jac_part = "-"
            else:
                if j_nplusjplus_touched[j_sp]:
                    jac_part = (
                        f"working_temp * J_nplusjplus[{j_sp}] - "
                    )
                else:
                    jac_part = "-"

            sp_part = (
                f"(energy_factor * (cp[{j_sp}] - cp[{num_s - 1}]) * "
                f"spec_rates[{k_sp}] * {sp_k.mw:.8e}))"
            )

            line += f"h[{k_sp}] * ({jac_part}{sp_part});\n"
            if jac_part != "-" or seen_sp[k_sp]:
                lines.append(line)
        finish_chunk_lines.append(lines)

    rxn_scratch = _reaction_scratch_declarations(reactions, num_s)
    rxn_names = _write_chunk_files(
        path, "jacob_rxn", rxn_chunk_lines, _JAC_RXN_CHUNK_SIZE,
        _RXN_CHUNK_PARAMS, rxn_scratch,
    )
    finish_names = _write_chunk_files(
        path, "jacob_finish", finish_chunk_lines,
        _JAC_SPECIES_CHUNK_SIZE, _FINISH_CHUNK_PARAMS, "",
    )

    (path / "jacob.h").write_text(
        _HEADER_GUARD.format("JACOB_HEAD")
        + '\n#include "header.h"\n'
        '#include "chem_utils.h"\n#include "rates.h"\n'
        "void eval_jacob (const double, const double, "
        "const double * __restrict__, double * __restrict__);\n"
        + "".join(
            f"void {name} ({_RXN_CHUNK_PARAMS});\n" for name in rxn_names
        )
        + "".join(
            f"void {name} ({_FINISH_CHUNK_PARAMS});\n"
            for name in finish_names
        )
        + "\n#endif\n"
    )

    debug_includes = (
        "#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n"
        if debug else ""
    )
    y_param = "y_in" if clamp_state else "y"
    body = [
        f'{debug_includes}'
        '#include "jacob.h"\n\n'
        "void eval_jacob (const double t, const double pres, "
        f"const double * __restrict__ {y_param}, "
        "double * __restrict__ jac) {\n\n"
    ]
    if clamp_state:
        body.append(state_clamp_lines(num_s))

    body.append("  double T = y[0];\n\n")
    body.append("  // average molecular weight\n  double mw_avg;\n")
    body.append("  // mass-averaged density\n  double rho;\n")
    body.append("  // species molar concentrations\n")
    body.append(f"  double conc[{num_s}];\n")
    body.append("  double y_N;\n")
    body.append(
        "  eval_conc(y[0], pres, &y[1], &y_N, &mw_avg, &rho, conc);\n\n"
    )
    if debug:
        body.append(
            "  // DEBUG: overwrite (not append) a small per-process\n"
            "  // file with the exact state fed to eval_rxn_rates() -\n"
            "  // unconditionally, not just when something \"looks\"\n"
            "  // out of range, so it catches whatever the real\n"
            "  // failure mode turns out to be. Overwriting keeps disk\n"
            "  // usage bounded across a real run (this runs on every\n"
            "  // Jacobian evaluation) - only the LAST state before a\n"
            "  // crash is ever needed, and that's exactly what\n"
            "  // survives. Same file dydt() writes (last writer wins)\n"
            "  // so whichever one actually crashes, its state is what\n"
            "  // ends up on disk.\n"
            "  {\n"
            '    const char *_dbg_dir = getenv("FOAM_CASE");\n'
            "    char _dbg_path[4096];\n"
            "    snprintf(_dbg_path, sizeof(_dbg_path), "
            '"%s/pyjac_debug_state_%d.log",\n'
            '             _dbg_dir ? _dbg_dir : ".", (int) getpid());\n'
            '    FILE *_dbgf = fopen(_dbg_path, "w");\n'
            "    if (_dbgf) {\n"
            '      fprintf(_dbgf, "SOURCE=jacob T=%.17g pres=%.17g '
            'rho=%.17g mw_avg=%.17g y_N=%.17g\\n",\n'
            "              T, pres, rho, mw_avg, y_N);\n"
            '      fprintf(_dbgf, "CONC");\n'
            f"      for (int _i = 0; _i < {num_s}; _i++) {{\n"
            '        fprintf(_dbgf, " %.17g", conc[_i]);\n'
            "      }\n"
            '      fprintf(_dbgf, "\\n");\n'
            "      fclose(_dbgf);\n"
            "    }\n"
            "  }\n\n"
        )

    body.append("  // evaluate reaction rates\n")
    body.append(f"  double fwd_rates[{num_r}];\n")
    if num_rev == 0:
        body.append("  double* rev_rates = 0;\n")
    else:
        body.append(f"  double rev_rates[{num_rev}];\n")
    body.append(
        "  eval_rxn_rates (T, pres, conc, fwd_rates, rev_rates);\n\n"
    )

    if num_pdep == 0:
        body.append("  double* pres_mod = 0;\n")
    else:
        body.append(f"  double pres_mod[{num_pdep}];\n")
    if num_pdep:
        body.append("  //get pressure modifications to reaction rates\n")
        body.append("  get_rxn_pres_mod (T, pres, conc, pres_mod);\n")
    body.append("\n")

    body.append(
        "  // evaluate rate of change of species molar concentration\n"
    )
    body.append(f"  double spec_rates[{num_s}] = {{0}};\n")
    body.append(
        "  eval_spec_rates (fwd_rates, rev_rates, pres_mod, "
        f"spec_rates, &spec_rates[{num_s - 1}]);\n\n"
    )

    body.append("  double rho_inv = 1.0 / rho;\n")
    body.append("  double J_nplusone[1] = {0};\n")
    body.append(f"  double J_nplusjplus[{num_s}] = {{0}};\n")
    body.append(f"  double dBdT[{num_s}] = {{0}};\n")
    body.append(dbdt_declaration_lines(species, reactions))

    for name in rxn_names:
        body.append(
            f"  {name}(T, pres, conc, fwd_rates, rev_rates, pres_mod, "
            "dBdT, mw_avg, rho_inv, jac, J_nplusone, J_nplusjplus);\n"
        )
    body.append("\n")

    # Partial derivatives of temperature (energy equation)
    body.append("  // species enthalpies\n")
    body.append(f"  double h[{num_s}];\n  eval_h(T, h);\n\n")
    body.append("  // species specific heats\n")
    body.append(f"  double cp[{num_s}];\n  eval_cp(T, cp);\n\n")
    body.append("  // average specific heat\n  double cp_avg;\n")

    cp_avg_lines = []
    cp_avg_line = "  cp_avg = "
    for isp, _sp in enumerate(species[:-1]):
        if len(cp_avg_line) > 70:
            cp_avg_lines.append(cp_avg_line + "\n")
            cp_avg_line = "   "
        cp_avg_line += f"(y[{isp + 1}] * cp[{isp}]) + "
    cp_avg_line += f"(y_N * cp[{num_s - 1}]);\n"
    cp_avg_lines.append(cp_avg_line)
    body.append("".join(cp_avg_lines))

    body.append("  jac[0] = 0.0;\n")
    touched[0] = True

    body.append("  double working_temp = (1.0 / cp_avg);\n")
    body.append(
        "  double energy_factor = 1.0 / (rho * cp_avg * cp_avg);\n"
    )

    body.append("  //Finish dYk / Yj's\n  //And dT/dYj's\n")
    for name in finish_names:
        body.append(
            f"  {name}(h, cp, spec_rates, mw_avg, rho_inv, "
            "working_temp, energy_factor, jac, J_nplusjplus);\n"
        )
    body.append("\n")

    body.append(write_dcp_dt(species))
    body.append("\n")
    body.append(write_dt_completion(species, j_nplusone_touched))
    body.append("} // end eval_jacob\n\n")

    (path / "jacob.c").write_text("".join(body))
    return touched


# Writes `chunk_lines` in groups of `chunk_size` as `<stem>_<i>.c` files
# with `void <stem>_<i>(<params>)`; returns the function names.
def _write_chunk_files(
    path: Path, stem: str, chunk_lines: List[List[str]],
    chunk_size: int, params: str, scratch: str,
) -> List[str]:
    names = []
    for chunk_i in range(0, len(chunk_lines), chunk_size):
        name = f"{stem}_{chunk_i // chunk_size}"
        names.append(name)
        group = chunk_lines[chunk_i:chunk_i + chunk_size]
        text = (
            '#include "jacob.h"\n\n'
            f"void {name} ({params}) {{\n\n"
            + scratch
            + "".join("".join(lines) for lines in group)
            + "}\n\n"
        )
        (path / f"{name}.c").write_text(text)
    return names
