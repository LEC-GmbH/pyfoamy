"""OpenFOAM-ready thermo/transport data from a Cantera mechanism."""

import warnings
from typing import Dict, List, Optional, Tuple

from pyfoamy.cantera.mechanism import Mechanism, require_cantera

try:
    from numpy.polynomial import Polynomial, polynomial
    import numpy as np
except ImportError:
    np = None
    Polynomial = None
    polynomial = None

try:
    import cantera as ct
except ImportError:
    ct = None

_TREF = 298.15
# OpenFOAM's standard pressure (etc/configDict pStd), not the
# mechanism's reference_pressure: janafThermo::s()/hPolynomialThermo::s()
# apply -R*ln(p/pStd) at runtime, so entropy is fitted at this pressure.
_P_STD = 1e5
_TSCALE = 1000.0  # avoids Vandermonde ill-conditioning in the Cp fit
# relative Cp/H/S gap at Tcommon: repair below this, raise above it.
_MAX_DISCONTINUITY = 0.01
# Below this, treat as clean (Cantera's NasaPoly2::validate is silent
# below ~1e-6).
_NOISE_FLOOR = 1e-5
# OpenFOAM's Polynomial<PolySize> default; the PolySize is part of the
# keyword (e.g. "muCoeffs<8>").
_POLY_DEGREE = 7
_POLY_SIZE = _POLY_DEGREE + 1
# sutherland has only 2 parameters: warn (not raise) when it can't
# track a complex specie's mu(T).
_SUTHERLAND_WARN_THRESHOLD = 0.05
# polynomial/logPolynomial coefficients can be large and sign-
# alternating without a bad fit; warn on the resulting mu/kappa error,
# same threshold as sutherland/hPolynomial.
_TRANSPORT_POLYNOMIAL_WARN_THRESHOLD = 0.05
# hPolynomial's Cp fit is sampled a bit beyond [T_low,T_high] to reduce
# edge error (too much margin hurts); only the nominal range is
# reported.
_H_POLYNOMIAL_MARGIN = 0.01
_H_POLYNOMIAL_WARN_THRESHOLD = 0.05


# --- NASA7 math -------------------------------------------------------
# Cp/Ru = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4
# H/(Ru T) = a0 + a1 T/2 + a2 T^2/3 + a3 T^3/4 + a4 T^4/5 + a5/T
# S/Ru = a0 ln(T) + a1 T + a2 T^2/2 + a3 T^3/3 + a4 T^4/4 + a6

# Cp/Ru from the first 5 coefficients; `T_scale` divides T first, for
# a fit still in scaled-T units (see _fit_cp_side).
def _cp_R(a, T, T_scale=1.0):
    return Polynomial(a[:5])(T / T_scale)


# H/(Ru T) divides the Cp/Ru integral by T once more, which
# introduces the a5/T term.
_T_POLY = Polynomial([0, 1]) if Polynomial is not None else None


# Returns (quotient Polynomial, a_const) of H/(Ru T), without a_const/T.
def _h_RT_polynomial_part(a0_4, k=0.0):
    quotient, remainder = divmod(Polynomial(a0_4).integ(k=[k]), _T_POLY)
    return quotient, remainder.coef[0]


# S/Ru's polynomial part without a0*ln(T): a plain integral.
def _s_R_polynomial_part(a1_4, k=0.0):
    return Polynomial(a1_4).integ(k=[k])


def _h_RT(a, T):
    poly, a5 = _h_RT_polynomial_part(a[:5], k=a[5])
    return poly(T) + a5 / T


def _s_R(a, T):
    poly = _s_R_polynomial_part(a[1:5], k=a[6])
    return a[0]*np.log(T) + poly(T)


def _solve_a6(a0_4, T, h_RT_target) -> float:
    poly, _ = _h_RT_polynomial_part(a0_4)
    return float(T * (h_RT_target - poly(T)))


def _solve_a7(a0_4, T, s_R_target) -> float:
    poly = _s_R_polynomial_part(a0_4[1:])
    return float(s_R_target - a0_4[0]*np.log(T) - poly(T))


def _relative_gap(low, high):
    denom = (abs(low) + abs(high)) / 2 or 1.0
    return abs(high - low) / denom


def _continuity_gap(low7, high7, Tcommon) -> float:
    return max(
        _relative_gap(_cp_R(low7, Tcommon), _cp_R(high7, Tcommon)),
        _relative_gap(_h_RT(low7, Tcommon), _h_RT(high7, Tcommon)),
        _relative_gap(_s_R(low7, Tcommon), _s_R(high7, Tcommon)),
    )


# --- shared polynomial fitting primitives (hPolynomial and transport) ---

def _fit_polynomial_coeffs(xs, ys, degree: int) -> List[float]:
    # Polynomial.fit works in a [-1,1] window; converting with
    # window=p.domain gives power-series coefficients in the original
    # x units, as OpenFOAM's Polynomial<PolySize> expects.
    p = Polynomial.fit(np.asarray(xs), np.asarray(ys), degree)
    return [float(c) for c in p.convert(domain=p.domain, window=p.domain).coef]


def _poly_key(name: str) -> str:
    # The coefficient count is part of the OpenFOAM keyword
    # ("muCoeffs<8>"), as in polynomialTransport.C/hPolynomialThermo.C.
    return f"{name}<{_POLY_SIZE}>"


# --- Cp/Ru curve fitting (scaled-T units, rescaled back afterwards) ---

def _fit_cp_side(Ts, cpR, Tc, anchor=None):
    T_scaled, Tc_scaled = Ts / _TSCALE, Tc / _TSCALE
    if anchor is None:
        # Unconstrained fit in scaled-T units, to match the anchored
        # branch's basis.
        p = Polynomial.fit(T_scaled, cpR, 4)
        return p.convert(domain=p.domain, window=p.domain).coef
    # The anchored branch enforces f(Tc_scaled) == anchor, which
    # Polynomial.fit cannot express; a basis vanishing at Tc_scaled
    # (T_scaled**p - Tc_scaled**p) makes it an unconstrained fit.
    A = np.vstack(
        [T_scaled**p - Tc_scaled**p for p in range(1, 5)]
    ).T
    rest, *_ = np.linalg.lstsq(A, cpR - anchor, rcond=None)
    a1 = anchor - sum(
        r * Tc_scaled**p for r, p in zip(rest, range(1, 5))
    )
    return np.array([a1, *rest])


def _rescale(a) -> List[float]:
    return [float(ai) / _TSCALE**i for i, ai in enumerate(a)]


def _fit_residual(Ts_low, y_low, Ts_high, y_high, Tc) -> float:
    a_low = _fit_cp_side(Ts_low, y_low, Tc)
    anchor = _cp_R(a_low, Tc, T_scale=_TSCALE)
    a_high = _fit_cp_side(Ts_high, y_high, Tc, anchor=anchor)
    res_low = y_low - _cp_R(a_low, Ts_low, T_scale=_TSCALE)
    res_high = y_high - _cp_R(a_high, Ts_high, T_scale=_TSCALE)
    return float(np.sqrt(np.mean(np.concatenate([res_low, res_high])**2)))


# Coarse-to-fine search for the Tcommon minimizing the summed
# two-piece fit residual of all `cp_funcs`. One function optimizes its
# own Tcommon (lump()); several share one (lump_group(), for species
# recombined at OpenFOAM runtime).
def _optimize_tcommon(
    cp_funcs, T_low, T_high, n_candidates=25, n_samples=25, n_rounds=3
) -> float:
    margin = 0.05 * (T_high - T_low)
    lo, hi = T_low + margin, T_high - margin
    best = 0.5 * (lo + hi)
    for _ in range(n_rounds):
        candidates = np.linspace(lo, hi, n_candidates)
        residuals = []
        for Tc in candidates:
            total = 0.0
            for cp_func in cp_funcs:
                Ts_low = np.linspace(T_low, Tc, n_samples)
                Ts_high = np.linspace(Tc, T_high, n_samples)
                y_low = np.array([cp_func(T) for T in Ts_low])
                y_high = np.array([cp_func(T) for T in Ts_high])
                total += (
                    _fit_residual(Ts_low, y_low, Ts_high, y_high, Tc) ** 2
                )
            residuals.append(total)
        best = candidates[int(np.argmin(residuals))]
        step = (hi - lo) / n_candidates
        lo = max(T_low + 1, best - 2*step)
        hi = min(T_high - 1, best + 2*step)
    return float(best)


def _janaf_cp_func(cp_mole_func):
    return lambda T: cp_mole_func(T) / ct.gas_constant


# Continuity-constrained two-piece JANAF fit of a Cp(T) curve at a
# fixed Tcommon, anchored to H/S reference values.
def _janaf_fit_at_tcommon(
    cp_func, h_mole_ref, s_mole_ref, T_low, T_high, Tc, n_samples=30,
) -> dict:
    h_RT_ref = h_mole_ref / (ct.gas_constant * _TREF)
    s_R_ref = s_mole_ref / ct.gas_constant

    Ts_low = np.linspace(T_low, Tc, n_samples)
    Ts_high = np.linspace(Tc, T_high, n_samples)
    y_low = np.array([cp_func(T) for T in Ts_low])
    y_high = np.array([cp_func(T) for T in Ts_high])

    a_low5 = _fit_cp_side(Ts_low, y_low, Tc)
    anchor = _cp_R(a_low5, Tc, T_scale=_TSCALE)
    a_high5 = _fit_cp_side(Ts_high, y_high, Tc, anchor=anchor)
    a_low5, a_high5 = _rescale(a_low5), _rescale(a_high5)

    a6_low = _solve_a6(a_low5, _TREF, h_RT_ref)
    a7_low = _solve_a7(a_low5, _TREF, s_R_ref)
    low = [*a_low5, a6_low, a7_low]

    a6_high = _solve_a6(a_high5, Tc, _h_RT(low, Tc))
    a7_high = _solve_a7(a_high5, Tc, _s_R(low, Tc))
    high = [*a_high5, a6_high, a7_high]

    gap = _continuity_gap(low, high, Tc)
    if gap >= 1e-6:
        raise RuntimeError(
            f"constrained JANAF fit is not continuous at Tcommon={Tc} "
            f"(gap={gap})"
        )

    return {
        "Tlow": T_low, "Thigh": T_high, "Tcommon": Tc,
        "highCpCoeffs": high, "lowCpCoeffs": low,
    }


# _janaf_fit_at_tcommon() with Tcommon optimized. JANAF is molar, so
# molWeight is unused.
def _janaf_fit(cp_mole_func, h_mole_ref, s_mole_ref, molWeight, T_low,
               T_high, n_samples=30) -> dict:
    cp_func = _janaf_cp_func(cp_mole_func)
    Tc = _optimize_tcommon([cp_func], T_low, T_high)
    return _janaf_fit_at_tcommon(
        cp_func, h_mole_ref, s_mole_ref, T_low, T_high, Tc, n_samples,
    )


def _is_nasa7(sp) -> bool:
    return isinstance(sp.thermo, ct.NasaPoly2)


# Single-range, unconstrained Cp_mass(T) fit; hPolynomialThermo
# derives h/s itself by integrating CpCoeffs and anchoring to hf/sf at
# Tstd.
def _h_polynomial_fit(
    cp_mole_func, h_mole_ref, s_mole_ref, molWeight, T_low, T_high,
    n_samples=30,
) -> dict:
    margin = _H_POLYNOMIAL_MARGIN * (T_high - T_low)

    def cp_mass_func(T):
        return cp_mole_func(T) / molWeight

    Ts = np.linspace(T_low - margin, T_high + margin, n_samples)
    cps = np.array([cp_mass_func(T) for T in Ts])
    coeffs = _fit_polynomial_coeffs(Ts, cps, _POLY_DEGREE)

    # Quality is checked over the nominal range, not the padded
    # sampling range.
    check_Ts = np.linspace(T_low, T_high, n_samples)
    real = np.array([cp_mass_func(T) for T in check_Ts])
    pred = polynomial.polyval(check_Ts, coeffs)
    max_rel_err = float(np.max(np.abs(pred - real) / real))
    if max_rel_err > _H_POLYNOMIAL_WARN_THRESHOLD:
        warnings.warn(
            f"hPolynomial Cp fit has {max_rel_err:.1%} max relative "
            f"error over [{T_low:.0f},{T_high:.0f}]K"
        )

    return {
        "hf": h_mole_ref / molWeight,
        "sf": s_mole_ref / molWeight,
        "Tlow": T_low, "Thigh": T_high,
        _poly_key("CpCoeffs"): coeffs,
    }


_THERMO_MODEL_FORMATTERS = {
    "janaf": _janaf_fit,
    "hPolynomial": _h_polynomial_fit,
}


def _check_formatter(registry: dict, name: str, kind: str):
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"unknown {kind} {name!r}, supported: {sorted(registry)}"
        ) from None


def _format_gap(gap: float) -> str:
    return f"{gap * 100:.3g}%"  # .2% rounds tiny-but-real gaps to "0.00%"


def _report_continuity(
    repaired: List[Tuple[str, float]], broken: List[Tuple[str, float]]
) -> None:
    if repaired:
        names = ", ".join(f"{n} ({_format_gap(g)})" for n, g in repaired)
        warnings.warn(f"repaired Cp/H/S discontinuity at Tcommon: {names}")
    if broken:
        names = ", ".join(f"{n} ({_format_gap(g)})" for n, g in broken)
        raise ValueError(
            f"Cp/H/S discontinuity at Tcommon exceeds "
            f"{_MAX_DISCONTINUITY:.0%} for: {names}"
        )


# --- transport fitting -------------------------------------------------
# sutherland: mu = As*sqrt(T)/(1+Ts/T); kappa is derived from mu and Cv
#   via Eucken (sutherlandTransportI.H). Linear fit: T^1.5/mu = T/As +
#   Ts/As, weighted by mu/T^1.5 so it minimises the relative error in mu.
# polynomial: independent 8-term polynomials for mu(T) and kappa(T).
# logPolynomial: same for ln(mu)/ln(kappa) vs ln(T), well conditioned
#   as ln(T) stays below ~9.

def _const_transport_fit(
    mu_func, kappa_func, cp_func, T_low, T_high, label,
) -> dict:
    mu, kappa, cp = mu_func(_TREF), kappa_func(_TREF), cp_func(_TREF)
    return {"mu": mu, "Pr": mu * cp / kappa}


def _sutherland_fit(
    mu_func, kappa_func, cp_func, T_low, T_high, label, n_samples=30
) -> dict:
    Ts = np.linspace(T_low, T_high, n_samples)
    mus = np.array([mu_func(T) for T in Ts])
    y = Ts**1.5 / mus
    A = np.vstack([Ts, np.ones_like(Ts)]).T
    (slope, intercept), *_ = np.linalg.lstsq(
        A / y[:, None], np.ones_like(y), rcond=None,
    )
    As = 1.0 / slope
    Ts_param = intercept * As

    pred = As * np.sqrt(Ts) / (1 + Ts_param / Ts)
    max_rel_err = float(np.max(np.abs(pred - mus) / mus))
    if max_rel_err > _SUTHERLAND_WARN_THRESHOLD:
        warnings.warn(
            f"{label}: sutherland transport fit has {max_rel_err:.1%} max "
            f"relative error over [{T_low:.0f},{T_high:.0f}]K - the "
            "2-parameter model may not suit this specie/range well"
        )
    return {"As": float(As), "Ts": float(Ts_param)}


def _warn_if_poor_transport_fit(
    model: str, name: str, pred, real, T_low: float, T_high: float,
    label: str,
) -> None:
    max_rel_err = float(np.max(np.abs(pred - real) / np.abs(real)))
    if max_rel_err > _TRANSPORT_POLYNOMIAL_WARN_THRESHOLD:
        warnings.warn(
            f"{label}: {model} {name} transport fit has {max_rel_err:.1%} max "
            f"relative error over [{T_low:.0f},{T_high:.0f}]K"
        )


def _polynomial_transport_fit(
    mu_func, kappa_func, cp_func, T_low, T_high, label, n_samples=30
) -> dict:
    Ts = np.linspace(T_low, T_high, n_samples)
    mus = np.array([mu_func(T) for T in Ts])
    kappas = np.array([kappa_func(T) for T in Ts])
    mu_coeffs = _fit_polynomial_coeffs(Ts, mus, _POLY_DEGREE)
    kappa_coeffs = _fit_polynomial_coeffs(Ts, kappas, _POLY_DEGREE)
    _warn_if_poor_transport_fit(
        "polynomial", "mu", polynomial.polyval(Ts, mu_coeffs), mus,
        T_low, T_high, label,
    )
    _warn_if_poor_transport_fit(
        "polynomial", "kappa", polynomial.polyval(Ts, kappa_coeffs),
        kappas, T_low, T_high, label,
    )
    return {
        _poly_key("muCoeffs"): mu_coeffs,
        _poly_key("kappaCoeffs"): kappa_coeffs,
    }


def _log_polynomial_transport_fit(
    mu_func, kappa_func, cp_func, T_low, T_high, label, n_samples=30
) -> dict:
    Ts = np.linspace(T_low, T_high, n_samples)
    log_T = np.log(Ts)
    mus = np.array([mu_func(T) for T in Ts])
    kappas = np.array([kappa_func(T) for T in Ts])
    mu_log_coeffs = _fit_polynomial_coeffs(log_T, np.log(mus), _POLY_DEGREE)
    kappa_log_coeffs = _fit_polynomial_coeffs(
        log_T, np.log(kappas), _POLY_DEGREE
    )
    _warn_if_poor_transport_fit(
        "logPolynomial", "mu",
        np.exp(polynomial.polyval(log_T, mu_log_coeffs)), mus,
        T_low, T_high, label,
    )
    _warn_if_poor_transport_fit(
        "logPolynomial", "kappa",
        np.exp(polynomial.polyval(log_T, kappa_log_coeffs)), kappas,
        T_low, T_high, label,
    )
    return {
        _poly_key("muLogCoeffs"): mu_log_coeffs,
        _poly_key("kappaLogCoeffs"): kappa_log_coeffs,
    }


_TRANSPORT_MODEL_FORMATTERS = {
    "const": _const_transport_fit,
    "sutherland": _sutherland_fit,
    "polynomial": _polynomial_transport_fit,
    "logPolynomial": _log_polynomial_transport_fit,
}


class ThermoDatabase:
    """Cantera-backed thermo/transport fits as OpenFOAM-ready dicts, for
    real, lumped and combustion-derived species.

    Composes over a `Mechanism`.
    """

    def __init__(
        self, mechanism, reference_pressure=None,
        thermo_model="janaf", transport_model="const",
    ):
        require_cantera()
        self.mechanism = Mechanism.coerce(
            mechanism, reference_pressure=reference_pressure,
        )
        self.thermo_model = thermo_model
        self.transport_model = transport_model

    @property
    def thermo_model(self) -> str:
        """The thermo model of every `species()`/`lump()` call.

        One per database: OpenFOAM's `thermoType` is one compiled type per
        phase.
        """
        return self._thermo_model

    @thermo_model.setter
    def thermo_model(self, value: str) -> None:
        _check_formatter(_THERMO_MODEL_FORMATTERS, value, "thermo_model")
        self._thermo_model = value

    @property
    def transport_model(self) -> str:
        """The transport model of every `species()`/`lump()` call."""
        return self._transport_model

    @transport_model.setter
    def transport_model(self, value: str) -> None:
        _check_formatter(
            _TRANSPORT_MODEL_FORMATTERS, value, "transport_model"
        )
        self._transport_model = value

    @property
    def thermo_type_entries(self) -> Dict[str, str]:
        """The models as `ThermoTypeSpec` keywords, to splat into
        `ThermoTypeSpec(..., **db.thermo_type_entries)` so fit and
        `thermoType` cannot drift apart.
        """
        return {
            "thermo": self.thermo_model,
            "transport": self.transport_model,
        }

    def species(
        self, names=None,
        T_low: Optional[float] = None, T_high: Optional[float] = None,
    ) -> Dict[str, dict]:
        """Real species as dicts, converted directly (JANAF from NASA7 only)
        unless the source is discontinuous at Tcommon: a small gap is
        repaired and a large one rejected, as in `lump()`. Other thermo
        models, or other source polynomials, always fit.

        Args:
            names: Species to return (default: all), case-insensitive; keys
                use the mechanism's spelling.
            T_low: Overrides the low temperature bound (default: each
                specie's own). A specie not covering `T_low`/`T_high` is
                refit via `lump()`, which has the same extrapolation caveat.
            T_high: Same, for the high temperature bound.
        """
        gas = self.mechanism.gas
        names = self.mechanism.resolve_names(
            names if names is not None else gas.species_names
        )

        if self.thermo_model != "janaf":
            return {
                name: self._lump_fit({name: 1.0}, "mole", T_low, T_high)
                for name in names
            }

        result, repaired, broken = {}, [], []
        for name in names:
            if not _is_nasa7(gas.species(name)):
                result[name] = self._lump_fit(
                    {name: 1.0}, "mole", T_low, T_high
                )
                continue
            gap = self._specie_gap(name)
            if gap > _MAX_DISCONTINUITY:
                broken.append((name, gap))
                continue
            if gap > 0.0:
                repaired.append((name, gap))
            if gap == 0.0 and T_low is None and T_high is None:
                result[name] = self._real_specie(name)[0]
            else:
                result[name] = self._lump_fit(
                    {name: 1.0}, "mole", T_low, T_high
                )
        _report_continuity(repaired, broken)
        return result

    def species_group(
        self, names=None,
        T_low: Optional[float] = None, T_high: Optional[float] = None,
    ) -> Dict[str, dict]:
        """Like `species()`, but refit against one shared, jointly optimized
        `Tcommon`.

        Needed for species blended by coefficient (`janafThermo::operator+=`/
        `operator*`, i.e. `coefficientMulticomponentMixture`, OpenFOAM's only
        gas-phase multicomponent mixture for JANAF/`perfectGas`): a mixed
        `Tcommon` gives the blend an `H(T)` discontinuity. See `lump_group()`.

        Args:
            names: Species to return (default: all), case-insensitive.
            T_low: Overrides the low temperature bound (default: intersection
                of the species' ranges).
            T_high: Same, for the high temperature bound.

        Returns:
            `{name: dict}` shaped like `species()`, including `"elements"`.
        """
        gas = self.mechanism.gas
        names = self.mechanism.resolve_names(
            names if names is not None else gas.species_names
        )
        return self.lump_group(
            {name: ({name: 1.0}, "mole") for name in names},
            T_low=T_low, T_high=T_high,
        )

    def _specie_gap(self, name) -> float:
        sp = self.mechanism.gas.species(name)
        if not _is_nasa7(sp):
            # No Tcommon; fits sample Cantera's own evaluation.
            return 0.0
        c = sp.thermo.coeffs  # [Tcommon, high a1..a7, low a1..a7]
        if c[0] == sp.thermo.max_temp:
            return 0.0
        gap = _continuity_gap(list(c[8:15]), list(c[1:8]), c[0])
        return 0.0 if gap < _NOISE_FLOOR else gap

    def _real_specie(self, name) -> Tuple[dict, float]:
        sp = self.mechanism.gas.species(name)
        c = sp.thermo.coeffs  # [Tcommon, high a1..a7, low a1..a7]
        Tcommon, high, low = c[0], list(c[1:8]), list(c[8:15])
        # a7 is entropy's integration constant referenced to this
        # mechanism's reference_pressure, not _P_STD as janafThermo::s()
        # assumes; the shift leaves Cp, H and continuity unchanged.
        shift = float(np.log(sp.thermo.reference_pressure / _P_STD))
        high[6] += shift
        low[6] += shift
        gap = self._specie_gap(name)
        thermo = {
            "specie": {"molWeight": sp.molecular_weight, "nMoles": 1},
            "thermodynamics": {
                "Tlow": sp.thermo.min_temp, "Thigh": sp.thermo.max_temp,
                "Tcommon": Tcommon, "highCpCoeffs": high,
                "lowCpCoeffs": low,
            },
            "transport": self._transport(
                {name: 1.0}, "mole", sp.thermo.min_temp, sp.thermo.max_temp,
            ),
            "elements": self.mechanism.elements(name),
        }
        return thermo, gap

    def lump(
        self, composition: Dict[str, float], basis: str = "mass",
        T_low: Optional[float] = None, T_high: Optional[float] = None,
    ) -> dict:
        """One lumped pseudo-specie from a `{real_species: fraction}` blend:
        a Tcommon-optimized, continuity-constrained JANAF refit of Cantera's
        blended Cp(T)/H/S.

        Args:
            composition: `{real_species: fraction}`, names case-insensitive.
            basis: `"mass"` or `"mole"`.
            T_low: Overrides the low bound (default: intersection of the
                constituents' ranges). Below a constituent's range this fits
                Cantera's extrapolation, not verified data.
            T_high: Same, for the high temperature bound.

        Raises:
            ValueError: If `T_low >= T_high`.
        """
        composition = self.mechanism.resolve_composition(composition)
        repaired, broken = [], []
        for name in composition:
            gap = self._specie_gap(name)
            if 0.0 < gap <= _MAX_DISCONTINUITY:
                repaired.append((name, gap))
            elif gap > _MAX_DISCONTINUITY:
                broken.append((name, gap))
        _report_continuity(repaired, broken)
        return self._lump_fit(composition, basis, T_low, T_high)

    # Cantera-side setup for one composition without a Tcommon: restricted
    # gas, T range/molWeight, `cp_mole_func(T)`, H/S references. Shared by
    # _lump_fit() and lump_group().
    def _prepare_lump_inputs(
        self, composition: Dict[str, float], basis: str,
        T_low: Optional[float], T_high: Optional[float],
    ) -> dict:
        names = list(composition)
        gas = self.mechanism.restricted(names)

        Y = Mechanism.mass_fractions_of(gas, composition, basis)
        if T_low is None:
            T_low = max(gas.species(n).thermo.min_temp for n in names)
        if T_high is None:
            T_high = min(gas.species(n).thermo.max_temp for n in names)
        # Callers pass Decimal (OpenFOAM dict scalars), which does not
        # mix with float arithmetic; numpy/Cantera need float.
        T_low = float(T_low)
        T_high = float(T_high)
        if T_low >= T_high:
            raise ValueError(f"T_low ({T_low}) must be < T_high ({T_high})")
        molWeight = gas.mean_molecular_weight

        pressure = self.mechanism.reference_pressure

        def cp_mole_func(T):
            gas.TPY = T, pressure, Y
            return gas.cp_mole

        gas.TPY = _TREF, pressure, Y
        h_mole_ref = gas.enthalpy_mole
        # Entropy reference is at _P_STD, see its comment.
        gas.TPY = _TREF, _P_STD, Y
        s_mole_ref = gas.entropy_mole

        return dict(
            T_low=T_low, T_high=T_high, molWeight=molWeight,
            cp_mole_func=cp_mole_func, h_mole_ref=h_mole_ref,
            s_mole_ref=s_mole_ref,
        )

    # The fit without the continuity pre-check, for species()'s repair
    # path, which already checked.
    def _lump_fit(
        self, composition: Dict[str, float], basis: str,
        T_low: Optional[float] = None, T_high: Optional[float] = None,
    ) -> dict:
        require_cantera()
        formatter = _THERMO_MODEL_FORMATTERS[self.thermo_model]
        setup = self._prepare_lump_inputs(composition, basis, T_low, T_high)

        thermodynamics = formatter(
            setup["cp_mole_func"], setup["h_mole_ref"], setup["s_mole_ref"],
            setup["molWeight"], setup["T_low"], setup["T_high"],
        )
        result = {
            "specie": {"molWeight": setup["molWeight"], "nMoles": 1},
            "thermodynamics": thermodynamics,
            "transport": self._transport(
                composition, basis, setup["T_low"], setup["T_high"],
            ),
        }
        # Elements are defined only for one real specie, not a blend.
        if len(composition) == 1:
            result["elements"] = self.mechanism.elements(
                next(iter(composition))
            )
        return result

    def lump_group(
        self, compositions: Dict[str, Tuple[Dict[str, float], str]],
        T_low: Optional[float] = None, T_high: Optional[float] = None,
    ) -> Dict[str, dict]:
        """Lumps several pseudo-species sharing one jointly optimized JANAF
        `Tcommon`.

        OpenFOAM recombines species by coefficient (`janafThermo::operator+=`/
        `operator*`; `coefficientMulticomponentMixture` is OpenFOAM-14's only
        gas-phase multicomponent mixture for JANAF/`perfectGas`, see
        `rhoFluidMulticomponentThermos.C`). It keeps the first operand's
        `Tcommon` and checks the others only in debug builds, so differing
        `Tcommon` values give a Cp/H discontinuity that can break the `T(h)`
        Newton inversion. See `species_group()` for plain real species.

        Only the `"janaf"` model has a `Tcommon`; other models lump each
        composition independently, as `lump()`.

        Args:
            compositions: `{name: (composition, basis)}`, as `lump()`.
            T_low: Overrides the low bound (default: intersection of all
                ranges).
            T_high: Same, for the high temperature bound.

        Returns:
            `{name: lumped_dict}` shaped like `lump()`.

        Raises:
            ValueError: If `T_low >= T_high` (after intersecting ranges).
        """
        resolved = {}
        repaired, broken = [], []
        for name, (composition, basis) in compositions.items():
            composition = self.mechanism.resolve_composition(composition)
            resolved[name] = (composition, basis)
            for specie_name in composition:
                gap = self._specie_gap(specie_name)
                if 0.0 < gap <= _MAX_DISCONTINUITY:
                    repaired.append((f"{name}/{specie_name}", gap))
                elif gap > _MAX_DISCONTINUITY:
                    broken.append((f"{name}/{specie_name}", gap))
        _report_continuity(repaired, broken)

        if self.thermo_model != "janaf":
            return {
                name: self._lump_fit(composition, basis, T_low, T_high)
                for name, (composition, basis) in resolved.items()
            }

        require_cantera()
        setups = {
            name: self._prepare_lump_inputs(
                composition, basis, T_low, T_high,
            )
            for name, (composition, basis) in resolved.items()
        }
        T_low_shared = max(s["T_low"] for s in setups.values())
        T_high_shared = min(s["T_high"] for s in setups.values())
        if T_low_shared >= T_high_shared:
            raise ValueError(
                f"T_low ({T_low_shared}) must be < T_high "
                f"({T_high_shared}) once every composition's own "
                "valid range is intersected"
            )

        Tc = _optimize_tcommon(
            [_janaf_cp_func(s["cp_mole_func"]) for s in setups.values()],
            T_low_shared, T_high_shared,
        )

        result = {}
        for name, setup in setups.items():
            thermodynamics = _janaf_fit_at_tcommon(
                _janaf_cp_func(setup["cp_mole_func"]), setup["h_mole_ref"],
                setup["s_mole_ref"], T_low_shared, T_high_shared, Tc,
            )
            composition, basis = resolved[name]
            entry = {
                "specie": {"molWeight": setup["molWeight"], "nMoles": 1},
                "thermodynamics": thermodynamics,
                "transport": self._transport(
                    composition, basis, T_low_shared, T_high_shared,
                ),
            }
            if len(composition) == 1:
                entry["elements"] = self.mechanism.elements(
                    next(iter(composition))
                )
            result[name] = entry
        return result

    def combustion_group(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
        basis: str = "mass",
        T_low: Optional[float] = None, T_high: Optional[float] = None,
    ) -> Dict[str, dict]:
        """The `fuel`, `oxidant` and `products` pseudo-species of OpenFOAM's
        inhomogeneous mixtures, lumped as `lump_group()`.

        Args:
            fuel: `{specie: fraction}`, names case-insensitive.
            oxidant: `{specie: fraction}`.
            basis: `"mass"` or `"mole"` for `fuel`/`oxidant`.
            T_low: As `lump_group()`.
            T_high: As `lump_group()`.

        Returns:
            `{"fuel": ..., "oxidant": ..., "products": ...}`; `products` is
            the stoichiometric `Mechanism.complete_combustion_products()`.
        """
        products = self.mechanism.complete_combustion_products(
            fuel, oxidant, basis=basis,
        )
        return self.lump_group(
            {
                "fuel": (fuel, basis),
                "oxidant": (oxidant, basis),
                "products": (products, "mass"),
            },
            T_low=T_low, T_high=T_high,
        )

    def _transport(
        self, composition: Dict[str, float], basis: str, T_low: float,
        T_high: float,
    ) -> dict:
        if self.mechanism.gas.transport_model == "none":
            raise ValueError(
                f"mechanism {self.mechanism.mechanism_path} has no transport "
                f"data for the {self.transport_model} transport fit"
            )
        formatter = _TRANSPORT_MODEL_FORMATTERS[self.transport_model]
        gas = self.mechanism.restricted(list(composition))
        Y = Mechanism.mass_fractions_of(gas, composition, basis)
        pressure = self.mechanism.reference_pressure

        def mu_func(T):
            gas.TPY = T, pressure, Y
            return gas.viscosity

        def kappa_func(T):
            gas.TPY = T, pressure, Y
            return gas.thermal_conductivity

        def cp_func(T):
            gas.TPY = T, pressure, Y
            return gas.cp_mass

        return formatter(
            mu_func, kappa_func, cp_func, T_low, T_high, "+".join(composition),
        )
