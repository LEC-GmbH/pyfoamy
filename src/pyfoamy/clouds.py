"""Builds a Lagrangian cloud's `<cloud>Properties` dict and its entries."""

from typing import Mapping, Optional, Sequence

from pyfoamy.include_entries import entry_dict
from pyfoamy.parsing.grammar import Compound, String

_PARCEL = (String("liblagrangianParcel.so"),)


def _bounded(type_name: str, settings: dict, q, minimum, maximum) -> dict:
    if minimum >= maximum:
        raise ValueError(
            f"distribution: minimum {minimum} must be smaller than "
            f"maximum {maximum}"
        )
    return {
        "type": type_name, **settings, "Q": q, "min": minimum,
        "max": maximum,
    }


def fixed_value(value) -> dict:
    """A `fixedValue` size distribution."""
    return {"type": "fixedValue", "value": value}


def multi_fixed_value(values) -> dict:
    """A `multiFixedValue` size distribution of `(size, share)` pairs."""
    return {
        "type": "multiFixedValue",
        "values": [tuple(pair) for pair in values],
    }


def tabulated_cumulative(distribution, *, q) -> dict:
    """A `tabulatedCumulative` size distribution.

    Args:
        distribution: `(size, cumulative fraction)` pairs.
        q: Size exponent of the weighting (0 number, 3 volume).
    """
    return {
        "type": "tabulatedCumulative", "Q": q,
        "distribution": [tuple(pair) for pair in distribution],
    }


def tabulated_density(distribution, *, q) -> dict:
    """A `tabulatedDensity` size distribution, see
    `tabulated_cumulative()`."""
    return {
        "type": "tabulatedDensity", "Q": q,
        "distribution": [tuple(pair) for pair in distribution],
    }


def exponential(lam, *, q, minimum, maximum) -> dict:
    """An `exponential` size distribution between two sizes."""
    return _bounded("exponential", {"lambda": lam}, q, minimum, maximum)


def normal(mu, sigma, *, q, minimum, maximum) -> dict:
    """A `normal` size distribution between two sizes."""
    return _bounded(
        "normal", {"mu": mu, "sigma": sigma}, q, minimum, maximum,
    )


def multi_normal(mu, sigma, strength, *, q, minimum, maximum) -> dict:
    """A `multiNormal` size distribution: one entry per normal
    distribution in each list."""
    return _bounded("multiNormal", {
        "mu": list(mu), "sigma": list(sigma), "strength": list(strength),
    }, q, minimum, maximum)


def rosin_rammler(d, n, *, q, minimum, maximum) -> dict:
    """A `RosinRammler` size distribution between two sizes."""
    return _bounded("RosinRammler", {"d": d, "n": n}, q, minimum, maximum)


def uniform(*, q, minimum, maximum) -> dict:
    """A `uniform` size distribution between two sizes."""
    return _bounded("uniform", {}, q, minimum, maximum)


def constant_velocity(u_mag, mass_total) -> dict:
    """The `constantVelocity` flow of a cone injection."""
    return {
        "flowType": "constantVelocity", "Umag": u_mag,
        "massTotal": mass_total,
    }


def pressure_driven_velocity(p_inj, mass_total) -> dict:
    """The `pressureDrivenVelocity` flow of a cone injection."""
    return {
        "flowType": "pressureDrivenVelocity", "Pinj": p_inj,
        "massTotal": mass_total,
    }


def flow_rate_and_discharge(
    cd, *, mass_flow_rate=None, mass_total=None, flow_rate_profile=None,
) -> dict:
    """The `flowRateAndDischarge` flow of a cone injection.

    Args:
        cd: Discharge coefficient, a number or `Function1` of the time
            since the start of injection, as the other functions.
        mass_flow_rate: The rate itself; or
        mass_total: the mass, distributed over time by
        flow_rate_profile: a `Function1` of any scale.

    Raises:
        ValueError: Unless either the rate or mass and profile are given.
    """
    by_mass = mass_total is not None and flow_rate_profile is not None
    if (mass_flow_rate is None) != by_mass or (
        (mass_total is None) != (flow_rate_profile is None)
    ):
        raise ValueError(
            "give mass_flow_rate, or mass_total with flow_rate_profile"
        )
    result = {"flowType": "flowRateAndDischarge", "Cd": cd}
    if by_mass:
        result["massTotal"] = mass_total
        result["flowRateProfile"] = flow_rate_profile
    else:
        result["massFlowRate"] = mass_flow_rate
    return result


def cone_injection(
    *, soi, duration, position, direction, theta_inner, theta_outer,
    size_distribution: Mapping, flow: Mapping, d_outer, d_inner=0,
    uniform_parcel_size: str = "nParticle", parcels_per_second=2e7,
) -> dict:
    """A `coneInjection` model injecting from a disc.

    Args:
        soi: Start of injection, in user time.
        theta_inner: Cone angles in degrees, numbers or `Function1` of
            the time since the start of injection.
        size_distribution: From one of the distribution builders.
        flow: From `constant_velocity()`, `pressure_driven_velocity()` or
            `flow_rate_and_discharge()`.
        d_outer: Diameters of the disc.
        uniform_parcel_size: `"nParticle"`, `"surfaceArea"` or `"volume"`.
    """
    return {
        "type": "coneInjection",
        "SOI": soi,
        "duration": duration,
        "uniformParcelSize": uniform_parcel_size,
        "injectionMethod": "disc",
        "dInner": d_inner,
        "dOuter": d_outer,
        "position": position,
        "direction": direction,
        "parcelsPerSecond": parcels_per_second,
        "thetaInner": theta_inner,
        "thetaOuter": theta_outer,
        "sizeDistribution": dict(size_distribution),
        **flow,
    }


def sub_model(entry: str, model: str, coeffs: Optional[Mapping] = None):
    """A sub-model selection with its `<model>Coeffs`.

    Args:
        entry: E.g. `"breakupModel"`.
        coeffs: None writes no coefficients dict.
    """
    result = {entry: model}
    if coeffs is not None:
        result[f"{model}Coeffs"] = dict(coeffs)
    return result


def spray_cloud(
    *, t0, injections: Mapping, liquid: Mapping, sub_models: Mapping,
    rho0=1000, cp0=4187, max_co=0.3, cpu_load: Optional[bool] = None,
) -> dict:
    """A `sprayCloud` of one liquid mixture, coupled to the gas.

    Sphere drag, Ranz-Marshall heat transfer, rebound at walls, no
    dispersion, collision or film.

    Args:
        t0: Temperature of the injected liquid.
        injections: Injection models by name.
        liquid: Mass fraction per liquid species.
        sub_models: Further `subModels` entries, e.g. from `sub_model()`;
            `sprayCloud` needs `breakupModel`, `atomisationModel` and
            `phaseChangeModel`.
        rho0: Read by OpenFOAM, replaced by the liquid's properties, as
        cp0: this.
        cpu_load: Sets `cpuLoad` for a load-balanced run.
    """
    result = {
        "type": "sprayCloud",
        "solution": {
            "coupled": True,
            "transient": True,
            "cellValueSourceCorrection": True,
            "maxCo": max_co,
            "sourceTerms": {"schemes": {
                name: Compound(["explicit", 1])
                for name in ("rho", "U", "Yi", "h", "radiation")
            }},
            "interpolationSchemes": {
                "rho": "cell", "U": "cellPoint", "mu": "cell",
                "T": "cell", "Cp": "cell", "kappa": "cell", "p": "cell",
            },
            "integrationSchemes": {"U": "Euler", "T": "analytical"},
        },
        "constantProperties": {
            "T0": t0, "rho0": rho0, "Cp0": cp0, "constantVolume": False,
        },
        "subModels": {
            "particleForces": {"sphereDrag": Compound([])},
            "injectionModels": dict(injections),
            "dispersionModel": "none",
            "patchInteractionModel": "standardWallInteraction",
            "heatTransferModel": "RanzMarshall",
            "compositionModel": "singlePhaseMixture",
            "surfaceFilmModel": "none",
            "stochasticCollisionModel": "none",
            "radiation": False,
            "standardWallInteractionCoeffs": {"type": "rebound"},
            "RanzMarshallCoeffs": {"BirdCorrection": True},
            "singlePhaseMixtureCoeffs": {
                "phases": ["liquid", dict(liquid)],
            },
            **sub_models,
        },
        "cloudFunctions": {},
    }
    if cpu_load is not None:
        result["cpuLoad"] = cpu_load
    return result


def clouds_model(clouds: Sequence[str]) -> dict:
    """The `clouds` fvModel coupling the named clouds to the gas."""
    return entry_dict("clouds", _PARCEL, {"clouds": list(clouds)})
