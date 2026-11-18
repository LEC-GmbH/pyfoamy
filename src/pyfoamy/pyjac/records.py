"""Plain data records `adapter` produces from a `Mechanism`."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

#: Universal gas constant [J/(kmol*K)] (pyJac's `chem_utilities.RU`),
#: kept so the generated C matches pyJac's; Cantera's and OpenFOAM's
#: differ by < 1e-6 relative.
GAS_CONSTANT_SI = 8314.4621

#: Standard atmosphere [Pa] (pyJac's `chem_utilities.PA`).
STANDARD_PRESSURE = 101325.0

NasaCoeffs = Tuple[float, float, float, float, float, float, float]
ArrheniusParams = Tuple[float, float, float]
PlogPoint = Tuple[float, float, float, float]


@dataclass
class Species:
    """A species' molecular weight and 7-coefficient NASA fit.

    Args:
        mw: Molecular weight [kg/kmol].
        low: NASA-7 coefficients `(a1..a7)` below `t_mid`.
        high: NASA-7 coefficients `(a1..a7)` above `t_mid`.
        t_mid: Switch temperature [K] between `low` and `high`.
    """

    name: str
    mw: float
    low: NasaCoeffs
    high: NasaCoeffs
    t_mid: float = 1000.0


@dataclass
class ChebyshevRate:
    """A Chebyshev pressure/temperature-dependent forward rate fit.

    Args:
        t_range: `(t_low, t_high)` [K] validity range.
        p_range: `(p_low, p_high)` [Pa] validity range.
        coefficients: `coefficients[i][j]` for temperature index `i`,
            pressure index `j`.
    """

    t_range: Tuple[float, float]
    p_range: Tuple[float, float]
    coefficients: List[List[float]]


@dataclass
class Reaction:
    """One reaction's stoichiometry and rate parameters.

    Args:
        reactants: Reactant species indices into the species list.
        reactant_nu: Stoichiometric coefficients, matching `reactants`.
        products: Product species indices.
        product_nu: Stoichiometric coefficients, matching `products`.
        A: Forward Arrhenius pre-exponential factor; ignored if
            `chebyshev` or `plog` is given.
        b: Forward Arrhenius temperature exponent.
        E: Forward Arrhenius activation temperature [K].
        reverse_params: Explicit reverse `(A, b, E)`; `None` derives the
            reverse rate from the equilibrium constant.
        third_body: A plain third-body (`M`) correction applies.
        pressure_dependent: A falloff (Lindemann/Troe/SRI) correction
            applies.
        plog: `(pressure, A, b, E)` points of a PLOG forward rate.
        third_body_efficiencies: `(species_index, efficiency)` for
            species with efficiency other than `1.0`.
        pdep_species: Compute the falloff `Pr` from this species'
            concentration alone.
        low_rate_params: Explicit low-pressure-limit `(A, b, E)`
            (falloff). A missing limit falls back to `A`/`b`/`E`, as in
            Chemkin.
        high_rate_params: Explicit high-pressure-limit `(A, b, E)`
            (chemically activated falloff).
        troe_params: `(alpha, T3, T1[, T2])` Troe blending parameters.
        sri_params: `(a, b, c[, d, e])` SRI blending parameters. Neither
            Troe nor SRI means Lindemann (`F = 1`).
    """

    reactants: List[int]
    reactant_nu: List[float]
    products: List[int]
    product_nu: List[float]
    A: float
    b: float
    E: float
    reversible: bool = False
    reverse_params: Optional[ArrheniusParams] = None
    third_body: bool = False
    pressure_dependent: bool = False
    chebyshev: Optional[ChebyshevRate] = None
    plog: Optional[List[PlogPoint]] = None
    third_body_efficiencies: Optional[List[Tuple[int, float]]] = None
    pdep_species: Optional[int] = None
    low_rate_params: Optional[ArrheniusParams] = None
    high_rate_params: Optional[ArrheniusParams] = None
    troe_params: Optional[Tuple[float, ...]] = None
    sri_params: Optional[Tuple[float, ...]] = None
