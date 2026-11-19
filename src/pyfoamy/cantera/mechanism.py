"""A loaded Cantera mechanism and the facts derived from it."""

import contextlib
import io
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

try:
    import cantera as ct
except ImportError as exc:
    ct = None
    _CANTERA_IMPORT_ERROR = exc
else:
    _CANTERA_IMPORT_ERROR = None


def require_cantera() -> None:
    """Raises `ImportError` if cantera isn't installed."""
    if ct is None:
        raise ImportError(
            "pyfoamy.cantera requires cantera - install with "
            "`pip install pyfoamy[thermo]`"
        ) from _CANTERA_IMPORT_ERROR


# Every {specie: fraction} dict this module returns drops species
# below this threshold.
_TRACE_THRESHOLD = 1e-9
# Reactant species below this are not blended in by partially_burnt().
_BLEND_THRESHOLD = 1e-6
_DRY_AIR_MOLE = {"O2": 0.21, "N2": 0.79}
_ELEMENT_SINKS = {"C": "CO2", "H": "H2O", "N": "N2", "S": "SO2"}
_TREF = 298.15
_PERMISSIVE_HINT = "(retry from_chemkin() with permissive=True to allow this)"


class Mechanism:
    """A loaded Cantera mechanism, shared by the other `pyfoamy.cantera`
    classes.
    """

    def __init__(self, mechanism_path, reference_pressure=None):
        require_cantera()
        # Kept apart from the resolved self._pressure so a mechanism
        # swap knows whether to keep it fixed or re-derive it; float()
        # as it is exposed via `reference_pressure`.
        self._reference_pressure_override = (
            None if reference_pressure is None else float(reference_pressure)
        )
        self.mechanism_path = mechanism_path

    @classmethod
    def coerce(cls, mechanism, reference_pressure=None) -> "Mechanism":
        """Returns a `Mechanism` from a path/name, `Solution` or `Mechanism`.

        Args:
            reference_pressure: Only valid unless `mechanism` is already a
                `Mechanism`.

        Raises:
            ValueError: If both a `Mechanism` and `reference_pressure` are
                given.
        """
        if isinstance(mechanism, cls):
            if reference_pressure is not None:
                raise ValueError(
                    "reference_pressure can only be given when "
                    "mechanism is a path, not an already-constructed "
                    "Mechanism - its reference pressure is already "
                    "fixed"
                )
            return mechanism
        return cls(mechanism, reference_pressure=reference_pressure)

    @classmethod
    def from_yaml(
        cls, mechanism_yaml: str, reference_pressure=None,
    ) -> "Mechanism":
        """Constructs a `Mechanism` from YAML text.

        Loads via a temporary file, deleted at once: building a `Solution`
        from a YAML string used ~90x more memory than from a path for a large
        mechanism (see `frozen_path()`). Nothing re-reads `mechanism_path`
        afterwards, so the dangling path is harmless.
        """
        require_cantera()
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False,
        ) as mechanism_file:
            mechanism_file.write(mechanism_yaml)
            path = mechanism_file.name
        try:
            return cls(path, reference_pressure=reference_pressure)
        finally:
            Path(path).unlink()

    @classmethod
    def from_chemkin(
        cls,
        mechanism_file,
        thermo_file=None,
        transport_file=None,
        out_path=None,
        permissive=None,
        reference_pressure=None,
    ) -> "Mechanism":
        """Converts a CHEMKIN mechanism to YAML with `ck2yaml` and loads it.

        Args:
            mechanism_file: CHEMKIN reactions/species file.
            thermo_file: Separate thermo file, unless the mechanism embeds a
                `THERMO` block.
            transport_file: CHEMKIN transport file.
            out_path: Keep the converted YAML here; by default it goes to a
                temporary file deleted after loading.
            permissive: Forwarded to `ck2yaml`, downgrading redundant
                species/thermo/transport declarations to warnings. Does not
                relax Cantera's duplicate-reaction check at load time.

        Raises:
            ValueError: If an input file is missing, `ck2yaml` rejects the
                input (its message is kept, with a `permissive=True` hint if
                that would help) or conversion fails otherwise.
            cantera.CanteraError: If the converted mechanism has undeclared
                duplicate reactions (mark them `DUPLICATE` in the source).
        """
        require_cantera()
        from cantera import ck2yaml

        mechanism_file = Path(mechanism_file)
        thermo_file = (
            Path(thermo_file) if thermo_file is not None else None
        )
        transport_file = (
            Path(transport_file) if transport_file is not None
            else None
        )
        for role, path in (
            ("mechanism_file", mechanism_file),
            ("thermo_file", thermo_file),
            ("transport_file", transport_file),
        ):
            if path is not None and not path.is_file():
                raise ValueError(f"chemkin {role} not found: {path}")

        def convert(yaml_path):
            # ck2yaml logs to stdout; its errors are repeated in InputError.
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    ck2yaml.convert(
                        str(mechanism_file),
                        thermo_file=(
                            str(thermo_file) if thermo_file else None
                        ),
                        transport_file=(
                            str(transport_file) if transport_file
                            else None
                        ),
                        out_name=str(yaml_path),
                        quiet=True,
                        permissive=permissive,
                    )
            except ck2yaml.InputError as exc:
                # Translate ck2yaml's CLI flag hint to this method's
                # keyword.
                if "--permissive" in str(exc):
                    exc.args = (f"{exc}\n\n{_PERMISSIVE_HINT}",)
                raise
            except Exception as exc:
                raise ValueError(
                    f"could not convert chemkin mechanism "
                    f"{mechanism_file} to Cantera format: {exc}"
                ) from exc
            if output.getvalue().strip():
                warnings.warn(f"ck2yaml: {output.getvalue().strip()}")

        return cls._from_converted(
            f"chemkin mechanism {mechanism_file}", convert, out_path,
            "the CHEMKIN DUPLICATE keyword", reference_pressure,
        )

    @classmethod
    def from_cti(
        cls, cti_file, out_path=None, reference_pressure=None,
    ) -> "Mechanism":
        """Converts a legacy Cantera CTI mechanism to YAML with `cti2yaml`
        and loads it.

        Args:
            cti_file: CTI mechanism file.
            out_path: Keep the converted YAML here; by default it goes to a
                temporary file deleted after loading.

        Raises:
            ImportError: If the installed Cantera no longer ships
                `cti2yaml`.
            ValueError: If the file is missing or `cti2yaml` rejects it (its
                message, with the offending line, is kept).
            cantera.CanteraError: If Cantera refuses the converted mechanism,
                e.g. for a species without data or undeclared duplicate
                reactions.
        """
        require_cantera()
        try:
            from cantera import cti2yaml
        except ImportError as exc:
            raise ImportError(
                f"CTI input needs a Cantera that ships cti2yaml, "
                f"not {ct.__version__}"
            ) from exc

        cti_file = Path(cti_file)
        if not cti_file.is_file():
            raise ValueError(f"cti mechanism file not found: {cti_file}")

        def convert(yaml_path):
            output = io.StringIO()
            try:
                # cti2yaml prints a syntax error's location, then exits.
                with contextlib.redirect_stdout(output), \
                        contextlib.redirect_stderr(output):
                    cti2yaml.convert(
                        filename=str(cti_file), output_name=str(yaml_path),
                    )
            except (Exception, SystemExit) as exc:
                raise ValueError(
                    f"could not convert cti mechanism {cti_file} to "
                    f"Cantera format: {output.getvalue().strip() or exc}"
                ) from exc

        return cls._from_converted(
            f"cti mechanism {cti_file}", convert, out_path,
            "options='duplicate'", reference_pressure,
        )

    @classmethod
    def _from_converted(
        cls, source, convert, out_path, duplicate_marker,
        reference_pressure,
    ) -> "Mechanism":
        keep = out_path is not None
        if keep:
            yaml_path = Path(out_path)
        else:
            with tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False,
            ) as yaml_file:
                yaml_path = Path(yaml_file.name)

        try:
            convert(yaml_path)
            try:
                return cls(
                    str(yaml_path),
                    reference_pressure=reference_pressure,
                )
            except ct.CanteraError as exc:
                note = (
                    f"{source} converted to Cantera YAML successfully, "
                    "but Cantera refused to load the result"
                )
                if "duplicate reaction" in str(exc).lower():
                    note += (
                        " - these are genuinely duplicate reactions "
                        "as far as Cantera is concerned; if "
                        f"intentional, mark them with {duplicate_marker} "
                        "in the source file"
                    )
                exc.args = (f"{note}:\n\n{exc}",)
                raise
        finally:
            if not keep:
                yaml_path.unlink(missing_ok=True)

    @property
    def mechanism_path(self):
        """The mechanism's path, or `None` if loaded from a `ct.Solution`.

        Reassigning loads a new mechanism in place (a `ct.Solution` is adopted
        directly). The reference pressure is re-derived from it unless one was
        given at construction.
        """
        return self._mechanism_path

    @mechanism_path.setter
    def mechanism_path(self, value) -> None:
        require_cantera()
        if isinstance(value, ct.Solution):
            self._gas = value
            self._mechanism_path = None
        else:
            self._gas = ct.Solution(value)
            self._mechanism_path = value
        self._pressure = (
            self.gas.reference_pressure
            if self._reference_pressure_override is None
            else self._reference_pressure_override
        )

    @property
    def mechanism_yaml(self) -> str:
        """The loaded mechanism serialized to YAML.

        Unlike re-reading `mechanism_path`, it matches what is loaded (also for
        a Cantera built-in name); for handing the mechanism to a
        `multiprocessing` worker.
        """
        require_cantera()
        return self.gas.write_yaml()

    @contextlib.contextmanager
    def frozen_path(self) -> Iterator[str]:
        """A temp-file path of this mechanism, for `multiprocessing` workers.

        Workers must load by path, never `yaml=<text>` (~90x memory blowup in
        Cantera for a large mechanism). The file is created once, so every
        worker of a sweep reads identical content.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w",
        ) as mechanism_file:
            mechanism_file.write(self.mechanism_yaml)
            mechanism_file.flush()
            yield mechanism_file.name

    @property
    def gas(self) -> "ct.Solution":
        """The live Cantera `Solution`, for same-process use."""
        return self._gas

    @property
    def reference_pressure(self) -> float:
        """The pressure property evaluation is anchored at."""
        return self._pressure

    def find_species(self, name: str) -> str:
        """Resolves `name` to this mechanism's spelling, case-insensitively.

        Mechanisms differ in species-name case (`"n2"`, `"N2"`, mixed); every
        hardcoded name here (`O2`, `N2`, `H2O`, the combustion sinks) and every
        caller composition is resolved through this.

        Raises:
            ValueError: If no or more than one species matches.
        """
        require_cantera()
        if name in self.gas.species_names:
            return name
        matches = [
            n for n in self.gas.species_names
            if n.lower() == name.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"{name!r} matches multiple species in this mechanism "
                f"case-insensitively: {matches} - use one of these "
                "exactly"
            )
        raise ValueError(f"no species named {name!r} in this mechanism")

    def resolve_names(self, names) -> List[str]:
        """`find_species()`, applied to every name in `names`."""
        return [self.find_species(name) for name in names]

    def resolve_composition(
        self, composition: Dict[str, float],
    ) -> Dict[str, float]:
        """`find_species()`, applied to every key of `composition`."""
        return {
            self.find_species(name): fraction
            for name, fraction in composition.items()
        }

    # Basis conversion and normalization via Cantera's X/Y setters.

    @staticmethod
    def _set_composition(gas, fractions: Dict[str, float], basis: str) -> None:
        if basis == "mole":
            gas.X = fractions
        elif basis == "mass":
            gas.Y = fractions
        else:
            raise ValueError(f"unknown basis {basis!r}, expected 'mass'/'mole'")

    @staticmethod
    def mass_fractions_of(
        gas: "ct.Solution", fractions: Dict[str, float], basis: str,
    ) -> Dict[str, float]:
        """Mass fractions of a composition on an arbitrary `Solution`."""
        Mechanism._set_composition(gas, fractions, basis)
        return gas.mass_fraction_dict(threshold=_TRACE_THRESHOLD)

    def _mole_fractions(
        self, fractions: Dict[str, float], basis: str,
    ) -> Dict[str, float]:
        # Always this mechanism's own gas, unlike mass_fractions_of().
        self._set_composition(self.gas, fractions, basis)
        return self.gas.mole_fraction_dict(threshold=_TRACE_THRESHOLD)

    def restricted(self, names: List[str]) -> "ct.Solution":
        """A `Solution` over only the given species (case-insensitive).

        Mixture-averaged evaluation scales worse than linearly with species
        count, and Cantera's transport fits are valid only over the
        temperature range common to all species of the full mechanism.
        """
        names = self.resolve_names(names)
        return ct.Solution(
            thermo="ideal-gas", transport_model=self.gas.transport_model,
            species=[self.gas.species(n) for n in names],
        )

    def elements(self, name: str) -> Dict[str, int]:
        """Integer atom counts of a real species (not a lumped one).

        Raises:
            ValueError: If an element count isn't a whole number.
        """
        elements = {}
        for element, count in self.gas.species(name).composition.items():
            if not count.is_integer():
                raise ValueError(
                    f"{name!r} has a non-integer {element!r} count "
                    f"({count}) - elements requires a single real "
                    "specie with a well-defined molecular formula"
                )
            elements[element] = int(count)
        return elements

    def molecular_weight(
        self, composition: Dict[str, float], basis: str = "mass",
    ) -> float:
        """Mean molecular weight of a composition in g/mol.

        Args:
            composition: `{specie: fraction}`, names case-insensitive.
            basis: `"mass"` or `"mole"`.
        """
        require_cantera()
        composition = self.resolve_composition(composition)
        gas = self.restricted(list(composition))
        self._set_composition(gas, composition, basis)
        return gas.mean_molecular_weight

    def _state(
        self, p: float, T: float, composition: Dict[str, float],
        basis: str,
    ) -> "ct.Solution":
        # Restricted, as in molecular_weight(): mixture-averaged
        # transport scales worse than linearly with species count and
        # its fit is valid only over the temperature range common to
        # all species.
        composition = self.resolve_composition(composition)
        gas = self.restricted(list(composition))
        self._set_composition(gas, composition, basis)
        # p/T go straight to Cantera's TP setter, which accepts Decimal.
        gas.TP = T, p
        return gas

    def viscosity(
        self, p: float, T: float, composition: Dict[str, float],
        basis: str = "mass",
    ) -> float:
        """Dynamic viscosity [Pa s] at `p` [Pa], `T` [K]."""
        require_cantera()
        return self._state(p, T, composition, basis).viscosity

    def density(
        self, p: float, T: float, composition: Dict[str, float],
        basis: str = "mass",
    ) -> float:
        """Mass density [kg/m^3] at pressure `p` [Pa], temperature `T` [K]."""
        require_cantera()
        return self._state(p, T, composition, basis).density

    def thermal_conductivity(
        self, p: float, T: float, composition: Dict[str, float],
        basis: str = "mass",
    ) -> float:
        """Mixture-averaged thermal conductivity [W/m/K] at `p`, `T`."""
        require_cantera()
        return self._state(p, T, composition, basis).thermal_conductivity

    def cp(
        self, p: float, T: float, composition: Dict[str, float],
        basis: str = "mass",
    ) -> float:
        """Mass-specific heat capacity [J/kg/K] at `p` [Pa], `T` [K]."""
        require_cantera()
        return self._state(p, T, composition, basis).cp_mass

    def properties(
        self, p: float, T: float, composition: Dict[str, float],
        basis: str = "mass",
    ) -> Dict[str, float]:
        """`density`/`viscosity`/`thermal_conductivity`/`cp` in one call.

        Cheaper than separate calls; same units. Returns a dict keyed by those
        names.
        """
        require_cantera()
        gas = self._state(p, T, composition, basis)
        return {
            "density": gas.density,
            "viscosity": gas.viscosity,
            "thermal_conductivity": gas.thermal_conductivity,
            "cp": gas.cp_mass,
        }

    def mass_fractions(
        self, composition: Dict[str, float], basis: str = "mass",
    ) -> Dict[str, float]:
        """Mass fractions of a composition, normalized to sum to 1.

        Args:
            composition: `{specie: fraction}`, names case-insensitive.
            basis: `"mass"` or `"mole"`.

        Returns:
            `{specie: mass_fraction}` in the mechanism's spelling.
        """
        require_cantera()
        composition = self.resolve_composition(composition)
        return self.mass_fractions_of(self.gas, composition, basis)

    def mole_fractions(
        self, composition: Dict[str, float], basis: str = "mass",
    ) -> Dict[str, float]:
        """Mole fractions of a composition; see `mass_fractions()`."""
        require_cantera()
        composition = self.resolve_composition(composition)
        return self._mole_fractions(composition, basis)

    def stoichiometric_air_fuel_ratio(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
        basis: str = "mass",
    ) -> float:
        """Stoichiometric oxidant/fuel mass ratio, as OpenFOAM's
        `stoichiometricAirFuelMassRatio`.

        Always a mass ratio; `basis` only interprets `fuel`/`oxidant`.
        """
        require_cantera()
        fuel = self.resolve_composition(fuel)
        oxidant = self.resolve_composition(oxidant)
        return self.gas.stoich_air_fuel_ratio(fuel, oxidant, basis=basis)

    def mixture_fraction(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
        basis: str = "mass", equivalence_ratio: Optional[float] = None,
        air_fuel_equivalence_ratio: Optional[float] = None,
    ) -> float:
        """Mass fraction of the mixture from the fuel stream.

        Args:
            fuel: `{specie: fraction}`, names case-insensitive.
            oxidant: `{specie: fraction}`.
            basis: `"mass"` or `"mole"` for `fuel`/`oxidant`.
            equivalence_ratio: Actual to stoichiometric fuel/oxidant ratio.
            air_fuel_equivalence_ratio: Its inverse (lambda).

        Raises:
            ValueError: Unless exactly one of the two ratios is given.
        """
        if (equivalence_ratio is None) == (
            air_fuel_equivalence_ratio is None
        ):
            raise ValueError(
                "mixture_fraction() needs exactly one of "
                "equivalence_ratio/air_fuel_equivalence_ratio"
            )
        afr_st = self.stoichiometric_air_fuel_ratio(fuel, oxidant, basis)
        if equivalence_ratio is not None:
            phi = float(equivalence_ratio)
            return phi / (afr_st + phi)
        return 1 / (float(air_fuel_equivalence_ratio) * afr_st + 1)

    def residual_fuel_fraction(
        self, mixture_fraction: float, fuel: Dict[str, float],
        oxidant: Dict[str, float], basis: str = "mass",
    ) -> float:
        """Fuel-stream mass fraction left by complete combustion, as
        OpenFOAM's `fres` of the inhomogeneous mixtures; 0 if lean.
        """
        Z = float(mixture_fraction)
        afr_st = self.stoichiometric_air_fuel_ratio(fuel, oxidant, basis)
        return max(Z - (1 - Z) / afr_st, 0)

    def partially_burnt(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
        mixture_fraction: float, burnt_fraction: float,
        basis: str = "mass",
    ) -> Dict[str, float]:
        """Blend of the unburnt mixture and its complete combustion
        products at one mixture fraction.

        Args:
            fuel: `{specie: fraction}`, names case-insensitive.
            oxidant: `{specie: fraction}`.
            mixture_fraction: Mass fraction from the fuel stream.
            burnt_fraction: Mass fraction of products, 0 (unburnt) to 1.
            basis: `"mass"` or `"mole"` for `fuel`/`oxidant`.

        Returns:
            `{specie: mass_fraction}`, normalized to sum to 1.
        """
        require_cantera()
        fuel = self.resolve_composition(fuel)
        oxidant = self.resolve_composition(oxidant)
        self.gas.set_mixture_fraction(
            mixture_fraction, fuel, oxidant, basis=basis,
        )
        unburnt = self.gas.mass_fraction_dict(threshold=_BLEND_THRESHOLD)
        burnt = self.complete_combustion_products(
            fuel, oxidant, basis=basis, mixture_fraction=mixture_fraction,
        )
        burnt_fraction = float(burnt_fraction)
        result = {}
        for part, weight in (
            (unburnt, 1 - burnt_fraction), (burnt, burnt_fraction),
        ):
            for name, value in part.items():
                if not np.isclose(weight * value, 0):
                    result[name] = result.get(name, 0) + weight * value
        total = sum(result.values())
        return {name: value / total for name, value in result.items()}

    def air(
        self, relative_humidity: Optional[float] = None,
        absolute_humidity: Optional[float] = None,
        T: Optional[float] = None, P: Optional[float] = None,
        basis: str = "mass",
    ) -> Dict[str, float]:
        """Dry or humid air composition.

        `relative_humidity` (0-1) needs `T`/`P` (saturation pressure via
        `cantera.Water()`); `absolute_humidity` (kg H2O/kg dry air) needs
        neither. Keys are the mechanism's spelling of `O2`/`N2`/`H2O`.
        """
        require_cantera()
        if relative_humidity is not None and absolute_humidity is not None:
            raise ValueError(
                "give at most one of relative_humidity/absolute_humidity"
            )
        # float(): Decimal (from an OpenFOAM dict) cannot mix with the
        # floats/numpy values in the arithmetic below.
        if relative_humidity is not None:
            relative_humidity = float(relative_humidity)
        if absolute_humidity is not None:
            absolute_humidity = float(absolute_humidity)
        if T is not None:
            T = float(T)
        if P is not None:
            P = float(P)
        dry_air_mole = self.resolve_composition(_DRY_AIR_MOLE)
        h2o = self.find_species("H2O")

        if absolute_humidity is not None:
            y_h2o = absolute_humidity / (1 + absolute_humidity)
            dry_mass = self.mass_fractions_of(self.gas, dry_air_mole, "mole")
            mass = {h2o: y_h2o}
            mass.update({n: (1-y_h2o)*v for n, v in dry_mass.items()})
            if basis == "mass":
                return mass
            return self._mole_fractions(mass, "mass")

        if relative_humidity is not None:
            if T is None or P is None:
                raise ValueError("relative_humidity requires T and P")
            water = ct.Water()
            water.TP = T, ct.one_atm  # P_sat depends on T only
            x_h2o = relative_humidity * water.P_sat / P
            mole = {h2o: x_h2o}
            mole.update({n: (1-x_h2o)*v for n, v in dry_air_mole.items()})
            if basis == "mole":
                return mole
            return self.mass_fractions_of(self.gas, mole, "mole")

        if basis == "mole":
            return dict(dry_air_mole)
        return self.mass_fractions_of(self.gas, dry_air_mole, "mole")

    def complete_combustion_products(
        self, fuel: Dict[str, float], oxidant: Dict[str, float],
        basis: str = "mass", equivalence_ratio: Optional[float] = None,
        mixture_fraction: Optional[float] = None,
    ) -> dict:
        """Composition of idealized complete combustion of fuel in oxidant.

        Not true equilibrium (no CO/OH/NO/H2/H/O from dissociation): a Cantera
        equilibrium restricted to the reactants plus each element's sink (C to
        CO2, H to H2O, N to N2, S to SO2) forces complete combustion.

        Args:
            fuel: `{specie: fraction}`, names case-insensitive.
            oxidant: `{specie: fraction}`.
            basis: `"mass"` or `"mole"` for `fuel`/`oxidant`.
            equivalence_ratio: Actual to stoichiometric fuel/oxidant ratio;
                defaults to `1.0` if `mixture_fraction` is not given.
            mixture_fraction: Mass fraction of the mixture from the fuel
                stream [kg fuel/kg mixture].

        Returns:
            `{specie: mass_fraction}` of the products.

        Raises:
            ValueError: If both `equivalence_ratio` and `mixture_fraction` are
                given.
        """
        require_cantera()
        if equivalence_ratio is not None and mixture_fraction is not None:
            raise ValueError(
                "complete_combustion_products() accepts at most one "
                "of equivalence_ratio/mixture_fraction, not both"
            )
        if equivalence_ratio is None and mixture_fraction is None:
            equivalence_ratio = 1.0

        fuel = self.resolve_composition(fuel)
        oxidant = self.resolve_composition(oxidant)
        reactant_names = list(fuel) + list(oxidant)
        sinks = self.combustion_sinks(reactant_names)
        restricted_names = sorted(set(reactant_names) | sinks)
        restricted = self.restricted(restricted_names)
        restricted.TP = _TREF, self._pressure
        if mixture_fraction is not None:
            restricted.set_mixture_fraction(
                mixture_fraction, fuel, oxidant, basis=basis,
            )
        else:
            restricted.set_equivalence_ratio(
                equivalence_ratio, fuel, oxidant, basis=basis,
            )
        restricted.equilibrate("HP")

        # No "not in fuel/oxidant" exclusion: a non-stoichiometric ratio
        # can leave reactant mass.
        return restricted.mass_fraction_dict(threshold=_TRACE_THRESHOLD)

    def combustion_sinks(self, reactant_names) -> set:
        """The complete-combustion product species of the reactants' elements."""
        elements = set()
        for name in reactant_names:
            elements.update(
                el for el, n in self.gas.species(name).composition.items()
                if n > 0
            )
        sinks = set()
        for el in elements:
            sink = _ELEMENT_SINKS.get(el)
            if sink is None:
                continue  # inert element (e.g. Ar/He) - no sink needed
            try:
                sink = self.find_species(sink)
            except ValueError:
                raise ValueError(
                    f"mechanism has no {sink!r} to hold element {el!r}"
                ) from None
            sinks.add(sink)
        return sinks
