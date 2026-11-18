"""Example: OpenFOAM thermo/transport data from a Cantera mechanism."""

from pyfoamy.cantera.species_data import ThermoDatabase

# Shipped with Cantera.
MECHANISM = "gri30.yaml"


def main():
    db = ThermoDatabase(
        MECHANISM, thermo_model="janaf", transport_model="const",
    )
    mech = db.mechanism

    # 1. A single species; only these get "elements".
    ch4 = db.species(["CH4"])["CH4"]
    print("CH4 molWeight:", ch4["specie"]["molWeight"])
    print("CH4 Tcommon:", ch4["thermodynamics"]["Tcommon"])
    print("CH4 elements:", ch4["elements"])

    # 1b. Molecular weight only.
    print("CH4 molecular_weight():", mech.molecular_weight({"CH4": 1.0}))

    # 2. A lumped pseudo-species, refit as one JANAF entry.
    natural_gas = {
        "CH4": 0.9, "C2H6": 0.05, "C3H8": 0.02, "CO2": 0.02, "N2": 0.01,
    }
    lumped = db.lump(natural_gas, basis="mole")
    print("natural gas molWeight:", lumped["specie"]["molWeight"])

    # 3. Complete-combustion products, lumped by mass.
    air = mech.air(basis="mole")
    composition = mech.complete_combustion_products(
        natural_gas, air, basis="mole", equivalence_ratio=1.0,
    )
    products = db.lump(composition, basis="mass")
    print("products molWeight:", products["specie"]["molWeight"])

    # 4. Stoichiometric air/fuel mass ratio.
    afr = mech.stoichiometric_air_fuel_ratio(natural_gas, air, basis="mole")
    print("stoichiometric air/fuel mass ratio:", afr)

    # 5. thermo/transport keywords for ThermoTypeSpec(**...).
    print("thermoType entries:", db.thermo_type_entries)


if __name__ == "__main__":
    main()
