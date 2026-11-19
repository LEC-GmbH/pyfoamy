"""pyfoamyPengRobinsonGas against Cantera's PengRobinsonPhase and itself,
and the perfect-gas specie properties of its coefficient mixture.

The test programs are in openfoam/test/. Quantities are
compared dimensionless, each made so with its own code's gas constant:
OpenFOAM's and Cantera's differ by 6e-7.
"""

import os
import shutil
import subprocess
from itertools import product

import pytest

ct = pytest.importorskip("cantera")
np = pytest.importorskip("numpy")

from pyfoamy import custom_lib  # noqa: E402
from pyfoamy._openfoam_build import source_dir  # noqa: E402
from pyfoamy.cantera import critical  # noqa: E402

pytestmark = pytest.mark.openfoam

# Name in nasa_gas.yaml, Tc [K], Pc [Pa], omega
SPECIES = {
    "IC8H18": ("C8H18,isooctane", 543.9, 2.57e6, 0.304),
    "NC7H16": ("C7H16,n-heptane", 540.2, 2.74e6, 0.350),
    "TOLUENE": ("C7H8", 591.75, 4.108e6, 0.264),
    "C2H5OH": ("C2H5OH", 513.9, 6.148e6, 0.645),
    "N2": ("N2", 126.1, 3.3944e6, 0.0403),
    "O2": ("O2", 154.58, 5.043e6, 0.022),
    "CO2": ("CO2", 304.2, 7.39e6, 0.228),
    "H2O": ("H2O", 647.13, 2.2055e7, 0.3449),
    "CH4": ("CH4", 190.7, 4.63e6, 0.011),
}

GASOLINE = {"IC8H18": 0.6, "NC7H16": 0.15, "TOLUENE": 0.25}
MIXTURES = {
    **{name: {name: 1.0} for name in SPECIES},
    "GAS": GASOLINE,
    **{
        f"GAS{round(100*f)}": {
            **{k: f*v for k, v in GASOLINE.items()}, "N2": 1 - f,
        }
        for f in (0.3, 0.6, 0.9)
    },
    "IC8N2": {"IC8H18": 0.1, "N2": 0.9},
    "E85N2": {"C2H5OH": 0.425, "IC8H18": 0.075, "N2": 0.5},
    "AIRCH4": {"CH4": 0.095, "O2": 0.19, "N2": 0.715},
    "BURNT": {"N2": 0.72, "CO2": 0.14, "H2O": 0.14},
}

# Up to 900 K every species' sqrt(a alpha) is positive: one-fluid mixing is
# Cantera's exactly. Above, mixtures differ (header, "Validity and limits").
LOW_Z_STATES = list(product(
    (1e5, 1e6, 3e6, 6e6, 1e7, 2e7),
    (300, 350, 400, 450, 500, 540, 560, 600, 700, 900),
))
HOT_STATES = list(product((5e6, 1.5e7, 2.5e7), (1500, 2000, 2500)))
FD_STATES = [(2e5, 440), (8e6, 450), (8e6, 1100), (2.5e7, 1100), (2e7, 2500)]

COLUMNS = (
    "Z hDepRT CpDepR CvDepR CpMCvR alphavT errAlphavT errCpR errCvR errPsi"
).split()


def _nasa_species():
    wanted = {data[0]: name for name, data in SPECIES.items()}
    found = {}
    for sp in ct.Species.list_from_file("nasa_gas.yaml"):
        if sp.name in wanted:
            found[wanted[sp.name]] = sp
    return found


@pytest.fixture(scope="module")
def nasa():
    return _nasa_species()


def _build(name, root):
    # A copy: wmake writes its objects next to the sources.
    shutil.copytree(source_dir("test", name), root/"src")
    env = {
        **os.environ,
        "PYFOAMY_LIB_SRC": str(
            source_dir("lib", os.environ["WM_PROJECT_VERSION"])
        ),
        "PYFOAMY_TEST_BIN": str(root),
    }
    build = subprocess.run(
        ["wmake", str(root/"src")], env=env, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-3000:]
    return root/f"Test-{name}"


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    return _build(
        "pyfoamyPengRobinsonGas", tmp_path_factory.mktemp("program"),
    )


def _species_entry(sp, Tc, Pc, omega, W=None):
    T_low, T_common, T_high = sp.input_data["thermo"]["temperature-ranges"]
    low, high = sp.input_data["thermo"]["data"]
    return f"""
    {{
        specie {{ molWeight {W or sp.molecular_weight!r}; }}
        equationOfState {{ Tc {Tc!r}; Pc {Pc!r}; omega {omega!r}; }}
        thermodynamics
        {{
            Tlow {T_low!r}; Thigh {T_high!r}; Tcommon {T_common!r};
            highCpCoeffs ({' '.join(map(repr, high))});
            lowCpCoeffs ({' '.join(map(repr, low))});
        }}
    }}"""


def _run(program, cwd, species, mixtures, states, extra=""):
    cwd.mkdir()
    body = "species\n{\n" + "".join(
        f"    {name}{entry}\n" for name, entry in species.items()
    ) + "}\nmixtures\n{\n" + "".join(
        f"    {name} {{ "
        + " ".join(f"{k} {x!r};" for k, x in X.items()) + " }\n"
        for name, X in mixtures.items()
    ) + "}\nstates (" + " ".join(f"({p!r} {T!r})" for p, T in states) \
        + ");\n" + extra
    (cwd/"input").write_text(body)
    run = subprocess.run(
        [str(program)], cwd=cwd, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    results = {}
    for line in (cwd/"results.dat").read_text().splitlines()[1:]:
        name, p, T, *values = line.split()
        results[name, float(p), float(T)] = dict(
            zip(COLUMNS, map(float, values))
        )
    return results


def _all_species(nasa):
    return {
        name: _species_entry(nasa[name], *SPECIES[name][1:])
        for name in SPECIES
    }


def _cantera(nasa, X, states):
    """Dimensionless departures by Cantera, from a gas state each."""
    defs = []
    for name in X:
        d = dict(nasa[name].input_data, name=name)
        _, Tc, Pc, omega = SPECIES[name]
        d["equation-of-state"] = {"model": "Peng-Robinson"}
        d["critical-parameters"] = {
            "critical-temperature": Tc, "critical-pressure": Pc,
            "acentric-factor": omega,
        }
        defs.append(ct.Species.from_dict(d))
    ig = ct.ThermoPhase(thermo="ideal-gas", species=defs)
    R = ct.gas_constant
    out = {}
    for p, T in states:
        # Cantera takes the root nearest its current density: a new phase
        # starts from a gas
        pr = ct.ThermoPhase(thermo="Peng-Robinson", species=defs)
        pr.TPX = T, p, X
        ig.TPX = T, p, X
        out[p, T] = {
            "Z": p/(pr.density_mole*R*T),
            "hDepRT": (pr.enthalpy_mole - ig.enthalpy_mole)/(R*T),
            "CpDepR": (pr.cp_mole - ig.cp_mole)/R,
            "CvDepR": (pr.cv_mole - ig.cv_mole)/R,
            "CpMCvR": (pr.cp_mole - pr.cv_mole)/R,
            "alphavT": pr.thermal_expansion_coeff*T,
        }
    return out


@pytest.fixture(scope="module")
def low_z(program, nasa, tmp_path_factory):
    return _run(
        program, tmp_path_factory.mktemp("lowZ")/"run", _all_species(nasa),
        MIXTURES, LOW_Z_STATES + HOT_STATES,
    )


@pytest.mark.parametrize("mixture", MIXTURES)
def test_departures_are_cantera_s(low_z, nasa, mixture):
    X = MIXTURES[mixture]
    reference = _cantera(nasa, X, LOW_Z_STATES)
    for (p, T), ref in reference.items():
        ours = low_z[mixture, p, T]
        for key, value in ref.items():
            assert ours[key] == pytest.approx(value, abs=1e-8), (key, p, T)


@pytest.mark.parametrize("mixture", MIXTURES)
def test_above_the_sign_change_only_mixtures_differ(low_z, nasa, mixture):
    X = MIXTURES[mixture]
    reference = _cantera(nasa, X, HOT_STATES)
    tolerance = 1e-8 if len(X) == 1 else 5e-3
    for (p, T), ref in reference.items():
        assert low_z[mixture, p, T]["Z"] == pytest.approx(
            ref["Z"], abs=tolerance
        ), (p, T)


def test_no_nan_on_the_low_z_grid(low_z):
    for key, values in low_z.items():
        assert np.isfinite([values[c] for c in COLUMNS[:6]]).all(), key


def test_derivatives_match_finite_differences(program, nasa, tmp_path):
    results = _run(
        program, tmp_path/"fd", _all_species(nasa), MIXTURES, FD_STATES,
    )
    for key, r in results.items():
        assert r["errAlphavT"] < 1e-6, key
        assert r["errCpR"] < 1e-6, key
        assert r["errCvR"] < 1e-6, key
        assert r["errPsi"] < 1e-12, key
        assert r["CpMCvR"] == pytest.approx(
            1 + r["CpDepR"] - r["CvDepR"], abs=1e-9
        ), key


def test_operator_eq_has_finite_coefficients(program, nasa, tmp_path):
    _run(program, tmp_path/"eq", _all_species(nasa), {"N2": {"N2": 1}},
         FD_STATES)
    Z_eq, Z_first = map(
        float, (tmp_path/"eq"/"operatorEq.dat").read_text().split()
    )
    assert np.isfinite(Z_eq)
    assert Z_eq == Z_first


def test_lumped_pseudo_specie_is_the_mixture(program, nasa, tmp_path):
    data = {
        name: critical.Critical(*SPECIES[name][1:]) for name in GASOLINE
    }
    lump = critical.lump(data, GASOLINE)
    W = sum(x*nasa[n].molecular_weight for n, x in GASOLINE.items())
    species = _all_species(nasa)
    species["LUMP"] = _species_entry(
        nasa["IC8H18"], lump.Tc, lump.Pc, lump.omega, W,
    )
    results = _run(
        program, tmp_path/"lump", species,
        {"GAS": GASOLINE, "LUMP": {"LUMP": 1}}, LOW_Z_STATES,
    )
    for p, T in LOW_Z_STATES:
        for key in ("Z", "hDepRT", "CpDepR"):
            assert results["LUMP", p, T][key] == pytest.approx(
                results["GAS", p, T][key], rel=1e-10, abs=1e-12
            ), (key, p, T)


def test_temperature_from_enthalpy_across_the_pseudo_critical_point(
    program, nasa, tmp_path,
):
    Ts = [450 + 2.5*i for i in range(101)]
    offsets = (-50, -10, -1, 1, 10, 50)
    extra = (
        "roundTrip { mixture GAS; p 3e6; "
        f"T ({' '.join(map(repr, Ts))}); "
        f"T0Offsets ({' '.join(map(repr, offsets))}); }}\n"
    )
    _run(program, tmp_path/"rt", _all_species(nasa), {"GAS": GASOLINE},
         [(3e6, 555)], extra)
    rows = np.loadtxt(tmp_path/"rt"/"roundTrip.dat")
    T, T0, T_back = rows.T
    failed = rows[~(np.abs(T_back - T) < 2e-4*T)]
    assert not len(failed), failed


def test_coefficient_mixture_species_are_perfect_gases(tmp_path, monkeypatch):
    # The checkout's library, not an installed one.
    monkeypatch.setattr(custom_lib, "_installed_lib", lambda: None)
    lib = custom_lib.ensure_built()
    program = _build("pyfoamyIdealSpecieThermo", tmp_path)
    case = tmp_path/"case"
    shutil.copytree(source_dir("test", "pyfoamyIdealSpecieThermo", "case"),
                    case)
    control = case/"system"/"controlDict"
    control.write_text(
        control.read_text().replace('"libpyfoamy.so"', f'"{lib}"')
    )
    for command in ("blockMesh", str(program)):
        run = subprocess.run(
            [command], cwd=case, capture_output=True, text=True,
        )
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "PASSED" in run.stdout
