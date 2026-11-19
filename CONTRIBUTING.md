# Contributing to pyfoamy

Bug reports and pull requests are welcome. For a larger change, open an issue
first so the design can be agreed before the work is done.

## Layout

- `src/pyfoamy/parsing/` - pyparsing grammar (`grammar.py`), list/field/
  polyMesh readers (numpy arrays: `PolyMesh`, `FieldValues`), decomposed
  cases (`decomposed.py`), `postProcessing/` time series and sets
  (`postprocessing.py`: `Table`), time directories, regions.
- `src/pyfoamy/` top level - `Case` (facade over `casefiles.py`: dictionary
  registry and writing; `include_entries.py`: `#includeFunc`/`#includeModel`/
  `#includeConstraint` registries; `reader.py`: mesh/field reads), `Runner`,
  field templates, boundary conditions (default sets keyed by `PatchRole`,
  patch names per case), function objects, `Function1` builders
  (`function1.py`), fvModels/fvConstraints, `dynamicMeshDict` entries
  (`dynamic_mesh.py`), native backend (`native.py`),
  custom OpenFOAM library builder (`custom_lib.py`).
- `src/pyfoamy/models/` - solver setups (`SolverSetup.build()` assembles the
  case from per-solver hooks), schemes, linear solvers, turbulence, thermo.
- `src/pyfoamy/cantera/` - mechanisms, species thermo/transport fits,
  reactions, the mechanism-conversion command line (`__main__.py`),
  laminar flame speed (`laminar_flame_speed.py` sweep and cache,
  `_lfs_*.py` solver, fits, validation; optional `cantera`; never
  imported by `pyfoamy/__init__.py`).
- `src/pyfoamy/pyjac/` - port of pyJac 1.0.6 (analytical Jacobian C source).
- `openfoam/` - OpenFOAM-linked C++ (`native/`, `lib/<version>/`),
  GPL-3.0-or-later, not in the wheel; built with `wmake` on first use from a
  temporary copy, found via `$PYFOAMY_OPENFOAM_SRC` or the checkout.

## Environment and tests

```sh
pip install -e '.[dev,thermo]'           # Python >= 3.10 virtualenv
python -m pytest                         # ~1 min, no OpenFOAM/tutorials
source <OpenFOAM install>/etc/bashrc     # Foundation 14 or dev
python -m pytest -m "openfoam or tutorials"   # ~6 min
python -m pytest -m slow                 # ~10 min, FreeFlame, C3Mech
python -m pytest -m ""                   # everything
```

Tutorial sweeps read `$PYFOAMY_TUTORIALS_ROOT`, else `$FOAM_TUTORIALS`.
Tests marked `openfoam`/`tutorials` skip when those are missing; mark new
tests accordingly. `tests/test_examples.py` runs every script in `examples/`;
a new example that needs OpenFOAM, Cantera or a long run gets its entry there.

## Ground truth

The OpenFOAM C++ source is the spec for the file format
(`$FOAM_SRC/OpenFOAM/db/IOstreams/token/`, `db/dictionary/**`,
`db/IOobject/`); `foamDictionary` is the oracle when parsing or formatting
behaviour is in doubt.

## Conventions

- Python >= 3.10, PEP8, 79 columns.
- Output that differs between OpenFOAM 14 and dev asks
  `openfoam_version()` (`openfoam_version.py`); the C++ in `openfoam/lib/`
  has one tree per version.
- Comments: only a non-obvious *why*, 1-2 lines. No history ("previously",
  "confirmed", dates, what was tried); that goes in the commit message.
- Docstrings: module one sentence; public API a summary line, plus
  Google-style `Args`/`Returns`/`Raises` only where names and types don't
  say it; private (`_`) helpers none.
- Error messages: one sentence, the offending value if useful.
- Fail fast: check preconditions before doing work or mutating state.
- No abstraction before the second or third use case.
- Commit messages explain why, not just what.
- A user-visible change gets a line in `CHANGELOG.md` under "Unreleased".

## Licence of contributions

Python code is contributed under the MIT licence (`LICENSE`); C++ in
`openfoam/` under GPL-3.0-or-later (`openfoam/LICENSE`).
