# pyfoamy

Python interface to OpenFOAM (OpenFOAM Foundation, version 14 and dev):
read, edit and write dictionaries, fields and meshes; set up cases from
Python; read results and `postProcessing/` output as numpy arrays; generate
thermophysical and chemistry data from Cantera mechanisms.

## Features

- **Dictionaries**: a parser for the OpenFOAM file format (directives,
  macros, `#if`, regular-expression keys, binary lists); edit and write back
  in OpenFOAM's layout.
- **Meshes and fields**: `polyMesh` and vol/surface/point fields, ascii or
  binary, as numpy arrays; decomposed cases are reconstructed in Python.
- **Post-processing**: function-object output under `postProcessing/` (time
  series, probes, sets) as tables, joined over restarts and chained cases;
  optional `to_pandas()`.
- **Case setup**: solver setups (`incompressibleFluid`, `fluid`,
  `multicomponentFluid`, `XiFluid`) that write schemes, solution and
  thermophysical dictionaries; typed boundary conditions, function objects,
  `Function1` entries, fvModels/fvConstraints, clouds and sprays,
  `dynamicMeshDict` and meshing dictionaries.
- **Running**: `Runner` runs OpenFOAM applications like `runApplication`/
  `runParallel`, or records them into a job script.
- **Thermo and chemistry** (optional, Cantera): JANAF and transport fits for
  species and lumped mixtures, reactions in OpenFOAM form, laminar flame speed
  tables with fits, analytical Jacobian C code (derived from pyJac).

## Installation

Python >= 3.10. From a checkout:

```sh
pip install .              # core: pyparsing, numpy
pip install '.[thermo]'    # + Cantera for thermo and chemistry
pip install '.[pandas]'    # + pandas for Table.to_pandas()
```

Reading and writing files needs no OpenFOAM. Running applications and the
C++ in `openfoam/` need a sourced OpenFOAM 14 or dev.

## Quick start

```python
from pyfoamy import Case

case = Case("cavity")
control = case.read_dict("system/controlDict")
control["endTime"] = 1
case.write_dict("system/controlDict", control)

mesh = case.reader.read_mesh_at(0.1)       # points, faces, owner, patches
U = case.reader.read_field_at(0.1, "U")    # U.internal: (n_cells, 3) array
print(mesh.n_cells, list(mesh.patches), U.internal.mean(axis=0))

table = case.read_time_series("probes")    # postProcessing/probes/*/
frame = table.to_pandas()
```

More in `examples/`: reading and editing dictionaries, reading meshes,
species data from a Cantera mechanism, converting a mechanism, and a complete
`XiFluid` case written from Python.

## Mechanism conversion

A Cantera YAML, legacy Cantera CTI or CHEMKIN mechanism to OpenFOAM's
`reactions` and `thermo` files (needs `.[thermo]`):

```sh
python -m pyfoamy.cantera gri30.yaml -o constant
python -m pyfoamy.cantera mech.inp --thermo therm.dat \
    --transport tran.dat -o constant
python -m pyfoamy.cantera old.cti --thermo-model hPolynomial \
    --transport-model logPolynomial --T-low 300 --T-high 3000
```

`#include "thermo"` in `constant/physicalProperties` and `#include
"reactions"` in `constant/chemistryProperties`. A CTI or CHEMKIN input is
also written as Cantera YAML next to them. By default all species are refit
to one shared `Tcommon` (`--per-species` keeps each NASA7 polynomial) with
Sutherland transport; `--help` lists the options, and
`examples/convert_mechanism.py` does the same from Python.

## OpenFOAM library

`openfoam/` holds C++ that pyfoamy builds with `wmake` on first use for the
sourced OpenFOAM: `libpyfoamy` (the `piecewisePhi` laminar flame speed, the
`nonUniformTable2` Function2, the `pyfoamyPengRobinsonGas` equation of state
with its thermo packages and chemistry, and for release 14 the `pLog`
reaction rate and XiFluid mixtures), which generated cases load, and a
native mesh and field reader; `openfoam/test/` has the test programs. It is
not part of the Python package.

## Status

Pre-1.0: the API can still change between minor versions. See
`CHANGELOG.md`.

## Contributing

See `CONTRIBUTING.md`.

## Licence

MIT (`LICENSE`). `src/pyfoamy/pyjac/` is derived from pyJac 1.0.6
(`LICENSE-pyJac`, MIT). The OpenFOAM-linked C++ in `openfoam/` is
GPL-3.0-or-later (`openfoam/LICENSE`) and not part of the Python package.
