# Changelog

pyfoamy follows [Semantic Versioning](https://semver.org); before 1.0 a
minor version can change the API.

## Unreleased

## 0.3.0 - 2026-11-18

First public release.

- Dictionary parser and writer for the OpenFOAM file format.
- `polyMesh` and field readers as numpy arrays, ascii and binary, decomposed
  cases reconstructed in Python; optional native reader.
- `postProcessing/` readers: time series over restarts and chained cases,
  sets, probes; `Table.to_pandas()`.
- `Case`, solver setups (`incompressibleFluid`, `fluid`,
  `multicomponentFluid`, `XiFluid`), boundary conditions, function objects,
  `Function1`, fvModels/fvConstraints, clouds and sprays, `dynamicMeshDict`
  and meshing dictionaries.
- `Runner`: live runs and recorded job scripts.
- Cantera-based thermo and chemistry: species fits, lumped mixtures,
  reactions, laminar flame speed tables and fits, pyJac-derived Jacobian
  code.
- OpenFOAM Foundation 14 and dev (`openfoam_version()`), with `libpyfoamy`
  for both.
