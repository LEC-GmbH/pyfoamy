import subprocess

import pytest

from pyfoamy import native
from pyfoamy.models import (
    CourantBlend,
    IncompressibleFluidSetup,
    LinearSolver,
    TurbulenceSpec,
)

P_SOLVER = LinearSolver("multigrid", tolerance=1e-7, rel_tol=0.01)
MOMENTUM_SOLVER = LinearSolver(
    "conjugate_gradient", tolerance=1e-6, rel_tol=0.1
)


def _words(entry):
    return str(entry).split()


def _build(setup, root, **kwargs):
    kwargs.setdefault("ddt_order", "steady")
    kwargs.setdefault("div_order", "firstOrder")
    kwargs.setdefault("p_solver", P_SOLVER)
    kwargs.setdefault("momentum_solver", MOMENTUM_SOLVER)
    return setup.build(root, **kwargs)


def test_required_fields_laminar():
    setup = IncompressibleFluidSetup(nu=1e-05)
    assert setup.required_fields() == ["p", "U"]


def test_required_fields_with_turbulence():
    setup = IncompressibleFluidSetup(
        nu=1e-05, turbulence=TurbulenceSpec("RAS", "kOmegaSST")
    )
    assert setup.required_fields() == ["p", "U", "nut", "k", "omega"]


def test_build_does_not_touch_disk_until_write(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    root = tmp_path / "myCase"
    case = _build(setup, root)
    assert not root.exists()
    case.write()
    assert root.is_dir()


def test_build_writes_a_bare_solver_entry_no_application(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    d = case["system/controlDict"]
    assert str(d["solver"]) == "incompressibleFluid"
    assert "application" not in d


def test_build_respects_run_control_arguments(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        start_time=0,
        end_time=10,
        delta_t=0.01,
    )
    d = case["system/controlDict"]
    assert int(d["startTime"]) == 0
    assert int(d["endTime"]) == 10
    assert float(d["deltaT"]) == 0.01


def test_build_writes_physical_properties(tmp_path):
    setup = IncompressibleFluidSetup(nu=2e-05)
    case = _build(setup, tmp_path / "myCase")
    d = case["constant/physicalProperties"]
    assert str(d["viscosityModel"]) == "constant"
    assert float(d["nu"]) == 2e-05


def test_build_writes_laminar_momentum_transport(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    d = case["constant/momentumTransport"]
    assert str(d["simulationType"]) == "laminar"
    assert "RAS" not in d


def test_build_writes_ras_momentum_transport(tmp_path):
    setup = IncompressibleFluidSetup(
        nu=1e-05, turbulence=TurbulenceSpec("RAS", "kEpsilon")
    )
    case = _build(setup, tmp_path / "myCase")
    d = case["constant/momentumTransport"]
    assert str(d["simulationType"]) == "RAS"
    ras = d["RAS"]
    assert str(ras["model"]) == "kEpsilon"
    assert ras["turbulence"].as_switch().value is True


def test_div_order_resolves_u_via_the_vector_table(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup, tmp_path / "myCase", div_order="secondOrderBounded"
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,U)"]) == ["Gauss", "limitedLinearV", "1"]


def test_div_order_accepts_literal_scheme_text(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup, tmp_path / "myCase", div_order="linearUpwindV grad(U)"
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,U)"]) == [
        "Gauss", "linearUpwindV", "grad(U)",
    ]


def test_div_order_accepts_a_courant_blend(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        div_order=CourantBlend(1, "secondOrderBounded", 10, "firstOrder"),
    )
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,U)"]) == [
        "Gauss", "cellCoBlended", "1", "limitedLinearV", "1",
        "10", "upwind",
    ]


def test_turbulence_fields_are_pinned_to_first_order_regardless_of_order(
    tmp_path,
):
    setup = IncompressibleFluidSetup(
        nu=1e-05, turbulence=TurbulenceSpec("RAS", "kOmegaSST")
    )
    case = _build(setup, tmp_path / "myCase", div_order="secondOrder")
    div = case["system/fvSchemes"]["divSchemes"]
    assert _words(div["div(phi,U)"]) == ["Gauss", "LUST", "grad(U)"]
    assert _words(div["div(phi,k)"]) == ["Gauss", "upwind"]
    assert _words(div["div(phi,omega)"]) == ["Gauss", "upwind"]
    assert "div(phi,nut)" not in div


def test_ddt_order_resolves_via_the_table(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase", ddt_order="secondOrder")
    ddt = case["system/fvSchemes"]["ddtSchemes"]
    assert _words(ddt["default"]) == ["backward"]


def test_ddt_order_accepts_literal_scheme_text(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup, tmp_path / "myCase", ddt_order="CrankNicolson 0.7"
    )
    ddt = case["system/fvSchemes"]["ddtSchemes"]
    assert _words(ddt["default"]) == ["CrankNicolson", "0.7"]


def test_build_defaults_grad_interpolation_and_sn_grad_schemes(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    schemes = case["system/fvSchemes"]
    assert _words(schemes["gradSchemes"]["default"]) == [
        "Gauss", "linear",
    ]
    assert _words(
        schemes["interpolationSchemes"]["default"]
    ) == ["linear"]
    assert _words(
        schemes["snGradSchemes"]["default"]
    ) == ["corrected"]
    assert _words(
        schemes["laplacianSchemes"]["default"]
    ) == ["Gauss", "linear", "corrected"]


def test_build_always_adds_interpolate_co_localmax(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    interp = case["system/fvSchemes"]["interpolationSchemes"]
    assert _words(interp["interpolate(Co)"]) == ["localMax"]


def test_build_allows_overriding_sn_grad_scheme_for_non_orthogonal_mesh(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup, tmp_path / "myCase", sn_grad_scheme="limited corrected 0.5"
    )
    schemes = case["system/fvSchemes"]
    assert _words(schemes["snGradSchemes"]["default"]) == [
        "limited", "corrected", "0.5",
    ]
    assert _words(schemes["laplacianSchemes"]["default"]) == [
        "Gauss", "linear", "limited", "corrected", "0.5",
    ]


def test_p_solver_resolves_as_symmetric_via_multigrid(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "GAMG"
    assert str(p["smoother"]) == "GaussSeidel"


def test_momentum_solver_resolves_as_asymmetric_via_conjugate_gradient(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    u = case["system/fvSolution"]["solvers"]["U"]
    assert str(u["solver"]) == "PBiCGStab"
    assert str(u["preconditioner"]) == "DILU"


def test_p_solver_conjugate_gradient_resolves_to_plain_pcg(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "conjugate_gradient", tolerance=1e-7, rel_tol=0.01
        ),
    )
    p = case["system/fvSolution"]["solvers"]["p"]
    assert str(p["solver"]) == "PCG"
    assert str(p["preconditioner"]) == "DIC"


def test_final_blocks_always_force_rel_tol_zero(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    assert int(solvers["pFinal"]["relTol"]) == 0
    assert int(solvers["UFinal"]["relTol"]) == 0


def test_min_iter_and_max_iter_are_optional_and_omitted_by_default(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    p = case["system/fvSolution"]["solvers"]["p"]
    assert "minIter" not in p
    assert "maxIter" not in p


def test_min_iter_and_max_iter_are_written_when_given(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "multigrid",
            tolerance=1e-7,
            rel_tol=0.01,
            min_iter=1,
            max_iter=50,
        ),
    )
    p = case["system/fvSolution"]["solvers"]["p"]
    assert int(p["minIter"]) == 1
    assert int(p["maxIter"]) == 50


def test_min_iter_final_defaults_to_min_iter(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "multigrid", tolerance=1e-7, rel_tol=0.01, min_iter=2
        ),
    )
    p_final = case["system/fvSolution"]["solvers"][
        "pFinal"
    ]
    assert int(p_final["minIter"]) == 2


def test_min_iter_final_overridable_independently(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        p_solver=LinearSolver(
            "multigrid",
            tolerance=1e-7,
            rel_tol=0.01,
            min_iter=2,
            min_iter_final=5,
        ),
    )
    p_final = case["system/fvSolution"]["solvers"][
        "pFinal"
    ]
    assert int(p_final["minIter"]) == 5


def test_symmetric_only_solver_rejected_for_momentum(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    with pytest.raises(ValueError, match="symmetric"):
        _build(
            setup,
            tmp_path / "myCase",
            momentum_solver=LinearSolver(
                "PCG", tolerance=1e-6, rel_tol=0.1
            ),
        )


def test_asymmetric_only_solver_rejected_for_pressure(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    with pytest.raises(ValueError, match="asymmetric"):
        _build(
            setup,
            tmp_path / "myCase",
            p_solver=LinearSolver(
                "PBiCGStab", tolerance=1e-7, rel_tol=0.01
            ),
        )


def test_pimple_correctors_are_optional_and_omitted_by_default(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    pimple = case["system/fvSolution"]["PIMPLE"]
    assert len(pimple) == 0


def test_pimple_correctors_are_written_when_given(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        n_outer_correctors=1,
        n_correctors=2,
        n_non_orthogonal_correctors=1,
        momentum_predictor=False,
    )
    pimple = case["system/fvSolution"]["PIMPLE"]
    assert int(pimple["nOuterCorrectors"]) == 1
    assert int(pimple["nCorrectors"]) == 2
    assert int(pimple["nNonOrthogonalCorrectors"]) == 1
    assert pimple["momentumPredictor"].as_switch().value is False


def test_pressure_reference_is_optional_and_omitted_by_default(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    pimple = case["system/fvSolution"]["PIMPLE"]
    assert "pRefCell" not in pimple
    assert "pRefValue" not in pimple


def test_pressure_reference_written_when_given(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup, tmp_path / "myCase", pressure_reference=(0, 0)
    )
    pimple = case["system/fvSolution"]["PIMPLE"]
    assert int(pimple["pRefCell"]) == 0
    assert int(pimple["pRefValue"]) == 0


def test_relaxation_factors_is_optional_and_omitted_by_default(
    tmp_path,
):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(setup, tmp_path / "myCase")
    assert "relaxationFactors" not in case["system/fvSolution"]


def test_relaxation_factors_written_when_given(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    case = _build(
        setup,
        tmp_path / "myCase",
        relaxation_factors={"fields": {"p": 0.3}, "equations": {"U": 0.7}},
    )
    relax = case["system/fvSolution"]["relaxationFactors"]
    assert float(relax["fields"]["p"]) == 0.3


def test_build_writes_solvers_for_every_transported_field(tmp_path):
    setup = IncompressibleFluidSetup(
        nu=1e-05, turbulence=TurbulenceSpec("RAS", "kOmegaSST")
    )
    case = _build(setup, tmp_path / "myCase")
    solvers = case["system/fvSolution"]["solvers"]
    for name in ["p", "pFinal", "U", "UFinal", "k", "kFinal",
                 "omega", "omegaFinal"]:
        assert name in solvers, f"missing solver block for {name!r}"
    assert "nut" not in solvers


def test_full_write_read_roundtrip(tmp_path):
    setup = IncompressibleFluidSetup(
        nu=1e-05, turbulence=TurbulenceSpec("RAS", "kOmegaSST")
    )
    root = tmp_path / "myCase"
    case = _build(setup, root, div_order="secondOrderBounded")
    case.write()

    reread = case.read_dict("constant/momentumTransport")
    assert str(reread["simulationType"]) == "RAS"
    assert (root / "system" / "controlDict").is_file()
    assert (root / "constant" / "physicalProperties").is_file()
    assert (root / "system" / "fvSchemes").is_file()
    assert (root / "system" / "fvSolution").is_file()


@pytest.mark.openfoam
def test_written_case_is_valid_per_foam_dictionary(tmp_path):
    setup = IncompressibleFluidSetup(
        nu=1e-05, turbulence=TurbulenceSpec("RAS", "kOmegaSST")
    )
    root = tmp_path / "myCase"
    case = _build(
        setup,
        root,
        div_order="secondOrderBounded",
        n_correctors=2,
        pressure_reference=(0, 0),
    )
    case.write()

    def foam_value(relative_path, entry):
        result = subprocess.run(
            [
                "foamDictionary",
                "-entry",
                entry,
                "-value",
                str(root / relative_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    assert foam_value("system/controlDict", "solver") == (
        "incompressibleFluid"
    )
    assert foam_value("constant/physicalProperties", "nu") == "1e-05"
    assert (
        foam_value("constant/momentumTransport", "RAS/model")
        == "kOmegaSST"
    )
    assert (
        foam_value("system/fvSolution", "solvers/k/solver")
        == "PBiCGStab"
    )
    assert (
        foam_value("system/fvSolution", "solvers/p/solver")
        == "GAMG"
    )
    assert (
        foam_value("system/fvSchemes", "divSchemes/div(phi,k)")
        == "Gauss upwind"
    )
    assert (
        foam_value(
            "system/fvSchemes", "interpolationSchemes/interpolate(Co)"
        )
        == "localMax"
    )
    assert foam_value("system/fvSolution", "PIMPLE/pRefCell") == "0"


@pytest.mark.openfoam
def test_control_dict_satisfies_real_time_construction(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-05)
    root = tmp_path / "myCase"
    case = _build(setup, root)
    case.write()

    # Time must construct from this controlDict; only the (absent)
    # mesh may fail.
    with pytest.raises(native.NativeError, match="points|polyMesh"):
        with native.open_case(tmp_path, "myCase"):
            pass


def test_build_rejects_gravity(tmp_path):
    setup = IncompressibleFluidSetup(nu=1e-5)
    with pytest.raises(ValueError, match="gravity"):
        _build(setup, tmp_path / "myCase", gravity=(0, 0, -9.81))
