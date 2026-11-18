import pytest

from pyfoamy import clouds, function_objects
from pyfoamy.parsing.grammar import Dictionary


def test_bounded_distributions_end_with_weighting_and_bounds():
    assert clouds.rosin_rammler(1e-4, 3, q=0, minimum=1e-6, maximum=2e-4) == {
        "type": "RosinRammler", "d": 1e-4, "n": 3, "Q": 0, "min": 1e-6,
        "max": 2e-4,
    }
    assert clouds.exponential(2, q=3, minimum=1, maximum=2)["lambda"] == 2
    assert clouds.multi_normal(
        (1, 2), (3, 4), (5, 6), q=0, minimum=1, maximum=2,
    )["strength"] == [5, 6]
    assert clouds.uniform(q=0, minimum=1, maximum=2)["type"] == "uniform"
    assert clouds.normal(1, 2, q=0, minimum=1, maximum=2)["sigma"] == 2


def test_bounded_distribution_needs_rising_bounds():
    with pytest.raises(ValueError, match="minimum 2 must be smaller"):
        clouds.uniform(q=0, minimum=2, maximum=2)


def test_unbounded_distributions():
    assert clouds.fixed_value(9e-5) == {"type": "fixedValue", "value": 9e-5}
    assert clouds.multi_fixed_value([[1, 0.3], [2, 0.7]])["values"] == [
        (1, 0.3), (2, 0.7),
    ]
    assert clouds.tabulated_cumulative([(1, 0), (2, 1)], q=3) == {
        "type": "tabulatedCumulative", "Q": 3,
        "distribution": [(1, 0), (2, 1)],
    }
    assert clouds.tabulated_density([(1, 2)], q=0)["type"] == (
        "tabulatedDensity"
    )


def test_flow_rate_and_discharge_takes_a_rate_or_mass_and_profile():
    assert clouds.flow_rate_and_discharge(0.9, mass_flow_rate=2) == {
        "flowType": "flowRateAndDischarge", "Cd": 0.9, "massFlowRate": 2,
    }
    assert clouds.flow_rate_and_discharge(
        0.9, mass_total=1e-6, flow_rate_profile=1,
    ) == {
        "flowType": "flowRateAndDischarge", "Cd": 0.9, "massTotal": 1e-6,
        "flowRateProfile": 1,
    }
    for arguments in (
        {}, {"mass_total": 1}, {"flow_rate_profile": 1},
        {"mass_flow_rate": 1, "mass_total": 1, "flow_rate_profile": 1},
    ):
        with pytest.raises(ValueError, match="mass_flow_rate, or"):
            clouds.flow_rate_and_discharge(0.9, **arguments)


def test_cone_injection_ends_with_its_flow():
    result = clouds.cone_injection(
        soi=708, duration=8, position=(0, 0, 0), direction=(0, 0, -1),
        theta_inner=0, theta_outer=10, d_outer=9e-5,
        size_distribution=clouds.fixed_value(9e-5),
        flow=clouds.constant_velocity(100, 1e-6),
        uniform_parcel_size="volume",
    )
    assert list(result)[:4] == [
        "type", "SOI", "duration", "uniformParcelSize",
    ]
    assert result["dInner"] == 0
    assert result["injectionMethod"] == "disc"
    assert list(result)[-3:] == ["flowType", "Umag", "massTotal"]
    assert clouds.pressure_driven_velocity(1e7, 1e-6)["Pinj"] == 1e7


def test_sub_model_writes_coefficients_only_if_given():
    assert clouds.sub_model("atomisationModel", "none") == {
        "atomisationModel": "none",
    }
    assert clouds.sub_model("breakupModel", "TAB", {"y0": 0}) == {
        "breakupModel": "TAB", "TABCoeffs": {"y0": 0},
    }


def test_spray_cloud_renders_and_parses_back():
    cloud = clouds.spray_cloud(
        t0=363, injections={"nozzle1": {"type": "coneInjection"}},
        liquid={"NC12H26": 1},
        sub_models=clouds.sub_model("breakupModel", "none"),
        cpu_load=False,
    )
    text = Dictionary(cloud).as_text
    assert "sphereDrag     ;" in text
    assert "phases         (liquid { NC12H26 1; });" in text
    assert "rho            explicit 1;" in text
    assert cloud["constantProperties"]["T0"] == 363
    assert cloud["subModels"]["breakupModel"] == "none"
    assert cloud["cpuLoad"] is False
    assert "cpuLoad" not in clouds.spray_cloud(
        t0=1, injections={}, liquid={}, sub_models={},
    )


def test_clouds_model_and_cloud_info_name_the_clouds():
    model = clouds.clouds_model(["cloud1"])
    assert model["clouds"] == ["cloud1"]
    assert 'libs           ("liblagrangianParcel.so");' in Dictionary(
        model).as_text
    info = function_objects.cloud_info(("cloud1", "cloud2"))
    assert info["type"] == "cloudInfo"
    assert info["clouds"] == ["cloud1", "cloud2"]
