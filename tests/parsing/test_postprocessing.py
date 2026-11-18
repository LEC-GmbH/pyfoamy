import shutil
from pathlib import Path

import numpy as np
import pytest

from pyfoamy.case import Case
from pyfoamy.parsing import (
    list_function_objects,
    list_times,
    read_sets,
    read_table,
    read_time_series,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "postprocessing"
ENGINE = FIXTURES / "engine"
CHAMBER = FIXTURES / "chamber"
CHAIN = FIXTURES / "chain"

PROBES = """\
# Probe 0 (0.1 0 0)
# Probe 1 (0.2 0 0)
# Time                    0                         1
0                         (1 2 3)                   (4 5 6)
0.1                       (7 8 9)                   (10 11 12)
"""


def test_field_value_header_and_columns():
    table = read_table(
        ENGINE / "postProcessing" / "cylinderVolFieldValueVolAverage"
        / "129" / "volFieldValue.dat"
    )
    assert table.header == {
        "Selection": "cylinderCells", "Cells": "1454117",
        "Volume": "5.445731e-04",
    }
    assert table.names[:3] == ["Time", "volAverage(p)", "volAverage(T)"]
    assert table.data.shape == (2, 10)
    assert table["Time"].tolist() == [129, 130]


def test_names_with_spaces_are_split_at_tabs():
    table = read_time_series(ENGINE, "LECmultiValveEngineState")
    assert table.names[1:3] == ["piston from TDC", "piston speed"]
    assert table.data.shape == (2, 11)


def test_vector_cells_are_split_into_components():
    table = read_time_series(ENGINE, "gapOrientedSum")
    assert table["orientedSum(Cf)_x"][0] == 5.781483e-05
    assert table["orientedSum(Cf)_z"][0] == -3.753259e-06
    assert "orientedSum(Cf)" not in table.names
    assert table.data.shape[1] == len(table.names)


def test_restart_segments_join_at_the_later_start():
    table = read_time_series(ENGINE, "timeStep")
    assert table["Time"].tolist() == [
        128.75, 128.7501, 128.7502, 800, 800.083333, 800.166667,
    ]
    assert [row for _, row in table.segments] == [0, 3]
    assert list_times(ENGINE, "timeStep") == [128.75, 800]


def test_later_segment_wins(tmp_path):
    case = Path(shutil.copytree(ENGINE, tmp_path / "case"))
    restart = case / "postProcessing" / "timeStep" / "800" / "timeStep.dat"
    restart.write_text(restart.read_text().replace(
        "1.201923e-05", "9.0e-05"
    ))
    table = read_time_series(case, "timeStep")
    assert table["deltaT"][table["Time"] == 800].tolist() == [9.0e-05]


def test_short_rows_are_dropped():
    table = read_time_series(CHAMBER, "chamberVolFieldValueVolIntegrate")
    assert table.dropped == 1
    assert table["Time"].tolist() == [0.1, 0.2]


def test_chained_cases_take_the_union_of_columns():
    table = read_time_series(
        [CHAIN / "fromColdFlow", CHAIN / "coldFlow"],
        "cylinderVolFieldValueVolAverage",
    )
    assert table.names[-1] == "volAverage(T.u)"
    assert table["Time"].tolist() == [
        180, 180.011905, 339.857143, 340, 340.142857,
    ]
    assert np.isnan(table["volAverage(T.u)"][:3]).all()
    assert not np.isnan(table["volAverage(T.u)"][3:]).any()
    assert table.segments[1][0].parts[-5] == "fromColdFlow"


def test_two_segments_at_one_start_raise():
    with pytest.raises(ValueError, match="start at 340"):
        read_time_series(
            [CHAIN / "fromColdFlow", CHAIN / "fromColdFlow"],
            "cylinderVolFieldValueVolAverage",
        )


def test_several_files_need_file(tmp_path):
    directory = tmp_path / "postProcessing" / "probes" / "0"
    directory.mkdir(parents=True)
    (directory / "U").write_text(PROBES)
    (directory / "p").write_text("# Time 0\n0 1\n")
    with pytest.raises(ValueError, match="pass file="):
        read_time_series(tmp_path, "probes")
    table = read_time_series(tmp_path, "probes", file="U")
    assert table.names == ["Time", "0_x", "0_y", "0_z", "1_x", "1_y", "1_z"]
    assert table.header == {"Probe 0": "(0.1 0 0)", "Probe 1": "(0.2 0 0)"}
    assert table["1_z"].tolist() == [6, 12]


def test_raw_sets_gz_and_csv(tmp_path):
    sets = read_sets(CHAMBER, "lineSample_samples", 70)
    assert sorted(sets) == ["x_IDX3_IDX3", "z_IDX8_IDX6"]
    assert sets["z_IDX8_IDX6"].names[:4] == ["z", "U_x", "U_y", "U_z"]
    assert sets["x_IDX3_IDX3"].data.shape == (3, 9)

    directory = tmp_path / "postProcessing" / "lines" / "0.5"
    directory.mkdir(parents=True)
    (directory / "a.csv").write_text("x,U_x,U_y,U_z,p\n0,1,2,3,4\n1,5,6,7,8\n")
    table = read_sets(tmp_path, "lines", 0.5)["a"]
    assert table.names == ["x", "U_x", "U_y", "U_z", "p"]
    assert table["p"].tolist() == [4, 8]


def test_other_set_formats_raise(tmp_path):
    directory = tmp_path / "postProcessing" / "lines" / "0"
    directory.mkdir(parents=True)
    (directory / "a.vtk").write_text("")
    with pytest.raises(ValueError, match="not a raw or csv set"):
        read_sets(tmp_path, "lines", 0)


def test_missing_output_raises():
    with pytest.raises(FileNotFoundError):
        read_time_series(ENGINE, "nothing")


def test_list_function_objects():
    assert list_function_objects(ENGINE) == [
        "LECmultiValveEngineState", "cylinderVolFieldValueVolAverage",
        "gapOrientedSum", "timeStep",
    ]
    assert list_function_objects(ENGINE / "missing") == []


def test_case_binds_its_root():
    case = Case(ENGINE)
    assert case.read_time_series("timeStep").data.shape == (6, 2)


def test_to_pandas():
    pytest.importorskip("pandas")
    frame = read_time_series(ENGINE, "timeStep").to_pandas()
    assert list(frame.columns) == ["Time", "deltaT"]
    assert len(frame) == 6
