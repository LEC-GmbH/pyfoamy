from pathlib import Path

import pytest

from pyfoamy.parsing.regions import Region, list_regions

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_list_regions_on_real_multiregion_fixture():
    regions = list_regions(FIXTURES / "multiregion_cylinder")
    assert regions == [
        Region("fluid", "multicomponentFluid"),
        Region("film", "film"),
    ]


def test_list_regions_on_single_region_case_raises(tmp_path):
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; "
        "object controlDict; }\n"
        "startFrom startTime;\n"
        "startTime 0;\n"
    )
    with pytest.raises(KeyError):
        list_regions(tmp_path)
