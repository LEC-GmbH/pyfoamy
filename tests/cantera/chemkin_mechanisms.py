"""Third-party CHEMKIN mechanisms, downloaded and cached on first use."""

import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

_CACHE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "chemkin" / "downloaded"
)

# name -> (base URL, {role: filename})
_MECHANISMS = {
    "c3mech": (
        "https://raw.githubusercontent.com/C3Mech/C3Mech/main/"
        "FULL_MODEL",
        {
            "mechanism": "C3MechV4.0.1.CKI",
            "thermo": "C3MechV4.0.1.THERM",
            "transport": "C3MechV4.0.1.TRAN",
        },
    ),
    "syngas_nox": (
        "https://universityofgalway.ie/media/researchcentres/"
        "combustionchemistrycentre/files/mechanismdownloads/"
        "hydrogen-nox",
        {
            "mechanism": "C0-C1_NOx.MECH",
            "thermo": "C0-C1_NOx.THERM",
            "transport": "C0-C1_NOx.TRAN",
        },
    ),
    "methanol": (
        "https://universityofgalway.ie/media/researchcentres/"
        "combustionchemistrycentre/files/mechanismdownloads/alcohols",
        {
            "mechanism": "Methanol_2016_mech.inp",
            "thermo": "Methanol_2016_therm.dat",
            "transport": "Methanol_2016_tran.dat",
        },
    ),
    # "c3mech" in portable CHEMKIN (LOWSP/TROESP expanded), which
    # converts; "oracle" is C3Mech's own Cantera YAML of it.
    "c3mech_cantera": (
        "https://raw.githubusercontent.com/C3Mech/C3Mech/main",
        {
            "mechanism": (
                "PRECOMPILED/C8%2B/C0-C8%2B_C5CY_C6CY_DMC%2BEC_N_PAH/"
                "Cantera/C3MechV4.0.1_4HZZ_C0-C8%2B_C5CY_C6CY_DMC%2B"
                "EC_N_PAH_LT-HT_Cantera.CKI"
            ),
            "thermo": "FULL_MODEL/C3MechV4.0.1.THERM",
            "transport": "FULL_MODEL/C3MechV4.0.1.TRAN",
            "oracle": (
                "PRECOMPILED/C8%2B/C0-C8%2B_C5CY_C6CY_DMC%2BEC_N_PAH/"
                "Cantera/C3MechV4.0.1_4HZZ_C0-C8%2B_C5CY_C6CY_DMC%2B"
                "EC_N_PAH_LT-HT_Cantera.yaml"
            ),
        },
    ),
}


def get(name: str) -> Optional[Dict[str, Path]]:
    """Returns role -> path for a `_MECHANISMS` entry, `None` offline."""
    base_url, files = _MECHANISMS[name]
    out_dir = _CACHE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role, filename in files.items():
        path = out_dir / Path(filename).name
        if not path.is_file():
            try:
                urllib.request.urlretrieve(f"{base_url}/{filename}", path)
            except (urllib.error.URLError, OSError):
                return None
        paths[role] = path
    return paths
