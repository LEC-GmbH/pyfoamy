"""Converts a reaction mechanism to OpenFOAM `reactions` and `thermo` files."""

import argparse
import contextlib
import sys
import warnings
from pathlib import Path

from pyfoamy.cantera.mechanism import _PERMISSIVE_HINT, Mechanism, ct
from pyfoamy.cantera.reactions import ReactionConverter
from pyfoamy.cantera.species_data import (
    _THERMO_MODEL_FORMATTERS,
    _TRANSPORT_MODEL_FORMATTERS,
    ThermoDatabase,
)
from pyfoamy.openfoam_version import SUPPORTED_VERSIONS, use_openfoam_version
from pyfoamy.parsing.grammar import Dictionary

_SUFFIX_FORMATS = {".yaml": "yaml", ".yml": "yaml", ".cti": "cti"}

_EPILOG = """\
examples:
  python -m pyfoamy.cantera gri30.yaml -o constant
  python -m pyfoamy.cantera mech.inp --thermo therm.dat \\
      --transport tran.dat -o constant
  python -m pyfoamy.cantera old.cti --thermo-model hPolynomial \\
      --transport-model logPolynomial --T-low 300 --T-high 3000

#include "thermo" in constant/physicalProperties and "reactions" in
constant/chemistryProperties.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyfoamy.cantera",
        description=(
            "Converts a Cantera YAML, legacy Cantera CTI or CHEMKIN "
            "mechanism to the OpenFOAM files DIR/reactions and DIR/thermo; "
            "a CTI or CHEMKIN input is also written as DIR/<MECH stem>.yaml."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mechanism", metavar="MECH",
        help="mechanism file, or a Cantera data file name such as gri30.yaml",
    )
    parser.add_argument(
        "-o", "--output-dir", metavar="DIR", type=Path, default=Path("."),
        help="output directory (default: .)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing outputs",
    )

    source = parser.add_argument_group("input")
    source.add_argument(
        "--format", choices=("yaml", "cti", "chemkin"),
        help="input format (default: from the suffix: .yaml/.yml, .cti, "
        "anything else CHEMKIN)",
    )
    source.add_argument(
        "--thermo", metavar="FILE", type=Path,
        help="CHEMKIN thermo file, unless MECH holds a THERMO block",
    )
    source.add_argument(
        "--transport", metavar="FILE", type=Path,
        help="CHEMKIN transport file",
    )
    source.add_argument(
        "--permissive", action="store_true",
        help="CHEMKIN: accept redundant species/thermo/transport "
        "declarations",
    )

    fit = parser.add_argument_group("thermo/transport fit")
    fit.add_argument(
        "--thermo-model", choices=sorted(_THERMO_MODEL_FORMATTERS),
        default="janaf", help="(default: %(default)s)",
    )
    fit.add_argument(
        "--transport-model", choices=sorted(_TRANSPORT_MODEL_FORMATTERS),
        default="sutherland", help="(default: %(default)s)",
    )
    fit.add_argument(
        "--per-species", action="store_true",
        help="fit each species on its own (janaf: its NASA7 polynomials "
        "unchanged) instead of refitting all to one shared Tcommon, which "
        "coefficient-blending mixtures need",
    )
    fit.add_argument(
        "--T-low", metavar="K", type=float,
        help="lower fit temperature (default: the species' common range)",
    )
    fit.add_argument(
        "--T-high", metavar="K", type=float,
        help="upper fit temperature (default: the species' common range)",
    )

    parser.add_argument(
        "--openfoam-version", choices=SUPPORTED_VERSIONS,
        help="write reaction types for this version (default: the sourced "
        "OpenFOAM's)",
    )
    return parser


def _load(args, fmt: str, yaml_out: Path) -> Mechanism:
    if fmt == "yaml":
        return Mechanism(args.mechanism)
    if fmt == "cti":
        return Mechanism.from_cti(args.mechanism, out_path=yaml_out)
    return Mechanism.from_chemkin(
        args.mechanism, thermo_file=args.thermo,
        transport_file=args.transport, out_path=yaml_out,
        permissive=args.permissive or None,
    )


def _convert(args, fmt: str) -> str:
    out = args.output_dir
    reactions_path, thermo_path = out / "reactions", out / "thermo"
    yaml_out = (
        None if fmt == "yaml" else out / f"{Path(args.mechanism).stem}.yaml"
    )
    outputs = [p for p in (reactions_path, thermo_path, yaml_out) if p]

    inputs = {
        Path(p).resolve()
        for p in (args.mechanism, args.thermo, args.transport) if p
    }
    clash = [str(p) for p in outputs if p.resolve() in inputs]
    if clash:
        raise ValueError(f"output would overwrite an input: {clash[0]}")
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.force:
        raise ValueError(
            f"output exists, --force overwrites it: {', '.join(existing)}"
        )

    out.mkdir(parents=True, exist_ok=True)
    mechanism = _load(args, fmt, yaml_out)
    reactions = ReactionConverter(mechanism).reactions_dict()
    database = ThermoDatabase(
        mechanism, thermo_model=args.thermo_model,
        transport_model=args.transport_model,
    )
    fit = database.species if args.per_species else database.species_group
    species = fit(T_low=args.T_low, T_high=args.T_high)

    reactions_path.write_text(
        Dictionary({"reactions": reactions}).as_text + "\n"
    )
    thermo_path.write_text(
        Dictionary({"species": list(species), **species}).as_text + "\n"
    )

    tcommon = {
        s["thermodynamics"].get("Tcommon") for s in species.values()
    } - {None}
    shared = (
        f", Tcommon {tcommon.pop():.6g} K" if len(tcommon) == 1 else ""
    )
    summary = (
        f"wrote {reactions_path} ({len(reactions)} reactions), "
        f"{thermo_path} ({len(species)} species, {args.thermo_model}/"
        f"{args.transport_model}{shared})"
    )
    return summary + (f", {yaml_out}" if yaml_out else "")


def _message(exc: BaseException) -> str:
    # ck2yaml's own text already names --permissive, which is also this
    # CLI's flag; Cantera frames its messages in lines of asterisks.
    lines = str(exc).replace(f"\n\n{_PERMISSIVE_HINT}", "").splitlines()
    return "\n".join(
        line for line in lines if line.strip("*") or not line
    ).strip()


def main(argv=None) -> int:
    """Runs the command line; returns the exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    fmt = args.format or _SUFFIX_FORMATS.get(
        Path(args.mechanism).suffix.lower(), "chemkin",
    )
    if fmt != "chemkin" and (
        args.thermo or args.transport or args.permissive
    ):
        parser.error(
            "--thermo/--transport/--permissive need a CHEMKIN input"
        )

    errors = (ValueError, OSError, ImportError)
    if ct is not None:
        from cantera import ck2yaml
        errors += (ct.CanteraError, ck2yaml.InputError)

    version = (
        use_openfoam_version(args.openfoam_version)
        if args.openfoam_version else contextlib.nullcontext()
    )
    with warnings.catch_warnings():
        warnings.showwarning = (
            lambda message, *_, **__: print(
                f"{parser.prog}: warning: {message}", file=sys.stderr,
            )
        )
        try:
            with version:
                print(_convert(args, fmt))
        except errors as exc:
            print(f"{parser.prog}: error: {_message(exc)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
