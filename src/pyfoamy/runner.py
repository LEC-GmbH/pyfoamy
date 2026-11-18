"""Runs (or records) OpenFOAM application invocations for a case."""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pyfoamy.parsing.grammar import Dictionary, to_grammar_value


def _render_bare_value(value) -> str:
    if isinstance(value, str):
        return value
    return str(to_grammar_value(value))


_SCRIPT_HEADER = (
    "#!/bin/sh\n"
    "cd ${0%/*} || exit 1    # Run from this directory\n"
    "\n"
    "# Source tutorial run functions\n"
    ". $WM_PROJECT_DIR/bin/tools/RunFunctions\n"
)


def _write_exports(f, env: Dict[str, str]) -> None:
    for name, value in env.items():
        f.write(
            f"export {shlex.quote(name)}={shlex.quote(str(value))}\n"
        )


def _flag_name(name: str) -> str:
    # Already camelCase: flags with acronyms ("scalarBCs") don't
    # round-trip through snake_case.
    if any(c.isupper() for c in name):
        return name
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


def _render_options(options: Optional[Dict[str, object]]) -> List[str]:
    result: List[str] = []
    for name, value in (options or {}).items():
        if value is False:
            continue
        result.append(f"-{_flag_name(name)}")
        if value is not None and value is not True:
            result.append(_render_bare_value(value))
    return result


def _render_args(args: Optional[list]) -> List[str]:
    return [_render_bare_value(a) for a in (args or [])]


class Runner:
    """Runs, or records a script of, OpenFOAM applications.

    `"live"` mode runs applications via `subprocess` with the log-naming,
    skip and flag semantics of `runApplication`/`runParallel`, and renames
    a failed/interrupted log to `.failed`/`.interrupted` so a re-run is not
    skipped. `"record"` mode executes nothing and appends the equivalent
    `runApplication`/`runParallel` line to a script.

    Args:
        root: Path to the case directory.
        env: Environment overlay on `os.environ` for every `run()`.
        mode: `"live"` or `"record"`.
        script_path: Script to append to; required for `"record"`.
        header: Text a new script starts with, in place of the default
            (shebang, `cd` to the script, `RunFunctions`); for a queue
            system's directives and a site's environment setup.
        rerun: Shell word recorded after `runApplication`/`runParallel`
            in every call that neither overwrites nor appends, unquoted:
            a variable of the header (`"$REDO"`) that the script sets to
            `-overwrite` to repeat steps whose logs exist.
        log_dir: Directory of the logs, relative to the case; default the
            case itself, as `runApplication`.
        mpi_command: The words a parallel application is started with in
            `"live"` mode; `"{np}"` is replaced by the number of ranks.
        serial_command: The words put before a serial application in
            `"live"` mode (`("srun", "-n", "1")` inside a batch job).
        np: MPI ranks of a parallel `run()` that gives none; default
            `numberOfSubdomains` of `system/decomposeParDict`.

    Raises:
        ValueError: If `mode` is unknown, or `"record"` has no
            `script_path`.
    """

    def __init__(
        self,
        root,
        env: Optional[Dict[str, str]] = None,
        mode: str = "live",
        script_path=None,
        header: Optional[str] = None,
        rerun: str = "",
        log_dir=None,
        mpi_command: Sequence[str] = ("mpirun", "-np", "{np}"),
        serial_command: Sequence[str] = (),
        np: Optional[int] = None,
    ):
        if mode not in ("live", "record"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "record" and script_path is None:
            raise ValueError("mode='record' requires script_path")

        self.root = Path(root)
        self.env = dict(env) if env else {}
        self.mode = mode
        self.script_path = (
            Path(script_path) if script_path is not None else None
        )
        self.header = header
        self.rerun = rerun
        self.log_dir = log_dir
        self.mpi_command = tuple(mpi_command)
        self.serial_command = tuple(serial_command)
        self.np = np

    def run(
        self,
        command: str,
        args: Optional[list] = None,
        options: Optional[Dict[str, object]] = None,
        parallel: bool = False,
        suffix: str = "",
        overwrite: bool = False,
        append: bool = False,
        np: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[int]:
        """Runs (or records) an application, like `runApplication`.

        Args:
            command: Application name, e.g. `"blockMesh"`.
            args: Positional arguments; non-`str` values are rendered like
                `options` values.
            options: `flag name -> value`. `snake_case` names become camelCase
                (`"no_overwrite"` -> `-noOverwrite`) unless they contain an
                uppercase letter (e.g. `scalarBCs`). `None` or `True` emits a
                bare flag, `False` omits it; other values go through
                `to_grammar_value`, so a list becomes `(a b c)`.
            parallel: Run under `mpirun` (`runParallel`).
            suffix: Log name `log.<appName>.<suffix>`.
            overwrite: Overwrite an existing log and re-run.
            append: Append to an existing log and re-run.
            np: MPI ranks; default `numberOfSubdomains` from
                `system/decomposeParDict`.
            env: Environment overlay for this call, over `self.env`.

        Returns:
            The exit code in `"live"` mode (`None` if skipped because the log
            exists); always `None` in `"record"` mode.

        Raises:
            ValueError: If `command` is empty.
            TypeError: If an argument or option value has no OpenFOAM
                rendering, e.g. `None` in `args`.
            FileNotFoundError: If the application is not found in `"live"`
                mode; no log is left behind.
            RuntimeError: If the application exits nonzero in `"live"` mode.
        """
        if not command:
            raise ValueError("command must not be empty")

        argv = [command] + _render_options(options) + _render_args(args)

        if self.mode == "record":
            line = self._record_line(
                argv, parallel, suffix, overwrite, append, np
            )
            self._append_to_script(line, env)
            return None

        return self._run_live(
            argv, parallel, suffix, overwrite, append, np, env
        )

    def raw(self, line: str, env: Optional[Dict[str, str]] = None) -> None:
        """Runs (or records) a shell command line as it is.

        For what `run()` would quote: globs, pipes, shell variables. No
        log is kept.

        Raises:
            RuntimeError: If the line exits nonzero in `"live"` mode.
        """
        if self.mode == "record":
            self._append_to_script(line, env)
            return
        result = subprocess.run(
            line, shell=True, cwd=self.root,
            env={**os.environ, **self.env, **(env or {})},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{line!r} failed with exit code {result.returncode}"
            )

    def set_entry(self, relative_path, entry: str, value) -> None:
        """Sets an entry of a dictionary on disk with `foamDictionary`.

        An edit between the applications of a script, where the dictionary
        has to change at that step and not when the case is written.

        Args:
            relative_path: Dictionary, relative to the case.
            entry: Entry name; `/` separates the levels
                (`"PIMPLE/nCorrectors"`).
            value: Rendered like an `options` value of `run()`; a `dict`
                becomes a sub-dictionary.
        """
        self._foam_dictionary(
            ["-entry", entry, "-set", _render_bare_value(value)],
            relative_path,
        )

    def remove_entry(self, relative_path, entry: str) -> None:
        """Removes an entry of a dictionary on disk with `foamDictionary`."""
        self._foam_dictionary(["-entry", entry, "-remove"], relative_path)

    def _foam_dictionary(self, arguments, relative_path) -> None:
        # foamDictionary rewrites the whole file: at its default
        # precision of 6 the other entries would lose digits.
        argv = [
            "foamDictionary", "-writePrecision", "15", *arguments,
            str(relative_path),
        ]
        if self.mode == "record":
            self._append_to_script(
                " ".join(shlex.quote(part) for part in argv), None
            )
            return
        result = subprocess.run(
            argv, cwd=self.root, env={**os.environ, **self.env},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"foamDictionary failed on {relative_path}: "
                f"{result.stderr.strip()}"
            )

    def _number_of_subdomains(self) -> int:
        path = self.root / "system" / "decomposeParDict"
        dictionary = Dictionary(path.read_text())
        return int(dictionary["numberOfSubdomains"])

    def log_path(self, command: str, suffix: str = "") -> Path:
        """The log of `run(command, suffix=suffix)` in `"live"` mode."""
        app_name = Path(command).name
        log_suffix = f"{app_name}.{suffix}" if suffix else app_name
        return self.root / (self.log_dir or "") / f"log.{log_suffix}"

    def _run_live(self, argv, parallel, suffix, overwrite, append, np,
                  env):
        log_path = self.log_path(argv[0], suffix)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.is_file() and not overwrite and not append:
            print(
                f"{argv[0]} already run in {self.root}: remove "
                f"{log_path.name} to re-run"
            )
            return None

        command = list(argv)
        if parallel:
            n = np if np is not None else self.np
            if n is None:
                n = self._number_of_subdomains()
            command = (
                [word.format(np=n) for word in self.mpi_command]
                + [command[0], "-parallel"] + command[1:]
            )
        else:
            command = list(self.serial_command) + command

        merged_env = {**os.environ, **self.env, **(env or {})}
        file_mode = "a" if append else "w"
        with open(log_path, file_mode) as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    env=merged_env,
                    stdin=(
                        subprocess.DEVNULL if parallel
                        else None
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except OSError:
                # A stale empty log would make every re-run a skip.
                if not append:
                    log.close()
                    log_path.unlink()
                raise
            except KeyboardInterrupt:
                log_path.rename(
                    log_path.with_name(
                        log_path.name + ".interrupted"
                    )
                )
                raise

        if result.returncode != 0:
            failed_path = log_path.with_name(
                log_path.name + ".failed"
            )
            log_path.rename(failed_path)
            raise RuntimeError(
                f"{argv[0]!r} failed with exit code "
                f"{result.returncode} - see {failed_path.name}"
            )
        return result.returncode

    def _record_line(self, argv, parallel, suffix, overwrite, append,
                     np) -> str:
        function = "runParallel" if parallel else "runApplication"
        flags = []
        if overwrite:
            flags.append("-overwrite")
        if append:
            flags.append("-append")
        if suffix:
            flags += ["-suffix", suffix]
        if parallel and np is not None:
            flags += ["-np", str(np)]
        parts = [shlex.quote(part) for part in flags + [str(a) for a in argv]]
        if self.rerun and not overwrite and not append:
            parts.insert(0, self.rerun)
        return " ".join([function] + parts)

    def _append_to_script(self, line, env):
        is_new = not self.script_path.is_file()
        self.script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.script_path, "a") as f:
            if is_new:
                if self.header is None:
                    f.write(_SCRIPT_HEADER)
                    f.write("\n")
                else:
                    f.write(self.header)
                _write_exports(f, self.env)
                if self.env:
                    f.write("\n")
            if env:
                # A subshell scopes the exports to this one call.
                f.write("(\n")
                _write_exports(f, env)
                f.write(line + "\n)\n")
            else:
                f.write(line + "\n")
        if is_new:
            self.script_path.chmod(0o755)
