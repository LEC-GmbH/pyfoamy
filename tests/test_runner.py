import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from pyfoamy import Case, Runner

PYTHON = sys.executable
TRUE_ARGS = ["-c", "pass"]
FAIL_ARGS = ["-c", "import sys; sys.exit(1)"]


def _write_command(*lines: str) -> list:
    return ["-c", "\n".join(lines)]


def _write_noop_script(tmp_path: Path) -> str:
    # Ignores its arguments, like an application given "-parallel".
    script = tmp_path / "noopApp"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_live_run_writes_a_log_file_at_the_case_root(tmp_path):
    runner = Runner(tmp_path)
    runner.run(PYTHON, args=_write_command("print('hello')"))
    log = tmp_path / f"log.{Path(sys.executable).name}"
    assert log.is_file()
    assert "hello" in log.read_text()


def test_live_run_uses_the_app_basename_and_suffix_for_the_log_name(
    tmp_path,
):
    runner = Runner(tmp_path)
    runner.run(PYTHON, args=TRUE_ARGS, suffix="first")
    assert (tmp_path / f"log.{Path(sys.executable).name}.first").is_file()


def test_live_run_skips_if_the_log_already_exists(tmp_path):
    runner = Runner(tmp_path)
    log = tmp_path / f"log.{Path(sys.executable).name}"
    log.write_text("previous run\n")
    result = runner.run(PYTHON, args=TRUE_ARGS)
    assert result is None
    assert log.read_text() == "previous run\n"


def test_live_run_overwrite_truncates_the_existing_log(tmp_path):
    runner = Runner(tmp_path)
    log = tmp_path / f"log.{Path(sys.executable).name}"
    log.write_text("previous run\n")
    runner.run(
        PYTHON, args=_write_command("print('new')"), overwrite=True
    )
    assert "previous run" not in log.read_text()
    assert "new" in log.read_text()


def test_live_run_append_keeps_the_existing_log_content(tmp_path):
    runner = Runner(tmp_path)
    log = tmp_path / f"log.{Path(sys.executable).name}"
    log.write_text("previous run\n")
    runner.run(PYTHON, args=_write_command("print('more')"), append=True)
    text = log.read_text()
    assert "previous run" in text
    assert "more" in text


def test_live_run_renames_the_log_and_raises_on_failure(tmp_path):
    runner = Runner(tmp_path)
    with pytest.raises(RuntimeError, match="failed"):
        runner.run(PYTHON, args=FAIL_ARGS)
    name = Path(sys.executable).name
    assert not (tmp_path / f"log.{name}").exists()
    assert (tmp_path / f"log.{name}.failed").is_file()


def test_live_run_passes_the_runner_env_overlay_to_the_subprocess(
    tmp_path,
):
    runner = Runner(tmp_path, env={"PYFOAMY_TEST_VAR": "fromRunner"})
    runner.run(PYTHON, args=_write_command(
        "import os",
        "print(os.environ.get('PYFOAMY_TEST_VAR', ''))",
    ))
    log = tmp_path / f"log.{Path(sys.executable).name}"
    assert "fromRunner" in log.read_text()


def test_live_run_per_call_env_overrides_the_runner_default(tmp_path):
    runner = Runner(tmp_path, env={"PYFOAMY_TEST_VAR": "fromRunner"})
    runner.run(
        PYTHON,
        args=_write_command(
            "import os",
            "print(os.environ.get('PYFOAMY_TEST_VAR', ''))",
        ),
        env={"PYFOAMY_TEST_VAR": "fromCall"},
    )
    log = tmp_path / f"log.{Path(sys.executable).name}"
    text = log.read_text()
    assert "fromCall" in text
    assert "fromRunner" not in text


def test_number_of_subdomains_reads_decompose_par_dict(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict("system/decomposeParDict", {"numberOfSubdomains": 4})
    runner = Runner(case.root)
    assert runner._number_of_subdomains() == 4


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="mpirun not on PATH"
)
def test_live_run_parallel_uses_number_of_subdomains(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict("system/decomposeParDict", {"numberOfSubdomains": 2})
    runner = Runner(case.root)
    script = _write_noop_script(tmp_path)
    runner.run(script, parallel=True)
    assert (case.root / f"log.{Path(script).name}").is_file()


def test_run_rejects_an_empty_command(tmp_path):
    runner = Runner(tmp_path)
    with pytest.raises(ValueError):
        runner.run("")


def test_unknown_mode_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        Runner(tmp_path, mode="parallel")


def test_record_mode_requires_a_script_path(tmp_path):
    with pytest.raises(ValueError, match="script_path"):
        Runner(tmp_path, mode="record")


def test_record_mode_does_not_execute_anything(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    result = runner.run(PYTHON, args=FAIL_ARGS)
    assert result is None
    assert not (tmp_path / f"log.{Path(sys.executable).name}").exists()


def test_record_mode_writes_a_real_allrun_style_header_once(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("blockMesh")
    runner.run("foamRun")
    text = script.read_text()
    assert text.startswith("#!/bin/sh\n")
    assert text.count("RunFunctions") == 1
    assert "runApplication blockMesh" in text
    assert "runApplication foamRun" in text


def test_record_mode_makes_the_script_executable(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("blockMesh")
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR


def test_record_mode_renders_parallel_runs_with_runParallel(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("foamRun", parallel=True, np=4)
    text = script.read_text()
    assert "runParallel -np 4 foamRun" in text


def test_record_mode_renders_overwrite_append_and_suffix_flags(
    tmp_path,
):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("blockMesh", overwrite=True, suffix="fine")
    text = script.read_text()
    assert "runApplication -overwrite -suffix fine blockMesh" in text


def test_record_mode_writes_runner_env_once_after_the_header(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(
        tmp_path, mode="record", script_path=script,
        env={"OMP_NUM_THREADS": "1"},
    )
    runner.run("blockMesh")
    runner.run("foamRun")
    text = script.read_text()
    assert text.count("export OMP_NUM_THREADS=1") == 1


def test_record_mode_writes_per_call_env_before_that_calls_line(
    tmp_path,
):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("blockMesh", env={"OMP_NUM_THREADS": "2"})
    text = script.read_text()
    lines = text.splitlines()
    export_index = lines.index("export OMP_NUM_THREADS=2")
    run_index = lines.index("runApplication blockMesh")
    assert export_index < run_index


def test_record_mode_scopes_per_call_env_to_that_call(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("blockMesh", env={"OMP_NUM_THREADS": "2"})
    runner.run("checkMesh")
    script.write_text(
        script.read_text().replace(
            ". $WM_PROJECT_DIR/bin/tools/RunFunctions",
            'runApplication() { echo "$1=${OMP_NUM_THREADS:-unset}"; }',
        )
    )
    result = subprocess.run(
        ["sh", str(script)], capture_output=True, text=True, check=True,
    )
    assert result.stdout.split() == ["blockMesh=2", "checkMesh=unset"]


def test_live_run_missing_executable_leaves_no_log(tmp_path):
    runner = Runner(tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.run("pyfoamyNoSuchApp")
    assert not (tmp_path / "log.pyfoamyNoSuchApp").exists()


def test_bool_options_render_as_bare_or_omitted_flags(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("decomposePar", options={"force": True, "cellDist": False})
    assert "runApplication decomposePar -force\n" in script.read_text()


def test_none_arg_is_rejected(tmp_path):
    runner = Runner(tmp_path, mode="record", script_path=tmp_path / "x")
    with pytest.raises(TypeError, match="None"):
        runner.run("blockMesh", args=[None])


def test_record_mode_renders_bare_and_valued_options(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run(
        "snappyHexMesh", options={"overwrite": None, "dict": "myDict"}
    )
    text = script.read_text()
    assert (
        "runApplication snappyHexMesh -overwrite -dict myDict" in text
    )


def test_record_mode_converts_snake_case_option_names_to_camel_case(
    tmp_path,
):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("snappyHexMesh", options={"no_overwrite": None})
    text = script.read_text()
    assert "-noOverwrite" in text


def test_record_mode_keeps_already_camel_case_option_names_verbatim(
    tmp_path,
):
    # A name with an uppercase letter (scalarBCs) is not converted.
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("setFields", options={"scalarBCs": None})
    text = script.read_text()
    assert "-scalarBCs" in text


def test_record_mode_renders_a_list_option_value_as_a_paren_group(
    tmp_path,
):
    # Matches real flags like surfaceRenamePatch's -point/
    # splitMeshRegions' -insidePoint ('(0 1 0)').
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("surfaceRenamePatch", options={"point": [0, 1, 0]})
    text = script.read_text()
    assert "-point '(0 1 0)'" in text


def test_record_mode_renders_a_numeric_option_value(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run("foamPostProcess", options={"time": 4})
    text = script.read_text()
    assert "-time 4" in text


def test_record_mode_renders_positional_args_after_options(tmp_path):
    script = tmp_path / "Allrun"
    runner = Runner(tmp_path, mode="record", script_path=script)
    runner.run(
        "surfaceTransformPoints",
        args=["translate=(1 0 0)", "in.stl", "out.stl"],
    )
    text = script.read_text()
    assert (
        "surfaceTransformPoints 'translate=(1 0 0)' in.stl out.stl"
        in text
    )


def test_case_run_lazily_constructs_a_live_runner(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.run(PYTHON, args=_write_command("print('via case')"))
    log = case.root / f"log.{Path(sys.executable).name}"
    assert "via case" in log.read_text()
    assert case.runner is not None
    assert case.runner.root == case.root


def test_case_run_uses_a_custom_runner_when_given(tmp_path):
    script = tmp_path / "Allrun"
    case = Case.create(tmp_path / "myCase")
    case.runner = Runner(case.root, mode="record", script_path=script)
    case.run("blockMesh")
    assert "runApplication blockMesh" in script.read_text()
    assert not any(case.root.glob("log.*"))


def _record(tmp_path, **kwargs):
    return Runner(
        tmp_path, mode="record", script_path=tmp_path / "Allrun", **kwargs
    )


def test_record_header_replaces_the_default(tmp_path):
    runner = _record(
        tmp_path, header="#!/bin/bash\n#SBATCH --ntasks=4\n",
        env={"OMP_NUM_THREADS": "1"},
    )
    runner.run("blockMesh")
    runner.run("checkMesh")
    assert (tmp_path / "Allrun").read_text() == (
        "#!/bin/bash\n#SBATCH --ntasks=4\n"
        "export OMP_NUM_THREADS=1\n\n"
        "runApplication blockMesh\nrunApplication checkMesh\n"
    )


def test_record_rerun_word_is_unquoted_and_skipped_on_overwrite(tmp_path):
    runner = _record(tmp_path, header="", rerun="$REDO")
    runner.run("blockMesh", suffix="a")
    runner.run("foamRun", parallel=True)
    runner.run("checkMesh", overwrite=True)
    runner.run("checkMesh", append=True)
    assert (tmp_path / "Allrun").read_text().splitlines() == [
        "runApplication $REDO -suffix a blockMesh",
        "runParallel $REDO foamRun",
        "runApplication -overwrite checkMesh",
        "runApplication -append checkMesh",
    ]


def test_record_raw_line_is_written_as_it_is(tmp_path):
    runner = _record(tmp_path, header="")
    runner.raw("md5sum 0/*.orig > sums")
    assert (tmp_path / "Allrun").read_text() == "md5sum 0/*.orig > sums\n"


def test_live_raw_line_runs_in_a_shell_at_the_case_root(tmp_path):
    runner = Runner(tmp_path, env={"WORD": "hello"})
    runner.raw("echo $WORD > out")
    assert (tmp_path / "out").read_text() == "hello\n"
    with pytest.raises(RuntimeError, match="exit code 3"):
        runner.raw("exit 3")


def test_record_dictionary_edits(tmp_path):
    runner = _record(tmp_path, header="")
    runner.set_entry("system/controlDict", "endTime", 0.5)
    runner.set_entry("system/fvSolution", "PIMPLE/models", False)
    runner.set_entry("system/functions", "probe", {"type": "probes"})
    runner.remove_entry("system/functions", "probe")
    prefix = "foamDictionary -writePrecision 15 -entry"
    assert (tmp_path / "Allrun").read_text() == (
        f"{prefix} endTime -set 0.5 system/controlDict\n"
        f"{prefix} PIMPLE/models -set false system/fvSolution\n"
        f"{prefix} probe -set '{{\n    type           probes;\n}}' "
        "system/functions\n"
        f"{prefix} probe -remove system/functions\n"
    )


@pytest.mark.openfoam
def test_live_dictionary_edits(tmp_path):
    (tmp_path / "system").mkdir()
    path = tmp_path / "system" / "controlDict"
    path.write_text(
        "FoamFile { format ascii; class dictionary; object controlDict; }\n"
        "endTime 1;\npi 3.14159265358979;\nfunctions { a { type x; } }\n"
    )
    runner = Runner(tmp_path)
    runner.set_entry("system/controlDict", "endTime", 0.5)
    runner.set_entry("system/controlDict", "functions/b", {"type": "y"})
    runner.remove_entry("system/controlDict", "functions/a")
    edited = Case(tmp_path).read_dict("system/controlDict")
    assert float(edited["endTime"]) == 0.5
    assert "3.14159265358979" in path.read_text()
    assert [str(entry.key) for entry in edited["functions"]] == ["b"]
    with pytest.raises(RuntimeError, match="foamDictionary failed"):
        runner.remove_entry("system/missing", "a")


def test_live_run_logs_into_the_log_directory(tmp_path):
    runner = Runner(tmp_path, log_dir="logs")
    runner.run(PYTHON, args=_write_command("print('hello')"), suffix="a")
    log = runner.log_path(PYTHON, "a")
    assert log == tmp_path / "logs" / f"log.{Path(PYTHON).name}.a"
    assert "hello" in log.read_text()


def test_live_run_starts_with_the_launcher_words(tmp_path):
    case = Case.create(tmp_path / "myCase")
    case.write_dict("system/decomposeParDict", {"numberOfSubdomains": 3})
    echo = shutil.which("echo")
    runner = Runner(
        case.root, mpi_command=(echo, "ranks", "{np}"),
        serial_command=(echo, "serial"),
    )
    runner.run("app", args=["x"], parallel=True)
    assert runner.log_path("app").read_text() == "ranks 3 app -parallel x\n"
    runner.run("app", args=["x"], suffix="one")
    assert runner.log_path("app", "one").read_text() == "serial app x\n"


def test_live_run_takes_the_ranks_of_the_runner(tmp_path):
    echo = shutil.which("echo")
    runner = Runner(tmp_path, mpi_command=(echo, "{np}"), np=5)
    runner.run("app", parallel=True)
    assert runner.log_path("app").read_text() == "5 app -parallel\n"
