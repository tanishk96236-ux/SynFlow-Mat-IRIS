"""The process must survive importing its own dependencies.

Two copies of Intel's OpenMP runtime reach this environment -- one bundled in
numpy's MKL, one in torch -- and MKL's response is to call abort() on the whole
process the moment the second one loads. There is no exception, no traceback
and no partial output: the interpreter is simply gone, and the shell reports
exit code 4.

That failure mode is why it has been expensive twice. The first time, 68 tests
covering DAO-G, the flow geometry, the velocity network, the pipeline and hull
calibration did not run for weeks, and I twice wrote the cause off as a broken
LAPACK. The second time it killed the KRhO2 reaction check after the structure
had been built and printed, which reads exactly like a bug in the chemistry.

The guard now lives in synflow/__init__.py, which imports nothing, so it runs
before any submodule can pull in torch. These tests check it stays there and
stays first.
"""

import pathlib
import re
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_importing_synflow_sets_the_openmp_guard():
    """conftest.py sets this too, so inside pytest it would pass either way.
    A subprocess with no conftest is what the scripts and scratch files
    actually get."""
    code = ("import os, sys; sys.path.insert(0, %r); import synflow; "
            "print(os.environ.get('KMP_DUPLICATE_LIB_OK'))"
            % str(_ROOT))
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "TRUE", out.stdout


def test_the_guard_runs_before_anything_can_import_torch():
    """Order is the whole point. Setting the variable after torch is already
    loaded does nothing at all -- MKL has read it by then -- so __init__.py
    must set it before its first import of anything from this package."""
    text = (_ROOT / "synflow" / "__init__.py").read_text(encoding="utf-8")

    guard = text.index("KMP_DUPLICATE_LIB_OK")
    for match in re.finditer(r"^\s*(?:from|import)\s+(\S+)", text, re.M):
        module = match.group(1)
        if module == "os":            # what the guard itself needs
            continue
        assert match.start() > guard, (
            "synflow/__init__.py imports {} at line {} -- before the OpenMP "
            "guard, which makes the guard useless if that import reaches "
            "torch".format(module, text[:match.start()].count("\n") + 1))


def test_a_torch_import_through_synflow_does_not_abort():
    """The end-to-end version: if this ever regresses, the subprocess dies
    without a traceback rather than failing an assertion, so check the exit
    code explicitly."""
    code = ("import sys; sys.path.insert(0, %r); "
            "import synflow, torch; print(torch.zeros(2).sum().item())"
            % str(_ROOT))
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, (
        "exit code {} -- a bare non-zero code with no traceback is the MKL "
        "abort:\n{}".format(out.returncode, out.stderr))
    assert out.stdout.strip() == "0.0"
