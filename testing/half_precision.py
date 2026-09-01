# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Run the CPU half-precision suite and enforce a reviewed failure baseline.

The complete half-precision suite is intentionally not a pull-request gate: a
number of tests exercise kernels that PyTorch does not provide on every CPU
version, and several modules are documented as partial support.  This module
keeps the known failures explicit while still making newly introduced failures
actionable in the scheduled workflow.  Test directories run in separate child
processes so large model tests cannot contaminate later groups or exhaust the
runner; a resource-terminated directory is retried one level deeper.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

import pytest

DEFAULT_BASELINE = Path("tests/half_precision_known_failures.txt")
DEFAULT_PATHS = ("tests/",)
_CHILD_TIMEOUT_SECONDS = 900


class _FailureCollector:
    """Collect failed test node IDs from pytest reports."""

    def __init__(self) -> None:
        self.failures: set[str] = set()

    def pytest_runtest_logreport(self, report: object) -> None:
        if getattr(report, "outcome", None) != "failed":
            return
        # A non-strict xfail is reported as ``skipped`` and is already excluded
        # by the outcome check. A strict xfail that unexpectedly passes is
        # reported as ``failed`` with ``wasxfail`` and must remain visible.
        nodeid = getattr(report, "nodeid", None)
        if nodeid:
            self.failures.add(str(nodeid))


def read_baseline(path: Path) -> set[str]:
    """Read and validate a newline-delimited node-ID baseline."""
    if not path.is_file():
        raise FileNotFoundError(f"half-precision baseline does not exist: {path}")

    entries: set[str] = set()
    duplicates: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in entries:
            duplicates.add(line)
        entries.add(line)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate entries in half-precision baseline: {duplicate_list}")
    return entries


def write_baseline(path: Path, failures: Iterable[str]) -> None:
    """Write a sorted baseline, retaining a short maintenance contract."""
    entries = sorted(set(failures))
    header = (
        "# Known CPU half-precision failures for the scheduled ratchet.\n"
        "# Scope: ubuntu-latest, Python 3.11, PyTorch 2.9.1, CPU, float16 and bfloat16.\n"
        "# Generated with: pixi run test-half-baseline\n"
        "# Keep entries sorted. Removing an entry records a newly passing test;\n"
        "# adding an entry requires a justification in the pull request.\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def compare_failures(actual: set[str], baseline: set[str]) -> tuple[set[str], set[str]]:
    """Return ``(new_failures, resolved_failures)`` for a test run."""
    return actual - baseline, baseline - actual


def _test_children(path: Path) -> list[Path]:
    """Return collectable child paths, excluding caches and helper modules."""
    if not path.is_dir():
        return []

    children = []
    for child in sorted(path.iterdir()):
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        if (child.is_dir() and _test_children(child)) or (
            child.is_file()
            and child.suffix == ".py"
            and (child.name.startswith("test_") or child.name.endswith("_test.py"))
        ):
            children.append(child)
    return children


def _initial_paths(paths: Sequence[str]) -> list[Path]:
    """Split test roots into independently runnable groups."""
    result = []
    for raw_path in paths:
        path = Path(raw_path)
        children = _test_children(path)
        result.extend(children if children else [path])
    return result


def _pytest_args(args: argparse.Namespace, paths: Sequence[str]) -> list[str]:
    return [
        "-q",
        "--tb=no",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        *paths,
    ]


def _write_failure_ids(path: Path, failures: Iterable[str]) -> None:
    """Write worker results without the human-facing baseline header."""
    entries = sorted(set(failures))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> int:
    """Run one group and leave its failure IDs for the parent process."""
    collector = _FailureCollector()
    exit_code = pytest.main(_pytest_args(args, args.paths), plugins=[collector])
    if args.worker_output is not None:
        _write_failure_ids(args.worker_output, collector.failures)

    if exit_code not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        print(f"half-precision pytest run did not complete (exit code {int(exit_code)})", file=sys.stderr)
        return 2
    return int(exit_code)


def _result_summary(output: str) -> str:
    """Extract the last useful pytest summary line from a worker log."""
    for line in reversed(output.splitlines()):
        text = line.strip()
        if " passed" in text or " failed" in text or " skipped" in text:
            return text
    return ""


def _run_child(args: argparse.Namespace, path: Path, output_path: Path) -> subprocess.CompletedProcess[str] | None:
    """Run one isolated pytest worker, returning ``None`` on timeout."""
    command = [
        sys.executable,
        "-m",
        "testing.half_precision",
        "--worker-output",
        str(output_path),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--",
        str(path),
    ]
    environment = os.environ.copy()
    environment["KORNIA_TEST_DEVICE"] = args.device
    environment["KORNIA_TEST_DTYPE"] = args.dtype
    environment["KORNIA_TEST_OPTIMIZER"] = ""
    try:
        result = subprocess.run(  # noqa: S603
            command,
            cwd=Path.cwd(),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=_CHILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"half-precision group timed out after {_CHILD_TIMEOUT_SECONDS}s: {path}", file=sys.stderr)
        if exc.stdout:
            print(str(exc.stdout)[-4000:], file=sys.stderr)
        return None

    summary = _result_summary(result.stdout)
    print(f"{path}: {summary or f'pytest exit {result.returncode}'}")
    return result


def _collect_group(
    args: argparse.Namespace,
    path: Path,
    temp_dir: Path,
    output_index: list[int],
    failures: set[str],
) -> bool:
    """Collect one path, retrying a killed directory at a finer granularity."""
    output_path = temp_dir / f"worker-{output_index[0]}.txt"
    output_index[0] += 1
    result = _run_child(args, path, output_path)
    if result is not None and result.returncode in (0, 1):
        try:
            failures.update(read_baseline(output_path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"invalid half-precision worker output for {path}: {exc}", file=sys.stderr)
            return False
        return True

    children = _test_children(path)
    if len(children) > 1:
        print(f"retrying {path} as {len(children)} smaller groups", file=sys.stderr)
        return all(_collect_group(args, child, temp_dir, output_index, failures) for child in children)

    if result is not None:
        if result.stdout:
            print(result.stdout[-4000:], file=sys.stderr)
        if result.stderr:
            print(result.stderr[-4000:], file=sys.stderr)
    return False


def _collect_failures(args: argparse.Namespace) -> set[str] | None:
    """Run all groups and return their union, or ``None`` if a group aborted."""
    failures: set[str] = set()
    output_index = [0]
    with TemporaryDirectory(prefix="kornia-half-precision-") as temporary:
        temp_dir = Path(temporary)
        for path in _initial_paths(args.paths):
            if not _collect_group(args, path, temp_dir, output_index, failures):
                return None
    return failures


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"newline-delimited known-failure file (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="replace the baseline with failures from this run",
    )
    parser.add_argument("--device", default="cpu", help="pytest device option (default: cpu)")
    parser.add_argument(
        "--dtype",
        default="float16,bfloat16",
        help="pytest dtype option (default: float16,bfloat16)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="test paths to run (default: tests/)",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["KORNIA_TEST_DEVICE"] = args.device
    os.environ["KORNIA_TEST_DTYPE"] = args.dtype
    os.environ["KORNIA_TEST_OPTIMIZER"] = ""
    if args.worker_output is not None:
        return _run_worker(args)

    failures = _collect_failures(args)
    if failures is None:
        print("half-precision test groups did not complete", file=sys.stderr)
        return 2

    if args.write_baseline:
        write_baseline(args.baseline, failures)
        print(f"wrote {len(failures)} known failures to {args.baseline}")
        return 0

    try:
        baseline = read_baseline(args.baseline)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    new_failures, resolved_failures = compare_failures(failures, baseline)
    print(f"observed {len(failures)} failures; baseline contains {len(baseline)}")
    if new_failures:
        print("new failures:")
        print("\n".join(f"  {nodeid}" for nodeid in sorted(new_failures)))
    if resolved_failures:
        print("resolved failures (remove deliberately in a follow-up):")
        print("\n".join(f"  {nodeid}" for nodeid in sorted(resolved_failures)))
    return 1 if new_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
