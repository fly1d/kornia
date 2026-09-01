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
actionable in the scheduled workflow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pytest

DEFAULT_BASELINE = Path("tests/half_precision_known_failures.txt")
DEFAULT_PATHS = ("tests/",)


class _FailureCollector:
    """Collect failed test node IDs from pytest reports."""

    def __init__(self) -> None:
        self.failures: set[str] = set()

    def pytest_runtest_logreport(self, report: object) -> None:
        if getattr(report, "outcome", None) != "failed":
            return
        # A non-strict xfail is represented as a failed report with ``wasxfail``.
        # It is an expected result and must not become a ratchet entry.
        if getattr(report, "wasxfail", False):
            return
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
        "# Generated with: pixi run test-half-baseline\n"
        "# Keep entries sorted. Removing an entry records a newly passing test;\n"
        "# adding an entry requires a justification in the pull request.\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def compare_failures(actual: set[str], baseline: set[str]) -> tuple[set[str], set[str]]:
    """Return ``(new_failures, resolved_failures)`` for a test run."""
    return actual - baseline, baseline - actual


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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    collector = _FailureCollector()

    # The half-precision ratchet is deliberately a non-compile CPU run.  The
    # environment assignment also makes direct invocations match CI, where the
    # reusable test workflow passes an empty optimizer value.
    os.environ["KORNIA_TEST_DEVICE"] = args.device
    os.environ["KORNIA_TEST_DTYPE"] = args.dtype
    os.environ["KORNIA_TEST_OPTIMIZER"] = ""
    pytest_args = [
        "-q",
        "--tb=no",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        *args.paths,
    ]
    exit_code = pytest.main(pytest_args, plugins=[collector])

    # A collection/configuration failure is not a test result and must never be
    # hidden by an unchanged baseline.  ExitCode.TESTS_FAILED (1) is the only
    # non-zero result that can be compared against the known-failure set.
    if exit_code not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        print(f"half-precision pytest run did not complete (exit code {int(exit_code)})", file=sys.stderr)
        return 2

    if args.write_baseline:
        write_baseline(args.baseline, collector.failures)
        print(f"wrote {len(collector.failures)} known failures to {args.baseline}")
        return 0

    try:
        baseline = read_baseline(args.baseline)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    new_failures, resolved_failures = compare_failures(collector.failures, baseline)
    print(f"observed {len(collector.failures)} failures; baseline contains {len(baseline)}")
    if new_failures:
        print("new failures:")
        print("\n".join(f"  {nodeid}" for nodeid in sorted(new_failures)))
    if resolved_failures:
        print("resolved failures (remove deliberately in a follow-up):")
        print("\n".join(f"  {nodeid}" for nodeid in sorted(resolved_failures)))
    return 1 if new_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
