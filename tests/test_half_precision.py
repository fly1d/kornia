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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from testing.half_precision import _FailureCollector, compare_failures, read_baseline, write_baseline


def test_failure_collector_keeps_real_failures_and_ignores_xfail() -> None:
    collector = _FailureCollector()

    collector.pytest_runtest_logreport(SimpleNamespace(outcome="failed", nodeid="tests/a.py::test_a"))
    collector.pytest_runtest_logreport(
        SimpleNamespace(outcome="failed", wasxfail="expected", nodeid="tests/a.py::test_expected")
    )
    collector.pytest_runtest_logreport(SimpleNamespace(outcome="passed", nodeid="tests/a.py::test_pass"))

    assert collector.failures == {"tests/a.py::test_a"}


def test_baseline_round_trip_is_sorted_and_comments_are_ignored(tmp_path) -> None:
    path = tmp_path / "known.txt"
    write_baseline(path, ["tests/z.py::test_z", "tests/a.py::test_a", "tests/z.py::test_z"])

    assert read_baseline(path) == {"tests/a.py::test_a", "tests/z.py::test_z"}
    assert path.read_text(encoding="utf-8").splitlines()[-2:] == [
        "tests/a.py::test_a",
        "tests/z.py::test_z",
    ]


def test_baseline_rejects_duplicate_entries(tmp_path) -> None:
    path = tmp_path / "known.txt"
    path.write_text("tests/a.py::test_a\ntests/a.py::test_a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        read_baseline(path)


def test_compare_failures_reports_new_and_resolved() -> None:
    new, resolved = compare_failures(
        {"tests/new.py::test_new", "tests/kept.py::test_kept"},
        {"tests/kept.py::test_kept", "tests/fixed.py::test_fixed"},
    )

    assert new == {"tests/new.py::test_new"}
    assert resolved == {"tests/fixed.py::test_fixed"}
