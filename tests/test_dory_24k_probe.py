"""The 24K probe reports evidence this host can actually produce.

Op-level placement is unobtainable here (MLComputePlan says "other" for every op
and cannot plan a stateful model), so the probe answers with a latency pair and
records the tool's raw output verbatim.  These tests pin that shape: no placement
percentage anywhere, the retry list kept whole, and a failed attempt never rounded
up into a pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_probe():
    path = REPO / "scripts" / "dory_24k_placement_probe.py"
    spec = importlib.util.spec_from_file_location("probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return load_probe()


def test_an_attempt_that_works_is_run_once(probe):
    records = probe.run_attempts([sys.executable, "-c", "print('ok')"], attempts=5)
    assert len(records) == 1
    assert records[0]["ok"] is True
    assert records[0]["returncode"] == 0
    assert "load_1m" in records[0]          # the machine is recorded with the attempt


def test_every_failed_attempt_keeps_its_error_text(probe):
    failing = [sys.executable, "-c",
               "import sys; sys.stderr.write('error code: -14' + chr(10)); sys.exit(3)"]
    records = probe.run_attempts(failing, attempts=4)
    assert len(records) == 4
    assert all(record["returncode"] == 3 for record in records)
    assert all(record["error_14"] is True for record in records)
    assert any("error code: -14" in line for line in records[0]["stderr_tail"])


def test_a_hung_attempt_is_a_failed_attempt(probe):
    records = probe.run_attempts(
        [sys.executable, "-c", "import time; time.sleep(10)"], attempts=1, timeout=1)
    assert records[0]["ok"] is False
    assert records[0]["returncode"] == -1
    assert "timeout" in records[0]["stderr_tail"][-1]


def test_the_verdict_is_labelled_as_latency_evidence(probe):
    fast = probe.latency_verdict(120.0, 20.0)
    assert fast["verdict"] == "ane"
    assert fast["ratio"] == 6.0
    assert "latency evidence only" in fast["method"]
    assert "not obtainable" in fast["method"]
    same = probe.latency_verdict(120.0, 118.0)
    assert same["verdict"] == "cpu"


def test_the_report_carries_no_placement_percentage(probe):
    attempt = {"attempt": 1, "ok": True, "returncode": 0, "seconds": 12.0,
               "load_1m": 3.0, "swap": "total = 0.00M  used = 0.00M  free = 0.00M"}
    measured = {
        "context_length": 24576, "tile_rows": [24576], "state_mib": 999.0,
        "latency": {"cpu_only": {"median_ms": 120.0}, "cpu_and_ne": {"median_ms": 20.0}},
        "verdict": probe.latency_verdict(120.0, 20.0),
        "compute_plan": {"cpu_and_ne": {"available": False, "usable": False,
                                        "devices": [], "raw": ["boom"], "lines": 1}},
    }
    report = probe.build_report([attempt], measured)
    text = json.dumps(report)
    for forbidden in ("placement_share", "ane_percent", "percent_ane", "ane_ops"):
        assert forbidden not in text
    assert report["attempts_used"] == 1
    assert report["verdict"]["verdict"] == "ane"
    assert report["compute_plan"]["cpu_and_ne"]["raw"] == ["boom"]
    assert any("not obtainable" in caveat for caveat in report["caveats"])


def test_no_completed_attempt_is_inconclusive_not_a_pass(probe):
    failed = [{"attempt": 1, "ok": False, "returncode": 3, "error_14": True,
               "stderr_tail": ["error code: -14"]}]
    report = probe.build_report(failed, None)
    assert report["verdict"]["verdict"] == "inconclusive"
    assert report["probe"] is None
    assert report["attempts_used"] == 1


def test_a_missing_package_records_what_the_tool_said(probe, tmp_path):
    plan = probe.compute_plan_raw(tmp_path / "nope.mlpackage")
    assert plan["available"] is False
    assert plan["usable"] is False
    assert plan["raw"] and "nope.mlpackage" in plan["raw"][0]
