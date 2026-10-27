"""The ANE gates must be able to fail.

The suite and the single-run scorer used to print numbers and return 0 whatever
those numbers were, so "the gate passed" only meant "the script finished".  These
tests hold the judging seam itself: the thresholds are the ticket's, a number
over them fails, a missing number fails, and the exit status follows the numbers.
"""

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import dory_ane_score  # noqa: E402
from scripts.dory_ane_gate_suite import build_parser as gate_parser  # noqa: E402
from scripts.dory_ane_gate_suite import summarize  # noqa: E402


def test_the_thresholds_are_the_tickets():
    assert dory_ane_score.MAX_MEAN_KLD == 0.01
    assert dory_ane_score.MIN_TOP1 == 0.98


def test_the_defaults_reach_the_command_line_and_can_be_overridden():
    args = gate_parser().parse_args(["--weights", "x"])
    assert args.max_kld == 0.01 and args.min_top1 == 0.98
    overridden = gate_parser().parse_args(["--weights", "x", "--max-kld", "0.02",
                                           "--min-top1", "0.9"])
    assert overridden.max_kld == 0.02 and overridden.min_top1 == 0.9


def test_a_good_result_passes():
    passed, failures = dory_ane_score.judge(0.004, 0.993)
    assert passed and failures == []
    assert dory_ane_score.judge_text(passed, failures) == "PASS"


def test_the_threshold_itself_is_inclusive():
    assert dory_ane_score.judge(0.01, 0.98)[0] is True
    assert dory_ane_score.judge(0.0100001, 0.98)[0] is False
    assert dory_ane_score.judge(0.005, 0.9799)[0] is False


@pytest.mark.parametrize("mean_kld, top1, expected", [
    (0.004, 0.993, True),
    (0.02, 0.993, False),
    (0.004, 0.90, False),
    (0.02, 0.90, False),
    (float("nan"), 0.993, False),
    (0.004, float("nan"), False),
    (float("inf"), 0.993, False),
])
def test_every_gate_is_judged(mean_kld, top1, expected):
    assert dory_ane_score.judge(mean_kld, top1)[0] is expected


def test_a_failure_says_which_gate_and_by_how_much():
    passed, failures = dory_ane_score.judge(0.03, 0.95)
    assert not passed
    assert len(failures) == 2
    assert "0.03000 > 0.01" in failures[0]
    assert "95.00 % < 98.00 %" in failures[1]


def _entry(**overrides):
    entry = {"gate": "decode2", "kind": "decode", "loops": 2, "rows": 128, "chunk": 0}
    entry.update(overrides)
    return entry


def test_summarize_exits_zero_when_every_gate_passes(capsys):
    results = [_entry(mean_kld=0.002, top1=0.99), _entry(gate="prefill2", chunk=1,
                                                        mean_kld=0.006, top1=0.985)]
    assert summarize(results) == 0
    output = capsys.readouterr().out
    assert "PASS" in output and "GATE SUITE: PASSED" in output
    assert all(entry["gate_passed"] for entry in results)


def test_summarize_exits_one_when_a_gate_fails(capsys):
    results = [_entry(mean_kld=0.002, top1=0.99),
               _entry(gate="decode4", mean_kld=0.02, top1=0.99)]
    assert summarize(results) == 1
    output = capsys.readouterr().out
    assert "FAIL" in output and "GATE SUITE: FAILED" in output
    assert results[1]["gate_passed"] is False
    assert results[0]["gate_passed"] is True


def test_a_gate_with_no_score_is_a_failure_not_a_silent_pass(capsys):
    missing_reference = _entry(gate="prefill4", skipped="missing reference x.kld")
    assert summarize([missing_reference]) == 1
    assert "FAIL" in capsys.readouterr().out
    # and one that produced numbers-shaped nothing at all
    assert summarize([_entry(gate="decode1")]) == 1


def test_summarize_honours_overridden_thresholds(capsys):
    results = [_entry(mean_kld=0.02, top1=0.99)]
    assert summarize(results) == 1
    assert summarize(results, max_kld=0.05) == 0
    assert not math.isnan(results[0]["mean_kld"])
