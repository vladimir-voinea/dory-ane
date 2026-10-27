"""Tests for the DANE-8 sweep harness (scripts/dory_sweep.py).

The point of these tests is the harness's one strict promise: it reports only
measured cells, and it refuses to render the table while any required cell is
missing.  So most of them build a synthetic measurement store, delete one cell,
and check that the report refuses and names exactly that cell.

Nothing here loads a model, touches the ANE or runs a conversion.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.dory_sweep as sweep  # noqa: E402
from anemll.eval.kld import write_candidate  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic measurement stores
# ---------------------------------------------------------------------------

def _kld(mean_kld, top1):
    return {"mean_kld": mean_kld, "kld_stderr": mean_kld / 10, "kld_p90": mean_kld,
            "kld_99": mean_kld, "kld_max": mean_kld, "top1": top1,
            "n_rows": 1024, "vocab_size": 131072}


def _speed(decode, prefill=300.0, rss=12 * 1024 ** 3):
    return {"decode_tok_s": decode, "prefill_tok_s": prefill, "peak_rss_bytes": rss,
            "prompt_tokens": 2048, "decoded_tokens": 24}


def _size(total):
    return {"total_bytes": total, "by_component": {"embeddings": total // 2,
                                                   "recurrent": total // 2},
            "context_length": 4096, "lut_blocks": 6, "lut_lm_head": 6,
            "model_dir": "/tmp/x", "files": 10}


def _energy(joules):
    return {"joules_per_token": joules, "mean_power_mw": 1000.0, "samples": 30,
            "tokens": 64, "seconds": 60.0}


#: Decode tok/s at the gate cell: lut6 is the fastest that also passes quality.
DECODE = {"lut4": 6.5, "lut6": 11.0, "lut8": 8.5}
#: Quality: lut4 misses the gate, lut6 and lut8 pass.
QUALITY = {"lut4": (0.09, 0.90), "lut6": (0.021, 0.972), "lut8": (0.006, 0.9915)}


def full_store(*, shipping="lut6") -> dict:
    cells = {}
    for name in sweep.CONFIG_NAMES:
        cells[sweep.size_key(name)] = _size(4 * 1024 ** 3)
        for loops in sweep.LOOPS:
            mean_kld, top1 = QUALITY[name]
            cells[sweep.kld_key(name, loops)] = _kld(mean_kld, top1)
        cells[sweep.speed_key("ane", name, sweep.GATE_CONTEXT, sweep.GATE_LOOPS)] = \
            _speed(DECODE[name])
    for context in sweep.CONTEXTS:
        for loops in sweep.LOOPS:
            cells[sweep.speed_key("ane", shipping, context, loops)] = _speed(11.0, 300.0)
            cells[sweep.speed_key("gpu", sweep.GPU_CONFIG, context, loops)] = _speed(22.0, 345.0)
    cells[sweep.energy_key("ane", shipping, sweep.GATE_CONTEXT, sweep.GATE_LOOPS)] = _energy(0.42)
    cells[sweep.energy_key("gpu", sweep.GPU_CONFIG, sweep.GATE_CONTEXT, sweep.GATE_LOOPS)] = \
        _energy(0.61)
    return {"schema": sweep.SCHEMA, "configs": {}, "cells": cells}


def missing_keys(results):
    return [need.key for need in sweep.missing_cells(results)]


# ---------------------------------------------------------------------------
# the required matrix
# ---------------------------------------------------------------------------

def test_a_complete_store_needs_nothing_and_has_the_ticket_matrix():
    results = full_store()
    assert missing_keys(results) == []
    keys = {need.key for need in sweep.required_cells(results)}
    assert len(keys) == len(sweep.required_cells(results)), "no duplicate cells"
    # ticket criterion 1: three LUT widths at three loop counts
    for name in sweep.CONFIG_NAMES:
        for loops in sweep.LOOPS:
            assert sweep.kld_key(name, loops) in keys
    # ticket criterion 2: four contexts at three loop counts, for the shipping config
    for context in sweep.CONTEXTS:
        for loops in sweep.LOOPS:
            assert sweep.speed_key("ane", "lut6", context, loops) in keys
    # ticket criterion 3: the same matrix on the GPU
    for context in sweep.CONTEXTS:
        for loops in sweep.LOOPS:
            assert sweep.speed_key("gpu", sweep.GPU_CONFIG, context, loops) in keys
    # ticket criterion 4
    assert len([key for key in keys if key.startswith("energy/")]) == 2


def test_every_required_cell_carries_the_command_that_fills_it():
    for need in sweep.required_cells(full_store()):
        assert need.fill.startswith("python scripts/dory_sweep.py"), need.key
        assert need.why, need.key


# ---------------------------------------------------------------------------
# refusing to emit a table
# ---------------------------------------------------------------------------

def test_an_empty_store_refuses_to_render_and_says_what_to_run():
    results = {"schema": sweep.SCHEMA, "configs": {}, "cells": {}}
    with pytest.raises(sweep.SweepIncomplete) as error:
        sweep.render_markdown(results)
    missing = error.value.missing
    measured = [need for need in sweep.required_cells(results) if need.key != "shipping"]
    assert len(missing) == len(measured)
    assert any("mean KLD" in need.what for need in missing)
    assert all(need.fill for need in missing)


@pytest.mark.parametrize("removed", [
    sweep.kld_key("lut8", 4),
    sweep.speed_key("ane", "lut6", 16384, 2),
    sweep.speed_key("gpu", sweep.GPU_CONFIG, 24576, 1),
    sweep.size_key("lut4"),
    sweep.energy_key("ane", "lut6", sweep.GATE_CONTEXT, sweep.GATE_LOOPS),
])
def test_one_missing_cell_refuses_the_whole_table_and_names_it(removed):
    results = full_store()
    del results["cells"][removed]
    with pytest.raises(sweep.SweepIncomplete) as error:
        sweep.render_markdown(results)
    assert [need.key for need in error.value.missing] == [removed]


def test_a_neighbouring_cell_is_never_interpolated():
    # every context but 8192 measured: the report must still refuse
    results = full_store()
    for loops in sweep.LOOPS:
        del results["cells"][sweep.speed_key("ane", "lut6", 8192, loops)]
    with pytest.raises(sweep.SweepIncomplete) as error:
        sweep.render_markdown(results)
    assert sorted(need.key for need in error.value.missing) == sorted(
        sweep.speed_key("ane", "lut6", 8192, loops) for loops in sweep.LOOPS)


def test_energy_can_be_left_pending_with_explicit_permission_only():
    results = full_store()
    for key in list(results["cells"]):
        if key.startswith("energy/"):
            del results["cells"][key]
    with pytest.raises(sweep.SweepIncomplete):
        sweep.render_markdown(results)
    table = sweep.render_markdown(results, energy_pending=True)
    assert "Energy is not measured yet" in table


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def test_the_shipping_configuration_is_the_fastest_that_passes_every_gate():
    results = full_store()
    # lut7 does not exist; lut8 is faster than lut6 but lut4 is fastest and fails
    assert sweep.choose_shipping(results) == "lut6"
    verdicts = {name: sweep.config_verdict(results, name) for name in sweep.CONFIG_NAMES}
    assert not verdicts["lut4"]["quality_ok"]      # KLD 0.09 > 0.05
    assert verdicts["lut6"]["quality_ok"] and verdicts["lut6"]["speed_ok"]
    assert verdicts["lut8"]["quality_ok"] and verdicts["lut8"]["speed_ok"]


def test_a_configuration_under_the_reassess_floor_is_never_chosen():
    results = full_store()
    results["cells"][sweep.speed_key("ane", "lut6", sweep.GATE_CONTEXT, sweep.GATE_LOOPS)] = \
        _speed(5.0)
    results["cells"][sweep.speed_key("ane", "lut8", sweep.GATE_CONTEXT, sweep.GATE_LOOPS)] = \
        _speed(7.0)
    assert sweep.choose_shipping(results) is None
    assert "REASSESS" in json.dumps([sweep.config_verdict(results, "lut6")]).upper() or \
        "reassess" in json.dumps(sweep.config_verdict(results, "lut6"))


def test_the_chosen_configuration_is_visible_in_the_rendered_table():
    table = sweep.render_markdown(full_store())
    assert "Shipping configuration: lut6" in table
    assert "11.0 tok/s at 24K / 2 loops" in table
    assert "0.02100" in table and "97.20 %" in table
    assert "| lut4 | 1 |" in table                 # the failing configuration is shown too
    assert "NOT MEASURED" not in table


def test_no_configuration_passing_means_no_shipping_claim():
    results = full_store()
    for name in sweep.CONFIG_NAMES:
        mean_kld, top1 = QUALITY["lut4"]
        for loops in sweep.LOOPS:
            results["cells"][sweep.kld_key(name, loops)] = _kld(mean_kld, top1)
    table = sweep.render_markdown(results, energy_pending=False)
    assert "NO CONFIGURATION MEETS EVERY GATE" in table
    assert "Shipping configuration" not in table


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def _run_cli(tmp_path, args):
    store = tmp_path / "sweep.json"
    return sweep.main(["--results", str(store)] + args)


def test_cli_check_refuses_on_an_empty_store(tmp_path, capsys):
    assert _run_cli(tmp_path, ["check"]) == 2
    output = capsys.readouterr().out
    assert "REFUSING TO EMIT A TABLE" in output
    assert "interpolated" in output


def test_cli_report_writes_only_when_complete(tmp_path):
    store = tmp_path / "sweep.json"
    sweep.save_results(full_store(), store)
    page = tmp_path / "page.md"
    page.write_text("# page" + chr(10) + sweep.BEGIN + chr(10) + sweep.END + chr(10))
    assert sweep.main(["--results", str(store), "report", "--out", str(page),
                       "--energy-pending"]) == 0
    text = page.read_text()
    assert "Shipping configuration: lut6" in text
    assert sweep.BEGIN in text and sweep.END in text


def test_cli_returns_three_when_nothing_passes(tmp_path, capsys):
    results = full_store()
    for name in sweep.CONFIG_NAMES:
        for loops in sweep.LOOPS:
            results["cells"][sweep.kld_key(name, loops)] = _kld(0.2, 0.8)
    store = tmp_path / "sweep.json"
    sweep.save_results(results, store)
    assert sweep.main(["--results", str(store), "check"]) == 3


# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------

def _fake_model_dir(tmp_path, *, lut_blocks=6, lut_lm_head=6, skip=None):
    import yaml

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    files = {}
    for name in ("embeddings", "recurrent", "lm_head"):
        files[name] = {}
        for function in ("infer", "prefill"):
            file_name = f"dory_{name}_{function}.mlpackage"
            files[name][function] = file_name
            if skip == file_name:
                continue
            package = model_dir / file_name
            package.mkdir()
            (package / "weights.bin").write_bytes(b"x" * 1024)
    meta = {
        "dory_meta": 1,
        "parameters": {"context_length": 4096, "batch_size": 64, "vocab_size": 131072,
                       "lut_blocks": lut_blocks, "lut_lm_head": lut_lm_head},
        "components": {name: {"files": value, "functions": list(value)}
                       for name, value in files.items()},
        "run_order": list(files),
    }
    (model_dir / "meta.yaml").write_text(yaml.safe_dump(meta))
    return model_dir


def test_sizes_are_measured_from_the_packages_on_disk(tmp_path):
    model_dir = _fake_model_dir(tmp_path)
    value = sweep.measure_sizes(model_dir, "lut6")
    assert value["total_bytes"] == 6 * 1024
    assert value["by_component"] == {"embeddings": 2048, "recurrent": 2048, "lm_head": 2048}
    assert value["lut_blocks"] == 6 and value["files"] == 6


def test_a_directory_whose_meta_disagrees_with_the_asked_configuration_is_refused(tmp_path):
    model_dir = _fake_model_dir(tmp_path, lut_blocks=4)
    with pytest.raises(SystemExit) as error:
        sweep.measure_sizes(model_dir, "lut6")
    assert "lut_blocks=4" in str(error.value)


def test_a_missing_package_is_refused_not_counted_as_zero(tmp_path):
    model_dir = _fake_model_dir(tmp_path, skip="dory_recurrent_infer.mlpackage")
    with pytest.raises(SystemExit) as error:
        sweep.measure_sizes(model_dir, "lut6")
    assert "missing packages" in str(error.value)


# ---------------------------------------------------------------------------
# quality: the scoring path is DANE-2's
# ---------------------------------------------------------------------------

def _candidate(tmp_path, name, logprobs, tokens, positions):
    return write_candidate(tmp_path / name, logprobs, positions, tokens,
                           meta={"trace": "test"})


def test_score_candidate_uses_the_documented_scorer(tmp_path):
    rows, vocab = 3, 8
    logprobs = np.log(np.full((rows, vocab), 1.0 / vocab))
    tokens = np.arange(16, dtype=np.int32)
    positions = np.arange(rows, dtype=np.int32)
    reference = _candidate(tmp_path, "ref.npz", logprobs, tokens, positions)
    identical = _candidate(tmp_path, "same.npz", logprobs, tokens, positions)
    value = sweep.score_candidate(reference, identical)
    assert value["mean_kld"] == pytest.approx(0.0, abs=1e-9)
    assert value["top1"] == pytest.approx(1.0)
    assert value["n_rows"] == rows and value["vocab_size"] == vocab

    other = logprobs.copy()
    other[0, 0] = np.log(0.9)
    other[0, 1:] = np.log(0.1 / (vocab - 1))
    different = _candidate(tmp_path, "other.npz", other, tokens, positions)
    shifted = sweep.score_candidate(reference, different)
    assert shifted["mean_kld"] > 0.0


def test_the_candidate_command_has_the_arguments_dory_chat_expects(tmp_path):
    command = sweep.dory_chat_score_command(
        "/models/lut6", tmp_path / "cand.npz", loops=4, chunk_size=1024, max_windows=2)
    assert command[1:4] == ["-m", "anemll.dory_chat", "score"]
    assert "--model-dir" in command and "/models/lut6" in command
    assert command[command.index("--loops") + 1] == "4"
    assert command[command.index("--max-windows") + 1] == "2"


# ---------------------------------------------------------------------------
# speed: the measurement loop, driven by a fake runtime
# ---------------------------------------------------------------------------

class FakeRuntime:
    def __init__(self, vocab=8):
        self.vocab = vocab
        self.loops = 2
        self.resets = 0
        self.prefills = []
        self.decodes = 0

    def set_loops(self, loops):
        self.loops = int(loops)
        return False

    def reset_state(self):
        self.resets += 1

    def prefill(self, ids, start, progress=None):
        self.prefills.append((len(ids), int(start)))
        return np.zeros((len(ids), self.vocab), dtype=np.float32)

    def decode(self, token, position):
        self.decodes += 1
        return np.zeros((self.vocab,), dtype=np.float32)


def test_measure_speed_fills_every_cell_it_is_asked_for():
    runtime = FakeRuntime()
    cells = sweep.measure_speed("/models/lut6", "lut6", [2048, 4096], [1, 2],
                                tokens=4, runtime_factory=lambda _: runtime)
    assert sorted(cells) == sorted(
        [sweep.speed_key("ane", "lut6", 2048, 1), sweep.speed_key("ane", "lut6", 2048, 2),
         sweep.speed_key("ane", "lut6", 4096, 1), sweep.speed_key("ane", "lut6", 4096, 2)])
    for value in cells.values():
        # the prompt leaves room for the decode steps: check_context reserves one
        # position, and a context-sized prompt overflowed before the first cell
        assert value["prompt_tokens"] in (2048 - 4, 4096 - 4)
        assert value["decode_headroom"] == 4
        assert value["prefill_tok_s"] > 0
        assert value["decode_tok_s"] > 0
        assert value["peak_rss_bytes"] > 0
    assert runtime.resets == 4            # one reset per cell
    assert runtime.prefills == [(2044, 0), (4092, 0), (2044, 0), (4092, 0)]


def test_the_workload_is_the_same_prompt_at_every_context():
    short = sweep.workload_ids(2048)
    long = sweep.workload_ids(4096)
    assert short.shape == (2048,) and long.shape == (4096,)
    assert np.array_equal(short, long[:2048])


# ---------------------------------------------------------------------------
# energy: the owner's powermetrics output
# ---------------------------------------------------------------------------

SAMPLE_LOG = chr(10).join([
    "*** Sampled system activity (Sat Oct 24 12:00:00 2026 +0300) (500.12ms elapsed) ***",
    "",
    "HW active residency:",
    "ANE Power: 1200 mW",
    "GPU Power: 450 mW",
    "CPU Power: 300 mW",
    "",
    "ANE Power: 1400 mW",
    "GPU Power: 500 mW",
    "CPU Power: 320 mW",
])


def test_powermetrics_samples_are_parsed_per_build():
    ane = sweep.parse_powermetrics(SAMPLE_LOG, "ane")
    assert ane["mean_power_mw"] == pytest.approx(1300.0)
    assert ane["samples"] == 2
    gpu = sweep.parse_powermetrics(SAMPLE_LOG, "gpu")
    assert gpu["mean_power_mw"] == pytest.approx(475.0)


def test_energy_per_token_is_power_times_time_over_tokens(tmp_path):
    log = tmp_path / "ane.txt"
    log.write_text(SAMPLE_LOG)
    value = sweep.energy_from_log(log, "ane", tokens=64, seconds=60.0)
    # 1.3 W * 60 s / 64 tokens
    assert value["joules_per_token"] == pytest.approx(1.3 * 60.0 / 64)
    assert value["tokens"] == 64


def test_energy_without_a_wall_time_is_refused(tmp_path):
    log = tmp_path / "ane.txt"
    log.write_text(SAMPLE_LOG)
    with pytest.raises(SystemExit):
        sweep.energy_from_log(log, "ane", tokens=64, seconds=0)


def test_a_log_without_the_sampler_is_refused(tmp_path):
    log = tmp_path / "empty.txt"
    log.write_text("nothing to see here")
    with pytest.raises(SystemExit):
        sweep.parse_powermetrics(log.read_text(), "ane")


# ---------------------------------------------------------------------------
# the GPU baseline must measure the same workload as the Neural Engine
# ---------------------------------------------------------------------------

def test_the_gpu_side_is_fed_the_same_token_ids_as_the_neural_engine():
    for context in sweep.CONTEXTS:
        gpu = sweep.gpu_prompt_tokens(context)
        ane = sweep.workload_ids(context)
        assert len(gpu) == context
        assert len(gpu) == len(ane)
        assert gpu == [int(token) for token in ane.tolist()]


def test_the_gpu_request_carries_token_ids_not_text(monkeypatch):
    import urllib.request

    captured = {}

    class Reply:
        def read(self):
            return json.dumps({"timings": {"prompt_n": 3, "predicted_n": 2}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data)
        captured["url"] = request.full_url
        return Reply()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = sweep.gpu_completion(49562, [11, 22, 33], n_predict=2)
    assert captured["url"].endswith("/completion")
    assert captured["body"]["prompt"] == [11, 22, 33]
    assert captured["body"]["n_predict"] == 2
    assert reply["timings"]["prompt_n"] == 3


def test_a_row_the_server_counted_differently_is_refused():
    sweep.check_prompt_count({"prompt_n": 2048}, 2048, 2048)      # fine
    with pytest.raises(SystemExit) as error:
        sweep.check_prompt_count({"prompt_n": 1900}, 2048, 2048)
    assert "1900" in str(error.value) and "2048" in str(error.value)
    with pytest.raises(SystemExit):
        sweep.check_prompt_count({}, 2048, 2048)                  # no count at all
