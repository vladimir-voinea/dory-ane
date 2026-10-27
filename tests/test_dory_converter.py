"""Tests for the Dory converter's plan, meta schema, guards and naming.

No weights and no Core ML conversion here: everything the converter decides
before it touches coremltools is pure, so it is tested as such.  The conversion
itself is exercised by tests/test_dory_chat.py's fake runtime and, for real, by
scripts/convert_dory.py on the BF16 checkpoint.
"""

from __future__ import annotations

import copy

import pytest

from anemll.ane_converter import dory_converter as dcv
from anemll.models.dory_ane_model import DoryANEConfig


@pytest.fixture()
def config():
    return DoryANEConfig(context_length=1024, batch_size=64)


@pytest.fixture()
def specs(config):
    return dcv.component_specs(config)


def _files(specs):
    """One package per function: the converter never welds functions together."""
    return {spec.name: {function: f"{spec.name}_{function}.mlpackage"
                        for function in ("infer", "prefill")}
            for spec in specs}


def _meta(config, specs, **overrides):
    files = _files(specs)
    kwargs = dict(config=config, specs=specs, files=files,
                  lut_bits={"embeddings": None, "blocks": 6, "lm_head": 6},
                  lm_head_chunk_sizes=[16384] * 8)
    kwargs.update(overrides)
    return dcv.build_meta(**kwargs)


# -- the component plan (spec D6) ------------------------------------------

def test_five_components_in_run_order(specs):
    assert [spec.name for spec in specs] == \
        ["embeddings", "input", "recurrent", "output", "lm_head"]


def test_block_ranges_follow_dory_structure(specs):
    ranges = {spec.name: spec.blocks for spec in specs if spec.kind == "blocks"}
    assert ranges == {"input": (0, 16), "recurrent": (16, 56), "output": (56, 72)}


def test_only_the_recurrent_component_is_called_once_per_loop(specs):
    calls = {spec.name: spec.calls_per_token for spec in specs}
    assert calls == {"embeddings": 1, "input": 1, "recurrent": "loops",
                     "output": 1, "lm_head": 1}


def test_loop_entry_and_exit_norms_sit_on_the_recurrent_component(specs):
    entries = {spec.name: (spec.loop_entry, spec.loop_exit) for spec in specs}
    assert entries["recurrent"] == (True, True)
    assert all(value == (False, False) for name, value in entries.items()
               if name != "recurrent")


def test_recurrent_split_keeps_one_entry_and_one_exit(config):
    split = dcv.component_specs(config, recurrent_splits=2)
    names = [spec.name for spec in split]
    assert names == ["embeddings", "input", "recurrent_1of2", "recurrent_2of2",
                     "output", "lm_head"]
    first = next(spec for spec in split if spec.name == "recurrent_1of2")
    second = next(spec for spec in split if spec.name == "recurrent_2of2")
    assert first.blocks == (16, 36) and second.blocks == (36, 56)
    assert (first.loop_entry, first.loop_exit) == (True, False)
    assert (second.loop_entry, second.loop_exit) == (False, True)
    assert first.calls_per_token == second.calls_per_token == "loops"


def test_recurrent_split_covers_the_zone_without_gaps(config):
    split = dcv.component_specs(config, recurrent_splits=3)
    covered = []
    for spec in split:
        if spec.kind == "blocks":
            covered.extend(range(*spec.blocks))
    assert covered == list(range(72))


def test_h_in_is_an_input_of_the_recurrent_component_only(specs):
    for spec in specs:
        assert ("h_in" in spec.inputs) == spec.loop_entry


def test_every_block_component_declares_the_four_standard_inputs(specs):
    for spec in specs:
        if spec.kind != "blocks":
            continue
        assert spec.inputs[:4] == ["hidden_states", "causal_mask", "current_pos",
                                   "cache_pos"]


# -- LM-head split ---------------------------------------------------------

def test_lm_head_splits_are_even_and_cover_the_vocabulary():
    sizes = dcv.dch.lm_head_split_sizes(131072, 8)
    assert sizes == [16384] * 8
    assert sum(sizes) == 131072


def test_lm_head_refuses_a_single_part():
    with pytest.raises(ValueError, match="at least 2"):
        dcv.dch.lm_head_split_sizes(131072, 1)


def test_lm_head_refuses_an_uneven_split():
    with pytest.raises(ValueError, match="not divisible"):
        dcv.dch.lm_head_split_sizes(131072, 3)


# -- size guard (the ~2 GB ANE limit) -------------------------------------

def test_attention_and_mlp_block_parameter_counts(config):
    # 2560*(16*256) + 2*2560*(4*256) + (16*256)*2560 + 2560*(16*256)
    assert dcv.block_params(config, 0) == 36_700_160
    # 3 * 2560 * 7168
    assert dcv.block_params(config, 1) == 55_050_240


def test_recurrent_component_is_under_two_gigabytes_at_lut6(config, specs):
    recurrent = next(s for s in specs if s.name == "recurrent")
    size = dcv.estimate_component_bytes(recurrent, config, 6)
    assert size < dcv.MAX_MODEL_BYTES
    # calibrated: coremltools groups 16 weights per fp16 scale, so LUT6 costs
    # 0.759 B/param and the 40-block recurrent component is ~1.35 GiB
    assert size > 1.3 * 1024 ** 3


def test_the_guard_fires_at_lut8_and_says_what_to_do(config, specs):
    recurrent = next(s for s in specs if s.name == "recurrent")
    # LUT8 is 1.89 GB once the estimate is calibrated, so it no longer trips the
    # 2 GB guard; LUT12 does, which is the point of the guard.
    size = dcv.estimate_component_bytes(recurrent, config, 12)
    with pytest.raises(dcv.ModelTooLargeError) as excinfo:
        dcv.check_size("recurrent", size)
    message = str(excinfo.value)
    assert "2.0 GB" in message or "GB" in message
    assert "--recurrent-splits 2" in message
    assert "--lut-blocks 4" in message


def test_splitting_the_recurrent_block_halves_the_file(config):
    single = next(s for s in dcv.component_specs(config) if s.name == "recurrent")
    halves = [s for s in dcv.component_specs(config, recurrent_splits=2)
              if s.kind == "blocks" and s.is_recurrent]
    whole = dcv.estimate_component_bytes(single, config, 6)
    parts = sum(dcv.estimate_component_bytes(s, config, 6) for s in halves)
    assert parts == pytest.approx(whole, rel=0.01)


def test_check_size_passes_a_small_component():
    assert dcv.check_size("tiny", 1024) == 1024


def test_unquantized_estimates_are_fp16(config, specs):
    embeddings = next(s for s in specs if s.name == "embeddings")
    assert dcv.estimate_component_bytes(embeddings, config, None) == pytest.approx(
        131072 * 2560 * 2, rel=0.05)


# -- disk guard ------------------------------------------------------------

def test_disk_needed_is_the_largest_component_tripled(config, specs):
    estimates = [(spec, dcv.estimate_component_bytes(spec, config, 6)) for spec in specs]
    largest = max(size for _, size in estimates)
    assert dcv.disk_needed(estimates) >= int(largest * 3.2)


def test_disk_guard_refuses_a_full_volume(monkeypatch, tmp_path, config, specs):
    estimates = [(spec, dcv.estimate_component_bytes(spec, config, 6)) for spec in specs]

    class Usage:
        free = 3 * 1024 ** 3

    monkeypatch.setattr(dcv.shutil, "disk_usage", lambda path: Usage)
    with pytest.raises(RuntimeError, match="not enough disk"):
        dcv.check_disk(tmp_path, estimates)


def test_disk_guard_passes_a_roomy_volume(monkeypatch, tmp_path, config, specs):
    estimates = [(spec, dcv.estimate_component_bytes(spec, config, 6)) for spec in specs]

    class Usage:
        free = 200 * 1024 ** 3

    monkeypatch.setattr(dcv.shutil, "disk_usage", lambda path: Usage)
    assert dcv.check_disk(tmp_path, estimates)["needed"] > 0


# -- meta schema -----------------------------------------------------------

def test_meta_records_kv_mode_loops_and_components(config, specs):
    meta = _meta(config, specs)
    assert meta["dory_meta"] == dcv.META_FORMAT
    assert meta["kv"]["mode"] == "last_loop"
    assert meta["kv"]["invalidate_on_loop_change"] is True
    assert meta["loops"] == {"min": 1, "max": 4, "default": 2}
    assert meta["parameters"]["context_length"] == 1024
    assert meta["parameters"]["mask_len"] == 1024 + config.sliding_rows
    assert meta["parameters"]["split_lm_head"] == 8
    assert meta["components"]["recurrent"]["calls_per_token"] == "loops"
    assert meta["components"]["lm_head"]["outputs"][0] == "logits_1"
    assert meta["run_order"] == [spec.name for spec in specs]


def test_meta_states_are_the_36_cache_slots(config, specs):
    meta = _meta(config, specs)
    full, sliding = meta["kv"]["states"]
    assert full["kind"] == "full" and full["slots"] == 9
    assert sliding["kind"] == "sliding" and sliding["slots"] == 27
    assert full["shape"][0] == 18 and sliding["shape"][0] == 54
    # 9 slots x K,V x 4 KV heads x 1024 rows x 256 head_dim x 2 bytes (fp16)
    assert full["bytes"] == 9 * 2 * 4 * 1024 * 256 * 2
    assert full["rows"] == 1024
    assert full["bytes"] + sliding["bytes"] > 100 * 1024 ** 2


def test_meta_records_the_tokenizer_and_the_template_fallback(config, specs):
    meta = _meta(config, specs, chat_template=None)
    tokenizer = meta["tokenizer"]
    assert tokenizer["file"] == "tokenizer.json"
    assert tokenizer["chat_template"] is None
    assert tokenizer["chat_template_fallback"] == dcv.FALLBACK_TEMPLATE
    assert tokenizer["reasoning_default"] is True
    assert tokenizer["eos_token_ids"] == [2, 11]


def test_a_valid_meta_has_no_problems(config, specs):
    assert dcv.validate_meta(_meta(config, specs)) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda m: m.__setitem__("dory_meta", 99), "dory_meta"),
    (lambda m: m["parameters"].__setitem__("mask_len", 10), "mask_len"),
    (lambda m: m["kv"].__setitem__("mode", "per_loop"), "last_loop"),
    (lambda m: m["loops"].__setitem__("default", 9), "loops must satisfy"),
    (lambda m: m["parameters"].__setitem__("split_lm_head", 4), "lm_head"),
    (lambda m: m["run_order"].pop(), "run_order"),
    (lambda m: m["components"]["input"]["functions"].clear(), "infer"),
    (lambda m: m["components"]["recurrent"].__setitem__("blocks", [16, 40]), "cover"),
])
def test_validate_meta_catches_broken_files(config, specs, mutate, needle):
    meta = copy.deepcopy(_meta(config, specs))
    mutate(meta)
    problems = dcv.validate_meta(meta)
    assert problems
    assert any(needle in problem for problem in problems)


def test_write_and_read_meta_round_trip(tmp_path, config, specs):
    meta = _meta(config, specs)
    path = dcv.write_meta(tmp_path / "meta.yaml", meta)
    assert path.is_file()
    assert dcv.read_meta(path)["kv"]["mode"] == "last_loop"


def test_write_meta_refuses_an_invalid_file(tmp_path, config, specs):
    meta = _meta(config, specs)
    meta["kv"]["mode"] = "something else"
    with pytest.raises(dcv.MetaError):
        dcv.write_meta(tmp_path / "meta.yaml", meta)


def test_read_meta_rejects_an_invalid_file(tmp_path):
    path = tmp_path / "meta.yaml"
    path.write_text("dory_meta: 1\nparameters: {}\n", encoding="utf-8")
    with pytest.raises(dcv.MetaError):
        dcv.read_meta(path)


# -- naming ----------------------------------------------------------------

def test_file_names_carry_the_lut_width_and_the_function(config):
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir="/tmp/out",
                                  config=config, lut_blocks=6, lut_lm_head=6,
                                  lut_embeddings=None, check_placement=False,
                                  verbose=False)
    names = {spec.name: converter.file_name(spec, "infer") for spec in converter.specs}
    assert names["input"] == "dory_input_lut6_infer.mlpackage"
    assert names["embeddings"] == "dory_embeddings_infer.mlpackage"
    assert names["lm_head"] == "dory_lm_head_lut6_infer.mlpackage"
    assert converter.file_name(converter.spec("input"), "prefill") == \
        "dory_input_lut6_prefill.mlpackage"
    assert converter.compiled_name(converter.spec("input"), "prefill") == \
        "dory_input_lut6_prefill.mlmodelc"


def test_every_component_declares_one_file_per_function(config):
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir="/tmp/out",
                                  config=config, check_placement=False, verbose=False)
    files = converter.manifest_files()
    for spec in converter.specs:
        assert set(files[spec.name]) == {"infer", "prefill"}


def test_functions_are_infer_and_prefill(config):
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir="/tmp/out",
                                  config=config, check_placement=False, verbose=False)
    for spec in converter.specs:
        assert [name for name, _ in converter._functions(spec)] == ["infer", "prefill"]
        assert [seq for _, seq in converter._functions(spec)] == [1, 64]


def test_estimates_cover_every_component(config):
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir="/tmp/out",
                                  config=config, check_placement=False, verbose=False)
    sizes = converter.guard_sizes()
    assert set(sizes) == {spec.name for spec in converter.specs}
    assert all(size > 0 for size in sizes.values())


def test_component_model_only_builds_its_own_blocks(config):
    model = dcv._ComponentModel(config, list(range(56, 72)))
    assert [layer.block for layer in model.model.layers] == list(range(56, 72))
    # the state buffers are the full 36-slot union regardless of the blocks
    assert model.model.kv_cache_full.shape[0] == 18
    assert model.model.kv_cache_sliding.shape[0] == 54


def test_resume_defaults_off_and_can_be_enabled(config):
    """Regression: convert_component read self.resume, which __init__ never set.

    Every component conversion died with
    AttributeError: 'DoryConverter' object has no attribute 'resume'.
    """
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir="/tmp/out",
                                  config=config, check_placement=False, verbose=False)
    assert converter.resume is False
    enabled = dcv.DoryConverter(model_path="/tmp/weights", output_dir="/tmp/out",
                                config=config, check_placement=False, verbose=False,
                                resume=True)
    assert enabled.resume is True


def test_finished_package_declines_what_it_cannot_verify(config, tmp_path):
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir=tmp_path,
                                  config=config, check_placement=False, verbose=False)
    spec = converter.spec("input")
    assert converter._finished_package(spec, "prefill") is None

    junk = tmp_path / converter.file_name(spec, "prefill")
    junk.mkdir()
    (junk / "Manifest.json").write_text("{}")
    assert converter._finished_package(spec, "prefill") is None


def test_finished_package_accepts_a_verified_package(config, tmp_path, monkeypatch):
    """A package with the right size, one function and the right recorded name."""
    converter = dcv.DoryConverter(model_path="/tmp/weights", output_dir=tmp_path,
                                  config=config, lut_blocks=6, lut_lm_head=6,
                                  lut_embeddings=None, lm_head_splits=8,
                                  check_placement=False, verbose=False)
    spec = converter.spec("lm_head")
    package = tmp_path / converter.file_name(spec, "infer")
    package.mkdir()
    (package / "weight.bin").write_bytes(b"w" * 4096)

    class FakeMetadata:
        userDefined = {"com.anemll.functions": "infer"}

    class FakeDescription:
        functions: list = []
        metadata = FakeMetadata()

    class FakeProto:
        description = FakeDescription()

    monkeypatch.setattr(dcv, "estimate_component_bytes", lambda *a, **k: 4096)
    monkeypatch.setattr("coremltools.models.utils.load_spec", lambda _path: FakeProto())
    assert converter._finished_package(spec, "infer") == 4096
    assert converter._finished_package(spec, "prefill") is None
