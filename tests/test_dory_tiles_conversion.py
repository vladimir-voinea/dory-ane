"""The tiled 24K path has to be convertible, not just constructible.

The review found the tiled layout unconvertible: states() iterated a hard-coded
pair of buffers, _state_descriptions and component_specs never mentioned tiles,
and the meta recorded no tile key although the runtime reads one.  These tests
pin the three surfaces to one another without converting anything: one state
description per tile, the meta keys the runtime reads, and input names that are
exactly the wrapper signature the converter traces.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from anemll.ane_converter import dory_converter as dcv
from anemll.models import dory_ane_model as dam
from anemll.models import dory_chunks as dch

TILES = (16384, 8192)


def tiled_config():
    return dam.DoryANEConfig(context_length=24576, batch_size=64,
                             full_cache_tiles=TILES, rope_tables_as_inputs=True)


def plain_config():
    return dam.DoryANEConfig(context_length=1024, batch_size=64)


def meta_and_specs(config):
    specs = dcv.component_specs(config)
    files = {spec.name: {function: f"{spec.name}_{function}.mlpackage"
                         for function in ("infer", "prefill")}
             for spec in specs}
    meta = dcv.build_meta(config=config, specs=specs, files=files,
                          lut_bits={"embeddings": None, "blocks": 6, "lm_head": 6},
                          lm_head_chunk_sizes=[config.vocab_size // 2] * 2,
                          chat_template=None)
    return meta, specs


def converter_for(config):
    """A DoryConverter with only the state it needs: no model, no weights."""
    converter = object.__new__(dcv.DoryConverter)
    converter.config = config
    return converter


def test_the_tiled_plan_declares_one_state_per_tile():
    config = tiled_config()
    descriptions = dcv._state_descriptions(config)
    assert [entry["buffer"] for entry in descriptions] == list(dch.state_attrs(config))
    assert [entry["name"] for entry in descriptions] == list(dch.state_names(config))
    assert [entry["rows"] for entry in descriptions] == [16384, 8192, 1280]
    assert [entry.get("tile") for entry in descriptions] == [1, 2, None]
    assert sum(entry["rows"] for entry in descriptions
               if entry["kind"] == "full") == config.context_length


def test_states_follow_the_config_and_not_a_hard_coded_pair():
    config = tiled_config()
    buffers = {attr: np.zeros((2, 4, rows, 8), dtype=np.float16)
               for attr, rows in zip(dch.state_attrs(config), (16384, 8192, 1280))}
    wrapper = types.SimpleNamespace(model=types.SimpleNamespace(**buffers))
    states = converter_for(config).states(wrapper)
    assert [state.name for state in states] == [f"model.{attr}"
                                                for attr in dch.state_attrs(config)]


def test_the_tiled_meta_records_what_the_runtime_reads():
    config = tiled_config()
    meta, _ = meta_and_specs(config)
    params = meta["parameters"]
    assert params["full_cache_tiles"] == list(TILES)
    assert params["full_cache_tiled"] is True
    assert params["rope_tables_as_inputs"] is True
    kv = meta["kv"]
    assert kv["tiled"] is True
    assert "write_index" in kv["write_frame"]
    assert [entry["buffer"] for entry in kv["states"]] == list(dch.state_attrs(config))
    assert dcv.validate_meta(meta) == []


def test_the_tiled_block_inputs_are_the_traced_signature():
    config = tiled_config()
    meta, specs = meta_and_specs(config)
    converter = converter_for(config)
    for spec in specs:
        if spec.kind != "blocks":
            continue
        wrapper = dch.traced_block_chunk_class(spec.tiled, spec.rope_tables)
        signature = [name for name in inspect.signature(wrapper.forward).parameters
                     if name != "self"]
        assert spec.inputs == dch.block_input_names(spec.tiled, spec.loop_entry,
                                                    spec.rope_tables)
        assert signature[:len(spec.inputs)] == spec.inputs
        assert meta["components"][spec.name]["inputs"] == spec.inputs
        declared = converter._coreml_inputs(spec, 64)
        assert [tensor.name for tensor in declared] == spec.inputs
        example = converter._example_inputs(spec, 64)
        assert [tuple(tensor.shape) for tensor in example] == \
            [dcv.input_spec(name, 64, config)[0] for name in spec.inputs]


def test_the_untiled_plan_is_unchanged():
    config = plain_config()
    meta, specs = meta_and_specs(config)
    assert [entry["buffer"] for entry in dcv._state_descriptions(config)] == \
        ["kv_cache_full", "kv_cache_sliding"]
    params = meta["parameters"]
    assert params["full_cache_tiled"] is False
    assert params["full_cache_tiles"] == [1024]
    assert params["rope_tables_as_inputs"] is False
    assert meta["kv"]["tiled"] is False
    assert dcv.validate_meta(meta) == []
    recurrent = [spec for spec in specs if spec.kind == "blocks" and spec.loop_entry][0]
    assert recurrent.inputs == ["hidden_states", "causal_mask", "current_pos",
                                "cache_pos", "h_in"]


def test_a_meta_with_one_state_too_few_is_refused():
    config = tiled_config()
    meta, _ = meta_and_specs(config)
    meta["kv"]["states"] = meta["kv"]["states"][:2]
    problems = dcv.validate_meta(meta)
    assert any("kv.states" in problem for problem in problems)


def test_the_tile_layout_is_parsed_and_validated():
    assert dcv.parse_tile_rows(None) is None
    assert dcv.parse_tile_rows("16384,8192") == (16384, 8192)
    assert dcv.parse_tile_rows(" 16384 , 8192 ") == (16384, 8192)
    with pytest.raises(SystemExit):
        dcv.parse_tile_rows("abc")



# -- the exported RoPE rows ------------------------------------------------

def test_the_converter_exports_the_rows_the_graph_used(tmp_path):
    config = tiled_config()
    path = dcv.write_rope_tables(config, tmp_path)
    assert path.name == dcv.ROPE_TABLE_FILE
    with np.load(path) as data:
        assert sorted(data.files) == sorted(dcv.ROPE_TABLE_KEYS)
        for key in dcv.ROPE_TABLE_KEYS:
            assert data[key].shape == (config.context_length + 1, config.head_dim)
            assert data[key].dtype == np.float16
    # one source of truth: the exported rows are the model's own table rows
    import torch

    full = dam.DoryANERotaryEmbedding(config.head_dim, config.rope_theta_full,
                                      config.context_length + 1, torch.float16)
    sliding = dam.DoryANERotaryEmbedding(config.head_dim, config.rope_theta_sliding,
                                         config.context_length + 1, torch.float16)
    with np.load(path) as data:
        assert np.array_equal(data["cos_full"], full.cos_cached.detach().cpu().numpy())
        assert np.array_equal(data["sin_sliding"], sliding.sin_cached.detach().cpu().numpy())


def test_the_meta_declares_the_sidecar():
    config = tiled_config()
    meta, _ = meta_and_specs(config)
    declared = meta["parameters"]["rope_tables"]
    assert declared["file"] == dcv.ROPE_TABLE_FILE
    assert declared["rows"] == config.context_length + 1
    assert declared["head_dim"] == config.head_dim
    assert declared["keys"] == list(dcv.ROPE_TABLE_KEYS)
    assert dcv.validate_meta(meta) == []


def test_a_meta_claiming_rope_inputs_without_a_usable_sidecar_is_refused():
    config = tiled_config()
    meta, _ = meta_and_specs(config)
    meta["parameters"]["rope_tables"] = None
    assert any("rope_tables" in problem for problem in dcv.validate_meta(meta))
    meta["parameters"]["rope_tables"] = {
        "file": dcv.ROPE_TABLE_FILE, "rows": config.context_length + 1,
        "head_dim": int(config.head_dim) + 1, "keys": list(dcv.ROPE_TABLE_KEYS)}
    assert any("head_dim" in problem for problem in dcv.validate_meta(meta))


def _runtime_with_sidecar(tmp_path, config):
    from anemll import dory_chat as chat

    dcv.write_rope_tables(config, tmp_path)
    runtime = chat.DoryRuntime.__new__(chat.DoryRuntime)
    runtime.model_dir = tmp_path
    runtime.params = {"rope_tables_as_inputs": True,
                      "rope_tables": dcv.rope_table_descriptor(config),
                      "context_length": int(config.context_length)}
    runtime.meta = {"parameters": runtime.params}
    runtime.rope_tables = runtime._load_rope_tables()
    return runtime


def test_the_runtime_slices_the_rows_the_converter_exported(tmp_path):
    import torch

    from anemll import dory_chat as chat

    config = tiled_config()
    runtime = _runtime_with_sidecar(tmp_path, config)
    out = runtime._rope_inputs(list(dcv.ROPE_TABLE_KEYS), 16384, 64)
    assert set(out) == set(dcv.ROPE_TABLE_KEYS)
    full = dam.DoryANERotaryEmbedding(config.head_dim, config.rope_theta_full,
                                      config.context_length + 1, torch.float16)
    cos, sin = full.table_rows(16384, 64)
    assert out["cos_full"].shape == (1, 1, 64, config.head_dim)
    assert np.array_equal(out["cos_full"], cos.detach().cpu().numpy())
    assert np.array_equal(out["sin_full"], sin.detach().cpu().numpy())
    with pytest.raises(chat.ContextOverflowError, match="RoPE rows"):
        runtime._rope_inputs(["cos_full"], int(config.context_length) - 8, 64)


def test_a_package_without_the_sidecar_is_refused(tmp_path):
    from anemll import dory_chat as chat

    runtime = chat.DoryRuntime.__new__(chat.DoryRuntime)
    runtime.model_dir = tmp_path
    runtime.params = {"rope_tables_as_inputs": True,
                      "rope_tables": {"file": dcv.ROPE_TABLE_FILE, "rows": 8,
                                      "head_dim": 4, "keys": list(dcv.ROPE_TABLE_KEYS)}}
    with pytest.raises(RuntimeError, match="missing"):
        runtime._load_rope_tables()
