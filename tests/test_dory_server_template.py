"""The prompt path against the real tokenizer and Dory's real chat template.

The server's HTTP surface is tested with a scripted engine, but the prompt it
*renders* has to be the model's own: Dory's template puts the tool definitions
in the system block, teaches the tool-call reply format, and closes a thinking
block when reasoning is off.  This test renders it with the BF16 repository's
tokenizer and template and checks the text and token ids that reach the model.

It needs the checkpoint's tokenizer files, which live outside the repo, so it is
marked integration and skips when they are not on disk.
"""

import os
import shutil
from pathlib import Path

import pytest

from anemll.dory_chat import ChatTokenizer
from anemll.openai_server.dory_engine import thinking_open
from anemll.openai_server.messages import to_template_messages
from anemll.openai_server.splitter import REASONING_CLOSE, REASONING_OPEN

pytestmark = pytest.mark.integration

NL = chr(10)
ASSISTANT_PROMPT = "<|im_start|>assistant"

SNAPSHOTS = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--"
    "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/snapshots"))

TOOLS = [{
    "type": "function",
    "function": {"name": "get_weather", "description": "Get the current weather",
                 "parameters": {"type": "object",
                                "properties": {"city": {"type": "string",
                                                        "description": "City name"}},
                                "required": ["city"]}},
}]


def _snapshot() -> Path:
    if not SNAPSHOTS.is_dir():
        pytest.skip("the BF16 checkpoint's tokenizer files are not on disk")
    for candidate in sorted(SNAPSHOTS.iterdir()):
        if (candidate / "tokenizer.json").is_file():
            return candidate
    pytest.skip("no snapshot carries tokenizer.json")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    snapshot = _snapshot()
    directory = tmp_path_factory.mktemp("dory-tokenizer")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja"):
        source = snapshot / name
        if source.is_file():
            shutil.copy2(source, directory / name)
    meta = {"tokenizer": {"chat_template": "chat_template.jinja",
                          "chat_template_fallback": None, "reasoning_default": True}}
    return ChatTokenizer(directory, meta)


def test_reasoning_on_leaves_an_open_thinking_block(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], reasoning=True)
    assert text.endswith(ASSISTANT_PROMPT + NL + REASONING_OPEN + NL)
    assert thinking_open(text) is True
    ids = tokenizer.encode(text)
    # '<|im_start|>assistant', newline, the open marker (12), newline
    assert ids[-3:].tolist() == [1010, 12, 1010]


def test_reasoning_off_closes_the_thinking_block_in_the_prompt(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], reasoning=False)
    assert text.endswith(ASSISTANT_PROMPT + NL + REASONING_OPEN + REASONING_CLOSE)
    assert thinking_open(text) is False
    ids = tokenizer.encode(text)
    assert ids[-2:].tolist() == [12, 13]


def test_tools_are_rendered_into_the_system_block(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the weather in Santa Clara?"}],
        reasoning=True, tools=TOOLS)
    assert "<tools>" in text and "get_weather" in text
    assert "<name>city</name>" in text
    assert '<required>["city"]</required>' in text
    assert text.index("<|im_start|>system") < text.index("<tools>")


def test_an_assistant_tool_call_round_trips_into_the_template(tokenizer):
    messages = to_template_messages([
        {"role": "user", "content": "weather in Santa Clara?"},
        {"role": "assistant", "content": None, "reasoning_content": "I need a tool.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": '{"city": "Santa Clara"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21 C"},
        {"role": "user", "content": "and tomorrow?"},
    ])
    text = tokenizer.apply_chat_template(messages, reasoning=True, tools=TOOLS)
    # the native call format the template teaches
    assert "<tool_call>" + NL + "<function=get_weather>" + NL + "<parameter=city>" in text
    assert "Santa Clara" in text
    assert "<tool_response>" + NL + "sunny, 21 C" + NL + "</tool_response>" in text
    # history reasoning is collapsed by the template (truncate_history_thinking)
    assert "I need a tool." not in text
    assert text.endswith(ASSISTANT_PROMPT + NL + REASONING_OPEN + NL)
    assert thinking_open(text) is True


def test_the_template_ignores_tools_when_none_are_given(tokenizer):
    text = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}],
                                         reasoning=True)
    assert "<tools>" not in text
    assert "You have access to the following functions" not in text
