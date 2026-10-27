"""Tool calling for Dory, in the format the model was actually trained on.

The chat template tells the model to answer a tool request with:

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

so a server has to do two things an ordinary engine does: turn that text into the
OpenAI tool_calls shape a client expects, and turn the client's tool result back into
the <tool_response> the template renders.  Both directions live here, with tests that
do not need a Neural Engine.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_NAME = re.compile(r"<function\s*=\s*([^>\s]+)\s*>", re.S)
_PARAM = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.S)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Split model text into (content, tool_calls) in the OpenAI shape.

    Anything before the first <tool_call> stays as content: the model is allowed to
    reason in prose first, and clients render that.  Values are kept as strings,
    because a model that writes 5 as "5" must not decide a schema's types for it;
    coercion is the tool's business.
    """
    matches = list(_CALL.finditer(text))
    if not matches:
        return text, []
    content = text[: matches[0].start()].strip()
    calls: List[Dict[str, Any]] = []
    for match in matches:
        block = match.group(1)
        name_match = _NAME.search(block)
        if not name_match:
            continue
        arguments: Dict[str, str] = {}
        for param in _PARAM.finditer(block):
            arguments[param.group(1)] = param.group(2).strip()
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:20],
            "type": "function",
            "function": {"name": name_match.group(1),
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    return content, calls


def normalise_tools(tools: Optional[list]) -> Optional[list]:
    """Accept both {"type":"function","function":{...}} and a bare function body."""
    if not tools:
        return None
    out = []
    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            out.append(tool)
        elif isinstance(tool, dict) and "name" in tool:
            out.append({"type": "function", "function": tool})
        else:
            out.append(tool)
    return out


def tool_result_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass tool results through in the shape the template renders.

    The template emits <tool_response> for role "tool" messages, and drops the
    tool_call_id itself, so nothing needs rewriting here -- this exists to document
    that and to be the single place to change if that ever differs.
    """
    return messages