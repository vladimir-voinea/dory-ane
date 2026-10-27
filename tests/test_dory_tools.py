from anemll.runtime.dory_tools import parse_tool_calls, normalise_tools


def test_no_tool_call_returns_the_text_untouched():
    text = "The capital of France is Paris."
    content, calls = parse_tool_calls(text)
    assert content == text and calls == []


def test_parses_the_template_format_into_openai_tool_calls():
    text = ("""Let me check the weather.
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
<parameter=units>
celsius
</parameter>
</function>
</tool_call>""")
    content, calls = parse_tool_calls(text)
    assert content == "Let me check the weather."
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    import json
    assert json.loads(call["function"]["arguments"]) == {
        "city": "Paris", "units": "celsius"}
    assert call["id"].startswith("call_")


def test_two_calls_in_one_reply_both_come_back():
    text = ("<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
            "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>")
    _, calls = parse_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_a_multi_line_value_survives():
    text = ("<tool_call><function=write>\n<parameter=body>\nline one\nline two\n"
            "</parameter>\n</function>\n</tool_call>")
    _, calls = parse_tool_calls(text)
    import json
    assert json.loads(calls[0]["function"]["arguments"])["body"] == "line one\nline two"


def test_bare_function_bodies_are_accepted_as_tools():
    tools = [{"name": "ping", "description": "x", "parameters": {}}]
    out = normalise_tools(tools)
    assert out[0]["type"] == "function" and out[0]["function"]["name"] == "ping"