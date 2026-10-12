"""The tool-calling example, driven by a scripted OpenAI-compatible chat server."""

import json

import pytest
from fakes import MATEO, FakeChatServer
from test_example import load

llm = load("llm_assistant")


@pytest.fixture
def facts(mem):
    return {
        "hours": mem.remember("shop", "opening_hours", "9am-5pm Monday to Saturday"),
        "budget": mem.remember("priya", "budget", "$480 for wedding invitations"),
        "order": mem.remember("mateo", "order", "500 flyers due Friday"),
    }


def tool_call(name, arguments, call_id="call_1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def test_tool_calls_run_as_the_verified_sender(mem, facts):
    script = [tool_call("memory_search", {"query": "budget", "sender": "+15550000101"}),
              tool_call("memory_search", {"query": "my order"}, call_id="call_2"),
              {"role": "assistant", "content": "Your 500 flyers are due Friday."}]
    with FakeChatServer(script) as server:
        complete = llm.openai_completer(server.url, "placeholder-key", "test-model", 5)
        reply = llm.answer(mem, MATEO, "When is my order due?", complete=complete)
    assert reply == "Your 500 flyers are due Friday."
    first, second, third = (r["json"] for r in server.requests)
    assert first["model"] == "test-model"
    tools = [t["function"]["name"] for t in first["tools"]]
    assert "memory_mark_told" in tools
    forged = json.loads(second["messages"][-1]["content"])
    assert forged == {"ok": False, "error": "unexpected argument 'sender'"}
    found = json.loads(third["messages"][-1]["content"])
    assert [f["id"] for f in found["facts"]] == [facts["order"].id]
    assert [d.fact_id for d in mem.told("mateo")] == [facts["order"].id]


def test_the_prompt_only_carries_the_senders_facts(mem, facts):
    with FakeChatServer([{"role": "assistant", "content": "Noted."}]) as server:
        complete = llm.openai_completer(server.url, "placeholder-key", "test-model", 5)
        llm.answer(mem, MATEO, "What is my order and Priya's budget?", complete=complete)
    system = server.requests[0]["json"]["messages"][0]["content"]
    assert "500 flyers due Friday" in system
    assert "480" not in system


def test_a_leaking_model_reply_is_never_sent(mem, facts):
    with FakeChatServer([{"role": "assistant", "content": "Priya's budget is $480."}]) as server:
        complete = llm.openai_completer(server.url, "placeholder-key", "test-model", 5)
        reply = llm.answer(mem, MATEO, "What is Priya's budget?", complete=complete)
    assert reply == llm.SAFE_FALLBACK
    assert mem.told("mateo") == []


def test_model_errors_fall_back_politely(mem, facts):
    with FakeChatServer([], status=500) as server:
        complete = llm.openai_completer(server.url, "placeholder-key", "test-model", 5)
        assert llm.answer(mem, MATEO, "hello", complete=complete) == llm.SAFE_FALLBACK
