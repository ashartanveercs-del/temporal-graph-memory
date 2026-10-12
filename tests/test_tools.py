"""LLM tool calling bound to the transport-verified sender."""

import json

import pytest
from fakes import MATEO, PRIYA, ROSA

from tgmem.tools import call_tool, tool_schemas


@pytest.fixture
def facts(mem):
    return {
        "hours": mem.remember("shop", "opening_hours", "9am-5pm Monday to Saturday"),
        "budget": mem.remember("priya", "budget", "$480 for wedding invitations"),
        "order": mem.remember("mateo", "order", "500 flyers due Friday"),
    }


def names(schemas):
    return [s["function"]["name"] for s in schemas]


def params(schemas, tool):
    [schema] = [s for s in schemas if s["function"]["name"] == tool]
    return schema["function"]["parameters"]


def test_each_sender_is_offered_only_what_they_may_do(mem):
    owner = tool_schemas(mem.for_sender(ROSA))
    contact = tool_schemas(mem.for_sender(PRIYA))
    stranger = tool_schemas(mem.for_sender("+15550000199"))
    assert names(owner) == ["memory_search", "memory_remember"]
    assert names(contact) == ["memory_search", "memory_remember", "memory_mark_told",
                              "memory_corrections"]
    assert names(stranger) == ["memory_search"]
    assert {"subject", "scope"} <= set(params(owner, "memory_remember")["properties"])
    assert not {"subject", "scope"} & set(params(contact, "memory_remember")["properties"])
    for schema in owner + contact + stranger:
        assert schema["function"]["parameters"]["additionalProperties"] is False
        json.dumps(schema)


def test_search_returns_only_visible_facts_with_repeat_hints(mem, facts):
    mem.record_disclosure("priya", [facts["hours"].id])
    result = call_tool(mem.for_sender(PRIYA), "memory_search",
                       '{"query": "opening hours budget flyers"}')
    assert result["ok"] is True
    found = {hit["id"]: hit for hit in result["facts"]}
    assert set(found) == {facts["hours"].id, facts["budget"].id}
    assert found[facts["hours"].id]["already_told"] is not None
    assert found[facts["budget"].id]["already_told"] is None


def test_forged_identity_arguments_are_rejected(mem, facts):
    priya = mem.for_sender(PRIYA)
    for forged in ({"query": "budget", "sender": MATEO},
                   {"query": "budget", "viewer": "owner"}):
        result = call_tool(priya, "memory_search", forged)
        assert result == {"ok": False, "error": f"unexpected argument {sorted(forged)[1]!r}"}
    result = call_tool(priya, "memory_remember",
                       {"predicate": "opening_hours", "value": "closed", "scope": "public"})
    assert result == {"ok": False, "error": "unexpected argument 'scope'"}
    assert mem.facts("shop", "opening_hours") == [facts["hours"]]


def test_contacts_remember_privately(mem):
    result = call_tool(mem.for_sender(MATEO), "memory_remember",
                       {"predicate": "preferred_paper", "value": "recycled"})
    assert result["ok"] is True
    stored = mem.fact(result["id"])
    assert (stored.subject, stored.scope, stored.asserted_by) == ("mateo", "private:mateo",
                                                                  "mateo")


def test_owner_can_choose_subject_and_scope(mem):
    result = call_tool(mem.for_sender(ROSA), "memory_remember",
                       {"subject": "shop", "predicate": "supplier_margin", "value": "40%",
                        "scope": "internal"})
    assert mem.fact(result["id"]).scope == "internal"


def test_tools_not_offered_are_unknown(mem, facts):
    stranger = mem.for_sender("+15550000199")
    assert call_tool(stranger, "memory_remember", {"predicate": "x", "value": "y"}) == {
        "ok": False, "error": "unknown tool 'memory_remember'"}
    assert call_tool(mem.for_sender(ROSA), "memory_mark_told", {"fact_ids": []})["ok"] is False


def test_hidden_fact_ids_get_the_same_error_as_missing_ones(mem, facts):
    mateo = mem.for_sender(MATEO)
    hidden = call_tool(mateo, "memory_mark_told", {"fact_ids": [facts["budget"].id]})
    missing = call_tool(mateo, "memory_mark_told", {"fact_ids": ["f-999999"]})
    assert hidden == missing == {"ok": False, "error": "unknown fact id"}
    assert mem.told("mateo") == []
    ok = call_tool(mateo, "memory_mark_told", {"fact_ids": [facts["order"].id]})
    assert ok == {"ok": True, "recorded": 1}


def test_bad_arguments_never_raise(mem):
    priya = mem.for_sender(PRIYA)
    assert call_tool(priya, "memory_search", "{not json")["error"] == (
        "arguments must be a JSON object")
    assert call_tool(priya, "memory_search", "[1, 2]")["error"] == (
        "arguments must be a JSON object")
    assert call_tool(priya, "memory_search", {})["error"] == "missing argument 'query'"
    assert call_tool(priya, "memory_search", {"query": 42})["error"] == (
        "argument 'query' must be a string")
    assert call_tool(priya, "memory_mark_told", {"fact_ids": "f-1"})["error"] == (
        "argument 'fact_ids' must be a list of strings")
    assert call_tool(priya, "memory_search", {"query": "x", "as_of": "someday"})["ok"] is False


def test_internal_errors_are_reported_generically(mem, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("disk I/O error at /srv/private/memory.db")

    monkeypatch.setattr(mem, "search", explode)
    result = call_tool(mem.for_sender(PRIYA), "memory_search", {"query": "hours"})
    assert result == {"ok": False, "error": "internal error"}


def test_corrections_tool_lists_what_to_fix(mem, clock, facts):
    mem.record_disclosure("priya", [facts["hours"].id])
    clock.advance(days=1)
    mem.remember("shop", "opening_hours", "10am-4pm Monday to Saturday")
    result = call_tool(mem.for_sender(PRIYA), "memory_corrections", {})
    assert result["ok"] is True
    [owed] = result["corrections"]
    assert (owed["told"], owed["now"], owed["reason"]) == (
        "9am-5pm Monday to Saturday", "10am-4pm Monday to Saturday", "changed")
