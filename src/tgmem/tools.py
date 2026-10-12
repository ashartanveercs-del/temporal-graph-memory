"""Function-calling tools for an LLM, bound to one sender.

Build the ``ScopedMemory`` from the sender your transport authenticated (the webhook's
verified phone number, the logged-in user), then give the model ``tool_schemas(scoped)``
and route its calls through ``call_tool(scoped, ...)``. The model never names a sender,
viewer or scope, so it cannot ask for someone else's memory:

* schemas only offer what this sender may do (a stranger can only search);
* unexpected arguments are rejected, so a forged ``"sender": ...`` does nothing;
* errors are generic where specifics would leak: a fact that exists but is hidden from
  this sender produces exactly the same error as one that does not exist;
* ``call_tool`` never raises; failures come back as ``{"ok": false, "error": ...}``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import audience
from .memory import NotFound, ScopedMemory, ScopeError
from .timeutil import parse_time, to_text

log = logging.getLogger("tgmem")

STRING = {"type": "string"}
STRINGS = {"type": "array", "items": {"type": "string"}}


@dataclass(frozen=True)
class _Tool:
    name: str
    description: str
    kinds: tuple[str, ...]  # viewer kinds the tool is offered to
    handler: Callable[..., dict]
    params: Callable[[ScopedMemory], tuple[dict, list[str]]]  # (properties, required)


def tool_schemas(scoped: ScopedMemory) -> list[dict]:
    """OpenAI-style function definitions for the tools this sender may use."""
    out = []
    for tool in _available(scoped):
        properties, required = tool.params(scoped)
        out.append({"type": "function", "function": {
            "name": tool.name, "description": tool.description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}})
    return out


def call_tool(scoped: ScopedMemory, name: str, arguments: str | dict) -> dict:
    tool = next((t for t in _available(scoped) if t.name == name), None)
    if tool is None:
        return _error(f"unknown tool {name!r}")
    try:
        args = _parse(arguments)
        properties, required = tool.params(scoped)
        _check(args, properties, required)
        return tool.handler(scoped, **args)
    except _BadArguments as exc:
        return _error(str(exc))
    except NotFound as exc:
        return _error(str(exc) if scoped.viewer.kind == "owner" else "unknown fact id")
    except ScopeError:
        return _error("not allowed for this sender")
    except ValueError as exc:
        return _error(str(exc))
    except Exception:
        log.exception("memory tool %s failed", name)
        return _error("internal error")


# -- handlers ------------------------------------------------------------------------------

def _search(scoped: ScopedMemory, query: str, as_of: str | None = None) -> dict:
    memory = scoped.memory
    when = parse_time(as_of, now=memory.now()) if as_of else None
    result = scoped.search(query, as_of=when)
    contact = scoped.viewer.node_id if scoped.viewer.kind == "contact" else None
    facts = []
    for hit in result.hits:
        told = audience.told_at(memory, contact, hit.fact) if contact else None
        facts.append({
            "id": hit.fact.id,
            "fact": hit.fact.text,
            "valid_from": to_text(hit.fact.valid_from),
            "valid_to": to_text(hit.fact.valid_to) if hit.fact.valid_to else None,
            "already_told": to_text(told) if told else None,
            "according_to": audience.attribution(memory, hit.fact),
        })
    return {"ok": True, "mode": result.mode, "facts": facts}


def _remember(scoped: ScopedMemory, predicate: str, value: str, subject: str | None = None,
              scope: str | None = None, valid_from: str | None = None,
              valid_to: str | None = None) -> dict:
    now = scoped.memory.now()
    fact = scoped.remember(
        predicate, value, subject=subject, scope=scope,
        valid_from=parse_time(valid_from, now=now) if valid_from else None,
        valid_to=parse_time(valid_to, now=now) if valid_to else None)
    return {"ok": True, "id": fact.id, "scope": fact.scope}


def _mark_told(scoped: ScopedMemory, fact_ids: list[str]) -> dict:
    return {"ok": True, "recorded": len(scoped.mark_told(fact_ids))}


def _corrections(scoped: ScopedMemory) -> dict:
    memory = scoped.memory
    owed = [{"told": audience.value_text(memory, c.told),
             "told_at": to_text(c.told_at),
             "now": audience.value_text(memory, c.current) if c.current else None,
             "reason": c.reason} for c in scoped.corrections_owed()]
    return {"ok": True, "corrections": owed}


# -- parameters ----------------------------------------------------------------------------

def _search_params(scoped: ScopedMemory) -> tuple[dict, list[str]]:
    return ({"query": {**STRING, "description": "What to look up, in plain words."},
             "as_of": {**STRING, "description": "Optional point in time, e.g. 2026-03-01."}},
            ["query"])


def _remember_params(scoped: ScopedMemory) -> tuple[dict, list[str]]:
    properties = {
        "predicate": {**STRING, "description": "Short attribute name, e.g. budget."},
        "value": {**STRING, "description": "The value, in plain words."},
        "valid_from": {**STRING, "description": "Optional start, e.g. 2026-04-01 or +7d."},
        "valid_to": {**STRING, "description": "Optional end, e.g. 2026-04-30."},
    }
    if scoped.viewer.kind != "owner":
        return properties, ["predicate", "value"]
    properties["subject"] = {**STRING, "description": "Id of the node the fact is about."}
    properties["scope"] = {**STRING, "description":
                           "public, internal, or private:<contact-id>. Defaults to private "
                           "for facts about a contact and public otherwise."}
    return properties, ["subject", "predicate", "value"]


def _ids_params(scoped: ScopedMemory) -> tuple[dict, list[str]]:
    return {"fact_ids": {**STRINGS, "description": "Ids of facts the reply stated."}}, [
        "fact_ids"]


def _no_params(scoped: ScopedMemory) -> tuple[dict, list[str]]:
    return {}, []


_TOOLS = (
    _Tool("memory_search", "Search what you know that you may share with this person.",
          ("owner", "contact", "anonymous"), _search, _search_params),
    _Tool("memory_remember", "Save a fact for later. Facts a contact tells you are kept "
          "private to them, as their own claim.", ("owner", "contact"), _remember,
          _remember_params),
    _Tool("memory_mark_told", "Record which facts you just told this person, so you do not "
          "repeat them.", ("contact",), _mark_told, _ids_params),
    _Tool("memory_corrections", "List things this person was told that are no longer true.",
          ("contact",), _corrections, _no_params),
)


def _available(scoped: ScopedMemory) -> list[_Tool]:
    return [tool for tool in _TOOLS if scoped.viewer.kind in tool.kinds]


# -- argument checking ---------------------------------------------------------------------

class _BadArguments(ValueError):
    pass


def _parse(arguments: str | dict) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            raise _BadArguments("arguments must be a JSON object") from None
    if not isinstance(arguments, dict):
        raise _BadArguments("arguments must be a JSON object")
    return arguments


def _check(args: dict[str, Any], properties: dict, required: list[str]) -> None:
    for key in sorted(args):
        if key not in properties:
            raise _BadArguments(f"unexpected argument {key!r}")
    for key in required:
        if key not in args:
            raise _BadArguments(f"missing argument {key!r}")
    for key, value in args.items():
        expected = properties[key]["type"]
        if expected == "string" and not isinstance(value, str):
            raise _BadArguments(f"argument {key!r} must be a string")
        if expected == "array" and not (isinstance(value, list)
                                        and all(isinstance(v, str) for v in value)):
            raise _BadArguments(f"argument {key!r} must be a list of strings")


def _error(message: str) -> dict:
    return {"ok": False, "error": message}
