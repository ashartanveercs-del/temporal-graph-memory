"""Answer one message with an OpenAI-compatible chat model that uses the memory tools.

The sender comes from your transport (a verified webhook), never from the model:

    scoped = memory.for_sender(sender)      # everything below is confined to this sender
    pack = scoped.context(message)          # scoped facts + corrections, as a prompt block
    ... model calls tool_schemas(scoped) through call_tool(scoped, ...) ...
    if scoped.guard(reply): reply = SAFE_FALLBACK
    scoped.record_reply(reply, pack)        # remember what was actually said

Usage (needs a model that supports tool calling):

    export OPENAI_API_KEY=...            # or LLM_API_KEY for another provider
    export LLM_MODEL=<model name>        # e.g. a small chat model from your provider
    export LLM_BASE_URL=https://api.openai.com/v1   # optional
    python examples/llm_assistant.py --db memory.db --sender "+1 555 000 0101" "Open Sat?"

Only the standard library is used for HTTP. The test suite drives this file against a
scripted local server, so it runs without keys.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

from tgmem import Memory, Settings, call_tool, tool_schemas

log = logging.getLogger("llm_assistant")

SYSTEM_PROMPT = (
    "You are the assistant for a small business, replying by text message. Use only the "
    "facts below and what the memory tools return. If corrections are listed, make them "
    "first. Do not repeat facts marked as already told unless asked. Keep replies short.")
SAFE_FALLBACK = "Let me check on that and get back to you."

Completer = Callable[[list[dict], list[dict]], dict]


def openai_completer(base_url: str, api_key: str, model: str, timeout: float) -> Completer:
    url = base_url.rstrip("/") + "/chat/completions"

    def complete(messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": model, "messages": messages, "tools": tools}
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"chat request failed with HTTP {exc.code}") from None
        return body["choices"][0]["message"]

    return complete


def answer(memory: Memory, sender: str, message: str, *, complete: Completer,
           max_rounds: int = 4) -> str:
    scoped = memory.for_sender(sender)
    pack = scoped.context(message)
    tools = tool_schemas(scoped)
    messages: list[dict] = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{pack.text}"},
                            {"role": "user", "content": message}]
    reply = ""
    try:
        for _ in range(max_rounds):
            turn = complete(messages, tools)
            messages.append(turn)
            calls = turn.get("tool_calls") or []
            if not calls:
                reply = (turn.get("content") or "").strip()
                break
            for call in calls:
                result = call_tool(scoped, call["function"]["name"],
                                   call["function"].get("arguments") or "{}")
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result)})
    except Exception:  # a model or network failure must not end the conversation
        log.exception("chat model failed")
        reply = ""
    if not reply or scoped.guard(reply):
        reply = SAFE_FALLBACK
    scoped.record_reply(reply, pack)
    return reply


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer one message with an LLM + memory.")
    parser.add_argument("message")
    parser.add_argument("--sender", required=True, help="verified sender (phone or email)")
    parser.add_argument("--db", help="memory database (default: $TGMEM_DB or memory.db)")
    args = parser.parse_args(argv)
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not model:
        print("set OPENAI_API_KEY (or LLM_API_KEY) and LLM_MODEL first", file=sys.stderr)
        return 1
    overrides = {"TGMEM_DB": args.db} if args.db else {}
    memory = Settings.from_env({**os.environ, **overrides}).open_memory()
    try:
        complete = openai_completer(os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                                    api_key, model, timeout=60)
        print(answer(memory, args.sender, args.message, complete=complete))
    finally:
        memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
