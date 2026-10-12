"""Shared builders and test doubles. Nothing here leaves the machine: the fake HTTP
servers listen on 127.0.0.1 only."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tgmem.embeddings import EmbeddingError, HashingEmbedder
from tgmem.model import PUBLIC, Fact, Node
from tgmem.timeutil import UTC

T0 = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)

ROSA = "+1 555 000 0100"  # the owner
PRIYA = "+1 555 000 0101"  # a customer
MATEO = "+1 555 000 0102"  # another customer


def make_node(node_id: str, name: str | None = None, *, kind: str = "thing",
              role: str | None = None, handles: tuple[str, ...] = (),
              aliases: tuple[str, ...] = (), created_at: datetime = T0) -> Node:
    return Node(id=node_id, name=name or node_id.replace("-", " ").title(), kind=kind,
                created_at=created_at, aliases=aliases, handles=handles, role=role)


def make_fact(fact_id: str = "f-1", **overrides) -> Fact:
    fields = dict(id=fact_id, subject="shop", predicate="opening_hours", value="9am-5pm",
                  object=None, scope=PUBLIC, valid_from=T0, valid_to=None, recorded_at=T0,
                  asserted_by="owner", text="Shop opening hours 9am-5pm")
    fields.update(overrides)
    return Fact(**fields)


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        pass  # clients that time out on purpose disconnect mid-response


class LoopbackJsonServer:
    """A JSON-over-HTTP endpoint on 127.0.0.1 (random port), served from a background
    thread. Subclasses decide the reply; every request is recorded in ``requests``."""

    def __init__(self, *, status: int = 200, delay: float = 0.0,
                 body: bytes | None = None) -> None:
        self.status, self.delay, self.body = status, delay, body
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (http.server naming)
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                server.requests.append({"path": self.path, "headers": dict(self.headers),
                                        "json": payload})
                time.sleep(server.delay)
                body = server.body if server.body is not None else server._reply(payload)
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # keep test output quiet
                pass

        self._httpd = _QuietServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.01}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def _reply(self, payload: dict) -> bytes:
        raise NotImplementedError

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class FakeEmbeddingServer(LoopbackJsonServer):
    """OpenAI-compatible ``POST /v1/embeddings``. Vectors come from ``HashingEmbedder`` so
    they are deterministic; items are returned in reverse order to check index sorting."""

    def __init__(self, *, dim: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim

    def _reply(self, payload: dict) -> bytes:
        if self.status != 200:
            return b'{"error": {"message": "upstream exploded"}}'
        vectors = HashingEmbedder(self.dim).embed(payload["input"])
        data = [{"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)]
        return json.dumps({"object": "list", "data": data[::-1],
                           "model": payload["model"]}).encode()


class FakeChatServer(LoopbackJsonServer):
    """OpenAI-compatible ``POST /v1/chat/completions`` that plays back scripted assistant
    messages, one per request, so a tool-calling loop can be tested without a model."""

    def __init__(self, script: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self.script = list(script)

    def _reply(self, payload: dict) -> bytes:
        if self.status != 200:
            return b'{"error": {"message": "model unavailable"}}'
        message = self.script.pop(0)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return json.dumps({"id": f"chatcmpl-{len(self.requests)}", "object": "chat.completion",
                           "model": payload["model"],
                           "choices": [{"index": 0, "message": message,
                                        "finish_reason": finish}]}).encode()


class FailingEmbedder:
    """Every call fails, like a provider that is down or a missing API key."""

    model_id = "failing-v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        raise EmbeddingError("provider unavailable")


class SwitchableEmbedder:
    """Wraps a real embedder; flip ``.down`` to simulate an outage. Counts calls."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.down = False
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.down:
            raise EmbeddingError("simulated outage")
        return self.inner.embed(texts)
