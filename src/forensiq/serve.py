"""Optional FastAPI receiver — the intake valve of the black box flight recorder.

Point any OpenTelemetry SDK at it: the OTLP/HTTP exporter POSTs JSON to
``/v1/traces`` (the standard OTel collector path; ``/v1/otlp/v1/traces`` is
accepted as an alias). Bodies are parsed with the tolerant OTLP parser and the
normalized traces are buffered as JSONL on disk with size-capped rotation —
a crash-safe drop file you can later feed to ``forensiq analyze``/``report``
or land in a :class:`~forensiq.ingest.duckdb_store.DuckDBTraceStore`.

Auth mirrors the sibling portfolio repos (AegisGate / SwarmResearch): raw keys
come from the ``FORENSIQ_API_KEYS`` env var (comma-separated), only SHA-256
hashes are ever stored or compared (constant-time), raw keys are never logged
and never echoed back. When the variable is unset the authenticator is
DISABLED (open receiver) with a startup warning, so a local dev setup keeps
working unchanged. ``/health`` is always exempt.

FastAPI is an optional dependency: ``pip install 'forensiq[serve]'``.
Importing this module without it is fine; :func:`create_app` raises an
actionable :class:`~forensiq.ingest.loaders.IngestError`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forensiq.ingest.loaders import IngestError
from forensiq.ingest.otlp import parse_otlp

if TYPE_CHECKING:  # only for type hints; never imported at runtime without the extra
    from fastapi import FastAPI

logger = logging.getLogger("forensiq.serve")

GENERIC_401 = {"detail": "Unauthorized: missing or invalid API key."}
EXEMPT_PATHS = {"/health"}
KEY_HEADER = "x-forensiq-key"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB per JSONL shard


def hash_key(raw: str) -> str:
    """SHA-256 of one raw key, hex-encoded. The only form kept at rest."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class APIKeyAuth:
    """Hold-at-rest key hashes; verify presented keys in constant time."""

    def __init__(self, raw_keys: str | None) -> None:
        self.enabled = bool(raw_keys and raw_keys.strip())
        if not self.enabled:
            self._hashes: frozenset[str] = frozenset()
            logger.warning(
                "FORENSIQ_API_KEYS is not set — receiver auth is DISABLED "
                "(open intake). Set it (comma-separated keys) before exposing "
                "the receiver beyond localhost."
            )
            return
        keys = [k.strip() for k in raw_keys.split(",")]
        self._hashes = frozenset(hash_key(k) for k in keys if k)
        logger.warning("Receiver auth ENABLED with %d key(s); only SHA-256 hashes are stored.", len(self._hashes))

    def verify(self, presented: str | None) -> bool:
        """True iff the presented key matches one enrolled key (constant-time)."""
        if not self.enabled:
            return True
        if presented is None:
            return False
        return any(hmac.compare_digest(hash_key(presented), h) for h in self._hashes)


class APIKeyMiddleware:
    """Pure-ASGI middleware requiring ``X-ForensiQ-Key`` on every route but /health."""

    def __init__(self, app: Any, auth: APIKeyAuth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not self.auth.enabled:
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if not self.auth.verify(headers.get(KEY_HEADER)):
            from fastapi.responses import JSONResponse  # deferred: only reachable with fastapi installed

            await JSONResponse(GENERIC_401, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class RotatingJSONLBuffer:
    """Append traces as JSONL; rotate to ``traces.NNNN.jsonl`` when the active
    shard exceeds ``max_bytes``. Crash-safe: every accepted trace is flushed
    line-by-line before the HTTP response is sent."""

    def __init__(self, directory: str | Path, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.dir = Path(directory)
        self.max_bytes = max(1024, int(max_bytes))  # floor: never rotate on every write
        self.active = self.dir / "traces.jsonl"
        self.written = 0
        self.rotations = 0

    def append(self, records: list[dict[str, Any]]) -> int:
        """Write one JSON object per line; returns lines written."""
        if not records:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        payloads = [json.dumps(r, default=str) for r in records]
        if self.active.exists() and self.active.stat().st_size + sum(len(p) + 1 for p in payloads) > self.max_bytes:
            self._rotate()
        with self.active.open("a", encoding="utf-8") as fh:
            for payload in payloads:
                fh.write(payload + "\n")
                fh.flush()
        self.written += len(payloads)
        return len(payloads)

    def _rotate(self) -> None:
        seq = self.rotations
        rotated = self.dir / f"traces.{seq:04d}.jsonl"
        while rotated.exists():  # defensive: pre-existing shards never overwritten
            seq += 1
            rotated = self.dir / f"traces.{seq:04d}.jsonl"
        self.active.rename(rotated)
        self.rotations = seq + 1
        logger.info("rotated OTLP buffer to %s", rotated.name)


def create_app(
    buffer_dir: str | Path = "forensiq-buffer",
    api_keys: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> FastAPI:
    """Build the receiver app.

    api_keys: comma-separated raw keys; None reads ``FORENSIQ_API_KEYS``.
    An empty/unset variable disables auth (with a warning) so local dev flows
    keep working.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise IngestError(
            "the trace receiver needs FastAPI: pip install 'forensiq[serve]'"
        ) from exc

    if api_keys is None:
        api_keys = os.environ.get("FORENSIQ_API_KEYS")
    auth = APIKeyAuth(api_keys)
    buffer = RotatingJSONLBuffer(buffer_dir, max_bytes=max_bytes)

    app = FastAPI(title="ForensiQ trace receiver", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.auth = auth
    app.state.buffer = buffer
    app.state.traces_received = 0
    app.add_middleware(APIKeyMiddleware, auth=auth)  # type: ignore[arg-type]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "forensiq-receiver",
            "auth_enabled": auth.enabled,
            "traces_received": app.state.traces_received,
        }

    async def _handle_export(request):
        # Plain Starlette handler (no FastAPI dependency injection): the
        # handler is registered for both OTLP paths and receives the raw
        # request, because the fastapi.Request type cannot be resolved from
        # this module's deferred imports.
        body = await request.body()
        try:
            result = parse_otlp(body)
        except IngestError as exc:
            return JSONResponse({"error": str(exc), "partialSuccess": None}, status_code=400)
        records = [t.model_dump(mode="json") for t in result.traces]
        buffer.append(records)
        app.state.traces_received += len(records)
        if result.skip.total_skipped:
            logger.warning(
                "OTLP export had %d malformed entries (first reason: %s)",
                result.skip.total_skipped,
                result.skip.reasons[0] if result.skip.reasons else "n/a",
            )
        # OTel SDKs expect an ExportTraceServiceResponse; partialSuccess stays
        # empty because rejected entries were already dropped upstream with a
        # logged reason, not rejected wholesale.
        return JSONResponse({"partialSuccess": {}}, status_code=200)

    # The standard OTLP/HTTP exporter path plus the task-specified alias.
    # Registered as raw Starlette routes (bypasses FastAPI signature analysis,
    # which cannot see the locally-imported Request type).
    from starlette.routing import Route

    app.router.routes.append(Route("/v1/traces", _handle_export, methods=["POST"]))
    app.router.routes.append(Route("/v1/otlp/v1/traces", _handle_export, methods=["POST"]))
    return app


def main_serve(host: str = "127.0.0.1", port: int = 4318, buffer_dir: str | Path = "forensiq-buffer") -> int:
    """Blocking entry point (``forensiq serve``). Raises IngestError when the
    [serve] extra is missing."""
    try:
        import uvicorn  # deferred: part of the [serve] extra
    except ImportError as exc:
        raise IngestError("serving needs uvicorn: pip install 'forensiq[serve]'") from exc
    app = create_app(buffer_dir=buffer_dir)
    logger.warning("ForensiQ receiver listening on http://%s:%d (OTLP/HTTP)", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
