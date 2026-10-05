"""OTLP receiver: auth (401/200), OTLP body acceptance, JSONL buffering +
rotation, health endpoint. All via the ASGI TestClient — no network."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from forensiq.serve import APIKeyAuth, create_app, hash_key
from tests.test_otlp import FIXTURE


@pytest.fixture()
def receiver(tmp_path):
    app = create_app(buffer_dir=tmp_path / "buf", api_keys="secret-key-1,secret-key-2")
    return TestClient(app), tmp_path / "buf"


def otlp_post(client: TestClient, key: str | None = None, path: str = "/v1/traces"):
    headers = {"X-ForensiQ-Key": key} if key else {}
    return client.post(path, json=FIXTURE, headers=headers)


class TestAuth:
    def test_missing_key_is_401(self, receiver):
        client, _ = receiver
        assert otlp_post(client).status_code == 401

    def test_wrong_key_is_401(self, receiver):
        client, _ = receiver
        resp = otlp_post(client, key="wrong-key")
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Unauthorized: missing or invalid API key."}

    def test_valid_key_is_200(self, receiver):
        client, _ = receiver
        resp = otlp_post(client, key="secret-key-1")
        assert resp.status_code == 200
        assert resp.json() == {"partialSuccess": {}}

    def test_second_enrolled_key_also_works(self, receiver):
        client, _ = receiver
        assert otlp_post(client, key="secret-key-2").status_code == 200

    def test_health_is_exempt(self, receiver):
        client, _ = receiver
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok" and body["auth_enabled"] is True

    def test_hash_only_at_rest(self):
        auth = APIKeyAuth("k1")
        assert hash_key("k1") in auth._hashes
        assert "k1" not in str(auth._hashes)


class TestDisabledAuth:
    def test_unset_keys_means_open_receiver(self, tmp_path, caplog):
        app = create_app(buffer_dir=tmp_path / "buf", api_keys=None)
        client = TestClient(app)
        assert otlp_post(client).status_code == 200  # no key needed
        assert client.get("/health").json()["auth_enabled"] is False

    def test_reads_forensiq_api_keys_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FORENSIQ_API_KEYS", "env-key")
        app = create_app(buffer_dir=tmp_path / "buf")
        client = TestClient(app)
        assert otlp_post(client).status_code == 401
        assert otlp_post(client, key="env-key").status_code == 200


class TestIngestAndBuffering:
    def test_traces_written_as_jsonl(self, receiver):
        client, buf = receiver
        otlp_post(client, key="secret-key-1")
        lines = (buf / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["trace_id"] == "5b8efff798038103d269b633813fc60c"
        assert record["spans"][0]["stage"] == "embed"

    def test_written_jsonl_reloads_through_jsonl_loader(self, receiver):
        from forensiq.ingest.loaders import JSONLTraceLoader

        client, buf = receiver
        otlp_post(client, key="secret-key-1")
        traces = JSONLTraceLoader().load(buf / "traces.jsonl")
        assert len(traces) == 1
        assert len(traces[0].spans) == 3

    def test_health_counts_traces(self, receiver):
        client, _ = receiver
        otlp_post(client, key="secret-key-1")
        assert client.get("/health").json()["traces_received"] == 1

    def test_both_endpoint_aliases(self, receiver):
        client, buf = receiver
        assert otlp_post(client, key="secret-key-1", path="/v1/otlp/v1/traces").status_code == 200
        assert otlp_post(client, key="secret-key-1", path="/v1/traces").status_code == 200
        lines = (buf / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_invalid_json_body_is_400(self, receiver):
        client, _ = receiver
        resp = client.post(
            "/v1/traces",
            content=b"{not json",
            headers={"X-ForensiQ-Key": "secret-key-1", "Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert "not valid JSON" in resp.json()["error"]

    def test_rotation_on_size_cap(self, tmp_path):
        app = create_app(buffer_dir=tmp_path / "buf", api_keys="k", max_bytes=3000)
        client = TestClient(app)
        for _ in range(5):
            assert otlp_post(client, key="k").status_code == 200
        active = tmp_path / "buf" / "traces.jsonl"
        rotated = sorted((tmp_path / "buf").glob("traces.*.jsonl"))
        assert active.stat().st_size <= 3000
        assert rotated, "at least one rotated shard must exist"
        total_lines = sum(
            len(p.read_text(encoding="utf-8").strip().splitlines())
            for p in [*rotated, active]
            if p.exists() and p.read_text(encoding="utf-8").strip()
        )
        assert total_lines == 5

    def test_malformed_spans_still_accepted_with_partial_success(self, receiver):
        client, buf = receiver
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {"spanId": "no-trace-id"},
                                {
                                    "traceId": "5b8efff798038103d269b633813fc60c",
                                    "spanId": "eee19b7ec3c1b174",
                                    "name": "embed",
                                    "startTimeUnixNano": "1544712660000000000",
                                },
                            ]
                        }
                    ]
                }
            ]
        }
        resp = client.post("/v1/traces", json=body, headers={"X-ForensiQ-Key": "secret-key-1"})
        assert resp.status_code == 200  # tolerant: one good span landed
        lines = (buf / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert len(json.loads(lines[0])["spans"]) == 1
