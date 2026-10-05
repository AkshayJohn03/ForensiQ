"""OTLP/JSON ingestion: fixtures, id/timestamp handling, stage mapping,
tolerant skipping. All offline."""

from __future__ import annotations

import json

import pytest

from forensiq.ingest.loaders import IngestError
from forensiq.ingest.otlp import parse_otlp, parse_otlp_json
from forensiq.ingest.schema import Trace

TRACE_ID = "5b8efff798038103d269b633813fc60c"
SPAN_ROOT = "eee19b7ec3c1b174"
SPAN_CHILD = "aaa19b7ec3c1b174"

# A realistic OTLP/JSON ExportTraceServiceRequest: KV-list attributes on the
# root span, map-form attributes on the hybrid span, gen_ai.* conventions on
# the generate span, an unknown-name span, one malformed span, one status=error.
FIXTURE: dict = {
    "resourceSpans": [
        {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "rag_showcase"}},
                    {"key": "deployment.environment", "value": {"stringValue": "prod"}},
                ]
            },
            "scopeSpans": [
                {
                    "scope": {"name": "forensiq.test", "version": "1.0"},
                    "spans": [
                        {
                            "traceId": TRACE_ID.upper(),
                            "spanId": SPAN_ROOT.upper(),
                            "name": "embed",
                            "kind": 1,
                            "startTimeUnixNano": "1544712660000000000",
                            "endTimeUnixNano": "1544712661000000000",
                            "attributes": [
                                {"key": "model", "value": {"stringValue": "MiniLM-L6"}},
                                {"key": "token_count", "value": {"intValue": "42"}},
                            ],
                        },
                        {
                            "traceId": TRACE_ID,
                            "spanId": SPAN_CHILD,
                            "parentSpanId": SPAN_ROOT.upper(),
                            "name": "hybrid",
                            "startTimeUnixNano": "1544712661000000000",
                            "endTimeUnixNano": "1544712661044000000",
                            # map-form attributes (some exporters emit this)
                            "attributes": {
                                "query": {"stringValue": "SAF-114 maintenance interval"},
                                "top_k": {"intValue": 5},
                                "hit_scores": {
                                    "arrayValue": {
                                        "values": [{"doubleValue": 0.81}, {"doubleValue": 0.44}]
                                    }
                                },
                            },
                        },
                        {
                            "traceId": TRACE_ID,
                            "spanId": "bbb19b7ec3c1b174",
                            "parentSpanId": SPAN_CHILD,
                            "name": "opaque_operation",
                            "startTimeUnixNano": "1544712661100000000",
                            "durationNanos": "900000000",  # duration instead of end time
                            "attributes": [
                                {
                                    "key": "gen_ai.usage.input_tokens",
                                    "value": {"intValue": "1204"},
                                },
                                {
                                    "key": "gen_ai.usage.output_tokens",
                                    "value": {"intValue": "210"},
                                },
                                {
                                    "key": "gen_ai.request.model",
                                    "value": {"stringValue": "qwen3:1.7b"},
                                },
                                {
                                    "key": "gen_ai.response.finish_reasons",
                                    "value": {"arrayValue": {"values": [{"stringValue": "stop"}]}},
                                },
                                {"key": "gen_ai.completion", "value": {"stringValue": "the answer"}},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
}


@pytest.fixture()
def parsed():
    return parse_otlp(FIXTURE)


class TestParsing:
    def test_single_trace_grouped_from_mixed_case_ids(self, parsed):
        # uppercase and lowercase hex of the same trace id must group into ONE trace
        assert len(parsed.traces) == 1
        assert parsed.traces[0].trace_id == TRACE_ID

    def test_hex_ids_canonicalized_lowercase(self, parsed):
        spans = parsed.traces[0].spans
        assert spans[0].span_id == SPAN_ROOT
        assert spans[0].span_id == spans[0].span_id.lower()

    def test_parent_mapping_and_root(self, parsed):
        spans = parsed.traces[0].spans
        assert spans[0].parent_id is None  # missing parentSpanId -> root
        assert spans[1].parent_id == SPAN_ROOT  # uppercase parent hex normalized
        assert spans[2].parent_id == SPAN_CHILD

    def test_nanosecond_timestamps_and_duration(self, parsed):
        trace = parsed.traces[0]
        assert trace.ts.isoformat() == "2018-12-13T14:51:00+00:00"
        assert trace.spans[0].duration_ms == pytest.approx(1000.0)
        assert trace.spans[1].duration_ms == pytest.approx(44.0)

    def test_duration_nanos_fallback(self, parsed):
        # third span has durationNanos instead of endTimeUnixNano
        assert parsed.traces[0].spans[2].duration_ms == pytest.approx(900.0)

    def test_kv_list_and_map_attribute_shapes(self, parsed):
        attrs0 = parsed.traces[0].spans[0].attrs
        assert attrs0["model"] == "MiniLM-L6"
        assert attrs0["token_count"] == 42  # string-encoded int -> int
        attrs1 = parsed.traces[0].spans[1].attrs
        assert attrs1["hit_scores"] == [0.81, 0.44]
        assert attrs1["top_k"] == 5

    def test_pipeline_name_from_resource(self, parsed):
        assert parsed.traces[0].pipeline_name == "rag_showcase"
        assert parsed.traces[0].meta["source"] == "otlp"
        assert parsed.traces[0].meta["resource"]["deployment.environment"] == "prod"

    def test_chronological_span_order(self, parsed):
        spans = parsed.traces[0].spans
        assert [s.name for s in spans] == ["embed", "hybrid", "opaque_operation"]


class TestStageMapping:
    def test_known_alias_names(self, parsed):
        spans = parsed.traces[0].spans
        assert spans[0].stage == "embed"
        assert spans[1].stage == "retrieve"  # hybrid -> retrieve

    def test_genai_attrs_promote_opaque_name_to_generate(self, parsed):
        # name is unrecognizable, but gen_ai.usage/prompt attrs mark an LLM call
        assert parsed.traces[0].spans[2].stage == "generate"

    def test_unknown_stage_falls_back_to_tool(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": TRACE_ID,
                                    "spanId": SPAN_ROOT,
                                    "name": "weird_custom_op",  # no pattern, no hint attrs
                                    "startTimeUnixNano": "1544712660000000000",
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        assert parse_otlp_json(body)[0].spans[0].stage == "tool"

    def test_openinference_span_kind(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": TRACE_ID,
                                    "spanId": SPAN_ROOT,
                                    "name": "my_chain_step",
                                    "startTimeUnixNano": "1544712660000000000",
                                    "attributes": [
                                        {
                                            "key": "openinference.span.kind",
                                            "value": {"stringValue": "RETRIEVER"},
                                        }
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        trace = parse_otlp_json(body)[0]
        assert trace.spans[0].stage == "retrieve"

    def test_genai_attributes_drive_generate(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": TRACE_ID,
                                    "spanId": SPAN_ROOT,
                                    "name": "invoke_model_xyz",  # no recognizable name pattern
                                    "startTimeUnixNano": "1544712660000000000",
                                    "attributes": [
                                        {"key": "gen_ai.prompt", "value": {"stringValue": "hi"}}
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        trace = parse_otlp_json(body)[0]
        assert trace.spans[0].stage == "generate"

    def test_vector_db_client_name_maps_to_retrieve(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": TRACE_ID,
                                    "spanId": SPAN_ROOT,
                                    "name": "qdrant.query",
                                    "startTimeUnixNano": "1544712660000000000",
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        trace = parse_otlp_json(body)[0]
        assert trace.spans[0].stage == "retrieve"


class TestCanonicalAttrOverlay:
    def test_genai_usage_to_span_fields(self, parsed):
        span = parsed.traces[0].spans[2]
        assert span.tokens_in == 1204
        assert span.tokens_out == 210
        assert span.attrs["model"] == "qwen3:1.7b"
        assert span.attrs["finish_reason"] == "stop"
        assert span.attrs["output"] == "the answer"

    def test_retrieval_documents_mapping(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": TRACE_ID,
                                    "spanId": SPAN_ROOT,
                                    "name": "retrieve",
                                    "startTimeUnixNano": "1544712660000000000",
                                    "attributes": [
                                        {
                                            "key": "retrieval.documents",
                                            "value": {
                                                "arrayValue": {
                                                    "values": [
                                                        {
                                                            "kvlistValue": {
                                                                "values": [
                                                                    {
                                                                        "key": "document.score",
                                                                        "value": {"doubleValue": 0.7},
                                                                    },
                                                                    {
                                                                        "key": "document.content",
                                                                        "value": {"stringValue": "chunk A"},
                                                                    },
                                                                ]
                                                            }
                                                        },
                                                        {
                                                            "kvlistValue": {
                                                                "values": [
                                                                    {
                                                                        "key": "document.score",
                                                                        "value": {"doubleValue": 0.2},
                                                                    },
                                                                    {
                                                                        "key": "document.content",
                                                                        "value": {"stringValue": "chunk B"},
                                                                    },
                                                                ]
                                                            }
                                                        },
                                                    ]
                                                }
                                            },
                                        }
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        span = parse_otlp_json(body)[0].spans[0]
        assert span.attrs["hit_count"] == 2
        assert span.attrs["hit_scores"] == [0.7, 0.2]
        assert "chunk A" in span.attrs["context"]


class TestTolerance:
    def test_malformed_spans_skipped_with_report(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {"spanId": "deadbeefdeadbeef", "name": "no_trace_id"},  # missing traceId
                                {"traceId": "zzzz-not-hex", "spanId": "deadbeefdeadbeef", "name": "bad_trace_id"},
                                {"traceId": TRACE_ID, "name": "no_span_id"},
                                {"traceId": TRACE_ID, "spanId": "0xdeadbeefdeadbeef", "name": "ok_prefixed",
                                 "startTimeUnixNano": "1544712660000000000"},
                                "not even a dict",
                            ]
                        }
                    ]
                }
            ]
        }
        result = parse_otlp(body)
        assert result.skip.skipped_spans == 4
        assert len(result.skip.reasons) == 4
        assert result.skip.reasons[0].startswith("resourceSpans[0]/scopeSpans[0]/spans[0]")
        # the one good span survives (0x-prefixed hex id converted)
        assert len(result.traces) == 1
        assert result.traces[0].spans[0].name == "ok_prefixed"
        assert result.traces[0].spans[0].span_id == "deadbeefdeadbeef"

    def test_malformed_scope_and_resource_counted(self):
        body = {
            "resourceSpans": [
                "not a dict",
                {"scopeSpans": "not a list"},
                {"scopeSpans": [{"spans": "not a list"}]},
            ]
        }
        result = parse_otlp(body)
        assert result.skip.skipped_resources == 2
        assert result.skip.skipped_scopes == 1
        assert result.traces == []

    def test_never_raises_on_partial_data(self):
        garbage = {
            "resourceSpans": [
                {"scopeSpans": [{"spans": [None, 3, {"traceId": TRACE_ID, "spanId": SPAN_ROOT}]}]}
            ]
        }
        result = parse_otlp(garbage)
        assert len(result.traces[0].spans) == 1
        assert result.skip.skipped_spans == 2

    def test_non_json_document_raises(self):
        with pytest.raises(IngestError):
            parse_otlp("{not json")

    def test_zero_parent_and_all_zero_parent_are_roots(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {"traceId": TRACE_ID, "spanId": SPAN_ROOT, "name": "root_zero",
                                 "parentSpanId": "0000000000000000",
                                 "startTimeUnixNano": "1544712660000000000"},
                                {"traceId": TRACE_ID, "spanId": SPAN_CHILD, "name": "root_empty",
                                 "parentSpanId": "", "startTimeUnixNano": "1544712660000000000"},
                            ]
                        }
                    ]
                }
            ]
        }
        spans = parse_otlp_json(body)[0].spans
        assert all(s.parent_id is None for s in spans)

    def test_error_status_and_timeout_names(self):
        body = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {"traceId": TRACE_ID, "spanId": SPAN_ROOT, "name": "generate",
                                 "status": {"code": "STATUS_CODE_ERROR", "message": "boom"},
                                 "startTimeUnixNano": "1544712660000000000"},
                                {"traceId": TRACE_ID, "spanId": SPAN_CHILD, "name": "generate_timeout",
                                 "startTimeUnixNano": "1544712660000000000"},
                            ]
                        }
                    ]
                }
            ]
        }
        spans = parse_otlp_json(body)[0].spans
        assert spans[0].status == "error"
        assert spans[0].attrs["error"] == "boom"
        assert spans[1].status == "timeout"

    def test_json_string_and_bytes_payloads(self):
        text = json.dumps(FIXTURE)
        from_str = parse_otlp_json(text)
        from_bytes = parse_otlp_json(text.encode("utf-8"))
        assert [t.trace_id for t in from_str] == [t.trace_id for t in from_bytes] == [TRACE_ID]

    def test_batch_of_requests(self):
        batch = [FIXTURE, {"resourceSpans": []}]
        assert len(parse_otlp_json(batch)) == 1

    def test_trace_model_roundtrip(self, parsed):
        assert isinstance(parsed.traces[0], Trace)
        dumped = parsed.traces[0].model_dump(mode="json")
        assert Trace.model_validate(dumped).trace_id == TRACE_ID
