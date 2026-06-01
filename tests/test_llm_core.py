"""Characterization tests for src/llm_core.py.

These pin the CURRENT behavior of the de-facto OpenAI/Anthropic transports and the
llm_core execution engine (httpx pool, cache, per-host dead-host cooldown, retry,
fallback) BEFORE the provider-adapter refactor (PR1b/1c) extracts them. Any change
to a captured request dict / parsed output / SSE chunk = a regression, not an
improvement.

Strategy:
  * Import the REAL `src.llm_core` (it imports only httpx/fastapi/stdlib, so it is
    immune to the `sys.modules` mocking other test files install for src.database /
    src.endpoint_resolver — verified: llm_core imports none of those).
  * Mock the HTTP layer with `respx` (intercepts both the module-level sync
    `httpx.post` and the shared async client created by `_get_http_client()`).
  * Assert on normalized request *dicts* (json.loads of the request body),
    headers, URL, and `request.extensions["timeout"]` — never byte-exact JSON.
"""
import json

import httpx
import pytest
import respx
from fastapi import HTTPException

import src.llm_core as llm_core


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


@pytest.fixture(autouse=True)
def _reset_llm_core_state():
    """Each test starts from a clean cache / dead-host / activity state.

    The shared async client is intentionally left in place — respx patches the
    transport at the class level, so the persistent pool is still intercepted.
    """
    llm_core._response_cache.clear()
    llm_core._dead_hosts.clear()
    llm_core._host_fails.clear()
    llm_core._model_activity.clear()
    yield
    llm_core._response_cache.clear()
    llm_core._dead_hosts.clear()
    llm_core._host_fails.clear()
    llm_core._model_activity.clear()


# ── SSE helpers ──────────────────────────────────────────────────────────────

def _sse(*events: str) -> bytes:
    """Join raw SSE event payloads (each already a full `data: ...` / `event:`
    line, without the trailing blank line) into a byte stream."""
    return ("".join(e + "\n\n" for e in events)).encode()


def _data(chunk: str) -> dict:
    """Parse the JSON object out of a `data: {...}\\n\\n` SSE chunk."""
    assert chunk.startswith("data: "), chunk
    body = chunk[len("data: "):].strip()
    return json.loads(body)


def _is_done(chunk: str) -> bool:
    return chunk.strip() == "data: [DONE]"


async def _collect(agen):
    return [c async for c in agen]


# ══════════════════════════════════════════════════════════════════════════════
# Pure helpers (no I/O) — cheap, robust, pin the transport-shaping primitives.
# ══════════════════════════════════════════════════════════════════════════════

class TestProviderDetection:
    def test_anthropic_by_host(self):
        assert llm_core._detect_provider("https://api.anthropic.com/v1") == "anthropic"

    def test_openai_default(self):
        assert llm_core._detect_provider("https://api.openai.com/v1") == "openai"
        assert llm_core._detect_provider("http://localhost:11434/v1") == "openai"

    def test_none_safe(self):
        assert llm_core._detect_provider(None) == "openai"

    @pytest.mark.parametrize("url,label", [
        ("https://api.anthropic.com/v1", "Anthropic"),
        ("https://api.openai.com/v1", "OpenAI"),
        ("https://api.x.ai/v1", "xAI"),
        ("https://openrouter.ai/api/v1", "OpenRouter"),
        ("https://api.groq.com/openai/v1", "Groq"),
        ("https://api.deepseek.com/v1", "DeepSeek"),
        ("http://localhost:11434/v1", "local endpoint"),
        ("http://127.0.0.1:8000/v1", "local endpoint"),
    ])
    def test_provider_label(self, url, label):
        assert llm_core._provider_label(url) == label

    def test_provider_label_unknown_falls_back_to_host(self):
        assert llm_core._provider_label("https://example.org/v1") == "example.org"


class TestMaxCompletionTokens:
    @pytest.mark.parametrize("model", ["o1", "o1-mini", "o3", "o4-mini", "gpt-4.5", "gpt-5", "gpt-5-mini"])
    def test_uses_max_completion_tokens(self, model):
        assert llm_core._uses_max_completion_tokens(model) is True

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4-turbo", "claude-sonnet-4", "llama3", ""])
    def test_uses_plain_max_tokens(self, model):
        assert llm_core._uses_max_completion_tokens(model) is False

    def test_path_prefixed_model(self):
        # `provider/o3` style routing names still trip the o3 rule via the `/o3` check.
        assert llm_core._uses_max_completion_tokens("openai/gpt-5") is True


class TestThinkingSupport:
    @pytest.mark.parametrize("model", ["qwen3-32b", "qwq-32b", "deepseek-r1", "deepseek-reasoner", "minimax-m1", "m2-reap"])
    def test_supports_thinking(self, model):
        assert llm_core._supports_thinking(model) is True

    @pytest.mark.parametrize("model", ["gpt-4o", "claude-sonnet-4", "llama3", ""])
    def test_no_thinking(self, model):
        assert llm_core._supports_thinking(model) is False


class TestNormalizeAnthropicUrl:
    def test_bare_host(self):
        assert llm_core._normalize_anthropic_url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"

    def test_v1_suffix(self):
        assert llm_core._normalize_anthropic_url("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/messages"

    def test_already_messages(self):
        assert llm_core._normalize_anthropic_url("https://api.anthropic.com/v1/messages") == "https://api.anthropic.com/v1/messages"

    def test_trailing_slash(self):
        assert llm_core._normalize_anthropic_url("https://api.anthropic.com/v1/") == "https://api.anthropic.com/v1/messages"


class TestBuildAnthropicHeaders:
    def test_bearer_converted_to_x_api_key(self):
        h = llm_core._build_anthropic_headers({"Authorization": "Bearer sk-ant-123"})
        assert h == {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "sk-ant-123",
        }

    def test_non_bearer_headers_passed_through(self):
        h = llm_core._build_anthropic_headers({"X-Custom": "v"})
        assert h["X-Custom"] == "v"
        assert h["anthropic-version"] == "2023-06-01"

    def test_no_headers(self):
        assert llm_core._build_anthropic_headers(None) == {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }


class TestParseAnthropicResponse:
    def test_first_text_block(self):
        data = {"content": [{"type": "text", "text": "hello"}]}
        assert llm_core._parse_anthropic_response(data) == "hello"

    def test_skips_non_text_blocks(self):
        data = {"content": [{"type": "tool_use", "id": "x"}, {"type": "text", "text": "after"}]}
        assert llm_core._parse_anthropic_response(data) == "after"

    def test_empty(self):
        assert llm_core._parse_anthropic_response({"content": []}) == ""


class TestConvertOpenAIContentToAnthropic:
    def test_passthrough_string(self):
        assert llm_core._convert_openai_content_to_anthropic("plain") == "plain"

    def test_text_block_passthrough(self):
        blocks = [{"type": "text", "text": "hi"}]
        assert llm_core._convert_openai_content_to_anthropic(blocks) == blocks

    def test_data_uri_image(self):
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC123"}}]
        out = llm_core._convert_openai_content_to_anthropic(blocks)
        assert out == [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "ABC123"},
        }]

    def test_external_url_image(self):
        blocks = [{"type": "image_url", "image_url": {"url": "https://x.com/a.png"}}]
        out = llm_core._convert_openai_content_to_anthropic(blocks)
        assert out == [{"type": "image", "source": {"type": "url", "url": "https://x.com/a.png"}}]


class TestCacheKey:
    def test_deterministic(self):
        a = llm_core._get_cache_key(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}], 1.0, 0)
        b = llm_core._get_cache_key(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}], 1.0, 0)
        assert a == b

    def test_sensitive_to_inputs(self):
        base = llm_core._get_cache_key(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}], 1.0, 0)
        assert base != llm_core._get_cache_key(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}], 0.5, 0)
        assert base != llm_core._get_cache_key(OPENAI_URL, "gpt-4o-mini", [{"role": "user", "content": "hi"}], 1.0, 0)
        assert base != llm_core._get_cache_key(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "yo"}], 1.0, 0)


class TestDeadHostHelpers:
    def test_host_key_scheme_netloc(self):
        assert llm_core._host_key("https://api.openai.com/v1/chat/completions") == "https://api.openai.com"
        assert llm_core._host_key("http://localhost:11434/v1/chat/completions") == "http://localhost:11434"

    def test_threshold_then_cool(self):
        url = "http://host-a:1234/v1"
        assert llm_core._mark_host_dead(url) is False   # 1st failure: grace
        assert llm_core._is_host_dead(url) is False
        assert llm_core._mark_host_dead(url) is True     # 2nd failure: cooled
        assert llm_core._is_host_dead(url) is True

    def test_clear_resets(self):
        url = "http://host-b:1234/v1"
        llm_core._mark_host_dead(url)
        llm_core._mark_host_dead(url)
        assert llm_core._is_host_dead(url) is True
        llm_core._clear_host_dead(url)
        assert llm_core._is_host_dead(url) is False
        # counter reset too: a single fail no longer trips it
        assert llm_core._mark_host_dead(url) is False

    def test_expired_cooldown_clears_on_check(self):
        url = "http://host-c:1234/v1"
        key = llm_core._host_key(url)
        llm_core._dead_hosts[key] = 1.0   # far in the past
        assert llm_core._is_host_dead(url) is False
        assert key not in llm_core._dead_hosts


class TestFormatUpstreamError:
    def test_401_rejected_key(self):
        msg = llm_core._format_upstream_error(401, '{"error":{"message":"bad key"}}', OPENAI_URL)
        assert "OpenAI rejected the API key" in msg
        assert "bad key" in msg

    def test_403_denied(self):
        assert "denied access (403)" in llm_core._format_upstream_error(403, "", OPENAI_URL)

    def test_404(self):
        assert "returned 404" in llm_core._format_upstream_error(404, "", ANTHROPIC_URL)

    def test_429(self):
        assert "rate-limited" in llm_core._format_upstream_error(429, "", OPENAI_URL)

    def test_5xx_outage(self):
        assert "is having an outage (HTTP 503)" in llm_core._format_upstream_error(503, "", OPENAI_URL)


# ══════════════════════════════════════════════════════════════════════════════
# Sync llm_call
# ══════════════════════════════════════════════════════════════════════════════

class TestLlmCallSyncOpenAI:
    @respx.mock
    def test_request_shape_and_parse(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})
        )
        out = llm_core.llm_call(
            OPENAI_URL, "gpt-4o",
            [{"role": "user", "content": "hello"}],
            headers={"Authorization": "Bearer sk-x"},
        )
        assert out == "answer"
        req = route.calls.last.request
        assert str(req.url) == OPENAI_URL
        assert json.loads(req.content) == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 1.0,
        }
        assert req.headers["content-type"] == "application/json"
        assert req.headers["authorization"] == "Bearer sk-x"
        # default sync timeout is 30s across the board
        assert req.extensions["timeout"] == {"connect": 30, "read": 30, "write": 30, "pool": 30}

    @respx.mock
    def test_system_consolidation_and_max_tokens(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
        )
        llm_core.llm_call(
            OPENAI_URL, "gpt-4o",
            [
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=100,
        )
        assert json.loads(route.calls.last.request.content) == {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "A\n\nB"},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 1.0,
            "max_tokens": 100,
        }

    @respx.mock
    def test_max_completion_tokens_for_gpt5(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
        )
        llm_core.llm_call(OPENAI_URL, "gpt-5", [{"role": "user", "content": "hi"}], max_tokens=50)
        body = json.loads(route.calls.last.request.content)
        assert body["max_completion_tokens"] == 50
        assert "max_tokens" not in body

    @respx.mock
    def test_headers_as_json_string_tolerated(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
        )
        llm_core.llm_call(
            OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}],
            headers='{"Authorization": "Bearer sk-json"}',
        )
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-json"

    @respx.mock
    def test_upstream_non_2xx_raises_502(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(HTTPException) as ei:
            llm_core.llm_call(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert ei.value.status_code == 502

    @respx.mock
    def test_connect_error_raises_502(self):
        respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(HTTPException) as ei:
            llm_core.llm_call(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert ei.value.status_code == 502

    @respx.mock
    def test_unexpected_schema_raises_502(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"nope": True}))
        with pytest.raises(HTTPException) as ei:
            llm_core.llm_call(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert ei.value.status_code == 502

    @respx.mock
    def test_response_is_cached(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "cached!"}}]})
        )
        args = (OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert llm_core.llm_call(*args) == "cached!"
        assert llm_core.llm_call(*args) == "cached!"
        assert route.call_count == 1   # second call served from cache


class TestLlmCallSyncAnthropic:
    @respx.mock
    def test_request_shape_and_parse(self):
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json={"content": [{"type": "text", "text": "hi there"}]})
        )
        out = llm_core.llm_call(
            ANTHROPIC_URL, "claude-sonnet-4",
            [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "hello"}],
            headers={"Authorization": "Bearer sk-ant-1"},
        )
        assert out == "hi there"
        req = route.calls.last.request
        assert str(req.url) == ANTHROPIC_URL
        assert json.loads(req.content) == {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4096,
            "temperature": 1.0,
            "system": "Be brief",
        }
        assert req.headers["x-api-key"] == "sk-ant-1"
        assert req.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in req.headers

    @respx.mock
    def test_explicit_max_tokens_overrides_default(self):
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json={"content": [{"type": "text", "text": "x"}]})
        )
        llm_core.llm_call(ANTHROPIC_URL, "claude-sonnet-4", [{"role": "user", "content": "hi"}], max_tokens=256)
        assert json.loads(route.calls.last.request.content)["max_tokens"] == 256

    @respx.mock
    def test_url_normalized_from_bare_host(self):
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json={"content": [{"type": "text", "text": "x"}]})
        )
        llm_core.llm_call("https://api.anthropic.com", "claude-sonnet-4", [{"role": "user", "content": "hi"}])
        assert str(route.calls.last.request.url) == ANTHROPIC_URL


# ══════════════════════════════════════════════════════════════════════════════
# Async llm_call_async
# ══════════════════════════════════════════════════════════════════════════════

class TestLlmCallAsyncOpenAI:
    @respx.mock
    async def test_request_shape_timeout_and_parse(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        out = await llm_core.llm_call_async(
            OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}],
            headers={"Authorization": "Bearer sk-x"},
        )
        assert out == "ok"
        req = route.calls.last.request
        assert json.loads(req.content) == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.0,
        }
        # async default timeout asymmetry: short connect, long read
        assert req.extensions["timeout"] == {"connect": 3.0, "read": 300.0, "write": 10.0, "pool": 5.0}

    @respx.mock
    async def test_non_2xx_uses_friendly_status(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "nope"}}))
        with pytest.raises(HTTPException) as ei:
            await llm_core.llm_call_async(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert ei.value.status_code == 401
        assert "rejected the API key" in ei.value.detail

    @respx.mock
    async def test_retries_then_fails_on_transport_error(self, monkeypatch):
        monkeypatch.setattr(llm_core.LLMConfig, "RETRY_DELAY", 0)
        route = respx.post(OPENAI_URL).mock(side_effect=httpx.ReadError("flaky"))
        with pytest.raises(HTTPException) as ei:
            await llm_core.llm_call_async(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert ei.value.status_code == 502
        assert route.call_count == llm_core.LLMConfig.MAX_RETRIES   # 3 attempts

    @respx.mock
    async def test_dead_host_cooldown_sequence(self):
        route = respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
        # 1st connect failure: grace (not yet cooled)
        with pytest.raises(HTTPException) as e1:
            await llm_core.llm_call_async(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert e1.value.status_code == 503
        assert not llm_core._is_host_dead(OPENAI_URL)
        # 2nd connect failure: host gets cooled
        with pytest.raises(HTTPException):
            await llm_core.llm_call_async(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert llm_core._is_host_dead(OPENAI_URL)
        # 3rd call: short-circuits BEFORE hitting the network
        with pytest.raises(HTTPException) as e3:
            await llm_core.llm_call_async(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert e3.value.status_code == 503
        assert "unreachable" in e3.value.detail
        assert route.call_count == 2   # 3rd never reached the transport

    @respx.mock
    async def test_success_clears_dead_host_counter(self):
        # one connect failure leaves a fail-count; a subsequent success clears it
        llm_core._mark_host_dead(OPENAI_URL)   # n=1, not cooled
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        await llm_core.llm_call_async(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}])
        assert llm_core._host_key(OPENAI_URL) not in llm_core._host_fails


class TestLlmCallAsyncAnthropic:
    @respx.mock
    async def test_request_shape(self):
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json={"content": [{"type": "text", "text": "yo"}]})
        )
        out = await llm_core.llm_call_async(
            ANTHROPIC_URL, "claude-sonnet-4", [{"role": "user", "content": "hi"}],
            headers={"Authorization": "Bearer sk-ant-1"},
        )
        assert out == "yo"
        req = route.calls.last.request
        assert json.loads(req.content) == {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "temperature": 1.0,
        }
        assert req.headers["x-api-key"] == "sk-ant-1"


# ══════════════════════════════════════════════════════════════════════════════
# Streaming — OpenAI-compatible
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamOpenAI:
    @respx.mock
    async def test_payload_shape_and_timeout(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, content=_sse('data: [DONE]'),
                                        headers={"content-type": "text/event-stream"})
        )
        await _collect(llm_core.stream_llm(
            OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}],
            headers={"Authorization": "Bearer sk-x"},
        ))
        req = route.calls.last.request
        assert json.loads(req.content) == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert req.extensions["timeout"] == {"connect": 3.0, "read": 300.0, "write": 30.0, "pool": 5.0}

    @respx.mock
    async def test_content_deltas_and_done(self):
        sse = _sse(
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: [DONE]',
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}]))
        assert _data(chunks[0]) == {"delta": "Hel"}
        assert _data(chunks[1]) == {"delta": "lo"}
        assert _is_done(chunks[-1])

    @respx.mock
    async def test_usage_chunk(self):
        sse = _sse(
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}',
            'data: [DONE]',
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}]))
        assert _data(chunks[0]) == {"type": "usage", "data": {"input_tokens": 10, "output_tokens": 20}}

    @respx.mock
    async def test_tool_call_accumulation(self):
        sse = _sse(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\\"loc\\""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"NYC\\"}"}}]}}]}',
            'data: [DONE]',
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}]))
        # tool_calls emitted just before [DONE]
        tool_evt = _data(chunks[-2])
        assert tool_evt == {
            "type": "tool_calls",
            "calls": [{"id": "call_1", "name": "get_weather", "arguments": '{"loc":"NYC"}'}],
        }
        assert _is_done(chunks[-1])

    @respx.mock
    async def test_doc_tool_streams_arg_deltas(self):
        sse = _sse(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"create_document","arguments":"{\\"t\\":1}"}}]}}]}',
            'data: [DONE]',
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}]))
        delta_evt = _data(chunks[0])
        assert delta_evt["type"] == "tool_call_delta"
        assert delta_evt["name"] == "create_document"
        assert delta_evt["arg_delta"] == '{"t":1}'

    @respx.mock
    async def test_thinking_model_repairs_stray_close_tag(self):
        sse = _sse('data: {"choices":[{"delta":{"content":"</think>done"}}]}', 'data: [DONE]')
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "qwen3-32b", [{"role": "user", "content": "hi"}]))
        assert _data(chunks[0]) == {"delta": "<think></think>done"}

    @respx.mock
    async def test_reasoning_content_marked_thinking(self):
        sse = _sse('data: {"choices":[{"delta":{"reasoning_content":"pondering"}}]}', 'data: [DONE]')
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "deepseek-r1", [{"role": "user", "content": "hi"}]))
        assert _data(chunks[0]) == {"delta": "pondering", "thinking": True}

    @respx.mock
    async def test_non_200_yields_error_event(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, text="upstream boom"))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}]))
        assert chunks[0].startswith("event: error")
        payload = json.loads(chunks[0].split("data: ", 1)[1])
        assert payload["status"] == 500

    @respx.mock
    async def test_connect_error_yields_503_error_event(self):
        respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
        chunks = await _collect(llm_core.stream_llm(OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}]))
        assert chunks[0].startswith("event: error")
        payload = json.loads(chunks[0].split("data: ", 1)[1])
        assert payload["status"] == 503

    @respx.mock
    async def test_tools_included_in_payload(self):
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=_sse('data: [DONE]')))
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        await _collect(llm_core.stream_llm(
            OPENAI_URL, "gpt-4o", [{"role": "user", "content": "hi"}], tools=tools,
        ))
        assert json.loads(route.calls.last.request.content)["tools"] == tools


# ══════════════════════════════════════════════════════════════════════════════
# Streaming — Anthropic
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamAnthropic:
    @respx.mock
    async def test_payload_marks_stream(self):
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, content=_sse('data: {"type":"message_stop"}'))
        )
        await _collect(llm_core.stream_llm(
            ANTHROPIC_URL, "claude-sonnet-4", [{"role": "user", "content": "hi"}],
            headers={"Authorization": "Bearer sk-ant-1"},
        ))
        req = route.calls.last.request
        body = json.loads(req.content)
        assert body["stream"] is True
        assert body["model"] == "claude-sonnet-4"
        assert req.headers["x-api-key"] == "sk-ant-1"

    @respx.mock
    async def test_text_deltas_usage_and_done(self):
        sse = _sse(
            'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":3}}',
            'data: {"type":"message_stop"}',
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(ANTHROPIC_URL, "claude-sonnet-4", [{"role": "user", "content": "hi"}]))
        assert _data(chunks[0]) == {"delta": "Hi"}
        assert _data(chunks[1]) == {"type": "usage", "data": {"input_tokens": 5, "output_tokens": 3}}
        assert _is_done(chunks[-1])

    @respx.mock
    async def test_tool_use_accumulation(self):
        sse = _sse(
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"loc\\""}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\\"NYC\\"}"}}',
            'data: {"type":"message_stop"}',
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(ANTHROPIC_URL, "claude-sonnet-4", [{"role": "user", "content": "hi"}]))
        tool_evt = _data(chunks[-2])
        assert tool_evt == {
            "type": "tool_calls",
            "calls": [{"id": "toolu_1", "name": "get_weather", "arguments": '{"loc":"NYC"}'}],
        }
        assert _is_done(chunks[-1])

    @respx.mock
    async def test_error_event(self):
        sse = _sse('data: {"type":"error","error":{"message":"overloaded"}}')
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm(ANTHROPIC_URL, "claude-sonnet-4", [{"role": "user", "content": "hi"}]))
        assert chunks[0].startswith("event: error")
        payload = json.loads(chunks[0].split("data: ", 1)[1])
        assert payload["error"] == "overloaded"
        assert payload["status"] == 400


# ══════════════════════════════════════════════════════════════════════════════
# Fallback wrappers
# ══════════════════════════════════════════════════════════════════════════════

class TestSyncFallback:
    @respx.mock
    def test_first_candidate_wins(self):
        r0 = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "primary"}}]})
        )
        out = llm_core.llm_call_with_fallback(
            [(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer a"}),
             ("https://api.openai.com/v2/chat/completions", "gpt-4o-mini", {})],
            [{"role": "user", "content": "hi"}],
        )
        assert out == "primary"
        assert r0.call_count == 1

    @respx.mock
    def test_falls_through_on_failure(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, text="down"))
        backup = "https://backup.example.com/v1/chat/completions"
        respx.post(backup).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "backup"}}]})
        )
        out = llm_core.llm_call_with_fallback(
            [(OPENAI_URL, "gpt-4o", {}), (backup, "m2", {})],
            [{"role": "user", "content": "hi"}],
        )
        assert out == "backup"

    def test_no_candidates_raises_503(self):
        with pytest.raises(HTTPException) as ei:
            llm_core.llm_call_with_fallback([], [{"role": "user", "content": "hi"}])
        assert ei.value.status_code == 503

    @respx.mock
    def test_all_fail_raises_last_error(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, text="down"))
        with pytest.raises(HTTPException):
            llm_core.llm_call_with_fallback(
                [(OPENAI_URL, "gpt-4o", {})], [{"role": "user", "content": "hi"}],
            )


class TestAsyncFallback:
    @respx.mock
    async def test_falls_through_on_failure(self):
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, text="down"))
        backup = "https://backup.example.com/v1/chat/completions"
        respx.post(backup).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "backup"}}]})
        )
        out = await llm_core.llm_call_async_with_fallback(
            [(OPENAI_URL, "gpt-4o", {}), (backup, "m2", {})],
            [{"role": "user", "content": "hi"}],
        )
        assert out == "backup"


class TestStreamFallback:
    @respx.mock
    async def test_pre_content_switch(self):
        # primary fails BEFORE any output -> swallow + switch to backup
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, text="down"))
        backup = "https://backup.example.com/v1/chat/completions"
        sse = _sse('data: {"choices":[{"delta":{"content":"from-backup"}}]}', 'data: [DONE]')
        respx.post(backup).mock(return_value=httpx.Response(200, content=sse))
        chunks = await _collect(llm_core.stream_llm_with_fallback(
            [(OPENAI_URL, "gpt-4o", {}), (backup, "m2", {})],
            [{"role": "user", "content": "hi"}],
        ))
        # no error chunk surfaced; backup content delivered
        assert not any(c.startswith("event: error") for c in chunks)
        assert _data(chunks[0]) == {"delta": "from-backup"}
        assert _is_done(chunks[-1])

    @respx.mock
    async def test_no_switch_after_content(self):
        # primary emits content THEN errors -> error passes through, no switch
        sse = _sse(
            'data: {"choices":[{"delta":{"content":"partial"}}]}',
        )
        # follow the content with a mid-stream provider error event by returning 200
        # then a content_block error is awkward for openai; instead emit content and
        # let the stream end -> then a SECOND candidate must NOT be tried.
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        backup = "https://backup.example.com/v1/chat/completions"
        backup_route = respx.post(backup).mock(
            return_value=httpx.Response(200, content=_sse('data: {"choices":[{"delta":{"content":"SHOULD-NOT-APPEAR"}}]}', 'data: [DONE]'))
        )
        chunks = await _collect(llm_core.stream_llm_with_fallback(
            [(OPENAI_URL, "gpt-4o", {}), (backup, "m2", {})],
            [{"role": "user", "content": "hi"}],
        ))
        assert _data(chunks[0]) == {"delta": "partial"}
        assert backup_route.call_count == 0   # never switched after real output
        assert not any("SHOULD-NOT-APPEAR" in c for c in chunks)

    async def test_no_candidates_yields_error(self):
        chunks = await _collect(llm_core.stream_llm_with_fallback([], [{"role": "user", "content": "hi"}]))
        assert chunks[0].startswith("event: error")
        payload = json.loads(chunks[0].split("data: ", 1)[1])
        assert payload["status"] == 503
