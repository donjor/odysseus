"""PR1c tests: EndpointRef, the static credential seam + OAuth guard, and proof
that the optional `ref=` path produces byte-identical requests to the legacy
(url, model, headers) path.
"""
import dataclasses
import json

import httpx
import pytest
import respx
from fastapi import HTTPException

import src.llm_core as llm_core
from src.providers import auth
from src.providers.endpoint_ref import EndpointRef
from src.providers.spec import ANTHROPIC_SPEC, OPENAI_SPEC, ProviderSpec

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# A hypothetical OAuth provider (the shape PR2 adds) — only used to exercise the guard.
OAUTH_SPEC = ProviderSpec(
    id="codex", label="ChatGPT", transport="openai_chat", auth_type="oauth",
    url_matchers=(), default_chat_path="/responses", model_list_mode="static",
)


@pytest.fixture(autouse=True)
def _reset_llm_core_state():
    for d in (llm_core._response_cache, llm_core._dead_hosts, llm_core._host_fails, llm_core._model_activity):
        d.clear()
    yield
    for d in (llm_core._response_cache, llm_core._dead_hosts, llm_core._host_fails, llm_core._model_activity):
        d.clear()


class TestEndpointRef:
    def test_from_legacy_static(self):
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer k"})
        assert ref.url == OPENAI_URL
        assert ref.model == "gpt-4o"
        assert ref.headers == {"Authorization": "Bearer k"}
        assert ref.endpoint_id is None       # legacy tuple carries no identity
        assert ref.owner is None
        assert ref.auth_type == "api_key"     # never OAuth-complete

    def test_from_legacy_minimal(self):
        ref = EndpointRef.from_legacy("http://localhost:11434/v1/chat/completions")
        assert ref.model is None and ref.headers is None and ref.auth_type == "api_key"

    def test_explicit_identity(self):
        ref = EndpointRef(url="u", model="m", headers={}, endpoint_id="ep1", owner="o", provider_id="openai", auth_type="oauth")
        assert ref.endpoint_id == "ep1" and ref.provider_id == "openai" and ref.auth_type == "oauth"

    def test_frozen(self):
        ref = EndpointRef.from_legacy("u", "m")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.url = "other"  # type: ignore[misc]


class TestStaticAuth:
    def test_sync_passes_headers_through(self):
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer k"})
        cred = auth.resolve_sync(ref, OPENAI_SPEC)
        assert cred.headers == {"Authorization": "Bearer k"}

    def test_sync_none_headers_is_empty(self):
        cred = auth.resolve_sync(EndpointRef.from_legacy("u", "m"), OPENAI_SPEC)
        assert cred.headers == {}

    def test_resolved_headers_are_a_copy(self):
        src_headers = {"Authorization": "Bearer k"}
        cred = auth.resolve_sync(EndpointRef.from_legacy("u", "m", src_headers), OPENAI_SPEC)
        assert cred.headers == src_headers
        assert cred.headers is not src_headers   # mutating the request can't corrupt the ref

    async def test_async_matches_sync(self):
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"x-api-key": "k"})
        sync = auth.resolve_sync(ref, ANTHROPIC_SPEC)
        asyncc = await auth.resolve(ref, ANTHROPIC_SPEC)
        assert sync.headers == asyncc.headers


class TestOAuthGuard:
    def test_legacy_ref_without_identity_raises(self):
        # OAuth provider + a ref that has no endpoint_id (came from a tuple) → refuse.
        with pytest.raises(HTTPException) as ei:
            auth.resolve_sync(EndpointRef.from_legacy("u", "m"), OAUTH_SPEC)
        assert ei.value.status_code == 500
        assert "identity" in ei.value.detail

    def test_with_identity_passes_guard_then_not_implemented(self):
        # Has identity → passes the guard, then hits the PR2 boundary (not 500).
        ref = EndpointRef(url="u", model="m", endpoint_id="ep1", auth_type="oauth")
        with pytest.raises(HTTPException) as ei:
            auth.resolve_sync(ref, OAUTH_SPEC)
        assert ei.value.status_code == 501

    async def test_async_guard(self):
        with pytest.raises(HTTPException) as ei:
            await auth.resolve(EndpointRef.from_legacy("u", "m"), OAUTH_SPEC)
        assert ei.value.status_code == 500

    def test_static_provider_never_guarded(self):
        # api_key providers resolve regardless of identity.
        assert auth.resolve_sync(EndpointRef.from_legacy("u", "m"), OPENAI_SPEC).headers == {}


class TestOptionalRefParam:
    """The load-bearing 1c guarantee: `ref=` supersedes url/model/headers and the
    legacy path (ref=None) is byte-identical to passing the equivalent tuple."""

    @respx.mock
    def test_sync_ref_supersedes_positional_args(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer sk-ref"})
        # positional url/model/headers are deliberately wrong — ref must win
        out = llm_core.llm_call("http://ignored", "ignored-model", [{"role": "user", "content": "hi"}],
                                headers={"Authorization": "Bearer WRONG"}, ref=ref)
        assert out == "ok"
        req = route.calls.last.request
        assert str(req.url) == OPENAI_URL
        assert req.headers["authorization"] == "Bearer sk-ref"
        assert json.loads(req.content)["model"] == "gpt-4o"

    @respx.mock
    def test_sync_ref_none_equivalent_to_tuple(self):
        msgs = [{"role": "user", "content": "hi"}]
        resp = {"choices": [{"message": {"content": "x"}}]}

        r1 = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=resp))
        llm_core.llm_call(OPENAI_URL, "gpt-4o", msgs, headers={"Authorization": "Bearer k"})
        legacy_req = json.loads(r1.calls.last.request.content)
        legacy_auth = r1.calls.last.request.headers["authorization"]

        llm_core._response_cache.clear()
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer k"})
        llm_core.llm_call("http://ignored", "ignored", msgs, ref=ref)
        ref_req = json.loads(r1.calls.last.request.content)
        ref_auth = r1.calls.last.request.headers["authorization"]

        assert legacy_req == ref_req
        assert legacy_auth == ref_auth

    @respx.mock
    async def test_async_ref_supersedes(self):
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer sk-ref"})
        out = await llm_core.llm_call_async("http://ignored", "ignored", [{"role": "user", "content": "hi"}], ref=ref)
        assert out == "ok"
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-ref"
        assert json.loads(route.calls.last.request.content)["model"] == "gpt-4o"

    @respx.mock
    async def test_stream_ref_supersedes(self):
        sse = b'data: {"choices":[{"delta":{"content":"yo"}}]}\n\ndata: [DONE]\n\n'
        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=sse))
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {"Authorization": "Bearer sk-ref"})
        chunks = [c async for c in llm_core.stream_llm("http://ignored", "ignored", [{"role": "user", "content": "hi"}], ref=ref)]
        assert any('"delta": "yo"' in c for c in chunks)
        req = route.calls.last.request
        assert str(req.url) == OPENAI_URL
        assert req.headers["authorization"] == "Bearer sk-ref"
        assert json.loads(req.content)["model"] == "gpt-4o"


class TestFallbackRejectsRef:
    """Buddy-review catch: a `ref=` must never reach a fallback wrapper. The
    wrappers forward **kwargs to a per-candidate entry call, so a single ref
    would supersede EVERY candidate's (url, model, headers) and silently
    collapse the chain onto one endpoint. Guarded with a TypeError until PR2
    lets candidates themselves be EndpointRefs. No current caller passes ref=,
    so this is purely defensive — it can never fire on today's call sites."""

    MSGS = [{"role": "user", "content": "hi"}]
    CANDS = [(OPENAI_URL, "gpt-4o", {}), ("http://fallback/v1/chat/completions", "m2", {})]

    def test_sync_fallback_rejects_ref(self):
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {})
        with pytest.raises(TypeError, match="does not accept ref="):
            llm_core.llm_call_with_fallback(self.CANDS, self.MSGS, ref=ref)

    async def test_async_fallback_rejects_ref(self):
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {})
        with pytest.raises(TypeError, match="does not accept ref="):
            await llm_core.llm_call_async_with_fallback(self.CANDS, self.MSGS, ref=ref)

    async def test_stream_fallback_rejects_ref(self):
        ref = EndpointRef.from_legacy(OPENAI_URL, "gpt-4o", {})
        with pytest.raises(TypeError, match="does not accept ref="):
            async for _ in llm_core.stream_llm_with_fallback(self.CANDS, self.MSGS, ref=ref):
                pass

    @respx.mock
    def test_sync_fallback_without_ref_still_works(self):
        # Guard must not disturb the normal (no-ref) fallback path.
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        out = llm_core.llm_call_with_fallback(self.CANDS, self.MSGS)
        assert out == "ok"
