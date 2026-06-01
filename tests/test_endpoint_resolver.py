"""Tests for endpoint_resolver — pure functions tested directly to avoid import pollution."""
import re
from urllib.parse import urlparse


# Copy the pure functions to test them without importing the full module.
# This avoids module cache conflicts with other test files that mock dependencies.

def normalize_base(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    for suffix in ["/models", "/chat/completions", "/completions", "/v1/messages"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url


def _detect_provider(url: str) -> str:
    if "anthropic.com" in (url or ""):
        return "anthropic"
    return "openai"


def build_chat_url(base: str) -> str:
    provider = _detect_provider(base)
    if provider == "anthropic":
        host = urlparse(base).hostname or ""
        if host.endswith("anthropic.com") and base.rstrip("/").endswith("/v1"):
            base = base.rstrip("/")[:-3].rstrip("/")
        return base + "/v1/messages"
    return base + "/chat/completions"


def build_headers(api_key, base: str) -> dict:
    if not api_key:
        return {}
    provider = _detect_provider(base)
    if provider == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {api_key}"}


class TestNormalizeBase:
    def test_strips_models(self):
        assert normalize_base("https://api.openai.com/v1/models") == "https://api.openai.com/v1"

    def test_strips_chat_completions(self):
        assert normalize_base("https://api.openai.com/v1/chat/completions") == "https://api.openai.com/v1"

    def test_strips_completions(self):
        assert normalize_base("https://api.openai.com/v1/completions") == "https://api.openai.com/v1"

    def test_strips_v1_messages(self):
        assert normalize_base("https://api.anthropic.com/v1/messages") == "https://api.anthropic.com"

    def test_trailing_slash(self):
        assert normalize_base("https://api.openai.com/v1/") == "https://api.openai.com/v1"

    def test_clean_url_unchanged(self):
        assert normalize_base("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_empty_string(self):
        assert normalize_base("") == ""

    def test_none_safe(self):
        assert normalize_base(None) == ""


class TestBuildChatUrl:
    def test_openai_style(self):
        assert build_chat_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"

    def test_anthropic_style(self):
        assert build_chat_url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"

    def test_anthropic_v1_base_does_not_double_v1(self):
        assert build_chat_url("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/messages"

    def test_local_endpoint(self):
        assert build_chat_url("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"


class TestBuildHeaders:
    def test_no_key(self):
        assert build_headers(None, "https://api.openai.com/v1") == {}

    def test_openai_bearer(self):
        assert build_headers("sk-abc", "https://api.openai.com/v1") == {"Authorization": "Bearer sk-abc"}

    def test_anthropic_headers(self):
        assert build_headers("sk-ant-abc", "https://api.anthropic.com") == {"x-api-key": "sk-ant-abc", "anthropic-version": "2023-06-01"}

    def test_empty_key(self):
        assert build_headers("", "https://api.openai.com/v1") == {}


# ──────────────────────────────────────────────────────────────────────────────
# Characterization against the REAL src/endpoint_resolver.py.
#
# The copy-based tests above document intended behavior but exercise copies, so
# they cannot catch a regression in the real module. PR1c reshapes exactly these
# functions (build_chat_url/build_headers delegate to provider/auth;
# resolve_endpoint becomes a thin wrapper). To get a true before/after safety net
# we load the real module HERE, under a private name, by file path — which is
# immune to the `sys.modules['src.endpoint_resolver'] = MagicMock()` that
# test_agent_loop / test_context_compactor install during collection (and which
# is precisely why the simpler tests above resort to copies).
# ──────────────────────────────────────────────────────────────────────────────
import importlib.util
import os

import pytest


def _load_real_endpoint_resolver():
    """Load src/endpoint_resolver.py under a private name, bypassing any mock
    sitting in sys.modules['src.endpoint_resolver']. Its own imports
    (`src.database` stub, real `src.llm_core`) resolve via sys.modules normally."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "endpoint_resolver.py",
    )
    spec = importlib.util.spec_from_file_location("_real_endpoint_resolver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_real_er = _load_real_endpoint_resolver()


class TestRealAnthropicApiRoot:
    def test_strips_v1_only_for_anthropic(self):
        assert _real_er._anthropic_api_root("https://api.anthropic.com/v1") == "https://api.anthropic.com"

    def test_preserves_v1_for_non_anthropic(self):
        # An OpenAI-compatible host's /v1 is meaningful and must NOT be stripped.
        assert _real_er._anthropic_api_root("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"

    def test_bare_anthropic_host_unchanged(self):
        assert _real_er._anthropic_api_root("https://api.anthropic.com") == "https://api.anthropic.com"

    def test_none_safe(self):
        assert _real_er._anthropic_api_root(None) == ""


class TestRealBuildChatUrl:
    @pytest.fixture(autouse=True)
    def _no_dns(self, monkeypatch):
        # build_chat_url calls resolve_url() (DNS + Tailscale). Pin it to identity
        # so these stay hermetic — DNS/Tailscale resolution is out of scope here.
        monkeypatch.setattr(_real_er, "resolve_url", lambda u: u)

    def test_openai_style(self):
        assert _real_er.build_chat_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"

    def test_anthropic_bare_host(self):
        assert _real_er.build_chat_url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"

    def test_anthropic_v1_does_not_double(self):
        assert _real_er.build_chat_url("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/messages"

    def test_openai_compatible_preserves_api_v1(self):
        assert _real_er.build_chat_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1/chat/completions"

    def test_local_endpoint(self):
        assert _real_er.build_chat_url("http://localhost:11434/v1") == "http://localhost:11434/v1/chat/completions"


class TestRealBuildHeaders:
    def test_no_key(self):
        assert _real_er.build_headers(None, "https://api.openai.com/v1") == {}

    def test_openai_bearer(self):
        assert _real_er.build_headers("sk-abc", "https://api.openai.com/v1") == {"Authorization": "Bearer sk-abc"}

    def test_anthropic(self):
        assert _real_er.build_headers("k", "https://api.anthropic.com") == {"x-api-key": "k", "anthropic-version": "2023-06-01"}


class TestRealNormalizeBase:
    @pytest.mark.parametrize("raw,expected", [
        ("https://api.openai.com/v1/models", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1/completions", "https://api.openai.com/v1"),
        ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1"),
        ("", ""),
        (None, ""),
    ])
    def test_normalize(self, raw, expected):
        assert _real_er.normalize_base(raw) == expected


class TestRealFirstChatModel:
    def test_skips_embedding_first(self):
        assert _real_er._first_chat_model(["text-embedding-ada-002", "gpt-4o"]) == "gpt-4o"

    def test_skips_multiple_non_chat(self):
        assert _real_er._first_chat_model(["whisper-1", "tts-1", "dall-e-3", "llama3"]) == "llama3"

    def test_all_non_chat_falls_back_to_first(self):
        assert _real_er._first_chat_model(["text-embedding-ada-002"]) == "text-embedding-ada-002"

    def test_empty_is_none(self):
        assert _real_er._first_chat_model([]) is None

    def test_first_already_chat(self):
        assert _real_er._first_chat_model(["gpt-4o", "text-embedding-ada-002"]) == "gpt-4o"


class TestResolveEndpointRefEquivalence:
    """PR1c: resolve_endpoint_ref() is canonical; resolve_endpoint() materializes
    the legacy tuple from it. The ~37 tuple-unpack sites must see no change."""

    def test_resolve_endpoint_materializes_ref(self, monkeypatch):
        from src.providers.endpoint_ref import EndpointRef
        sentinel = EndpointRef(
            url="https://x/v1/chat/completions", model="m1",
            headers={"Authorization": "Bearer k"}, endpoint_id="ep1",
            owner="o", provider_id="openai", auth_type="api_key",
        )
        monkeypatch.setattr(_real_er, "resolve_endpoint_ref", lambda *a, **k: sentinel)
        assert _real_er.resolve_endpoint("default", fallback_url="fb") == (
            "https://x/v1/chat/completions", "m1", {"Authorization": "Bearer k"},
        )

    def test_fallback_path_yields_static_from_legacy_ref(self, monkeypatch):
        # Force settings load to fail → resolve_endpoint_ref returns a from_legacy ref
        # (static, no identity), and resolve_endpoint materializes the fallback tuple.
        import sys
        import types

        bad = types.ModuleType("src.settings")

        def _boom(*a, **k):
            raise RuntimeError("no settings")

        bad.load_settings = _boom
        bad.get_user_setting = _boom
        monkeypatch.setitem(sys.modules, "src.settings", bad)

        ref = _real_er.resolve_endpoint_ref("default", fallback_url="u", fallback_model="m", fallback_headers={"h": "v"})
        assert (ref.url, ref.model, ref.headers) == ("u", "m", {"h": "v"})
        assert ref.endpoint_id is None
        assert ref.auth_type == "api_key"

        assert _real_er.resolve_endpoint("default", fallback_url="u", fallback_model="m", fallback_headers={"h": "v"}) == ("u", "m", {"h": "v"})
