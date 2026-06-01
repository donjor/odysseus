"""Provider registry — URL → ProviderSpec → Transport.

Absorbs the old `llm_core._detect_provider` / `_provider_label`. `llm_core`
re-exports `detect_provider` (string id) and `provider_label` under their legacy
underscored names for the ~19 call sites that import them.
"""
from urllib.parse import urlparse

from src.providers.anthropic_messages import AnthropicMessagesTransport
from src.providers.codex_responses import CodexResponsesTransport
from src.providers.openai_chat import OpenAIChatTransport
from src.providers.spec import BUILTIN_SPECS, OPENAI_SPEC, ProviderSpec

# Singleton transports — pure and stateless, safe to share.
_TRANSPORTS = {
    "openai_chat": OpenAIChatTransport(),
    "anthropic_messages": AnthropicMessagesTransport(),
    "codex_responses": CodexResponsesTransport(),
}


def detect_spec(url: str) -> ProviderSpec:
    """Resolve the ProviderSpec for an endpoint URL (OpenAI is the default)."""
    for spec in BUILTIN_SPECS:
        if spec.url_matchers and spec.matches(url):
            return spec
    return OPENAI_SPEC


def get_transport(spec_or_id):
    """Resolve a Transport from a ProviderSpec or a transport id."""
    tid = spec_or_id.transport if isinstance(spec_or_id, ProviderSpec) else spec_or_id
    return _TRANSPORTS[tid]


def get_transport_for_url(url: str):
    return get_transport(detect_spec(url))


def detect_provider(url: str) -> str:
    """Legacy string contract: 'anthropic' or 'openai'."""
    return detect_spec(url).id


def provider_label(url: str) -> str:
    """Human-friendly provider name for error messages."""
    u = (url or "").lower()
    if "anthropic.com" in u:
        return "Anthropic"
    if "api.x.ai" in u or "x.ai/" in u:
        return "xAI"
    if "openai.com" in u:
        return "OpenAI"
    if "openrouter.ai" in u:
        return "OpenRouter"
    if "groq.com" in u:
        return "Groq"
    if "mistral.ai" in u:
        return "Mistral"
    if "deepseek.com" in u:
        return "DeepSeek"
    if "googleapis.com" in u or "generativelanguage" in u:
        return "Google"
    if "together.xyz" in u or "together.ai" in u:
        return "Together"
    if "fireworks.ai" in u:
        return "Fireworks"
    if "localhost" in u or "127.0.0.1" in u:
        return "local endpoint"
    try:
        host = urlparse(url).hostname or "provider"
        return host
    except Exception:
        return "provider"
