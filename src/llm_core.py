# src/llm_core.py
import httpx
import asyncio
import time
import json
import logging
import hashlib
from fastapi import HTTPException
from typing import Optional, Dict, List

from src.providers import registry, auth
from src.providers.endpoint_ref import EndpointRef
from src.providers.spec import ANTHROPIC_MODELS
from src.providers.openai_chat import (
    uses_max_completion_tokens as _uses_max_completion_tokens,
    supports_thinking as _supports_thinking,
)
from src.providers.anthropic_messages import (
    convert_openai_content_to_anthropic as _convert_openai_content_to_anthropic,
    build_anthropic_payload as _build_anthropic_payload,
    build_anthropic_headers as _build_anthropic_headers,
    parse_anthropic_response as _parse_anthropic_response,
    normalize_anthropic_url as _normalize_anthropic_url,
)

logger = logging.getLogger(__name__)

# ── Provider framework seam ──
# Provider identity + the de-facto OpenAI/Anthropic transports now live in
# `src/providers/`. `llm_core` stays the execution engine: it owns the httpx
# pool, response cache, per-host dead-host cooldown, retry, and fallback, and
# dispatches request shaping / response parsing / stream decoding to transports.
# The names below are re-exported so the ~19 modules that import them from
# `src.llm_core` (endpoint_resolver, ai_interaction, model_routes, email_*, …)
# keep working unchanged.
_detect_provider = registry.detect_provider
_provider_label = registry.provider_label


class LLMConfig:
    """Configuration constants for LLM operations."""
    DEFAULT_TIMEOUT = 30
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_MAX_TOKENS = 0
    MAX_RETRIES = 3
    RETRY_DELAY = 0.5
    STREAM_TIMEOUT = 300


# Cache for LLM responses
def _get_cache_key(url: str, model: str, messages: List[Dict],
                   temperature: float, max_tokens: int) -> str:
    """Generate cache key for LLM requests."""
    hashable_messages = []
    for msg in messages:
        sorted_items = tuple(sorted(msg.items()))
        hashable_messages.append(sorted_items)

    content = json.dumps({
        'url': url,
        'model': model,
        'messages': hashable_messages,
        'temp': temperature,
        'max_tokens': max_tokens
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

_response_cache = {}

# Dead-host cooldown: maps host (scheme://host:port) -> unix ts when cooldown expires.
# When a connect to a host fails, we mark it dead for DEAD_HOST_COOLDOWN seconds so
# subsequent calls fail instantly instead of waiting on the connect timeout. Keeps
# one unreachable upstream from jamming chat across the rest of the app.
#
# But a SINGLE transient blip (local model briefly busy, a momentary
# Tailscale hiccup) used to trip a full 60s lockout — the user saw a
# 503 and thought the model died when it was fine a second later. So:
#   - require FAIL_THRESHOLD consecutive failures before cooling
#   - shorter cooldown so recovery is quick
#   - any success resets the failure counter immediately
DEAD_HOST_COOLDOWN = 20.0
_HOST_FAIL_THRESHOLD = 2
_dead_hosts: Dict[str, float] = {}
_host_fails: Dict[str, int] = {}
_model_activity: Dict[str, float] = {}

def _model_activity_key(url: str, model: str) -> str:
    return f"{(url or '').strip().rstrip()}|{(model or '').strip()}"

def note_model_activity(url: str, model: str):
    """Record that a real upstream request used this endpoint/model."""
    if not url or not model:
        return
    _model_activity[_model_activity_key(url, model)] = time.time()

def seconds_since_model_activity(url: str, model: str) -> Optional[float]:
    """Seconds since the endpoint/model was last used in this process."""
    ts = _model_activity.get(_model_activity_key(url, model))
    if not ts:
        return None
    return max(0.0, time.time() - ts)

def _host_key(url: str) -> str:
    from urllib.parse import urlsplit
    s = urlsplit(url)
    return f"{s.scheme}://{s.netloc}" if s.scheme and s.netloc else url

def _is_host_dead(url: str) -> bool:
    key = _host_key(url)
    exp = _dead_hosts.get(key)
    if exp is None:
        return False
    if time.time() >= exp:
        _dead_hosts.pop(key, None)
        return False
    return True

def _mark_host_dead(url: str) -> bool:
    """Record a connect failure. Only actually cools the host after
    _HOST_FAIL_THRESHOLD consecutive failures. Returns True if the host
    is now cooled (so callers can log accurately), False if it's still
    within its allowed-failure grace."""
    key = _host_key(url)
    n = _host_fails.get(key, 0) + 1
    _host_fails[key] = n
    if n >= _HOST_FAIL_THRESHOLD:
        _dead_hosts[key] = time.time() + DEAD_HOST_COOLDOWN
        return True
    return False

def _clear_host_dead(url: str) -> None:
    key = _host_key(url)
    _dead_hosts.pop(key, None)
    _host_fails.pop(key, None)


# Shared async HTTP client. Reusing one client keeps connections warm:
# repeat calls to api.anthropic.com / api.openai.com / openrouter skip the
# 100-500ms TCP+TLS handshake. Lazy init so we bind to the running event loop.
_http_client: Optional[httpx.AsyncClient] = None
_http_limits = httpx.Limits(max_connections=100, max_keepalive_connections=30, keepalive_expiry=30.0)

def _get_http_client() -> httpx.AsyncClient:
    """Return process-wide AsyncClient. Per-request timeout is passed at call time."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(limits=_http_limits, http2=False)
    return _http_client

def _get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached response if it exists."""
    return _response_cache.get(cache_key)

def _set_cached_response(cache_key: str, response: str) -> None:
    """Store response in cache."""
    if len(_response_cache) > 128:
        keys_to_remove = list(_response_cache.keys())[:64]
        for key in keys_to_remove:
            del _response_cache[key]
    _response_cache[cache_key] = response


def _format_upstream_error(status: int, body: bytes | str, url: str) -> str:
    """Turn an upstream HTTP error into a user-readable sentence.

    Auth failures (401/403) become 'xAI rejected the API key' etc., so the UI
    stops showing raw JSON like '{"error":{"message":"User not found."}}'.
    """
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8", errors="replace")
        except Exception:
            body = str(body)
    provider = _provider_label(url)
    # Try to pull a message out of the body
    detail = ""
    try:
        j = json.loads(body) if body else {}
        if isinstance(j, dict):
            err = j.get("error") or j
            if isinstance(err, dict):
                detail = (err.get("message") or err.get("detail") or "").strip()
            elif isinstance(err, str):
                detail = err.strip()
    except Exception:
        detail = (body or "").strip()[:240]

    if status in (401, 403):
        msg = f"{provider} rejected the API key"
        if status == 403:
            msg = f"{provider} denied access (403)"
        if detail:
            msg += f" — {detail}"
        msg += ". Check Model Endpoints → {} and re-paste the key.".format(provider)
        return msg
    if status == 404:
        return f"{provider} returned 404 — check the base URL and model name." + (f" ({detail})" if detail else "")
    if status == 429:
        return f"{provider} rate-limited the request (429)." + (f" {detail}" if detail else "")
    if status >= 500:
        return f"{provider} is having an outage (HTTP {status})." + (f" {detail}" if detail else "")
    return f"{provider} returned HTTP {status}" + (f": {detail}" if detail else "")


def _consolidate_system_messages(messages: List[Dict]) -> List[Dict]:
    """Copy messages and merge all system messages into a single one at the start.

    Some models (e.g. Qwen3.5) reject system messages that aren't first. Shared by
    every entry fn so the request the transport sees is identical across them.
    """
    messages_copy = [msg.copy() for msg in messages]
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m["content"])
        else:
            non_sys.append(m)
    if sys_parts:
        return [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    return non_sys


def list_model_ids(base_chat_url: str, timeout: int = LLMConfig.DEFAULT_TIMEOUT, headers: Optional[Dict] = None) -> List[str]:
    """List available model IDs from an endpoint."""
    if _detect_provider(base_chat_url) == "anthropic":
        return list(ANTHROPIC_MODELS)
    try:
        h = {}
        if headers:
            h.update(headers)
        r = httpx.get(base_chat_url.replace("/chat/completions", "/models"), headers=h, timeout=timeout)
        r.raise_for_status()
        return [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
    except Exception:
        return []

def normalize_model_id(endpoint_url: str, requested: str, timeout: int = LLMConfig.DEFAULT_TIMEOUT) -> Optional[str]:
    """Normalize a model ID to match available models."""
    avail = list_model_ids(endpoint_url, timeout)
    if not avail:
        return None
    if requested in avail:
        return requested
    import os as _os
    req_base = _os.path.basename(requested.rstrip("/"))
    for a in avail:
        if _os.path.basename(a.rstrip("/")) == req_base:
            return a
    return None

def llm_call(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
             max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None,
             timeout: int = LLMConfig.DEFAULT_TIMEOUT, prompt_type: Optional[str] = None,
             ref: Optional[EndpointRef] = None) -> str:
    """Synchronous LLM call with optional prompt type enhancement.

    `ref` (optional) supersedes url/model/headers with a resolved `EndpointRef`.
    When omitted, an `EndpointRef.from_legacy(url, model, headers)` is built so
    behavior is byte-identical for the existing tuple callers.
    """
    # Tolerate headers that arrive as a JSON string (some sessions stored them
    # double-encoded) — otherwise the transport's h.update() throws "dictionary
    # update sequence element #0 has length 1; 2 is required". Sync-only quirk.
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except Exception:
            headers = None
    if not isinstance(headers, dict):
        headers = None

    if ref is None:
        ref = EndpointRef.from_legacy(url, model, headers)

    messages_copy = _consolidate_system_messages(messages)

    spec = registry.detect_spec(ref.url)
    transport = registry.get_transport(spec)
    cache_key = _get_cache_key(ref.url, ref.model, messages_copy, temperature, max_tokens)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    creds = auth.resolve_sync(ref, spec)
    target_url = transport.target_url(ref.url)
    h = transport.build_headers(creds.headers)
    payload = transport.build_payload(ref.model, messages_copy, temperature, max_tokens)
    try:
        note_model_activity(target_url, ref.model)
        r = httpx.post(target_url, headers=h, json=payload, timeout=timeout)
    except Exception as e:
        raise HTTPException(502, f"POST {target_url} failed: {e}")
    if not r.is_success:
        raise HTTPException(502, f"Upstream {target_url} -> {r.status_code}: {r.text}")
    data = r.json()
    try:
        response = transport.parse_response(data)
        _set_cached_response(cache_key, response)
        return response
    except Exception:
        raise HTTPException(502, f"Unexpected schema from {target_url}: {str(data)[:400]}")


def llm_call_with_fallback(candidates, messages, **kwargs) -> str:
    """Sync `llm_call` with an ordered fallback chain.

    `candidates` is a list of (url, model, headers). The first one that returns
    without an exception wins. Connection / 5xx-style failures fall through to
    the next candidate. The dead-host cooldown inside `llm_call` makes repeat
    attempts at an offline primary effectively free.
    """
    cands = [c for c in (candidates or []) if c and c[0] and c[1]]
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    last_err = None
    for i, (url, model, headers) in enumerate(cands):
        try:
            return llm_call(url, model, messages, headers=headers, **kwargs)
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next")
            continue
    raise last_err if last_err else HTTPException(503, "All fallback candidates failed")


async def llm_call_async_with_fallback(candidates, messages, **kwargs) -> str:
    """Async variant of `llm_call_with_fallback` — same semantics."""
    cands = [c for c in (candidates or []) if c and c[0] and c[1]]
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    last_err = None
    for i, (url, model, headers) in enumerate(cands):
        try:
            return await llm_call_async(url, model, messages, headers=headers, **kwargs)
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next")
            continue
    raise last_err if last_err else HTTPException(503, "All fallback candidates failed")


async def llm_call_async(
    url: str,
    model: str,
    messages: List[Dict],
    temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
    max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS,
    headers: Optional[Dict] = None,
    timeout: int = LLMConfig.STREAM_TIMEOUT,
    max_retries: int = LLMConfig.MAX_RETRIES,
    prompt_type: Optional[str] = None,
    ref: Optional[EndpointRef] = None,
) -> str:
    """Asynchronous LLM call using httpx with connection pooling, timeout, retry logic, and performance logging.

    `ref` (optional) supersedes url/model/headers; when omitted, built from the
    legacy tuple so existing callers are byte-identical.
    """
    if ref is None:
        ref = EndpointRef.from_legacy(url, model, headers)

    messages_copy = _consolidate_system_messages(messages)

    spec = registry.detect_spec(ref.url)
    transport = registry.get_transport(spec)
    cache_key = _get_cache_key(ref.url, ref.model, messages_copy, temperature, max_tokens)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    creds = await auth.resolve(ref, spec)
    target_url = transport.target_url(ref.url)
    h = transport.build_headers(creds.headers)
    payload = transport.build_payload(ref.model, messages_copy, temperature, max_tokens)

    if _is_host_dead(target_url):
        raise HTTPException(503, f"Upstream {_host_key(target_url)} marked unreachable (cooldown active)")

    call_timeout = httpx.Timeout(connect=3.0, read=float(timeout), write=10.0, pool=5.0)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        start = time.time()
        try:
            note_model_activity(target_url, ref.model)
            client = _get_http_client()
            r = await client.post(target_url, headers=h, json=payload, timeout=call_timeout)
            duration = time.time() - start
            if not r.is_success:
                friendly = _format_upstream_error(r.status_code, r.text, target_url)
                logger.warning(
                    f"LLM async call to {target_url} failed in {duration:.2f}s "
                    f"(attempt {attempt}): HTTP {r.status_code} {friendly}"
                )
                raise HTTPException(r.status_code, friendly)
            logger.info(f"LLM async call to {target_url} succeeded in {duration:.2f}s (attempt {attempt})")
            _clear_host_dead(target_url)
            data = r.json()
            try:
                response = transport.parse_response(data)
                _set_cached_response(cache_key, response)
                return response
            except Exception:
                raise HTTPException(502, f"Unexpected schema from {target_url}: {str(data)[:400]}")
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            duration = time.time() - start
            _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
            logger.warning(f"LLM async connect to {target_url} failed after {duration:.2f}s: {e}{_tail}")
            raise HTTPException(503, f"Cannot reach {_host_key(target_url)}: {e}")
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            duration = time.time() - start
            logger.warning(f"LLM async call attempt {attempt} failed after {duration:.2f}s: {e}")
            if attempt >= max_retries:
                raise HTTPException(502, f"POST {target_url} failed after {max_retries} attempts: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)

async def stream_llm(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
                     max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None,
                     timeout: int = LLMConfig.STREAM_TIMEOUT, prompt_type: Optional[str] = None,
                     tools: Optional[List[Dict]] = None, ref: Optional[EndpointRef] = None):
    """Stream LLM responses with improved error handling.

    Yields SSE chunks:
      - data: {"delta": "text"}           — text content
      - data: {"type": "tool_calls", ...}  — accumulated native tool calls (before DONE)
      - event: error                       — errors
      - data: [DONE]                       — end of stream

    `llm_core` owns the wire (httpx stream lifecycle, cooldown, error mapping); the
    transport's stateful `StreamDecoder` turns each raw SSE line into the normalized
    chunks above. `ref` (optional) supersedes url/model/headers; when omitted it is
    built from the legacy tuple so existing callers are byte-identical.
    """
    if ref is None:
        ref = EndpointRef.from_legacy(url, model, headers)

    messages_copy = _consolidate_system_messages(messages)

    spec = registry.detect_spec(ref.url)
    transport = registry.get_transport(spec)
    target_url = transport.target_url(ref.url)
    creds = await auth.resolve(ref, spec)
    h = transport.build_headers(creds.headers)
    payload = transport.build_payload(ref.model, messages_copy, temperature, max_tokens, stream=True, tools=tools)

    # Short connect timeout: a reachable peer answers SYN in <100ms even on
    # Tailscale. 3s is plenty; 30s let one dead upstream wedge the UI.
    stream_timeout = httpx.Timeout(connect=3.0, read=float(timeout), write=30.0, pool=5.0)

    if _is_host_dead(target_url):
        yield f'event: error\ndata: {json.dumps({"error": f"Upstream {_host_key(target_url)} unreachable (cooldown active)", "status": 503})}\n\n'
        return
    note_model_activity(target_url, ref.model)

    decoder = transport.stream_decoder(ref.model)
    try:
        client = _get_http_client()
        async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
            _clear_host_dead(target_url)
            if r.status_code != 200:
                raw = (await r.aread()).decode(errors="replace")
                friendly = _format_upstream_error(r.status_code, raw, target_url)
                yield f'event: error\ndata: {json.dumps({"status": r.status_code, "text": friendly, "raw": raw[:500]})}\n\n'
                return
            async for line in r.aiter_lines():
                chunks, stop = decoder.decode_line(line)
                for chunk in chunks:
                    yield chunk
                if stop:
                    return
            # End of stream (no explicit terminal event received)
            for chunk in decoder.finalize():
                yield chunk
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        _cooled = _mark_host_dead(target_url)
        _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
        logger.warning(f"{spec.id} stream connect to {target_url} failed: {e}{_tail}")
        yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
    except httpx.ReadTimeout:
        yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
    except httpx.NetworkError:
        yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502})}\n\n'
    except Exception as e:
        logger.error(f"{spec.id} stream error: {e}")
        yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502})}\n\n'


async def stream_llm_with_fallback(candidates, messages, **kwargs):
    """Wrap stream_llm with an ordered fallback chain.

    `candidates` is a list of (url, model, headers). Each is tried in order,
    but only retried on a *pre-content* failure — i.e. an ``event: error``
    that arrives before any assistant text / tool-call data has been yielded.
    Once a candidate has emitted real output we never switch (that would
    duplicate streamed tokens); a later error from that candidate passes
    through unchanged. The dead-host cooldown in stream_llm makes repeat
    attempts at an offline primary effectively instant.

    Yields the same SSE chunk protocol as stream_llm.
    """
    cands = [c for c in (candidates or []) if c and c[0] and c[1]]
    if not cands:
        yield f'event: error\ndata: {json.dumps({"error": "No model endpoint configured", "status": 503})}\n\n'
        return

    last_error = None
    for i, (url, model, headers) in enumerate(cands):
        is_last = (i == len(cands) - 1)
        emitted = False
        retried = False
        async for chunk in stream_llm(url, model, messages, headers=headers, **kwargs):
            if chunk.startswith("event: error"):
                if not emitted and not is_last:
                    # Pre-content failure with fallbacks left — swallow and
                    # move to the next candidate.
                    last_error = chunk
                    retried = True
                    if i == 0:
                        logger.warning(f"[fallback] primary {model} failed before output; trying fallback")
                    else:
                        logger.warning(f"[fallback] candidate {model} failed; trying next")
                    break
                yield chunk
                continue
            # Any data chunk other than the terminal [DONE] means real output.
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                emitted = True
            yield chunk
        if not retried:
            return  # candidate finished (success, or terminal error already sent)
    # Every candidate failed pre-content — surface the last error.
    if last_error:
        yield last_error
