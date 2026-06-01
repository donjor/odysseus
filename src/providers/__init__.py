"""Provider-adapter framework.

A registry of `ProviderSpec`s + pure `Transport`s extracted from what used to be
hardcoded `if provider == "anthropic"` branches in `llm_core`. The split:

  * **Transports are PURE** — no I/O, no global state. They only shape requests
    (`target_url` / `build_payload` / `build_headers`), parse responses
    (`parse_response`), and normalize streaming (`stream_decoder`).
  * **`llm_core` stays the execution engine** — it owns the httpx pool, response
    cache, per-host dead-host cooldown, retry, and fallback. It calls into
    transports; transports never call back into `llm_core` (no import cycle).

PR1 ships the two de-facto builtins (OpenAI-compatible chat, Anthropic messages)
plus a static-key auth seam (`providers.auth`). A later PR adds the
ChatGPT-subscription / codex transport + OAuth refresh on top of this seam.
"""
