"""Credential interface + dispatch + the OAuth guard.

`Credential.headers` are merged into the outgoing request by the transport's
`build_headers`. For static-key providers the resolved credential is simply the
headers carried on the `EndpointRef`. OAuth resolution (token fetch/refresh) is
deferred to a later PR; this module only enforces the boundary today.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import HTTPException

from src.providers.endpoint_ref import EndpointRef
from src.providers.spec import ProviderSpec


@dataclass(frozen=True)
class Credential:
    headers: Dict[str, str] = field(default_factory=dict)


def _effective_auth_type(ref: EndpointRef, spec: Optional[ProviderSpec]) -> str:
    """The spec is authoritative; the ref's auth_type is a fallback hint."""
    if spec is not None:
        return spec.auth_type
    return ref.auth_type


def _guard_oauth(ref: EndpointRef, spec: Optional[ProviderSpec]) -> None:
    """OAuth providers MUST have endpoint identity (so the token can be looked up
    and refreshed). A ref built from a legacy tuple has none — fail loudly rather
    than silently re-freezing a bearer token that can never refresh."""
    if _effective_auth_type(ref, spec) == "oauth" and not ref.endpoint_id:
        pid = getattr(spec, "id", None) or ref.provider_id or "?"
        raise HTTPException(
            500,
            f"Provider '{pid}' uses OAuth but this endpoint ref has no identity "
            f"(it came from a legacy tuple). Resolve it via resolve_endpoint_ref().",
        )


async def resolve(ref: EndpointRef, spec: Optional[ProviderSpec] = None) -> Credential:
    """Async credential resolution. Static-key today; OAuth refresh lands here."""
    _guard_oauth(ref, spec)
    if _effective_auth_type(ref, spec) == "oauth":
        # PR2: look up the token by ref.endpoint_id, refresh if expired, build
        # the provider's auth headers (single-flight per endpoint).
        raise HTTPException(501, "OAuth credential resolution is not implemented yet")
    from src.providers.auth import static  # lazy: avoids auth-package import cycle
    return static.resolve(ref)


def resolve_sync(ref: EndpointRef, spec: Optional[ProviderSpec] = None) -> Credential:
    """Sync sibling for the 3 sync `llm_call` callers. Static-key only.

    OAuth endpoints are unreachable from sync paths until a later PR decides
    sync-refresh vs migrating those callers — this is a documented boundary.
    """
    _guard_oauth(ref, spec)
    if _effective_auth_type(ref, spec) == "oauth":
        raise HTTPException(
            501,
            "OAuth endpoints require async credential resolution and are "
            "unreachable from sync llm_call until a later PR",
        )
    from src.providers.auth import static  # lazy: avoids auth-package import cycle
    return static.resolve(ref)
