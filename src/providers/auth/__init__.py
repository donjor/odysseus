"""Credential resolution seam.

`llm_core` resolves a `Credential` from an `EndpointRef` right before each call.
PR1 ships the static-key impl only (the headers frozen on the ref ARE the
credential). The seam is async (`resolve`) with a sync sibling (`resolve_sync`)
for the 3 sync `llm_call` callers; a later PR adds OAuth fetch/refresh + a token
store behind the same interface, keyed on `EndpointRef.endpoint_id`.
"""
from src.providers.auth.credentials import Credential, resolve, resolve_sync

__all__ = ["Credential", "resolve", "resolve_sync"]
