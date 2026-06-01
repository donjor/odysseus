"""Anthropic native Messages-API transport.

Pure request/response shaping (OpenAI-style messages -> Anthropic format, Bearer
-> x-api-key, /v1/messages URL normalization, multimodal image conversion) plus a
stateful streaming decoder. Extracted verbatim from the `provider == "anthropic"`
branches of `llm_core` so behavior is byte-identical (see tests/test_llm_core.py).
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

from src.providers.common import DOC_TOOLS

logger = logging.getLogger(__name__)


def convert_openai_content_to_anthropic(content):
    """Convert OpenAI multimodal content blocks to Anthropic format.

    Converts image_url blocks (data URI) -> Anthropic image blocks.
    Passes text blocks through unchanged.
    """
    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if not isinstance(block, dict):
            converted.append(block)
            continue
        if block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            # Parse data URI: data:image/<fmt>;base64,<data>
            if url.startswith("data:"):
                try:
                    header, b64_data = url.split(",", 1)
                    media_type = header.split(";")[0].replace("data:", "")
                except (ValueError, IndexError):
                    continue
                converted.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
            else:
                # External URL — use Anthropic's URL source
                converted.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        elif block.get("type") == "text":
            converted.append(block)
        else:
            converted.append(block)
    return converted


def build_anthropic_payload(model, messages, temperature, max_tokens, stream=False, tools=None):
    """Convert OpenAI-style messages to Anthropic format."""
    system_parts = []
    chat_messages = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m["content"])
        elif m.get("role") == "tool":
            # Convert OpenAI tool result to Anthropic format
            chat_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }],
            })
        elif m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            # Convert OpenAI assistant tool_calls to Anthropic format
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            chat_messages.append({"role": "assistant", "content": content})
        else:
            # Convert multimodal content (image_url → image) for Anthropic
            content = convert_openai_content_to_anthropic(m["content"])
            chat_messages.append({"role": m["role"], "content": content})
    payload = {
        "model": model,
        "messages": chat_messages,
        "max_tokens": max_tokens if max_tokens and max_tokens > 0 else 4096,
        "temperature": temperature,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if stream:
        payload["stream"] = True
    # Convert OpenAI-format tools to Anthropic format
    if tools:
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function":
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        if anthropic_tools:
            payload["tools"] = anthropic_tools
    return payload


def build_anthropic_headers(headers):
    """Convert Bearer auth to x-api-key for Anthropic."""
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if headers:
        for k, v in headers.items():
            if k.lower() == "authorization" and isinstance(v, str) and v.startswith("Bearer "):
                h["x-api-key"] = v[7:]
            else:
                h[k] = v
    return h


def parse_anthropic_response(data: dict) -> str:
    """Extract text from Anthropic response."""
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def normalize_anthropic_url(url: str) -> str:
    """Ensure Anthropic URL points to /v1/messages."""
    url = url.rstrip("/")
    if url.endswith("/v1/messages"):
        return url
    if url.endswith("/v1"):
        return url + "/messages"
    return url + "/v1/messages"


class AnthropicStreamDecoder:
    """Stateful decoder for an Anthropic Messages SSE stream.

    Accumulates tool_use content blocks and surfaces them in OpenAI-compatible
    `tool_calls` shape at message_stop, along with token usage.
    """

    def __init__(self, model: str):
        self._input_tokens = 0
        self._output_tokens = 0
        # Track tool_use blocks: {index: {id, name, arguments}}
        self._tool_blocks: Dict[int, Dict] = {}
        self._block_idx = -1
        self._block_type = ""

    def finalize(self) -> List[str]:
        # End of stream with no explicit message_stop: just terminate.
        return ["data: [DONE]\n\n"]

    def decode_line(self, line: str) -> Tuple[List[str], bool]:
        if not line or not line.startswith("data: "):
            return [], False
        data = line[6:].strip()
        if not data or not data.startswith("{"):
            return [], False

        out: List[str] = []
        try:
            j = json.loads(data)
            evt = j.get("type", "")
            if evt == "content_block_start":
                self._block_idx = j.get("index", self._block_idx + 1)
                cb = j.get("content_block", {})
                self._block_type = cb.get("type", "text")
                if self._block_type == "tool_use":
                    self._tool_blocks[self._block_idx] = {
                        "id": cb.get("id", f"call_{self._block_idx}"),
                        "name": cb.get("name", ""),
                        "arguments": "",
                    }
            elif evt == "content_block_delta":
                delta = j.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        out.append(f'data: {json.dumps({"delta": text})}\n\n')
                elif delta_type == "input_json_delta":
                    # Accumulate tool arguments JSON
                    idx = j.get("index", self._block_idx)
                    if idx in self._tool_blocks:
                        partial = delta.get("partial_json", "")
                        self._tool_blocks[idx]["arguments"] += partial
                        # Stream tool arg deltas for doc tools
                        if partial and self._tool_blocks[idx].get("name") in DOC_TOOLS:
                            out.append(f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": self._tool_blocks[idx]["name"], "arg_delta": partial})}\n\n')
            elif evt == "message_start":
                self._input_tokens = j.get("message", {}).get("usage", {}).get("input_tokens", 0)
            elif evt == "message_delta":
                self._output_tokens = j.get("usage", {}).get("output_tokens", 0)
            elif evt == "message_stop":
                # Emit accumulated tool calls in OpenAI-compatible format
                if self._tool_blocks:
                    calls = []
                    for idx in sorted(self._tool_blocks):
                        tb = self._tool_blocks[idx]
                        calls.append({
                            "id": tb["id"],
                            "name": tb["name"],
                            "arguments": tb["arguments"],
                        })
                    out.append(f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n')
                if self._input_tokens or self._output_tokens:
                    out.append(f'data: {json.dumps({"type": "usage", "data": {"input_tokens": self._input_tokens, "output_tokens": self._output_tokens}})}\n\n')
                out.append("data: [DONE]\n\n")
                return out, True
            elif evt == "error":
                err_msg = j.get("error", {}).get("message", "Unknown error")
                out.append(f'event: error\ndata: {json.dumps({"error": err_msg, "status": 400})}\n\n')
                return out, True
        except json.JSONDecodeError:
            return [], False
        return out, False


class AnthropicMessagesTransport:
    id = "anthropic_messages"

    def target_url(self, url: str) -> str:
        return normalize_anthropic_url(url)

    def build_payload(self, model, messages, temperature, max_tokens, *, stream=False, tools=None) -> Dict:
        return build_anthropic_payload(model, messages, temperature, max_tokens, stream=stream, tools=tools)

    def build_headers(self, headers: Optional[Dict]) -> Dict:
        return build_anthropic_headers(headers)

    def parse_response(self, data: Dict) -> str:
        return parse_anthropic_response(data)

    def stream_decoder(self, model: str) -> AnthropicStreamDecoder:
        return AnthropicStreamDecoder(model)
