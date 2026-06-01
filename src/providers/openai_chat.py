"""OpenAI-compatible chat transport (the default for any non-Anthropic host).

Pure request/response shaping + a stateful streaming decoder. Extracted verbatim
from the inline `provider != "anthropic"` branches of `llm_core` so behavior is
byte-identical (see tests/test_llm_core.py).
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

from src.providers.common import DOC_TOOLS

logger = logging.getLogger(__name__)

# Models that require `max_completion_tokens` instead of `max_tokens`.
_MAX_COMPLETION_TOKENS_MODELS = {"o1", "o3", "o4", "gpt-4.5", "gpt-5"}


def uses_max_completion_tokens(model: str) -> bool:
    """Check if a model requires max_completion_tokens instead of max_tokens."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) or f"/{p}" in m for p in _MAX_COMPLETION_TOKENS_MODELS)


# Models that support structured thinking — may output </think> without opening tag.
_THINKING_MODEL_PATTERNS = ("qwen3", "qwq", "deepseek-r1", "deepseek-reasoner", "minimax", "m2-reap")


def supports_thinking(model: str) -> bool:
    """Check if model supports structured thinking output."""
    if not model:
        return False
    m = model.lower()
    return any(p in m for p in _THINKING_MODEL_PATTERNS)


class OpenAIStreamDecoder:
    """Stateful decoder for an OpenAI-compatible SSE stream.

    Accumulates native tool_calls across chunks, repairs the stray `</think>`
    that some thinking backends emit, and surfaces usage chunks.
    """

    def __init__(self, model: str):
        self._tc_acc: Dict[int, Dict] = {}   # index -> {id, name, arguments}
        self._thinking_model = supports_thinking(model)
        self._first_content_sent = False

    def _emit_tool_calls(self) -> Optional[str]:
        if not self._tc_acc:
            return None
        calls = [self._tc_acc[i] for i in sorted(self._tc_acc)]
        return f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'

    def finalize(self) -> List[str]:
        out: List[str] = []
        tc_event = self._emit_tool_calls()
        if tc_event:
            out.append(tc_event)
        out.append("data: [DONE]\n\n")
        return out

    def decode_line(self, line: str) -> Tuple[List[str], bool]:
        if not line:
            return [], False
        if not line.startswith("data: "):
            return [], False

        data = line[6:].strip()
        if data == "[DONE]":
            return self.finalize(), True

        out: List[str] = []
        try:
            if data.strip():
                if data.startswith("{"):
                    j = json.loads(data)
                    # Usage chunk (from stream_options)
                    _choices = j.get("choices") or []
                    _delta0 = _choices[0].get("delta") if _choices else None
                    if "usage" in j and _delta0 in (None, {}, {"content": None}):
                        u = j["usage"]
                        out.append(f'data: {json.dumps({"type": "usage", "data": {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}})}\n\n')
                    elif "choices" in j:
                        delta = j["choices"][0].get("delta", {})
                        if isinstance(delta, dict):
                            # Reasoning tokens (VLLM --reasoning-parser, e.g. Qwen3/DeepSeek-R1)
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                out.append(f'data: {json.dumps({"delta": reasoning, "thinking": True})}\n\n')
                            content = delta.get("content", "")
                            if content:
                                # Some thinking backends start normal content with a
                                # stray closing tag. Repair only that shape; do not
                                # wrap every first token for model families like
                                # MiniMax, which often stream ordinary answers.
                                if self._thinking_model and not self._first_content_sent and content.lstrip().lower().startswith("</think"):
                                    content = "<think>" + content
                                self._first_content_sent = True
                                out.append(f'data: {json.dumps({"delta": content})}\n\n')
                            # Native tool calls — accumulate across chunks
                            for tc in delta.get("tool_calls", []):
                                idx = tc.get("index", 0)
                                if idx not in self._tc_acc:
                                    self._tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                                if tc.get("id"):
                                    self._tc_acc[idx]["id"] = tc["id"]
                                func = tc.get("function", {})
                                if func.get("name"):
                                    self._tc_acc[idx]["name"] = func["name"]
                                if "arguments" in func:
                                    self._tc_acc[idx]["arguments"] += func["arguments"]
                                    # Stream tool arg deltas for doc tools
                                    if func["arguments"] and self._tc_acc[idx].get("name") in DOC_TOOLS:
                                        out.append(f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": self._tc_acc[idx]["name"], "arg_delta": func["arguments"]})}\n\n')
                    elif "text" in j:
                        if j["text"]:
                            out.append(f'data: {json.dumps({"delta": j["text"]})}\n\n')
                else:
                    if data.strip():
                        out.append(f'data: {json.dumps({"delta": data})}\n\n')
        except Exception as e:
            logger.error(f"Error parsing stream data: {e}")
            return out, False
        return out, False


class OpenAIChatTransport:
    id = "openai_chat"

    def target_url(self, url: str) -> str:
        return url

    def build_payload(self, model, messages, temperature, max_tokens, *, stream=False, tools=None) -> Dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            if tools:
                payload["tools"] = tools
        return payload

    def build_headers(self, headers: Optional[Dict]) -> Dict:
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        return h

    def parse_response(self, data: Dict) -> str:
        return data["choices"][0]["message"]["content"]

    def stream_decoder(self, model: str) -> OpenAIStreamDecoder:
        return OpenAIStreamDecoder(model)
