"""litellm wrapper with per-stage model resolution."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, resolve_api_key
from ..logger import get

logger = get("openchronicle.writer")


@dataclass
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class PingResult:
    stage: str
    model: str
    ok: bool
    latency_ms: int | None
    error: str | None
    mocked: bool = False


def call_llm(
    cfg: Config,
    stage: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_mode: bool = False,
) -> Any:
    """Invoke litellm for the given stage.

    Returns a raw LiteLLM response for Chat Completions-compatible providers and
    an ``LLMResponse`` for providers that need an internal adapter.
    Respects OPENCHRONICLE_LLM_MOCK=1 for tests.
    """
    if os.environ.get("OPENCHRONICLE_LLM_MOCK") == "1":
        return _mock_response(stage, messages, tools, json_mode)

    import litellm  # imported lazily to keep CLI startup fast

    model_cfg = cfg.model_for(stage)
    kwargs: dict[str, Any] = {
        "model": model_cfg.model,
        "messages": messages,
    }
    if model_cfg.base_url:
        kwargs["api_base"] = model_cfg.base_url
    api_key = resolve_api_key(model_cfg)
    if api_key:
        kwargs["api_key"] = api_key
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model_cfg.max_tokens:
        kwargs["max_tokens"] = model_cfg.max_tokens

    logger.debug("llm call stage=%s model=%s", stage, model_cfg.model)
    if _uses_responses_api(model_cfg.model):
        return _call_responses_api(litellm, kwargs, json_mode=json_mode)
    return litellm.completion(**kwargs)


def _uses_responses_api(model: str) -> bool:
    return model.startswith("chatgpt/")


def _call_responses_api(litellm: Any, kwargs: dict[str, Any], *, json_mode: bool = False) -> LLMResponse:
    params: dict[str, Any] = {
        "model": kwargs["model"],
        "input": _responses_input(kwargs.get("messages") or []),
    }
    if kwargs.get("tools"):
        params["tools"] = _responses_tools(kwargs["tools"])
        params["tool_choice"] = kwargs.get("tool_choice", "auto")
    if json_mode:
        params["text"] = {"format": {"type": "json_object"}}
    if kwargs.get("api_base"):
        params["api_base"] = kwargs["api_base"]
    if kwargs.get("api_key"):
        params["api_key"] = kwargs["api_key"]
    if kwargs.get("timeout"):
        params["timeout"] = kwargs["timeout"]
    return _responses_to_llm_response(litellm.responses(**params))


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        fn = tool["function"]
        converted.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or msg.get("id") or "call_unknown",
                    "output": content or "",
                }
            )
            continue
        if role == "system":
            role = "developer"
        if role in {"developer", "user", "assistant"} and content:
            items.append({"role": role, "content": _responses_content_parts(content)})
        for call in msg.get("tool_calls") or []:
            fn = _get(call, "function", {}) or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": _get(call, "id", None) or "call_unknown",
                    "name": _get(fn, "name", ""),
                    "arguments": _get(fn, "arguments", "{}") or "{}",
                }
            )
    return items


def _responses_content_parts(content: Any) -> list[dict[str, str]]:
    """Return ChatGPT-compatible Responses content parts.

    The ChatGPT subscription backend expects message content to be a list of
    typed parts, even for plain text. Bare string content can trip backend
    validation before LiteLLM normalizes it.
    """
    if isinstance(content, list):
        return content
    return [{"type": "input_text", "text": str(content)}]


def _responses_to_llm_response(resp: Any) -> LLMResponse:
    text_parts: list[str] = []
    calls: list[ToolCall] = []

    # LiteLLM/OpenAI Responses objects expose text in a few shapes depending on
    # provider/model/version. Prefer the official convenience field when it is
    # present, then fall back to walking output message content. The previous
    # adapter only handled one nested shape, which made health checks look green
    # while reducer/extractor calls saw empty text.
    top_level_text = _get(resp, "output_text", "")
    if isinstance(top_level_text, str) and top_level_text.strip():
        text_parts.append(top_level_text)

    for item in _as_list(_get(resp, "output", [])):
        typ = _get(item, "type", "")
        if typ == "function_call":
            try:
                args = json.loads(_get(item, "arguments", "{}") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(
                ToolCall(
                    id=_get(item, "call_id", None) or _get(item, "id", None),
                    name=_get(item, "name", ""),
                    arguments=args,
                )
            )
            continue

        if typ in {"output_text", "text"}:
            text = _get(item, "text", "")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
            continue

        content = _get(item, "content", None)
        if content is not None:
            for part in _as_list(content):
                text = _get(part, "text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
            continue

        text = _get(item, "text", "")
        if isinstance(text, str) and text.strip():
            text_parts.append(text)

    # Defensive fallback for Responses shims that return a Chat Completions-like
    # object even when called through litellm.responses().
    if not text_parts:
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        if text.strip():
            text_parts.append(text)

    return LLMResponse(text="\n".join(p for p in text_parts if p), tool_calls=calls)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _mock_response(stage: str, messages, tools, json_mode):
    """Minimal stub for offline tests. Customize via OPENCHRONICLE_LLM_MOCK_JSON."""
    override = os.environ.get("OPENCHRONICLE_LLM_MOCK_JSON")
    content = override if override else '{"worth_writing": false, "brief_reason": "mock"}'

    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Choice:
        def __init__(self, msg):
            self.message = msg
            self.finish_reason = "stop"

    class _Resp:
        def __init__(self, choices):
            self.choices = choices

    return _Resp([_Choice(_Msg(content))])


def extract_text(response: Any) -> str:
    if isinstance(response, LLMResponse):
        return response.text
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def ping_stage(cfg: Config, stage: str, *, timeout: float = 5.0) -> PingResult:
    """Send a tiny round-trip request to the stage's configured model.

    Returns a PingResult with success, latency, and a short error label on
    failure. Honors OPENCHRONICLE_LLM_MOCK=1 by returning a mocked-ok result
    without touching the network. Never raises — `status` and similar
    informational callers must remain non-fatal.
    """
    model_cfg = cfg.model_for(stage)
    if os.environ.get("OPENCHRONICLE_LLM_MOCK") == "1":
        return PingResult(
            stage=stage, model=model_cfg.model, ok=True,
            latency_ms=0, error=None, mocked=True,
        )

    try:
        import litellm  # lazy import — keeps CLI startup fast
    except ImportError as exc:
        return PingResult(
            stage=stage, model=model_cfg.model, ok=False,
            latency_ms=None, error=f"ImportError: {exc}",
        )

    kwargs: dict[str, Any] = {
        "model": model_cfg.model,
        "messages": [{"role": "user", "content": "Reply with 'ok'."}],
        "max_tokens": 64,
        "timeout": timeout,
    }
    if model_cfg.base_url:
        kwargs["api_base"] = model_cfg.base_url
    api_key = resolve_api_key(model_cfg)
    if api_key:
        kwargs["api_key"] = api_key

    start = time.monotonic()
    try:
        if _uses_responses_api(model_cfg.model):
            resp = _call_responses_api(litellm, kwargs)
            text = extract_text(resp).strip()
            if not text:
                raise ValueError("Responses ping returned empty text")
        else:
            litellm.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001
        label = type(exc).__name__
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        if msg:
            label = f"{label}: {msg[:240]}"
        return PingResult(
            stage=stage, model=model_cfg.model, ok=False,
            latency_ms=None, error=label,
        )
    latency_ms = int((time.monotonic() - start) * 1000)
    return PingResult(
        stage=stage, model=model_cfg.model, ok=True,
        latency_ms=latency_ms, error=None,
    )


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, LLMResponse):
        return [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ]
    try:
        calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        return []
    out: list[dict[str, Any]] = []
    for c in calls:
        fn = getattr(c, "function", None) or c.get("function", {})
        args_raw = getattr(fn, "arguments", None) if hasattr(fn, "arguments") else fn.get("arguments")
        name = getattr(fn, "name", None) if hasattr(fn, "name") else fn.get("name")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        out.append(
            {
                "id": getattr(c, "id", None) or c.get("id"),
                "name": name,
                "arguments": args,
            }
        )
    return out
