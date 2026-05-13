from __future__ import annotations

from types import SimpleNamespace

import pytest

from openchronicle.config import Config, ModelConfig
from openchronicle.writer import llm as llm_mod


def _cfg(model: str, max_tokens: int | None = None) -> Config:
    return Config(models={"default": ModelConfig(model=model, api_key="", api_key_env="", max_tokens=max_tokens)})


def _responses_message(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
    )


def _responses_tool_call(name: str, arguments: str = '{"path":"user-preferences.md"}'):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_123",
                name=name,
                arguments=arguments,
            )
        ]
    )


def test_chatgpt_models_use_litellm_responses_not_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCHRONICLE_LLM_MOCK", raising=False)
    import litellm

    calls: list[dict] = []

    def fake_responses(**kwargs):
        calls.append(kwargs)
        return _responses_message("ok")

    def fake_completion(**kwargs):  # pragma: no cover - should never be called
        raise AssertionError(f"completion should not be called for chatgpt models: {kwargs}")

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "completion", fake_completion)

    resp = llm_mod.call_llm(
        _cfg("chatgpt/gpt-5.3-chat-latest"),
        "reducer",
        messages=[{"role": "user", "content": "Reply ok"}],
        json_mode=True,
    )

    assert llm_mod.extract_text(resp) == "ok"
    assert calls[0]["model"] == "chatgpt/gpt-5.3-chat-latest"
    assert calls[0]["input"] == [{"role": "user", "content": "Reply ok"}]
    assert "max_tokens" not in calls[0]
    assert "max_output_tokens" not in calls[0]
    assert "max_completion_tokens" not in calls[0]
    assert calls[0]["text"] == {"format": {"type": "json_object"}}


def test_chatgpt_responses_promotes_system_messages_to_developer_and_strips_token_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCHRONICLE_LLM_MOCK", raising=False)
    import litellm

    calls: list[dict] = []

    def fake_responses(**kwargs):
        calls.append(kwargs)
        return _responses_message("ok")

    monkeypatch.setattr(litellm, "responses", fake_responses)

    llm_mod.call_llm(
        _cfg("chatgpt/gpt-5.4", max_tokens=2048),
        "timeline",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Summarize."},
        ],
    )

    assert calls[0]["input"] == [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "Summarize."},
    ]
    assert "max_output_tokens" not in calls[0]


def test_chatgpt_responses_function_calls_are_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCHRONICLE_LLM_MOCK", raising=False)
    import litellm

    captured_tools: list[list[dict]] = []

    def fake_responses(**kwargs):
        captured_tools.append(kwargs["tools"])
        return _responses_tool_call("append", '{"path":"user-preferences.md","content":"x","tags":["y"]}')

    monkeypatch.setattr(litellm, "responses", fake_responses)

    resp = llm_mod.call_llm(
        _cfg("chatgpt/gpt-5.3-chat-latest", max_tokens=2048),
        "classifier",
        messages=[{"role": "user", "content": "Classify"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "append",
                    "description": "Append memory",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert captured_tools[0] == [
        {
            "type": "function",
            "name": "append",
            "description": "Append memory",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert llm_mod.extract_text(resp) == ""
    assert llm_mod.extract_tool_calls(resp) == [
        {
            "id": "call_123",
            "name": "append",
            "arguments": {"path": "user-preferences.md", "content": "x", "tags": ["y"]},
        }
    ]


def test_non_chatgpt_models_still_use_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCHRONICLE_LLM_MOCK", raising=False)
    import litellm

    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))])

    monkeypatch.setattr(litellm, "completion", fake_completion)

    resp = llm_mod.call_llm(
        _cfg("openai/gpt-4.1-mini", max_tokens=123),
        "timeline",
        messages=[{"role": "user", "content": "Reply ok"}],
    )

    assert llm_mod.extract_text(resp) == "ok"
    assert calls[0]["model"] == "openai/gpt-4.1-mini"
    assert calls[0]["max_tokens"] == 123
