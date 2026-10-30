"""Tests for blue_horizon/agents/_llm.py."""

# ruff: noqa: S101

import pytest
from langchain_core.messages import HumanMessage

from blue_horizon.agents._llm import build_chat_model
from blue_horizon.config import BookingLlmConfig

_MAX_OUTPUT_TOKENS = 8000


def test_output_token_cap_reaches_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap is sent as the Responses API's `max_output_tokens`.

    Checked on the request payload rather than on the client's attributes,
    because LangChain renames the field when it builds a Responses API call.
    A cap that never reached the wire would leave a runaway response bounded
    only by the client timeout, which does not stop the provider billing it.

    Args:
        monkeypatch: Supplies a placeholder API key so the client can be built.

    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = BookingLlmConfig(
        model="gpt-5-mini",
        reasoning_effort="medium",
        timeout_s=20.0,
        max_retries=2,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    payload = build_chat_model(cfg)._get_request_payload([HumanMessage("hi")])  # noqa: SLF001

    assert payload["max_output_tokens"] == _MAX_OUTPUT_TOKENS
