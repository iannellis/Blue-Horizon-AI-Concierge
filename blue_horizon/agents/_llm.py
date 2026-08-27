"""Shared chat-model construction for agent LLM clients.

The info responder/parser, the booking agent, and the orchestration router
each build a `ChatOpenAI` client from the same four config fields. This
module gives that construction one home instead of four call sites that must
be kept in sync by hand.
"""

from __future__ import annotations

from typing import Protocol

from langchain_openai import ChatOpenAI


class LlmConfig(Protocol):
    """Structural type for the config fields `build_chat_model` needs.

    `InfoLlmConfig`, `BookingLlmConfig`, and `OrchestrationLlmConfig` each
    satisfy this without inheriting from it.

    Attributes:
        model: Chat model identifier.
        reasoning_effort: Reasoning effort hint (e.g. low, medium, high).
        timeout_s: Client-side request timeout in seconds.
        max_retries: Retry budget for transient LLM failures.

    """

    model: str
    reasoning_effort: str
    timeout_s: float
    max_retries: int


def build_chat_model(cfg: LlmConfig) -> ChatOpenAI:
    """Build a `ChatOpenAI` client from an agent's LLM configuration section.

    Args:
        cfg: LLM configuration exposing `model`, `reasoning_effort`,
            `timeout_s`, and `max_retries` (e.g. `InfoLlmConfig`,
            `BookingLlmConfig`, `OrchestrationLlmConfig`).

    Returns:
        ChatOpenAI: Configured chat model client.

    """
    return ChatOpenAI(
        model=cfg.model,
        reasoning={"effort": cfg.reasoning_effort},
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )
