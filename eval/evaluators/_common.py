"""Shared constants and utilities for Blue Horizon evaluators."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from eval._utils import truncate
from eval.models import ExampleTurn, TurnOutput, _validate_list

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langsmith.schemas import Example, Run

try:  # Optional dependency for LangChain Gemini integration.
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI

    _LANGCHAIN_GEMINI_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # noqa: BLE001
    _ChatGoogleGenerativeAI = None
    _LANGCHAIN_GEMINI_IMPORT_ERROR = _exc

# Required tool names for info agent turns
_INFO_REQUIRED_TOOLS = (
    "query_faq",
    "query_amenities",
    "query_services",
    "merge",
)

# Tool names used by rooms agent SQL generation
_SQL_TOOL_NAMES = ("run_sql",)



def _iter_turn_outputs(run: Run) -> list[TurnOutput]:
    """Extract and parse the turn_outputs list from a LangSmith run.

    Args:
        run: LangSmith run object.

    Returns:
        List of parsed turn outputs (empty if missing or invalid; a
        malformed entry is dropped individually rather than failing the
        whole run -- see `eval.models._validate_list`).

    """
    outputs = run.outputs or {}
    return _validate_list(TurnOutput, outputs.get("turn_outputs"))


def _get_example_turns(example: Example) -> list[ExampleTurn]:
    """Extract and parse the turns list from a LangSmith example input.

    Args:
        example: LangSmith example object.

    Returns:
        List of parsed example turns (empty if missing or invalid).

    """
    inputs = example.inputs or {}
    return _validate_list(ExampleTurn, inputs.get("turns"))


def _rag_extract_turn_inputs(
    run: Run,
    example: Example,
) -> tuple[list[TurnOutput], list[ExampleTurn], list[object]]:
    """Extract turn data from run outputs and example inputs.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.

    Returns:
        Tuple of (turn_outputs, example_turns, reference_answers).

    """
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)

    inputs = example.inputs or {}
    reference_answers = inputs.get("reference_answers") or []
    if not isinstance(reference_answers, list):
        reference_answers = []

    return turn_outputs, example_turns, reference_answers


def _rag_extract_reference(
    example_turn: ExampleTurn,
    reference_answers: list[object],
    index: int,
    max_chars: int,
) -> str | None:
    """Extract a reference answer for a turn if available.

    Args:
        example_turn: Parsed example turn, potentially carrying a reference.
        reference_answers: Optional list of reference answers from example inputs.
        index: Turn index used to lookup reference list entries.
        max_chars: Maximum length for the reference string.

    Returns:
        Truncated reference string if available, otherwise None.

    """
    reference: object = example_turn.reference
    if reference is None and index < len(reference_answers):
        reference = reference_answers[index]
    if reference is None:
        return None
    text = truncate(reference, max_chars)
    return text or None


async def _call_judge_llm_structured[T: BaseModel](
    *,
    prompt: str,
    model: str,
    response_model: type[T],
) -> T:
    """Call the judge LLM and parse its response into a structured model.

    Uses the chat model's native structured-output support (Gemini's
    ``json_schema`` mode, the library default) instead of prompting for JSON
    and hand-parsing/validating the response: the provider enforces the
    schema, so there is no markdown fence to strip and no ad hoc payload
    validation to write -- a malformed response raises here rather than
    silently returning invalid data.

    Args:
        prompt: Prompt text to send to the judge.
        model: Model name to use for the judge.
        response_model: Pydantic model describing the expected response shape.

    Returns:
        A validated instance of `response_model`.

    Raises:
        RuntimeError: If the model is unavailable on the Developer API.

    """
    llm = _get_judge_llm(model)
    structured_llm = llm.with_structured_output(response_model)
    try:
        result = await structured_llm.ainvoke(prompt)
    except Exception as exc:
        if _is_model_unavailable_error(exc):
            msg = (
                "Judge model is unavailable on Developer API. "
                "Switch to Vertex AI for this model."
            )
            raise RuntimeError(msg) from exc
        raise
    # with_structured_output(response_model) (no include_raw) always returns
    # an instance of response_model at runtime; its declared return type is
    # the broader `dict | BaseModel` shared by every schema form the method
    # accepts (a plain dict, a TypedDict, or a Pydantic model).
    return cast("T", result)


@cache
def _get_judge_llm(model: str) -> BaseChatModel:
    """Lazily initialize and return the shared LangChain judge model.

    A single client is reused across every evaluator that calls an LLM judge
    (rubric grading, unbacked-success-claim adjudication, ...), keyed by
    model name so a config change between runs picks up a fresh client
    rather than reusing a stale one. ``ChatGoogleGenerativeAI(...)`` is a
    synchronous constructor, so caching needs no lock: `functools.cache` is
    a single-threaded, GIL-protected dict lookup, and this module has no
    thread pool crossing it.

    Args:
        model: Model name to use for the judge.

    Returns:
        LangChain chat model instance.

    Raises:
        RuntimeError: If the LangChain Gemini integration is unavailable.

    """
    if _ChatGoogleGenerativeAI is None:
        msg = "langchain-google-genai is required for judge evaluation."
        raise RuntimeError(msg) from _LANGCHAIN_GEMINI_IMPORT_ERROR
    return _ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
    )


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Check whether an exception indicates the model is unavailable.

    Args:
        exc: Exception raised during the judge request.

    Returns:
        True if the exception suggests the model is unavailable.

    """
    message = str(exc).lower()
    triggers = (
        "not found",
        "model",
        "404",
        "permission",
        "developer api",
        "not available",
    )
    return any(trigger in message for trigger in triggers)
