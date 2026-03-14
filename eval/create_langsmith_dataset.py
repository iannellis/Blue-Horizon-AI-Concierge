"""Create or append a LangSmith dataset from JSONL cases."""

from __future__ import annotations

import argparse
import json
from itertools import batched
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langsmith import Client
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()


def _raise(message: str) -> None:
    """Raise a ValueError with a provided message.

    Args:
        message: Error message to include in the exception.

    Raises:
        ValueError: Always raised with the provided message.

    """
    raise ValueError(message)


class TurnInputModel(BaseModel):
    """Represent a single user turn in a multi-turn evaluation case.

    Attributes:
        user: End-user utterance for the turn.
        expected_route: Expected agent route for the turn (rooms/info/refuse).
        expect_injection: Whether the turn is an injection attempt.

    """

    model_config = ConfigDict(extra="allow")

    user: str = Field(..., min_length=1)
    expected_route: Literal["info", "rooms", "refuse"]
    expect_injection: bool = False


class CaseInputModel(BaseModel):
    """Represent a full multi-turn evaluation case.

    Attributes:
        case_id: Unique identifier for the case.
        turns: Ordered list of turns that make up the case.
        tags: Optional tags used for dataset filtering.

    """

    model_config = ConfigDict(extra="allow")

    case_id: str = Field(..., min_length=1)
    turns: list[TurnInputModel] = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)


def _format_validation_error(exc: ValidationError) -> str:
    """Format Pydantic validation errors for line-level reporting.

    Args:
        exc: Pydantic validation error instance.

    Returns:
        A concise, semicolon-delimited error message.

    """
    pieces: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        if loc:
            pieces.append(f"{loc}: {msg}")
        else:
            pieces.append(str(msg))
    return "; ".join(pieces) or "Invalid case schema."


def load_cases(path: Path) -> list[CaseInputModel]:
    """Load and validate JSONL cases from disk.

    Args:
        path: Path to a JSONL file containing one case per line.

    Returns:
        List of validated case objects.

    """
    if not path.exists():
        _raise(f"Cases file not found: {path}")

    cases: list[CaseInputModel] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                case = json.loads(raw)
            except json.JSONDecodeError as exc:
                snippet = raw.strip().replace("\t", " ")[:120]
                _raise(
                    f"Line {line_no}: invalid JSON: {exc.msg}. Snippet: {snippet}",
                )
            else:
                try:
                    validated = CaseInputModel.model_validate(case)
                except ValidationError as exc:
                    _raise(f"Line {line_no}: {_format_validation_error(exc)}")
                else:
                    if validated.case_id in seen_ids:
                        _raise(
                            f"Line {line_no}: duplicate case_id '{validated.case_id}'.",
                        )
                    seen_ids.add(validated.case_id)
                    cases.append(validated)

    if not cases:
        _raise("Cases file contained no examples.")
    return cases


def _find_dataset_id(client: Client, dataset_name: str) -> str | None:
    """Locate a LangSmith dataset by name.

    Args:
        client: LangSmith client instance.
        dataset_name: Name to search for.

    Returns:
        The dataset ID string if found; otherwise ``None``.

    """
    for dataset in client.list_datasets():
        if dataset.name == dataset_name:
            return str(dataset.id)
    return None


def main() -> None:
    """Create or append a LangSmith dataset from JSONL cases."""
    parser = argparse.ArgumentParser(
        description="Create or append a LangSmith dataset from JSONL cases.",
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name.")
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=Path("eval/datasets/cases_stub.jsonl"),
        help="Path to JSONL cases.",
    )
    parser.add_argument("--description", default=None, help="Dataset description.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append if dataset exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Examples per batch.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases_path)
    client = Client()

    dataset_id = _find_dataset_id(client, args.dataset_name)
    if dataset_id is None:
        dataset = client.create_dataset(
            dataset_name=args.dataset_name,
            description=args.description,
        )
        dataset_id = str(dataset.id)
    elif not args.append:
        _raise(
            f"Dataset '{args.dataset_name}' already exists. "
            "Use --append or choose a new dataset name.",
        )

    examples = [
        {
            "inputs": case.model_dump(),
            "metadata": {
                "case_id": case.case_id,
                "tags": case.tags,
                "n_turns": len(case.turns),
            },
        }
        for case in cases
    ]

    if args.batch_size <= 0:
        _raise("--batch-size must be a positive integer.")

    examples_uploaded = 0
    for batch in batched(examples, args.batch_size, strict=False):
        client.create_examples(dataset_id=dataset_id, examples=batch)
        examples_uploaded += len(batch)

    route_counts = {"info": 0, "rooms": 0, "refuse": 0}
    for case in cases:
        for turn in case.turns:
            route_counts[turn.expected_route] += 1

    print(f"cases_loaded={len(cases)}")  # noqa: T201
    print(f"examples_uploaded={examples_uploaded}")  # noqa: T201
    print(  # noqa: T201
        "routes_seen="
        f"info:{route_counts['info']} "
        f"rooms:{route_counts['rooms']} "
        f"refuse:{route_counts['refuse']}",
    )


if __name__ == "__main__":
    main()
