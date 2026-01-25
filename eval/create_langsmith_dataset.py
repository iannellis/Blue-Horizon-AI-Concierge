"""Create or append a LangSmith dataset from JSONL cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langsmith import Client
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


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
        expected_route: Expected agent route for the turn (rooms/info/none).
        expect_injection: Whether the turn is an injection attempt.
        injection_grade_rubric: Optional rubric for grading injection handling.

    """

    model_config = ConfigDict(extra="forbid")

    user: str = Field(..., min_length=1)
    expected_route: Literal["rooms", "info", "none"]
    expect_injection: bool
    injection_grade_rubric: str | None = None

    @field_validator("injection_grade_rubric")
    @classmethod
    def _validate_rubric(cls, value: str | None) -> str | None:
        """Validate optional injection rubric content.

        Args:
            value: Rubric text or ``None`` when omitted.

        Returns:
            The validated rubric text (or ``None``).

        Raises:
            ValueError: If a rubric is provided but is empty.

        """
        if value is None:
            return value
        if not value.strip():
            msg = "injection_grade_rubric must be a non-empty string."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_injection_rubric(self) -> TurnInputModel:
        """Require rubric when injection is expected.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If injection is expected but no rubric is provided.

        """
        if self.expect_injection and not self.injection_grade_rubric:
            msg = "injection_grade_rubric is required when expect_injection is true."
            raise ValueError(msg)
        return self


class CaseInputModel(BaseModel):
    """Represent a full multi-turn evaluation case.

    Attributes:
        case_id: Unique identifier for the case.
        turns: Ordered list of turns that make up the case.
        tags: Optional tags used for dataset filtering.

    """

    model_config = ConfigDict(extra="forbid")

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
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            _raise(f"Line {line_no}: invalid JSON: {exc.msg}")
        else:
            try:
                validated = CaseInputModel.model_validate(case)
            except ValidationError as exc:
                _raise(f"Line {line_no}: {_format_validation_error(exc)}")
            else:
                if validated.case_id in seen_ids:
                    _raise(f"Line {line_no}: duplicate case_id '{validated.case_id}'.")
                seen_ids.add(validated.case_id)
                cases.append(validated)

    if not cases:
        _raise("Cases file contained no examples.")
    return cases


def _chunked(
    items: Iterable[dict[str, object]],
    batch_size: int,
) -> Iterable[list[dict[str, object]]]:
    """Yield items in fixed-size batches.

    Args:
        items: Iterable of items to batch.
        batch_size: Maximum size of each batch.

    Yields:
        Lists containing up to ``batch_size`` items.

    """
    batch: list[dict[str, object]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
        "--append", action="store_true", help="Append if dataset exists.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Examples per batch.",
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

    for batch in _chunked(examples, args.batch_size):
        client.create_examples(dataset_id=dataset_id, examples=batch)


if __name__ == "__main__":
    main()
