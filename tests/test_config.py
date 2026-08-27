"""Tests for the shared configuration parsing."""
# ruff: noqa: S101

import importlib.resources as importlib_resources
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from blue_horizon.config import AppConfig, NeonConfig, NonNegInt, PositiveInt
from blue_horizon.config import FrozenModel as _FrozenModel

EXPECTED_BATCH_SIZE = 64
EXPECTED_MAX_ROWS = 50
EXPECTED_HEALTH_CHECK_INTERVAL_S = 30
EXPECTED_ROOMS_TOP_K = 4
EXPECTED_DATA_PATH = Path("data/pandas")
EXPECTED_PROPOSAL_TTL_S = 1800.0
EXPECTED_SEEDED_CUSTOMER_COUNT = 15

SAMPLE_APP_CONFIG: dict[str, object] = {
    "orchestration": {
        "llm": {
            "model": "gpt-5-nano",
            "reasoning_effort": "low",
            "timeout_s": 15.0,
            "max_retries": 2,
        },
        "orchestration": {
            "init_retry_base_s": 2.0,
            "init_retry_max_s": 60.0,
            "router_timeout_s": 30.0,
            "info_timeout_s": 60,
            "booking_timeout_s": 60,
        },
        "prompts": {
            "folder": "system_prompts",
            "orchestration_prompt_filename": "orchestration_prompt.txt",
        },
        "messages": {
            "refusal": (
                "I'm sorry, I cannot help with that query. "
                "I can only provide information about the hotel and "
                "help with room bookings."
            ),
            "error": (
                "Sorry - there was a problem processing your request. Please try again."
            ),
            "unavailable": (
                "Sorry - the system isn't available at the moment. "
                "Please try again shortly."
            ),
        },
    },
    "info": {
        "retrieval": {
            "top_k": 4,
            "vector_dims": 1536,
            "retriever_cache_max": 64,
            "max_context_items": 20,
        },
        "embeddings": {
            "model": "text-embedding-3-small",
            "batch_size": EXPECTED_BATCH_SIZE,
            "timeout_s": 20.0,
            "max_retries": 2,
        },
        "llm": {
            "model": "gpt-5.2",
            "reasoning_effort": "medium",
            "timeout_s": 20.0,
            "max_retries": 2,
        },
        "redis": {
            "connect_timeout_s": 2.0,
            "socket_timeout_s": 4.0,
            "health_check_interval_s": EXPECTED_HEALTH_CHECK_INTERVAL_S,
            "retry_max_retries": 2,
            "retry_backoff_base_s": 0.1,
            "retry_backoff_max_s": 1.0,
        },
        "prompts": {
            "folder": "system_prompts",
            "system_prompt_filename": "information_prompt.txt",
            "parser_prompt_filename": "information_parser.txt",
        },
    },
    "booking": {
        "llm": {
            "model": "gpt-5-mini",
            "reasoning_effort": "low",
            "timeout_s": 20.0,
            "max_retries": 2,
        },
        "prompts": {
            "folder": "system_prompts",
            "system_prompt_filename": "rooms_sql_prompt.txt",
        },
        "agent": {"top_k": EXPECTED_ROOMS_TOP_K},
        "db": {
            "pool": {
                "min_size": 0,
                "max_size": 10,
                "timeout_s": 10.0,
                "max_idle_s": 240.0,
            },
            "guardrails": {
                "max_rows": EXPECTED_MAX_ROWS,
                "allow_only_hotel_tables": True,
            },
            "retry": {
                "max_transient_retries": 1,
                "transient_retry_backoff_s": 0.15,
            },
        },
        "proposals": {"ttl_s": EXPECTED_PROPOSAL_TTL_S},
    },
    "load_data": {
        "information_redis": {"data_path": str(EXPECTED_DATA_PATH)},
        "booking_pgsql": {
            "data_path": str(EXPECTED_DATA_PATH),
            "seeded_customer_count": EXPECTED_SEEDED_CUSTOMER_COUNT,
        },
    },
    "neon": {
        "project_id": "test-project-id",
        "branch_name": "Production",
        "lock_retry_attempts": 8,
        "lock_retry_delay_s": 5.0,
    },
}


EXPECTED_NEON_LOCK_RETRY_ATTEMPTS = 8
EXPECTED_NEON_LOCK_RETRY_DELAY_S = 5.0


def test_parse_app_config_from_dict() -> None:
    """Verify dict-based parsing returns the expected values."""
    cfg = AppConfig.model_validate(SAMPLE_APP_CONFIG)
    assert cfg.orchestration.llm.model == "gpt-5-nano"
    assert cfg.info.embeddings.batch_size == EXPECTED_BATCH_SIZE
    assert cfg.booking.db.guardrails.max_rows == EXPECTED_MAX_ROWS
    assert cfg.booking.agent.top_k == EXPECTED_ROOMS_TOP_K
    assert cfg.booking.proposals.ttl_s == EXPECTED_PROPOSAL_TTL_S
    assert cfg.load_data.information_redis.data_path == EXPECTED_DATA_PATH
    assert (
        cfg.load_data.booking_pgsql.seeded_customer_count
        == EXPECTED_SEEDED_CUSTOMER_COUNT
    )
    assert cfg.neon.branch_name == "Production"
    assert cfg.neon.lock_retry_attempts == EXPECTED_NEON_LOCK_RETRY_ATTEMPTS
    assert cfg.neon.lock_retry_delay_s == EXPECTED_NEON_LOCK_RETRY_DELAY_S


def test_load_packaged_app_config() -> None:
    """Load the packaged app_config via importlib.resources and parse it.

    Asserts shape/type sanity, not specific literal values. This test's job
    is to catch a schema-breaking change -- e.g. a typo'd TOML key that
    Pydantic would otherwise silently ignore -- not to pin today's tunables.
    A deliberate retune in app_config.toml (seeded_customer_count, a
    timeout, a retry count, ...) should never fail this test; that
    round-trip is already covered, against values this module controls, by
    `test_parse_app_config_from_dict`.
    """
    data = tomllib.loads(
        importlib_resources.files("blue_horizon")
        .joinpath("app_config.toml")
        .read_text(encoding="utf-8"),
    )
    cfg = AppConfig.model_validate(data)
    assert cfg.orchestration.messages.unavailable
    assert cfg.info.redis.health_check_interval_s > 0
    assert cfg.booking.agent.top_k > 0
    assert cfg.booking.proposals.ttl_s > 0
    assert isinstance(cfg.load_data.booking_pgsql.data_path, Path)
    assert cfg.load_data.booking_pgsql.seeded_customer_count > 0
    assert cfg.neon.branch_name
    assert cfg.neon.lock_retry_attempts >= 1
    assert cfg.neon.lock_retry_delay_s >= 0


class _PositiveModel(_FrozenModel):
    """Minimal model exercising `PositiveInt` (clamp to 1) in isolation."""

    a: PositiveInt


class _NonNegModel(_FrozenModel):
    """Minimal model exercising `NonNegInt` (clamp to 0) in isolation."""

    a: NonNegInt


class TestClampedTypes:
    """`PositiveInt`/`NonNegInt` coerce out-of-range values instead of rejecting them.

    This is the behaviour the former per-class ``@field_validator(...,
    mode="before")`` methods provided; moving the clamp into a reusable
    ``Annotated`` type must not silently turn coercion into rejection.
    """

    def test_positive_int_clamps_negative_to_one(self) -> None:
        """A negative value is raised to 1, not rejected."""
        assert _PositiveModel(a=-5).a == 1

    def test_positive_int_clamps_zero_to_one(self) -> None:
        """Zero is raised to 1, not rejected."""
        assert _PositiveModel(a=0).a == 1

    def test_positive_int_passes_through_in_range_value(self) -> None:
        """A value already >= 1 is left unchanged."""
        assert _PositiveModel(a=7).a == 7  # noqa: PLR2004

    def test_non_neg_int_clamps_negative_to_zero(self) -> None:
        """A negative value is raised to 0, not rejected."""
        assert _NonNegModel(a=-5).a == 0

    def test_non_neg_int_passes_through_zero(self) -> None:
        """Zero is a valid value on its own and is left unchanged."""
        assert _NonNegModel(a=0).a == 0


class TestFrozenModel:
    """`FrozenModel` subclasses reject attribute mutation after construction."""

    def test_assignment_after_construction_raises(self) -> None:
        """Mutating a field on an already-built instance raises."""
        model = _PositiveModel(a=3)
        with pytest.raises(ValidationError):
            model.a = 9


class TestNeonConfigClamping:
    """`NeonConfig.lock_retry_attempts` still clamps via the shared `PositiveInt`."""

    def test_lock_retry_attempts_clamps_below_one(self) -> None:
        """A configured retry count below 1 is raised to 1, matching prior behaviour."""
        cfg = NeonConfig(
            project_id="p", branch_name="b", lock_retry_attempts=-3,
        )
        assert cfg.lock_retry_attempts == 1
