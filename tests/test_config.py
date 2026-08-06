"""Tests for the shared configuration parsing."""
# ruff: noqa: S101

import importlib.resources as importlib_resources
import tomllib
from pathlib import Path

from blue_horizon.config import AppConfig

EXPECTED_BATCH_SIZE = 64
EXPECTED_MAX_ROWS = 50
EXPECTED_HEALTH_CHECK_INTERVAL_S = 30
EXPECTED_ROOMS_TOP_K = 4
EXPECTED_DATA_PATH = Path("data/pandas")

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
                "retry_writes_on_transient_errors": False,
            },
        },
    },
    "load_data": {
        "information_redis": {"data_path": str(EXPECTED_DATA_PATH)},
        "booking_pgsql": {"data_path": str(EXPECTED_DATA_PATH)},
    },
    "neon": {
        "project_id": "test-project-id",
        "branch_name": "Working",
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
    assert cfg.load_data.information_redis.data_path == EXPECTED_DATA_PATH
    assert cfg.neon.branch_name == "Working"
    assert cfg.neon.lock_retry_attempts == EXPECTED_NEON_LOCK_RETRY_ATTEMPTS
    assert cfg.neon.lock_retry_delay_s == EXPECTED_NEON_LOCK_RETRY_DELAY_S


def test_load_packaged_app_config() -> None:
    """Load the packaged app_config via importlib.resources and parse it."""
    data = tomllib.loads(
        importlib_resources.files("blue_horizon")
        .joinpath("app_config.toml")
        .read_text(encoding="utf-8"),
    )
    cfg = AppConfig.model_validate(data)
    assert cfg.orchestration.messages.unavailable.startswith("Sorry")
    assert cfg.info.redis.health_check_interval_s == EXPECTED_HEALTH_CHECK_INTERVAL_S
    assert cfg.booking.agent.top_k == EXPECTED_ROOMS_TOP_K
    assert cfg.load_data.booking_pgsql.data_path == EXPECTED_DATA_PATH
    assert cfg.neon.branch_name == "Working"
    assert cfg.neon.lock_retry_attempts == EXPECTED_NEON_LOCK_RETRY_ATTEMPTS
    assert cfg.neon.lock_retry_delay_s == EXPECTED_NEON_LOCK_RETRY_DELAY_S
