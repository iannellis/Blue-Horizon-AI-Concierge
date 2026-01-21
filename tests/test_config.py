"""Tests for the shared configuration parsing."""

import importlib.resources as importlib_resources
import tomllib

from blue_horizon.config import parse_app_config

SAMPLE_APP_CONFIG: dict[str, object] = {
    "orchestration": {
        "llm": {
            "model": "gpt-5-nano",
            "temperature": 0.0,
            "reasoning_effort": "low",
            "timeout_s": 15.0,
            "max_retries": 2,
        },
        "orchestration": {
            "init_retry_base_s": 2.0,
            "init_retry_max_s": 60.0,
            "router_timeout_s": 30.0,
            "info_timeout_s": 60,
            "rooms_timeout_s": 60,
        },
        "prompts": {
            "folder": "system_prompts",
            "orchestration_prompt_filename": "orchestration_prompt.txt",
        },
        "messages": {
            "refusal": "I'm sorry, I cannot help with that query. I can only provide information about the hotel and help with room bookings.",
            "error": "Sorry - there was a problem processing your request. Please try again.",
            "unavailable": "Sorry - the system isn't available at the moment. Please try again shortly.",
        },
    },
    "info": {
        "retrieval": {
            "top_k": 4,
            "vector_dims": 1536,
            "retriever_cache_max": 64,
        },
        "embeddings": {
            "model": "text-embedding-3-small",
            "batch_size": 64,
            "timeout_s": 20.0,
            "max_retries": 2,
        },
        "llm": {
            "model": "gpt-5.2",
            "temperature": 0.0,
            "reasoning_effort": "medium",
            "timeout_s": 20.0,
            "max_retries": 2,
        },
        "redis": {
            "connect_timeout_s": 2.0,
            "socket_timeout_s": 4.0,
            "health_check_interval_s": 30,
        },
        "prompts": {
            "folder": "system_prompts",
            "system_prompt_filename": "information_prompt.txt",
        },
    },
    "rooms_sql": {
        "llm": {
            "model": "gpt-5-mini",
            "temperature": 0.0,
            "reasoning_effort": "low",
            "timeout_s": 20.0,
            "max_retries": 2,
        },
        "prompts": {
            "folder": "system_prompts",
            "system_prompt_filename": "rooms_sql_prompt.txt",
        },
        "agent": {"dialect": "PostgreSQL", "top_k": 4},
        "db": {
            "pool": {"min_size": 1, "max_size": 10, "timeout_s": 10.0},
            "guardrails": {"max_rows": 50, "allow_only_hotel_tables": True},
            "timeouts": {"statement_timeout_ms": 10000},
            "retry": {
                "max_transient_retries": 1,
                "transient_retry_backoff_s": 0.15,
                "retry_writes_on_transient_errors": False,
            },
        },
    },
}


def test_parse_app_config_from_dict() -> None:
    """Verify dict-based parsing returns the expected values."""

    cfg = parse_app_config(SAMPLE_APP_CONFIG)

    assert cfg.orchestration.llm.model == "gpt-5-nano"
    assert cfg.info.embeddings.batch_size == 64
    assert cfg.rooms_sql.db.guardrails.max_rows == 50


def test_load_packaged_app_config() -> None:
    """Load the packaged app_config via importlib.resources and parse it."""

    data = tomllib.loads(
        importlib_resources.files("blue_horizon")
        .joinpath("app_config.toml")
        .read_text(encoding="utf-8"),
    )
    cfg = parse_app_config(data)

    assert cfg.orchestration.messages.unavailable.startswith("Sorry")
    assert cfg.info.redis.health_check_interval_s == 30
    assert cfg.rooms_sql.agent.dialect == "PostgreSQL"
