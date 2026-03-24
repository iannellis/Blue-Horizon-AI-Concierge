"""Tests for the information Redis loader helpers.

These tests stay pure with respect to Redis and embeddings. They cover
schema validation and vectorized node normalization only.
"""

# ruff: noqa: S101

from pathlib import Path

import pandas as pd
import pytest

from blue_horizon.load_data.information_redis import (
    _AMENITIES_SCHEMA,
    get_amenities_nodes,
    get_faq_nodes,
    get_services_nodes,
    load_dataframe,
)


class TestLoadDataFrame:
    """load_dataframe validates pickled datasets with Pandera schemas."""

    def test_invalid_numeric_value_exits(self, tmp_path: Path) -> None:
        """Pandera rejects non-coercible numeric values instead of defaulting them."""
        path = tmp_path / "amenities.pkl"
        pd.DataFrame(
            {
                "name": ["Spa Access"],
                "description": ["All day spa access"],
                "category": ["wellness"],
                "price": ["free"],
                "duration": [60],
                "availability": ["daily"],
                "booking_required": ["True"],
                "min_notice_hours": [2],
            },
        ).to_pickle(path)

        with pytest.raises(SystemExit):
            load_dataframe("Amenities", path, _AMENITIES_SCHEMA)


class TestFaqNodes:
    """get_faq_nodes vectorizes string defaults before TextNode creation."""

    def test_missing_values_default_cleanly(self) -> None:
        """Missing FAQ string fields are filled with empty/general defaults."""
        df = pd.DataFrame(
            {
                "question": [None],
                "answer": [None],
                "category": [None],
                "subcategory": [None],
                "keywords": [None],
                "last_updated": [None],
            },
            index=["faq-1"],
        )

        node = get_faq_nodes(df)[0]

        assert node.text == "Question:\n\n\nAnswer:\n"
        assert node.metadata == {
            "category": "general",
            "subcategory": "general",
            "keywords": "",
            "last_updated": "",
        }


class TestAmenitiesNodes:
    """get_amenities_nodes vectorizes defaults for text and numeric metadata."""

    def test_missing_values_default_cleanly(self) -> None:
        """Missing amenity values are defaulted without row-wise helper calls."""
        df = pd.DataFrame(
            {
                "name": ["Spa Access"],
                "description": [None],
                "category": [None],
                "price": [None],
                "duration": [None],
                "availability": [None],
                "booking_required": [None],
                "min_notice_hours": [None],
            },
            index=["amenity-1"],
        )

        node = get_amenities_nodes(df)[0]

        assert node.text == "Name:Spa Access\n\nDescription:\n"
        assert node.metadata == {
            "category": "general",
            "price": 0.0,
            "duration": 0.0,
            "availability": "",
            "booking_required": "False",
            "min_notice_hours": 0.0,
        }


class TestServicesNodes:
    """get_services_nodes vectorizes defaults and duration normalization."""

    def test_missing_values_default_cleanly(self) -> None:
        """Missing service values default cleanly and duration becomes numeric."""
        df = pd.DataFrame(
            {
                "name": ["Airport Transfer"],
                "description": [None],
                "service_type": [None],
                "duration_minutes": [None],
                "price": [None],
                "department": [None],
                "booking_required": [None],
                "min_notice_hours": [None],
            },
            index=["service-1"],
        )

        node = get_services_nodes(df)[0]

        assert node.text == "Name:Airport Transfer\n\nDescription:\n"
        assert node.metadata == {
            "service_type": "general",
            "duration": 0.0,
            "price": 0.0,
            "department": "general",
            "booking_required": "False",
            "min_notice_hours": 0.0,
        }
