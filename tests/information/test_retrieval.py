"""Tests for information-agent retrieval helpers.

build_filters and build_index_schema are pure functions; no I/O required.
IndexSchema.from_dict is patched to isolate the dict-construction logic from
redisvl internals.
"""

# ruff: noqa: S101

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from blue_horizon.agents.information.models import Source
from blue_horizon.agents.information.retrieval import (
    build_filters,
    build_index_schema,
    build_information_index_schemas,
    build_source_index_schema,
)

# ---------------------------------------------------------------------------
# build_filters
# ---------------------------------------------------------------------------


class TestBuildFilters:
    """build_filters constructs the correct MetadataFilters from constraints."""

    def test_all_none_returns_none(self) -> None:
        """With no constraints, None is returned."""
        assert build_filters() is None

    def test_booking_required_true(self) -> None:
        """booking_required=True produces an EQ filter with value 'True'."""
        result = build_filters(booking_required=True)
        assert result is not None
        assert len(result.filters) == 1
        f = result.filters[0]
        assert isinstance(f, MetadataFilter)
        assert f.key == "booking_required"
        assert f.operator == FilterOperator.EQ
        assert f.value == "True"

    def test_booking_required_false(self) -> None:
        """booking_required=False produces an EQ filter with value 'False'."""
        result = build_filters(booking_required=False)
        assert result is not None
        f = result.filters[0]
        assert f.value == "False"

    def test_min_price(self) -> None:
        """min_price produces a GTE filter on the price key."""
        result = build_filters(min_price=50.0)
        assert result is not None
        f = result.filters[0]
        assert isinstance(f, MetadataFilter)
        assert f.key == "price"
        assert f.operator == FilterOperator.GTE
        assert f.value == pytest.approx(50.0)

    def test_max_price(self) -> None:
        """max_price produces a LTE filter on the price key."""
        result = build_filters(max_price=200.0)
        assert result is not None
        f = result.filters[0]
        assert f.key == "price"
        assert f.operator == FilterOperator.LTE
        assert f.value == pytest.approx(200.0)

    def test_price_range_produces_two_filters(self) -> None:
        """min_price + max_price together produce two price filters."""
        result = build_filters(min_price=20.0, max_price=100.0)
        assert result is not None
        assert len(result.filters) == 2  # noqa: PLR2004
        operators = {f.operator for f in result.filters}
        assert FilterOperator.GTE in operators
        assert FilterOperator.LTE in operators

    def test_max_notice_hours(self) -> None:
        """max_notice_hours produces a LTE filter on min_notice_hours."""
        result = build_filters(max_notice_hours=24)
        assert result is not None
        f = result.filters[0]
        assert f.key == "min_notice_hours"
        assert f.operator == FilterOperator.LTE
        assert f.value == 24  # noqa: PLR2004

    def test_min_duration_minutes(self) -> None:
        """min_duration_minutes produces a GTE filter on duration."""
        result = build_filters(min_duration_minutes=30)
        assert result is not None
        f = result.filters[0]
        assert f.key == "duration"
        assert f.operator == FilterOperator.GTE
        assert f.value == 30  # noqa: PLR2004

    def test_max_duration_minutes(self) -> None:
        """max_duration_minutes produces a LTE filter on duration."""
        result = build_filters(max_duration_minutes=90)
        assert result is not None
        f = result.filters[0]
        assert f.key == "duration"
        assert f.operator == FilterOperator.LTE
        assert f.value == 90  # noqa: PLR2004

    def test_duration_range_produces_two_filters(self) -> None:
        """Min + max duration together produce two duration filters."""
        result = build_filters(min_duration_minutes=30, max_duration_minutes=90)
        assert result is not None
        assert len(result.filters) == 2  # noqa: PLR2004

    def test_all_constraints_produce_six_filters(self) -> None:
        """All six parameters each contribute one filter."""
        result = build_filters(
            booking_required=False,
            min_price=10.0,
            max_price=500.0,
            max_notice_hours=48,
            min_duration_minutes=15,
            max_duration_minutes=120,
        )
        assert result is not None
        assert len(result.filters) == 6  # noqa: PLR2004

    def test_filters_use_and_condition(self) -> None:
        """Multiple filters are AND-combined (default MetadataFilters condition)."""
        result = build_filters(min_price=10.0, max_price=100.0)
        assert result is not None
        assert result.condition == FilterCondition.AND

    def test_returns_metadata_filters_type(self) -> None:
        """Any non-None result is a MetadataFilters instance."""
        result = build_filters(booking_required=True)
        assert isinstance(result, MetadataFilters)

    def test_price_value_cast_to_float(self) -> None:
        """Integer price inputs are stored as float."""
        result = build_filters(min_price=50)  # type: ignore[arg-type]
        assert result is not None
        assert isinstance(result.filters[0].value, float)


# ---------------------------------------------------------------------------
# build_index_schema
# ---------------------------------------------------------------------------


class TestBuildIndexSchema:
    """build_index_schema builds the correct schema dict and delegates to redisvl."""

    def _captured_dict(
        self,
        *,
        name: str = "test_index",
        prefix: str = "test:",
        extra_fields: list | None = None,
        vector_dims: int = 128,
    ) -> dict:
        """Call build_index_schema with IndexSchema.from_dict patched; return the dict.

        Args:
            name: Index name.
            prefix: Key prefix.
            extra_fields: Additional fields.
            vector_dims: Vector dimensionality.

        Returns:
            The schema_dict argument captured from the IndexSchema.from_dict call.

        """
        if extra_fields is None:
            extra_fields = []
        with patch(
            "blue_horizon.agents.information.retrieval.IndexSchema.from_dict",
        ) as mock_from_dict:
            mock_from_dict.return_value = MagicMock()
            build_index_schema(
                name=name,
                prefix=prefix,
                extra_fields=extra_fields,
                vector_dims=vector_dims,
            )
            return mock_from_dict.call_args[0][0]

    def test_index_name(self) -> None:
        """schema_dict carries the supplied index name."""
        d = self._captured_dict(name="my_idx")
        assert d["index"]["name"] == "my_idx"

    def test_index_prefix(self) -> None:
        """schema_dict carries the supplied key prefix."""
        d = self._captured_dict(prefix="docs:")
        assert d["index"]["prefix"] == "docs:"

    def test_required_fields_present(self) -> None:
        """id, doc_id, text, and vector fields are always included."""
        d = self._captured_dict()
        field_names = [f["name"] for f in d["fields"]]
        assert "id" in field_names
        assert "doc_id" in field_names
        assert "text" in field_names
        assert "vector" in field_names

    def test_vector_field_dims(self) -> None:
        """The vector field attrs carry the correct dims value."""
        d = self._captured_dict(vector_dims=512)
        vector_field = next(f for f in d["fields"] if f["name"] == "vector")
        assert vector_field["attrs"]["dims"] == 512  # noqa: PLR2004

    def test_vector_field_algorithm(self) -> None:
        """The vector field always uses HNSW."""
        d = self._captured_dict()
        vector_field = next(f for f in d["fields"] if f["name"] == "vector")
        assert vector_field["attrs"]["algorithm"] == "hnsw"

    def test_vector_field_distance_metric(self) -> None:
        """The vector field uses cosine distance."""
        d = self._captured_dict()
        vector_field = next(f for f in d["fields"] if f["name"] == "vector")
        assert vector_field["attrs"]["distance_metric"] == "cosine"

    def test_extra_fields_appended(self) -> None:
        """Extra fields are appended after the four base fields."""
        extra = [{"type": "tag", "name": "category"}]
        d = self._captured_dict(extra_fields=extra)
        field_names = [f["name"] for f in d["fields"]]
        assert "category" in field_names
        assert len(d["fields"]) == 5  # noqa: PLR2004  # 4 base + 1 extra

    def test_no_extra_fields_gives_four_fields(self) -> None:
        """Without extra fields exactly four base fields are present."""
        d = self._captured_dict(extra_fields=[])
        assert len(d["fields"]) == 4  # noqa: PLR2004


# ---------------------------------------------------------------------------
# build_source_index_schema / build_information_index_schemas
# ---------------------------------------------------------------------------


class TestSharedInformationSchemas:
    """Shared schema builders keep runtime and ETL schema definitions aligned."""

    def test_build_source_index_schema_uses_source_defaults(self) -> None:
        """Each source expands to its expected name, prefix, and metadata fields."""
        with patch(
            "blue_horizon.agents.information.retrieval.build_index_schema",
        ) as mock_build_index_schema:
            mock_build_index_schema.return_value = MagicMock()

            build_source_index_schema(source=Source.SERVICES, vector_dims=256)

        mock_build_index_schema.assert_called_once_with(
            name="services",
            prefix="services",
            extra_fields=[
                {"type": "tag", "name": "service_type"},
                {"type": "numeric", "name": "price"},
                {"type": "numeric", "name": "duration"},
                {"type": "numeric", "name": "min_notice_hours"},
                {"type": "tag", "name": "department"},
                {"type": "tag", "name": "booking_required"},
            ],
            vector_dims=256,
        )

    def test_build_information_index_schemas_returns_one_schema_per_source(
        self,
    ) -> None:
        """All information sources receive a schema from the shared builder."""
        with patch(
            "blue_horizon.agents.information.retrieval.build_source_index_schema",
        ) as mock_build_source_index_schema:
            mock_build_source_index_schema.side_effect = (
                lambda *, source, vector_dims: f"{source.value}:{vector_dims}"
            )

            schemas = build_information_index_schemas(vector_dims=1536)

        assert schemas == {
            Source.FAQ: "faq:1536",
            Source.AMENITIES: "amenities:1536",
            Source.SERVICES: "services:1536",
        }
