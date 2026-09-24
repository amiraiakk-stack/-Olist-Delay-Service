"""Great Expectations validation for the request contract."""

import logging
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import (
    DataContextConfig,
    InMemoryStoreBackendDefaults,
    ProgressBarsConfig,
)

from .validation import ValidationResult

# quiet GE's own INFO-level logging
logging.getLogger("great_expectations").setLevel(logging.WARNING)


def build_expectation_suite(data_config: dict[str, Any]) -> Any:
    suite = gx.ExpectationSuite(name="olist_inference_input")
    required_columns = data_config.get("required_columns", [])
    max_missing_ratio = float(data_config.get("max_missing_ratio", 1.0))

    for column in required_columns:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column, mostly=1.0 - max_missing_ratio
            )
        )
    for column, type_name in data_config.get("column_types", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(column=column, type_=type_name)
        )
    for column, bounds in data_config.get("numeric_ranges", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=bounds[0], max_value=bounds[1], mostly=1.0
            )
        )
    for column, allowed in data_config.get("categorical_values", {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=allowed)
        )
    return suite


class GreatExpectationsValidator:
    """Validates a request DataFrame against the configured expectation suite."""

    def __init__(self, data_config: dict[str, Any]):
        context = gx.get_context(
            project_config=DataContextConfig(
                store_backend_defaults=InMemoryStoreBackendDefaults(),
                progress_bars=ProgressBarsConfig(globally=False),
            ),
            mode="ephemeral",
        )
        self._suite = build_expectation_suite(data_config)
        data_source = context.data_sources.add_pandas("olist_inference_pandas")
        data_asset = data_source.add_dataframe_asset(name="olist_inference_requests")
        self._batch_definition = data_asset.add_batch_definition_whole_dataframe(
            "olist_inference_batch"
        )

    def validate(self, frame: pd.DataFrame) -> ValidationResult:
        batch = self._batch_definition.get_batch(batch_parameters={"dataframe": frame})
        result = batch.validate(self._suite)
        errors = [
            f"{item.expectation_config.type}:{item.expectation_config.kwargs.get('column')}"
            for item in result.results
            if not item.success
        ]
        return ValidationResult(valid=bool(result.success), errors=errors)
