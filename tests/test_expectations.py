import numpy as np
import pandas as pd

from src.expectations import GreatExpectationsValidator


def data_config() -> dict:
    return {
        "required_columns": ["total_price", "customer_state"],
        "column_types": {"total_price": "float64", "customer_state": "str"},
        "numeric_ranges": {"total_price": [0, 1000]},
        "categorical_values": {"customer_state": ["SP", "RJ"]},
        "max_missing_ratio": 0.0,
    }


def good_frame() -> pd.DataFrame:
    return pd.DataFrame([{"total_price": 100.0, "customer_state": "SP"}])


def test_valid_frame_passes():
    validator = GreatExpectationsValidator(data_config())
    result = validator.validate(good_frame())
    assert result.valid
    assert result.errors == []


def test_missing_required_column_is_rejected():
    validator = GreatExpectationsValidator(data_config())
    frame = good_frame().drop(columns=["customer_state"])
    result = validator.validate(frame)
    assert not result.valid
    assert any("expect_column_to_exist" in error for error in result.errors)


def test_wrong_column_type_is_rejected():
    validator = GreatExpectationsValidator(data_config())
    frame = good_frame()
    frame["total_price"] = frame["total_price"].astype(object)
    frame.loc[0, "total_price"] = "not-a-number"
    result = validator.validate(frame)
    assert not result.valid
    assert any("expect_column_values_to_be_of_type" in error for error in result.errors)


def test_out_of_range_value_is_rejected():
    validator = GreatExpectationsValidator(data_config())
    frame = good_frame()
    frame.loc[0, "total_price"] = 5000.0
    result = validator.validate(frame)
    assert not result.valid
    assert any("expect_column_values_to_be_between" in error for error in result.errors)


def test_unexpected_category_is_rejected():
    validator = GreatExpectationsValidator(data_config())
    frame = good_frame()
    frame.loc[0, "customer_state"] = "XX"
    result = validator.validate(frame)
    assert not result.valid
    assert any("expect_column_values_to_be_in_set" in error for error in result.errors)


def test_missing_ratio_exceeded_is_rejected():
    validator = GreatExpectationsValidator(data_config())
    frame = good_frame()
    frame["total_price"] = np.nan
    result = validator.validate(frame)
    assert not result.valid
    assert any("expect_column_values_to_not_be_null" in error for error in result.errors)
