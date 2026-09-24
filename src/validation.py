from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def validate_input(frame: pd.DataFrame, data_config: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    required = data_config.get("required_columns", [])
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        errors.append(f"missing_columns={missing_columns}")
        return ValidationResult(False, errors)
    max_missing_ratio = float(data_config.get("max_missing_ratio", 1.0))
    for column in required:
        if frame[column].isna().mean() > max_missing_ratio:
            errors.append(f"missing_ratio_exceeded={column}")
    for column, bounds in data_config.get("numeric_ranges", {}).items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty and (values < bounds[0]).any():
            errors.append(f"below_minimum={column}")
        if not values.empty and (values > bounds[1]).any():
            errors.append(f"above_maximum={column}")
    for column, allowed in data_config.get("categorical_values", {}).items():
        unexpected = set(frame[column].dropna().astype(str)) - set(allowed)
        if unexpected:
            errors.append(f"unexpected_categories={column}:{sorted(unexpected)}")
    return ValidationResult(not errors, errors)
