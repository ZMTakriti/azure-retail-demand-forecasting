"""Unit tests for ETL schema validation."""

# Expected columns in the transformed long-format DataFrame
EXPECTED_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "store_id",
    "cat_id",
    "state_id",
    "d",
    "sales",
    "day_num",
    "cal_date",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
]

# Required columns for modeling
REQUIRED_FOR_MODEL = ["item_id", "store_id", "day_num", "sales"]


class TestSchemaExpectations:
    """Tests for validating schema expectations."""

    def test_expected_columns_defined(self):
        """Verify expected columns list is not empty."""
        assert len(EXPECTED_COLUMNS) > 0

    def test_required_columns_subset(self):
        """Required modeling columns should be subset of expected columns."""
        for col in REQUIRED_FOR_MODEL:
            assert col in EXPECTED_COLUMNS, f"Required column '{col}' not in expected schema"

    def test_no_duplicate_columns(self):
        """Schema should not have duplicate column names."""
        assert len(EXPECTED_COLUMNS) == len(set(EXPECTED_COLUMNS))

    def test_day_column_present(self):
        """Day identifier column must be present."""
        assert "d" in EXPECTED_COLUMNS or "day_num" in EXPECTED_COLUMNS

    def test_sales_column_present(self):
        """Sales target column must be present."""
        assert "sales" in EXPECTED_COLUMNS


def validate_dataframe_schema(df_columns: list[str]) -> tuple[bool, list[str]]:
    """
    Validate that a DataFrame has the expected schema.

    Parameters
    ----------
    df_columns : list[str]
        List of column names from a DataFrame.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of missing columns)
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df_columns]
    return len(missing) == 0, missing


class TestSchemaValidation:
    """Tests for the schema validation helper function."""

    def test_valid_schema(self):
        """Valid schema should pass validation."""
        is_valid, missing = validate_dataframe_schema(EXPECTED_COLUMNS)
        assert is_valid
        assert missing == []

    def test_extra_columns_ok(self):
        """Extra columns should not cause validation to fail."""
        columns = EXPECTED_COLUMNS + ["extra_col", "another_col"]
        is_valid, missing = validate_dataframe_schema(columns)
        assert is_valid

    def test_missing_columns_detected(self):
        """Missing columns should be detected."""
        columns = ["id", "item_id"]  # Missing many columns
        is_valid, missing = validate_dataframe_schema(columns)
        assert not is_valid
        assert "sales" in missing
        assert "store_id" in missing

    def test_empty_columns(self):
        """Empty column list should fail validation."""
        is_valid, missing = validate_dataframe_schema([])
        assert not is_valid
        assert len(missing) == len(EXPECTED_COLUMNS)
