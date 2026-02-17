"""
ETL transformations for M5 sales data.

This module contains functions to transform the raw M5 sales data
from wide format to long format suitable for time series analysis.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import DateType


def read_sales_raw(spark: SparkSession, raw_path: str) -> DataFrame:
    """
    Read raw sales_train_evaluation.csv from ADLS.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    raw_path : str
        Base path to raw container (e.g., abfss://raw@account.dfs.core.windows.net).

    Returns
    -------
    DataFrame
        Raw sales DataFrame with wide format (d_1, d_2, ..., d_1913 columns).
    """
    file_path = f"{raw_path}/sales_train_evaluation.csv"
    return spark.read.option("header", "true").csv(file_path)


def filter_by_store(df: DataFrame, store_id: str) -> DataFrame:
    """
    Filter DataFrame to a single store.

    Parameters
    ----------
    df : DataFrame
        Sales DataFrame with store_id column.
    store_id : str
        Store identifier (e.g., 'CA_1', 'CA_2', 'TX_1').

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only rows for the specified store.
    """
    return df.filter(col("store_id") == store_id)


def wide_to_long(df: DataFrame) -> DataFrame:
    """
    Transform sales data from wide format to long format.

    Converts columns d_1, d_2, ..., d_N into rows with (d, sales) pairs
    using SQL stack() function for efficient unpivoting.

    Parameters
    ----------
    df : DataFrame
        Sales DataFrame with id columns and d_* value columns.

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns:
        [id, item_id, dept_id, store_id, cat_id, state_id, d, sales]
    """
    id_cols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]
    value_cols = [c for c in df.columns if c.startswith("d_")]

    # Build stack expression: stack(N, 'd_1', d_1, 'd_2', d_2, ...)
    stack_args = ", ".join([f"'{c}', {c}" for c in value_cols])
    stack_expr = f"stack({len(value_cols)}, {stack_args}) as (d, sales)"

    df_long = df.select(*id_cols, *value_cols).selectExpr(*id_cols, stack_expr)

    # Cast sales to integer
    df_long = df_long.withColumn("sales", col("sales").cast("int"))

    return df_long


def add_day_number(df: DataFrame) -> DataFrame:
    """
    Add day_num column by extracting numeric part from d column.

    Parameters
    ----------
    df : DataFrame
        Long-format DataFrame with 'd' column (e.g., 'd_1', 'd_100').

    Returns
    -------
    DataFrame
        DataFrame with additional 'day_num' integer column.
    """
    return df.withColumn("day_num", regexp_replace("d", "d_", "").cast("int"))


def read_calendar(spark: SparkSession, raw_path: str) -> DataFrame:
    """
    Read calendar.csv from the raw container.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    raw_path : str
        Base path to raw container.

    Returns
    -------
    DataFrame
        Calendar DataFrame with columns: d, cal_date, wday, month, year,
        event_name_1, event_type_1.
    """
    file_path = f"{raw_path}/calendar.csv"
    df_cal = spark.read.option("header", "true").csv(file_path)
    df_cal = df_cal.select(
        col("d"),
        col("date").alias("cal_date"),
        col("wday").cast("int"),
        col("month").cast("int"),
        col("year").cast("int"),
        col("event_name_1"),
        col("event_type_1"),
    )
    df_cal = df_cal.withColumn("cal_date", col("cal_date").cast(DateType()))
    return df_cal


def join_calendar(df_long: DataFrame, df_cal: DataFrame) -> DataFrame:
    """
    Join long-format sales with calendar data on the 'd' column.

    Parameters
    ----------
    df_long : DataFrame
        Long-format sales DataFrame with 'd' column.
    df_cal : DataFrame
        Calendar DataFrame from read_calendar.

    Returns
    -------
    DataFrame
        Sales DataFrame enriched with cal_date, wday, month, year,
        event_name_1, event_type_1.
    """
    return df_long.join(df_cal, on="d", how="left")


def transform_sales_to_long(
    spark: SparkSession,
    raw_path: str,
    store_id: str = "CA_1",
    add_day_num: bool = True,
    calendar_path: str | None = None,
) -> DataFrame:
    """
    Full ETL pipeline: read raw data, filter store, convert to long format.

    This is the main entry point for the ETL transformation.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    raw_path : str
        Base path to raw container.
    store_id : str, optional
        Store to filter to (default: 'CA_1').
    add_day_num : bool, optional
        Whether to add day_num column (default: True).
    calendar_path : str, optional
        Base path to raw container for calendar.csv. When provided, enriches
        the output with calendar columns (cal_date, wday, month, year,
        event_name_1, event_type_1). Default: None (no calendar join).

    Returns
    -------
    DataFrame
        Transformed long-format DataFrame ready for analysis/modeling.
    """
    df_raw = read_sales_raw(spark, raw_path)
    df_store = filter_by_store(df_raw, store_id)
    df_long = wide_to_long(df_store)

    if add_day_num:
        df_long = add_day_number(df_long)

    if calendar_path is not None:
        df_cal = read_calendar(spark, calendar_path)
        df_long = join_calendar(df_long, df_cal)

    return df_long


def write_parquet(
    df: DataFrame, output_path: str, partition_by: list[str] | None = None, mode: str = "overwrite"
) -> None:
    """
    Write DataFrame to Parquet format.

    Parameters
    ----------
    df : DataFrame
        DataFrame to write.
    output_path : str
        Destination path (e.g., abfss://curated@.../m5_daily_ca1/).
    partition_by : list[str], optional
        Columns to partition by (default: None).
    mode : str, optional
        Write mode (default: 'overwrite').
    """
    writer = df.write.mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.parquet(output_path)
