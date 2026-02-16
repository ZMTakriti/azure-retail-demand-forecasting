"""
ETL transformations for M5 sales data.

This module contains functions to transform the raw M5 sales data
from wide format to long format suitable for time series analysis.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, regexp_replace


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


def transform_sales_to_long(
    spark: SparkSession, raw_path: str, store_id: str = "CA_1", add_day_num: bool = True
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

    Returns
    -------
    DataFrame
        Transformed long-format DataFrame ready for analysis/modeling.

    Example
    -------
    >>> df = transform_sales_to_long(spark, raw_path, store_id="CA_1")
    >>> df.printSchema()
    root
     |-- id: string
     |-- item_id: string
     |-- dept_id: string
     |-- store_id: string
     |-- cat_id: string
     |-- state_id: string
     |-- d: string
     |-- sales: integer
     |-- day_num: integer
    """
    df_raw = read_sales_raw(spark, raw_path)
    df_store = filter_by_store(df_raw, store_id)
    df_long = wide_to_long(df_store)

    if add_day_num:
        df_long = add_day_number(df_long)

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
