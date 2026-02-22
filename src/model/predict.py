"""Batch prediction: generate forecasts and write to the prediction store.

This module loads a trained LightGBM model, generates predictions for
all items in a store, and writes results to Azure SQL DB.
"""

import json

import pandas as pd

from src.db.connection import get_jdbc_properties, get_jdbc_url


def delete_model_version(model_version: str) -> None:
    """Delete all data for a model version from forecasts, sales_history, and model_runs.

    Call this before re-writing a version to prevent duplicate rows.

    Parameters
    ----------
    model_version : str
        Version tag to purge (e.g. "v6.0").
    """
    from src.db.connection import get_connection  # noqa: PLC0415

    conn = get_connection()
    cursor = conn.cursor()
    for table in ("forecasts", "sales_history", "model_runs"):
        cursor.execute(f"DELETE FROM {table} WHERE model_version = %s", (model_version,))
    conn.commit()
    cursor.close()
    conn.close()


def write_forecasts(
    predictions: pd.DataFrame,
    model_version: str,
) -> int:
    """Write forecast rows to the Azure SQL DB `forecasts` table via Spark JDBC.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: item_id, store_id, forecast_date, predicted_sales.
    model_version : str
        Version tag for this model run (e.g. "v1.0").

    Returns
    -------
    int
        Number of rows inserted.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415

    df = predictions.copy()
    df["model_version"] = model_version

    spark = SparkSession.getActiveSession()
    spark_df = spark.createDataFrame(df)

    (
        spark_df.write.format("jdbc")
        .option("url", get_jdbc_url())
        .option("dbtable", "forecasts")
        .options(**get_jdbc_properties())
        .mode("append")
        .save()
    )

    return len(df)


def write_sales_history(
    history: pd.DataFrame,
    model_version: str,
) -> int:
    """Write validation-period actuals to the `sales_history` table via Spark JDBC.

    Parameters
    ----------
    history : pd.DataFrame
        Must have columns: item_id, store_id, sale_date, actual_sales.
    model_version : str
        Version tag for this model run.

    Returns
    -------
    int
        Number of rows inserted.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415

    df = history.copy()
    df["model_version"] = model_version

    spark = SparkSession.getActiveSession()
    spark_df = spark.createDataFrame(df)

    (
        spark_df.write.format("jdbc")
        .option("url", get_jdbc_url())
        .option("dbtable", "sales_history")
        .options(**get_jdbc_properties())
        .mode("append")
        .save()
    )

    return len(df)


def log_model_run(
    model_version: str,
    mae: float,
    rmse: float,
    weighted_mae: float,
    horizon_days: int,
    num_items: int,
    store_id: str,
    parameters: dict | None = None,
) -> None:
    """Log a model training run to the `model_runs` table via Spark JDBC.

    After writing, promotes this version to active (is_active = 1) and
    deactivates all other versions.

    Parameters
    ----------
    model_version : str
        Unique version tag (e.g. "v1.0").
    mae : float
        Mean absolute error on holdout.
    rmse : float
        Root mean squared error on holdout.
    weighted_mae : float
        Volume-weighted MAE across items.
    horizon_days : int
        Number of days forecasted.
    num_items : int
        Number of items forecasted.
    store_id : str
        Store identifier (e.g. "CA_1").
    parameters : dict, optional
        Model hyperparameters to log as JSON.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415

    df = pd.DataFrame(
        [
            {
                "model_version": model_version,
                "mae": mae,
                "rmse": rmse,
                "weighted_mae": weighted_mae,
                "horizon_days": horizon_days,
                "num_items": num_items,
                "store_id": store_id,
                "parameters": json.dumps(parameters) if parameters else None,
            }
        ]
    )

    spark = SparkSession.getActiveSession()
    spark_df = spark.createDataFrame(df)

    (
        spark_df.write.format("jdbc")
        .option("url", get_jdbc_url())
        .option("dbtable", "model_runs")
        .options(**get_jdbc_properties())
        .mode("append")
        .save()
    )

    # Promote this version to active; deactivate all others
    from src.db.connection import get_connection  # noqa: PLC0415

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE model_runs SET is_active = 0 WHERE model_version != %s", (model_version,)
    )
    cursor.execute("UPDATE model_runs SET is_active = 1 WHERE model_version = %s", (model_version,))
    conn.commit()
    cursor.close()
    conn.close()
