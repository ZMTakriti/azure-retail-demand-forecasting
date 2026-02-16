-- Analytics Queries for the Prediction Store
-- Run these against Azure SQL DB to inspect forecasts and model performance.

-- 1. Latest forecasts for a specific item
SELECT item_id, store_id, forecast_date, predicted_sales, model_version
FROM forecasts
WHERE item_id = 'FOODS_3_090'
  AND store_id = 'CA_1'
ORDER BY forecast_date;

-- 2. All model runs ordered by performance
SELECT run_id, model_version, trained_at, smape, mae, horizon_days, num_items
FROM model_runs
ORDER BY smape ASC;

-- 3. Latest model run details
SELECT TOP 1 *
FROM model_runs
ORDER BY trained_at DESC;

-- 4. Forecast summary by item (latest model version)
SELECT
    f.item_id,
    f.store_id,
    COUNT(*) AS forecast_days,
    AVG(f.predicted_sales) AS avg_predicted_sales,
    MIN(f.forecast_date) AS first_date,
    MAX(f.forecast_date) AS last_date
FROM forecasts f
INNER JOIN (
    SELECT TOP 1 model_version
    FROM model_runs
    ORDER BY trained_at DESC
) m ON f.model_version = m.model_version
GROUP BY f.item_id, f.store_id
ORDER BY f.item_id;

-- 5. Compare predictions across model versions for one item
SELECT
    f.model_version,
    m.smape AS model_smape,
    AVG(f.predicted_sales) AS avg_prediction,
    COUNT(*) AS num_forecasts
FROM forecasts f
JOIN model_runs m ON f.model_version = m.model_version
WHERE f.item_id = 'FOODS_3_090'
  AND f.store_id = 'CA_1'
GROUP BY f.model_version, m.smape
ORDER BY m.trained_at DESC;

-- 6. Top 10 items by highest predicted demand (latest model)
SELECT TOP 10
    f.item_id,
    SUM(f.predicted_sales) AS total_predicted_sales
FROM forecasts f
INNER JOIN (
    SELECT TOP 1 model_version
    FROM model_runs
    ORDER BY trained_at DESC
) m ON f.model_version = m.model_version
GROUP BY f.item_id
ORDER BY total_predicted_sales DESC;
