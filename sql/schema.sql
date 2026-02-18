-- Prediction Store Schema for Azure SQL Database
-- Run this once against your Azure SQL DB to create the tables.
--
-- Usage:
--   sqlcmd -S your-server.database.windows.net -d your-db -U your-user -P your-pass -i sql/schema.sql
--   OR run in Azure Portal Query Editor

-- Store batch forecast results from model inference
CREATE TABLE forecasts (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    item_id         VARCHAR(30)   NOT NULL,
    store_id        VARCHAR(10)   NOT NULL,
    forecast_date   DATE          NOT NULL,
    predicted_sales FLOAT         NOT NULL,
    model_version   VARCHAR(20)   NOT NULL,
    created_at      DATETIME      DEFAULT GETDATE()
);

-- Index for the most common query pattern: lookup by item + store
CREATE INDEX ix_forecasts_item_store
    ON forecasts (item_id, store_id, forecast_date);

-- Index for filtering by model version (useful for comparing runs)
CREATE INDEX ix_forecasts_model_version
    ON forecasts (model_version);

-- Store metadata for each model training run
CREATE TABLE model_runs (
    run_id          INT IDENTITY(1,1) PRIMARY KEY,
    model_version   VARCHAR(20)   NOT NULL UNIQUE,
    trained_at      DATETIME      DEFAULT GETDATE(),
    mae             FLOAT,
    rmse            FLOAT,
    horizon_days    INT           NOT NULL,
    num_items       INT,
    store_id        VARCHAR(10),
    parameters      NVARCHAR(MAX) -- JSON blob with hyperparameters
);
