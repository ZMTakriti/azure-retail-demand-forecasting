# Azure Retail Demand Forecasting

![CI](https://github.com/ZMTakriti/azure-retail-demand-forecasting/actions/workflows/ci.yml/badge.svg)

End-to-end demand forecasting pipeline on Azure using the [Walmart M5 dataset](https://www.kaggle.com/c/m5-forecasting-accuracy). Raw sales data is ingested to ADLS Gen2, a PySpark ETL on Databricks produces curated Parquet, a LightGBM model generates batch forecasts written to an Azure SQL prediction store, a FastAPI endpoint serves results, and a Streamlit dashboard visualises them.

**[Live Dashboard →](https://zmt-azure-retail-demand-forecasting.streamlit.app/)**

## Architecture

```
                         Azure Cloud (rg-m5-forecast-dev)
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  ADLS Gen2 (stgm5forecastdev)                                        │
 │  ┌─────────────┐    ┌──────────────┐                                 │
 │  │  raw/       │    │  curated/    │                                 │
 │  │  M5 CSVs    │    │  Parquet     │                                 │
 │  └──────┬──────┘    └──────▲───────┘                                 │
 │         │                  │                                         │
 │         ▼                  │                                         │
 │  ┌────────────────────────────────┐                                  │
 │  │  Databricks (dbw-m5-forecast)  │                                  │
 │  │  PySpark ETL + LightGBM Train  │                                  │
 │  └─────────────┬──────────────────┘                                  │
 │                │ batch predictions                                   │
 │                ▼                                                     │
 │  ┌────────────────────────────────┐      ┌────────────────────────┐  │
 │  │  Azure SQL DB                  │◄─────│  FastAPI               │  │
 │  │  (prediction store)            │      │  (App Service F1)      │  │
 │  │  forecasts + model_runs tables │──────►  GET /forecast         │  │
 │  └────────────────────────────────┘      └───────────┬────────────┘  │
 │                                                      │               │
 └──────────────────────────────────────────────────────┼───────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │  Streamlit       │
                                              │  Dashboard       │
                                              └──────────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Storage | ADLS Gen2 | Raw CSV + curated Parquet zones |
| Compute | Azure Databricks (PySpark) | ETL transformations + model training |
| Model | LightGBM | Gradient-boosted demand forecasting |
| Prediction Store | Azure SQL Database | Batch forecasts + model run metadata |
| API | FastAPI on App Service (F1) | REST endpoint serving forecasts |
| Dashboard | Streamlit | Interactive sales + forecast visualization |
| CI/CD | GitHub Actions | Test, lint, deploy on push |

## Dataset

The [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset contains ~30K Walmart product sales time series across 10 stores in 3 US states. This project starts with store CA_1 (California) for faster iteration, with multi-store expansion planned.

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Cloud + Repo Scaffolding | Done | ADLS, Databricks, repo structure |
| 2. Ingestion + ETL | Done | PySpark ETL, curated Parquet |
| 3. Modeling + MLOps | Done | LightGBM training, prediction store, batch inference |
| 4. Serving + Dashboard | Done | FastAPI API, Streamlit dashboard, CI/CD deploy |
