# Azure Data Factory - Pipeline Setup

ADF orchestrates the Databricks ETL notebook on a schedule. All configuration
is portal-only; nothing is committed as code.

## Prerequisites

- Azure Data Factory instance in the same subscription
- Azure Databricks workspace with the repo synced
- ADLS Gen2 storage account (`stgm5forecastdev`) with `raw` and `curated` containers
- Databricks secret scope `m5-scope` containing the ADLS account key + SQL credentials

## 1. Create Linked Service - ADLS Gen2

1. In ADF Studio, go to **Manage > Linked services > + New**.
2. Select **Azure Data Lake Storage Gen2**.
3. Set authentication to **Account key**.
4. Choose the storage account (`stgm5forecastdev`).
5. Test connection and save.

## 2. Create Linked Service - Databricks

1. **Manage > Linked services > + New** > **Azure Databricks**.
2. Select the Databricks workspace.
3. Choose **New job cluster** (or an existing interactive cluster for dev).
4. Set cluster configuration:
   - Node type: `Standard_DS3_v2` (or similar)
   - Workers: 1 (single-node is fine for CA_1 data volume)
   - Spark version: latest LTS (e.g., 13.3 LTS)
5. Test connection and save.

## 3. Create Pipeline

1. **Author > Pipelines > + New pipeline**.
2. Name it `etl-m5-train`.
3. Add two pipeline parameters (top-level, not activity-level):
   - `model_version` (String) — e.g. `v7.0`
   - `store_id` (String, default `CA_1`)
4. Drag a **Databricks Notebook** activity onto the canvas.
5. Configure the activity:
   - **Linked service**: select the Databricks linked service from step 2.
   - **Notebook path**: `/Repos/<user>/azure-demand-forecasting/notebooks/etl_m5_databricks`
   - **Base parameters**: add two entries:
     - `model_version` → `@pipeline().parameters.model_version`
     - `store_id` → `@pipeline().parameters.store_id`
6. Validate and publish.

## 4. Configure Trigger

### Manual (dev)

Use **Trigger now** from the pipeline toolbar. You will be prompted to fill in
`model_version` (e.g. `v7.0`) and `store_id` (`CA_1`) before the run starts.

### Scheduled (prod)

1. **Add trigger > New/Edit** on the pipeline.
2. Choose **Schedule**.
3. Set recurrence (e.g., weekly or monthly — M5 is a static dataset).
4. In the trigger parameters, set `model_version` to a dynamic expression:
   `@formatDateTime(trigger().startTime, 'v'yyyy-MM-dd'')` — produces e.g. `v2026-03-01`.
5. Set `store_id` to `CA_1` (or parameterise for multi-store later).
6. Activate and publish.

## 5. Monitoring

- **Monitor > Pipeline runs** shows execution history.
- Click a run to see the Databricks notebook output link.
- Failed runs surface the Spark driver log from Databricks.
- If a run fails mid-way, cell 13 can be re-run safely — it loads metrics and
  data from ADLS checkpoints and calls `delete_model_version` before writing.
