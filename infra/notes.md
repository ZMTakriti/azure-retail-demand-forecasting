# Azure Data Factory - Pipeline Setup

ADF orchestrates the Databricks ETL notebook on a schedule. All configuration
is portal-only; nothing is committed as code.

## Prerequisites

- Azure Data Factory instance in the same subscription
- Azure Databricks workspace with the repo synced
- ADLS Gen2 storage account (`stgm5forecastdev`) with `raw` and `curated` containers
- Databricks secret scope `m5-scope` containing the ADLS account key

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
2. Name it `etl-m5-daily`.
3. Drag a **Databricks Notebook** activity onto the canvas.
4. Configure the activity:
   - **Linked service**: select the Databricks linked service from step 2.
   - **Notebook path**: `/Repos/<user>/azure-demand-forecasting/notebooks/etl_m5_databricks`
   - Leave base parameters empty (the notebook reads config from its own cells).
5. Validate and publish.

## 4. Configure Trigger

### Manual (dev)

Use **Trigger now** from the pipeline toolbar to run on demand.

### Scheduled (prod)

1. **Add trigger > New/Edit** on the pipeline.
2. Choose **Schedule**.
3. Set recurrence (e.g., daily at 02:00 UTC).
4. Activate and publish.

## 5. Monitoring

- **Monitor > Pipeline runs** shows execution history.
- Click a run to see the Databricks notebook output link.
- Failed runs surface the Spark driver log from Databricks.
