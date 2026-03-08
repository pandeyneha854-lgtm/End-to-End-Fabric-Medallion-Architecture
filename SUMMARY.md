# Project Summary: Microsoft Fabric Medallion Architecture (PySpark)

This project delivers an end-to-end Bronze–Silver–Gold pipeline in a Microsoft Fabric Lakehouse using PySpark and Delta Lake. It generates a realistic mock e-commerce dataset, writes raw data to Bronze, applies data quality rules in Silver, and builds a Gold star schema with Slowly Changing Dimensions (SCD Type 2) and a daily-grain fact table suitable for Power BI DirectLake.

## What The Pipeline Does
1. Generates 5,000 mock e-commerce transactions with UUIDs, inconsistent timestamp formats, negative quantity anomalies, null prices, and customer locations.
2. Writes Bronze data exactly as received, adding `ingestion_timestamp` for lineage.
3. Cleans and validates Silver data by standardizing timestamps, removing invalid quantities, imputing missing prices by product median, and removing duplicate transactions.
4. Builds Gold tables:
   - `Dim_Customer` with SCD Type 2 tracking for customer location changes.
   - `Dim_Date` as a dense calendar dimension for time intelligence.
   - `Fact_Sales` aggregated to daily grain per customer and product with total revenue.
5. Applies DirectLake-friendly strategies: date partitioning on the fact table and periodic `OPTIMIZE` with `ZORDER`.

## Data Layers And Tables
1. Bronze
   - `bronze_transactions`: raw append-only Delta table with ingestion metadata.
2. Silver
   - `silver_transactions`: cleaned Delta table with standardized timestamps and imputed prices.
3. Gold
   - `Dim_Customer`: SCD Type 2 dimension with `valid_from`, `valid_to`, and `is_current`.
   - `Dim_Date`: calendar dimension derived from fact date bounds.
   - `Fact_Sales`: daily-grain fact with `Total_Quantity` and `Total_Revenue`.

## Key Design Choices
1. Timestamp standardization uses `coalesce` across multiple `to_timestamp` patterns to handle inconsistent raw formats efficiently.
2. Median price imputation uses `percentile_approx`, which scales well for large datasets in Fabric.
3. SCD Type 2 is implemented with the Delta Lake `MERGE` API to expire old records and insert new versions in a single transaction.
4. Fact table is partitioned by `Sales_Date` to improve pruning and reduce scan cost in DirectLake.
5. `OPTIMIZE ... ZORDER BY` is recommended for compaction and data skipping on common filters.

## How To Run
1. Open `medallion_pyspark_notebook.py` in a Fabric notebook or compatible Spark environment.
2. Run cells in order to generate the mock data and build Bronze, Silver, and Gold tables.
3. Run `OPTIMIZE` steps periodically (not every micro-batch) for DirectLake performance.

## Notes
1. The notebook is intentionally verbose and line‑by‑line commented for clarity.
2. Table names are stable and can be bound directly into Power BI DirectLake models.
