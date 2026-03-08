# COMMAND ----------
# Import Spark SQL functions as F for concise, optimized column expressions in Fabric
from pyspark.sql import functions as F
# Import Spark SQL types for explicit schema control when needed
from pyspark.sql import types as T
# Import Spark SQL window functions to support ordered per-customer logic
from pyspark.sql import Window
# Import DeltaTable API to perform SCD Type 2 MERGE logic efficiently
from delta.tables import DeltaTable

# COMMAND ----------
# Set a reproducible seed so the mock dataset stays stable across reruns
seed = 42

# COMMAND ----------
# Create a base DataFrame with exactly 5,000 rows to drive deterministic mock data creation
base_df = spark.range(5000).withColumnRenamed("id", "row_id")

# COMMAND ----------
# Define a reusable array of customer locations to sample from
locations_array = F.array(
    F.lit("New York"),
    F.lit("California"),
    F.lit("Texas"),
    F.lit("Washington"),
    F.lit("Florida"),
    F.lit("Illinois"),
    F.lit("Massachusetts"),
    F.lit("Colorado"),
    F.lit("Georgia"),
    F.lit("Arizona")
)

# COMMAND ----------
# Generate a mock raw dataset with inconsistent timestamps, null prices, and negative quantity anomalies
raw_df = (
    base_df
    # Generate a UUID for each transaction to mimic real transaction identifiers
    .withColumn("Transaction_ID", F.expr("uuid()"))
    # Create Customer_ID values between 1 and 500
    .withColumn("Customer_ID", (F.floor(F.rand(seed) * 500) + F.lit(1)).cast(T.IntegerType()))
    # Create Product_ID values between 1 and 100
    .withColumn("Product_ID", (F.floor(F.rand(seed + 1) * 100) + F.lit(1)).cast(T.IntegerType()))
    # Generate a base event timestamp from a randomized Unix epoch offset
    .withColumn(
        "base_ts",
        F.to_timestamp(
            F.from_unixtime(
                F.unix_timestamp(F.current_timestamp())
                - (F.rand(seed + 2) * F.lit(365 * 86400)).cast(T.IntegerType())
            )
        )
    )
    # Format base timestamp into multiple string formats to simulate inconsistent source data
    .withColumn("ts_format_1", F.date_format(F.col("base_ts"), "yyyy-MM-dd HH:mm:ss"))
    # Second timestamp format for variation (common US format)
    .withColumn("ts_format_2", F.date_format(F.col("base_ts"), "MM/dd/yyyy HH:mm"))
    # Third timestamp format with slashes and date only
    .withColumn("ts_format_3", F.date_format(F.col("base_ts"), "yyyy/MM/dd"))
    # Fourth timestamp format with month name to increase parsing complexity
    .withColumn("ts_format_4", F.date_format(F.col("base_ts"), "dd-MMM-yyyy HH:mm:ss"))
    # Randomly choose one of the timestamp formats for each row
    .withColumn(
        "Timestamp",
        F.when(F.rand(seed + 3) < 0.25, F.col("ts_format_1"))
        .when(F.rand(seed + 4) < 0.50, F.col("ts_format_2"))
        .when(F.rand(seed + 5) < 0.75, F.col("ts_format_3"))
        .otherwise(F.col("ts_format_4"))
    )
    # Create a base positive quantity between 1 and 5
    .withColumn("base_qty", (F.floor(F.rand(seed + 6) * 5) + F.lit(1)).cast(T.IntegerType()))
    # Introduce negative anomalies in quantity with a small probability
    .withColumn(
        "Quantity",
        F.when(F.rand(seed + 7) < 0.05, -F.col("base_qty")).otherwise(F.col("base_qty"))
    )
    # Generate a base price between 5 and 500
    .withColumn("base_price", F.round(F.rand(seed + 8) * F.lit(495) + F.lit(5), 2))
    # Inject nulls into price with a small probability to simulate missing values
    .withColumn(
        "Price",
        F.when(F.rand(seed + 9) < 0.05, F.lit(None).cast(T.DoubleType()))
        .otherwise(F.col("base_price").cast(T.DoubleType()))
    )
    # Assign a customer location from a fixed list to emulate a small reference set
    .withColumn(
        "Customer_Location",
        F.element_at(locations_array, (F.floor(F.rand(seed + 10) * F.lit(10)) + F.lit(1)).cast(T.IntegerType()))
    )
    # Select only the required columns in the requested order
    .select(
        "Transaction_ID",
        "Customer_ID",
        "Product_ID",
        "Timestamp",
        "Quantity",
        "Price",
        "Customer_Location"
    )
)

# COMMAND ----------
# Add an ingestion timestamp to capture when the raw data landed in the Lakehouse
bronze_df = raw_df.withColumn("ingestion_timestamp", F.current_timestamp())

# COMMAND ----------
# Persist raw data exactly as received into the Bronze Delta table (append-only)
(
    bronze_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable("bronze_transactions")
)

# COMMAND ----------
# Read the Bronze table for Silver transformations
bronze_read_df = spark.table("bronze_transactions")

# COMMAND ----------
# Parse inconsistent timestamp strings into a single TimestampType column using coalesce for robustness
silver_ts_df = bronze_read_df.withColumn(
    "Timestamp",
    F.coalesce(
        F.to_timestamp(F.col("Timestamp"), "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(F.col("Timestamp"), "MM/dd/yyyy HH:mm"),
        F.to_timestamp(F.col("Timestamp"), "yyyy/MM/dd"),
        F.to_timestamp(F.col("Timestamp"), "dd-MMM-yyyy HH:mm:ss")
    )
)

# COMMAND ----------
# Filter out invalid or anomalous quantities (<= 0) to enforce business rules
silver_filtered_df = silver_ts_df.filter(F.col("Quantity") > F.lit(0))

# COMMAND ----------
# Compute median price per Product_ID using percentile_approx for scalable median calculation
median_price_df = (
    silver_filtered_df
    .groupBy("Product_ID")
    .agg(F.expr("percentile_approx(Price, 0.5)").alias("median_price"))
)

# COMMAND ----------
# Join median prices back and impute null prices using coalesce for fast null handling
silver_imputed_df = (
    silver_filtered_df
    .join(median_price_df, on="Product_ID", how="left")
    .withColumn("Price", F.coalesce(F.col("Price"), F.col("median_price")))
    .drop("median_price")
)

# COMMAND ----------
# Drop exact duplicate rows based on Transaction_ID to ensure uniqueness
silver_deduped_df = silver_imputed_df.dropDuplicates(["Transaction_ID"])

# COMMAND ----------
# Write clean Silver data as a managed Delta table (overwrite for repeatable pipeline runs)
(
    silver_deduped_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("silver_transactions")
)

# COMMAND ----------
# Read the Silver table for Gold transformations
silver_read_df = spark.table("silver_transactions")

# COMMAND ----------
# Create a window to select the most recent location per customer for initial dimension load
latest_customer_window = Window.partitionBy("Customer_ID").orderBy(F.col("Timestamp").desc())

# COMMAND ----------
# Build the initial customer dimension with the most recent location per customer
dim_customer_initial_df = (
    silver_read_df
    # Rank each customer record by most recent timestamp
    .withColumn("rn", F.row_number().over(latest_customer_window))
    # Keep only the latest record per customer
    .filter(F.col("rn") == F.lit(1))
    # Select only business attributes for the dimension
    .select("Customer_ID", "Customer_Location")
    # Add SCD Type 2 columns to track history
    .withColumn("valid_from", F.current_timestamp())
    .withColumn("valid_to", F.to_timestamp(F.lit("9999-12-31")))
    .withColumn("is_current", F.lit(True))
)

# COMMAND ----------
# Save the initial Dim_Customer Delta table (overwrite to keep demo repeatable)
(
    dim_customer_initial_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("Dim_Customer")
)

# COMMAND ----------
# Simulate a new batch of customer updates arriving from an upstream system
dim_customer_current_df = spark.table("Dim_Customer").filter(F.col("is_current") == F.lit(True))

# COMMAND ----------
# Select a subset of existing customers to update their location
existing_customer_updates_df = (
    dim_customer_current_df
    # Randomly sample ~10% of current customers for updates
    .sample(withReplacement=False, fraction=0.10, seed=seed)
    # Generate a new location using a different random choice
    .withColumn(
        "Customer_Location",
        F.when(
            F.col("Customer_Location")
            == F.element_at(locations_array, F.lit(1)),
            F.element_at(locations_array, F.lit(2))
        ).otherwise(F.element_at(locations_array, F.lit(1)))
    )
    # Add an update timestamp to track when this change was received
    .withColumn("update_ts", F.current_timestamp())
    # Keep only the columns needed for the updates feed
    .select("Customer_ID", "Customer_Location", "update_ts")
)

# COMMAND ----------
# Create a small set of brand-new customers that do not yet exist in the dimension
new_customer_updates_df = (
    spark.range(501, 521)
    # Rename the generated id to Customer_ID to match the dimension schema
    .withColumnRenamed("id", "Customer_ID")
    # Assign a random location to each new customer
    .withColumn(
        "Customer_Location",
        F.element_at(locations_array, (F.floor(F.rand(seed + 11) * F.lit(10)) + F.lit(1)).cast(T.IntegerType()))
    )
    # Add an update timestamp for consistency with the updates feed
    .withColumn("update_ts", F.current_timestamp())
)

# COMMAND ----------
# Union existing and new updates into a single updates DataFrame
customer_updates_df = existing_customer_updates_df.unionByName(new_customer_updates_df)

# COMMAND ----------
# Prepare a current-only snapshot of the dimension for SCD comparisons
dim_customer_current_snapshot_df = (
    spark.table("Dim_Customer")
    .filter(F.col("is_current") == F.lit(True))
    .select("Customer_ID", "Customer_Location")
)

# COMMAND ----------
# Join updates to the current dimension to detect changes and new records
updates_joined_df = (
    customer_updates_df.alias("u")
    .join(dim_customer_current_snapshot_df.alias("d"), on="Customer_ID", how="left")
    .select(
        F.col("u.Customer_ID").alias("Customer_ID"),
        F.col("u.Customer_Location").alias("new_location"),
        F.col("u.update_ts").alias("update_ts"),
        F.col("d.Customer_Location").alias("current_location"),
        F.col("d.Customer_ID").alias("dim_customer_id")
    )
)

# COMMAND ----------
# Flag rows that are new or have a changed location to drive SCD Type 2 updates
changes_df = (
    updates_joined_df
    .withColumn(
        "is_changed",
        F.when(F.col("dim_customer_id").isNull(), F.lit(True))
        .when(F.col("new_location") != F.col("current_location"), F.lit(True))
        .otherwise(F.lit(False))
    )
    .filter(F.col("is_changed") == F.lit(True))
)

# COMMAND ----------
# Define an open-ended valid_to timestamp to represent current records
open_ended_ts = F.to_timestamp(F.lit("9999-12-31"))

# COMMAND ----------
# Create the staged updates for MERGE: match rows for updates and non-match rows for inserts
updates_for_match_df = (
    changes_df
    # Merge key equals Customer_ID to match current records
    .withColumn("merge_key", F.col("Customer_ID"))
    # Rename the new location column to match the dimension schema
    .withColumn("Customer_Location", F.col("new_location"))
    # Add SCD columns for inserts
    .withColumn("valid_from", F.col("update_ts"))
    .withColumn("valid_to", open_ended_ts)
    .withColumn("is_current", F.lit(True))
    # Select the exact columns used in the merge staging table
    .select(
        "merge_key",
        "Customer_ID",
        "Customer_Location",
        "update_ts",
        "valid_from",
        "valid_to",
        "is_current",
        "dim_customer_id"
    )
)

# COMMAND ----------
# Add extra rows with null merge_key so changed records also get inserted as new versions
updates_for_insert_df = (
    updates_for_match_df
    # Keep only rows that correspond to existing customers (changes, not new)
    .filter(F.col("dim_customer_id").isNotNull())
    # Null merge_key ensures these rows won't match and will be inserted
    .withColumn("merge_key", F.lit(None).cast(T.IntegerType()))
)

# COMMAND ----------
# Union the update-match and insert-only rows into a single staged updates DataFrame
staged_updates_df = updates_for_match_df.unionByName(updates_for_insert_df)

# COMMAND ----------
# Load the Dim_Customer Delta table for SCD Type 2 MERGE operations
dim_customer_delta = DeltaTable.forName(spark, "Dim_Customer")

# COMMAND ----------
# Execute SCD Type 2 MERGE: expire old records and insert new versions where changes exist
(
    dim_customer_delta.alias("dim")
    .merge(
        staged_updates_df.alias("upd"),
        "dim.Customer_ID = upd.merge_key"
    )
    # Close out current records when a location change is detected
    .whenMatchedUpdate(
        condition="dim.is_current = true AND dim.Customer_Location <> upd.Customer_Location",
        set={
            "is_current": "false",
            "valid_to": "upd.update_ts"
        }
    )
    # Insert new records for new customers and for changed customers (via null merge_key rows)
    .whenNotMatchedInsert(
        values={
            "Customer_ID": "upd.Customer_ID",
            "Customer_Location": "upd.Customer_Location",
            "valid_from": "upd.valid_from",
            "valid_to": "upd.valid_to",
            "is_current": "upd.is_current"
        }
    )
    .execute()
)

# COMMAND ----------
# Build the Fact_Sales table by aggregating to daily grain per customer and product
fact_sales_df = (
    silver_read_df
    # Extract the date portion to aggregate at the daily level
    .withColumn("Sales_Date", F.to_date(F.col("Timestamp")))
    # Compute revenue per row to avoid repeating calculations in the aggregation
    .withColumn("Row_Revenue", F.col("Quantity") * F.col("Price"))
    # Group to daily grain for DirectLake-friendly fact size
    .groupBy("Sales_Date", "Customer_ID", "Product_ID")
    # Aggregate totals for quantity and revenue
    .agg(
        F.sum("Quantity").alias("Total_Quantity"),
        F.sum("Row_Revenue").alias("Total_Revenue")
    )
)

# COMMAND ----------
# Derive the min and max dates to generate a contiguous Dim_Date range
date_bounds_row = (
    fact_sales_df
    # Compute the earliest and latest dates present in the fact table
    .agg(
        F.min("Sales_Date").alias("min_date"),
        F.max("Sales_Date").alias("max_date")
    )
    # Collect the bounds to the driver for range construction
    .collect()[0]
)

# COMMAND ----------
# Extract min and max dates as Python date objects for Dim_Date generation
min_date = date_bounds_row["min_date"]

# COMMAND ----------
# Extract max date to complete the Dim_Date range
max_date = date_bounds_row["max_date"]

# COMMAND ----------
# Build Dim_Date as a dense calendar dimension for Power BI time intelligence
dim_date_df = (
    spark
    # Create a single-row DataFrame with the min and max bounds
    .createDataFrame([(min_date, max_date)], ["min_date", "max_date"])
    # Generate a continuous sequence of dates between bounds
    .select(F.explode(F.sequence("min_date", "max_date", F.expr("interval 1 day"))).alias("Date"))
    # Create an integer DateKey (YYYYMMDD) for star-schema joins
    .withColumn("DateKey", F.date_format(F.col("Date"), "yyyyMMdd").cast(T.IntegerType()))
    # Add year for easy slicing
    .withColumn("Year", F.year(F.col("Date")))
    # Add month number for hierarchical visuals
    .withColumn("Month", F.month(F.col("Date")))
    # Add month name for friendly labels
    .withColumn("Month_Name", F.date_format(F.col("Date"), "MMMM"))
    # Add quarter for fiscal grouping
    .withColumn("Quarter", F.quarter(F.col("Date")))
    # Add day of month for completeness
    .withColumn("Day", F.dayofmonth(F.col("Date")))
    # Add day of week number (1=Sunday, 7=Saturday in Spark)
    .withColumn("DayOfWeek", F.dayofweek(F.col("Date")))
    # Add day name for usability in reports
    .withColumn("Day_Name", F.date_format(F.col("Date"), "EEEE"))
    # Flag weekends to help common analytical filters
    .withColumn("Is_Weekend", F.col("DayOfWeek").isin([1, 7]))
)

# COMMAND ----------
# Persist Dim_Date as a Delta table (small dimension, no partitioning needed)
(
    dim_date_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("Dim_Date")
)

# COMMAND ----------
# Persist the Fact_Sales Delta table for Power BI DirectLake mode
(
    fact_sales_df
    .write
    .format("delta")
    # Partition by Sales_Date to improve pruning for time-sliced DirectLake queries
    .partitionBy("Sales_Date")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("Fact_Sales")
)

# COMMAND ----------
# Run OPTIMIZE periodically (not on every micro-batch) to compact files for DirectLake
spark.sql("OPTIMIZE Fact_Sales ZORDER BY (Customer_ID, Product_ID)")

# COMMAND ----------
# Run OPTIMIZE on Silver for better downstream read performance when filters are customer/product heavy
spark.sql("OPTIMIZE silver_transactions ZORDER BY (Customer_ID, Product_ID)")
