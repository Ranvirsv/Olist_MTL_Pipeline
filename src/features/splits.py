import duckdb
import pandas as pd
import numpy as np
from loguru import logger
import os
from dotenv import load_dotenv
from haversine import haversine_distance

load_dotenv()

root_dir = os.getenv("PROJECT_ROOT")

def main():
    logger.info("Connecting to DuckDB...")

    conn = duckdb.connect(f"{root_dir}/data/olist.duckdb", read_only=True)

    logger.info("Loading data...")
    df = conn.execute("SELECT * FROM analytical_base_table").fetchdf()
    df['geo_distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])

    p99 = df['delivery_days'].quantile(0.99)

    df_model = df[df['delivery_days'] <= p99].copy()
    df_model = df_model.sort_values('order_purchase_timestamp').reset_index(drop=True)

    df_model = df_model[df_model['order_purchase_timestamp'] >= '2017-01-01']

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Capped shape: {df_model.shape}")

    os.makedirs(f"{root_dir}/data/splits", exist_ok=True)

    logger.info("Saving df_model...")
    df_model.to_parquet(f"{root_dir}/data/splits/df_model.parquet", index=False)

    n = len(df_model)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_model.iloc[:train_end]
    val_df   = df_model.iloc[train_end:val_end]
    test_df  = df_model.iloc[val_end:]

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {val_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    logger.info("Saving splits...")
    train_df.to_parquet(f"{root_dir}/data/splits/train.parquet", index=False)
    val_df.to_parquet(f"{root_dir}/data/splits/val.parquet", index=False)
    test_df.to_parquet(f"{root_dir}/data/splits/test.parquet", index=False)

    ## Close connection
    logger.info("Closing connection...")
    conn.close()

if __name__ == "__main__":
    main()
    