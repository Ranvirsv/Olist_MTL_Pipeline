import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pipeline import build_preprocessor
import os
from loguru import logger
from sklearn.preprocessing import StandardScaler
import joblib

load_dotenv()

def process_targets(df):
    ## Target A: Delivery Days
    ## For regression tasks, PyTorch needs data in float32 format
    y_delivery_days = df['delivery_days'].astype("float32").values

    ## Target B: Review Scores
    ## For classification task, we need to binarize the targets, Classification - needs int64 for PyTorch
    ## Anything below 4 is a 0, and anything 4 or above is a 1
    y_review_score = (df['review_score'] >= 4).astype("int64").values

    return y_delivery_days, y_review_score

def main():
    root_dir = os.getenv("PROJECT_ROOT")

    logger.info("Loading data...")
    train_df = pd.read_parquet(f"{root_dir}/data/splits/train.parquet")
    val_df = pd.read_parquet(f"{root_dir}/data/splits/val.parquet")
    test_df = pd.read_parquet(f"{root_dir}/data/splits/test.parquet")

    train_df = train_df.dropna(subset=['delivery_days', 'review_score']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['delivery_days', 'review_score']).reset_index(drop=True)
    test_df = test_df.dropna(subset=['delivery_days', 'review_score']).reset_index(drop=True)

    target_cols = ['delivery_days', 'review_score']

    X_train = train_df.drop(columns=target_cols)
    X_val = val_df.drop(columns=target_cols)
    X_test = test_df.drop(columns=target_cols)

    logger.info("Building preprocessor...")
    preprocessor = build_preprocessor()

    logger.info("Fitting and Transforming Train Data...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    logger.info("Fitting Validation and Test Data...")
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    ## Process targets

    y_train_delivery_days, y_train_review_score = process_targets(train_df)
    y_val_delivery_days, y_val_review_score = process_targets(val_df)
    y_test_delivery_days, y_test_review_score = process_targets(test_df)

    ## Normalize delivery targets — fit on train only, transform val/test
    logger.info("Normalizing delivery targets...")
    delivery_scaler = StandardScaler()
    y_train_delivery_days = delivery_scaler.fit_transform(y_train_delivery_days.reshape(-1, 1)).flatten().astype("float32")
    y_val_delivery_days = delivery_scaler.transform(y_val_delivery_days.reshape(-1, 1)).flatten().astype("float32")
    y_test_delivery_days = delivery_scaler.transform(y_test_delivery_days.reshape(-1, 1)).flatten().astype("float32")

    logger.info("Saving preprocessed data...")

    os.makedirs(f"{root_dir}/data/preprocessed", exist_ok=True)

    np.save(f"{root_dir}/data/preprocessed/train_features.npy", X_train_preprocessed)
    np.save(f"{root_dir}/data/preprocessed/val_features.npy", X_val_preprocessed)
    np.save(f"{root_dir}/data/preprocessed/test_features.npy", X_test_preprocessed)

    np.save(f"{root_dir}/data/preprocessed/train_delivery_days.npy", y_train_delivery_days)
    np.save(f"{root_dir}/data/preprocessed/val_delivery_days.npy", y_val_delivery_days)
    np.save(f"{root_dir}/data/preprocessed/test_delivery_days.npy", y_test_delivery_days)

    np.save(f"{root_dir}/data/preprocessed/train_review_score.npy", y_train_review_score)
    np.save(f"{root_dir}/data/preprocessed/val_review_score.npy", y_val_review_score)
    np.save(f"{root_dir}/data/preprocessed/test_review_score.npy", y_test_review_score)


    # Also save to src/models/inference/ so the scalers are git-tracked and available for deployment
    os.makedirs(f"{root_dir}/src/models/inference", exist_ok=True)
    joblib.dump(delivery_scaler, f"{root_dir}/src/models/inference/delivery_scaler.pkl")
    joblib.dump(preprocessor, f"{root_dir}/src/models/inference/feature_preprocessor.pkl")
    logger.info("Scalers saved to data/preprocessed/ and src/models/inference/ (git-tracked)")

if __name__ == "__main__":
    main()