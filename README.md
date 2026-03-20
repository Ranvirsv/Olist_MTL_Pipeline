<div align="center">
  
# 📦 Olist Multi-Task Learning (MTL) Pipeline

**A Full-Stack Data Product for E-commerce Logistics and Customer Satisfaction**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5.0-yellow.svg)](https://duckdb.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-ranvirsv.com-C8A96E.svg)](https://ranvirsv.com/work/olist-mtl-pipeline)

</div>

---

## 🎯 The End Goal and The "Why"

### 🏢 The Business Objective

In e-commerce, **logistics and customer satisfaction are intrinsically linked**. A delayed shipment almost guarantees a negative review, regardless of product quality.

The goal of this project is to build a system that simultaneously predicts two critical outcomes for a delivered order:

1. **Delivery Time (Regression)**: How many days the package should have taken to arrive?
2. **Customer Satisfaction (Classification)**: Will the customer leave a positive review (score >= 4), or a negative one?

### 🧠 The Architectural "Why"

Standard models fail here. Treating these as two completely separate models ignores the heavy correlation between shipping delays and angry customers. Conversely, a rigid Shared-bottom model suffers from **negative transfer** because predicting numerical "days" and predicting human "sentiment" require fundamentally different mathematical representations.

**The Solution:** We are building a **Multi-gate Mixture-of-Experts (MMoE)**. This allows the neural network to share broad latent features (like geographic distance or freight cost) while utilizing independent gates to protect the task-specific heads from cross-contamination.

### ⚙️ The Engineering "Why"

A Jupyter notebook cannot run in production. This project demonstrates the jump from academic data science to a **Lead Machine Learning Engineer** role by integrating serverless SQL, strict data validation, version control, and high-performance serving to handle the entire lifecycle of enterprise data.

---

## 🛠 The Tech Stack

| Layer                          | Tool                                                   | Status    |
| ------------------------------ | ------------------------------------------------------ | --------- |
| **Environment & Dependencies** | `uv` _(Rust-based, reproducible virtual environments)_ | ✅ Active |
| **Data Engineering (ETL)**     | `DuckDB` _(Serverless analytical SQL)_                 | ✅ Active |
| **Data Contracts**             | `Pydantic` _(Strict schema validation)_                | ✅ Active |
| **Statistical Research**       | `scipy.stats` _(Hypothesis testing)_                   | ✅ Active |
| **Visualization**              | `matplotlib` + `seaborn` _(Custom Plotter library)_    | ✅ Active |
| **Data Versioning**            | `DVC` _(Git for massive datasets)_                     | ✅ Active |
| **Deep Learning**              | `PyTorch` _(Custom MMoE architecture)_                 | ✅ Active |
| **Experiment Tracking**        | `MLflow` _(Hyperparameters & loss curves)_             | ✅ Active |
| **Model Export**               | `ONNX Runtime` _(Optimized inference)_                 | ✅ Active |
| **Serving**                    | `FastAPI` _(REST API)_                                 | ✅ Active |

---

## 📂 Project Structure

```
Olist_MTL_Pipeline/
├── main.py                           # Pipeline entry point
├── pyproject.toml                    # uv project config & dependencies
├── uv.lock                           # Reproducible dependency lock
│
├── data/                             # ⚠️ Entire folder is gitignored
│   ├── raw/                          # 9 raw Olist CSV files
│   ├── olist.duckdb                  # Built DuckDB data warehouse
│   ├── splits/
│   │   ├── df_model.parquet          # Full modelling-ready dataset
│   │   ├── train.parquet             # Training split
│   │   ├── val.parquet               # Validation split
│   │   └── test.parquet              # Test split
│   └── preprocessed/                 # Preprocessed data (NumPy arrays)
│       ├── train_features.npy        # Training split features
│       ├── val_features.npy          # Validation split features
│       ├── test_features.npy         # Test split features
│       ├── train_review_score.npy    # Training split review score targets (binary)
│       ├── val_review_score.npy      # Validation split review score targets
│       ├── test_review_score.npy     # Test split review score targets
│       ├── train_delivery_days.npy   # Training split delivery days targets (z-scored)
│       ├── val_delivery_days.npy     # Validation split delivery days targets
│       ├── test_delivery_days.npy    # Test split delivery days targets
│       └── delivery_scaler.pkl       # StandardScaler fitted on train delivery targets
│
├── sql/
│   └── build_db.sql                  # DuckDB warehouse DDL & analytical view
│
├── src/
│   ├── etl.py                        # ETL orchestrator + data validation
│   ├── schemas.py                    # Pydantic data contracts
│   ├── serve/
│   │   └── app.py                    # FastAPI serving endpoint (ONNX inference)
│   ├── models/
│   │   ├── dataset.py                # Dataset class for PyTorch (with NaN handling)
│   │   ├── MTLLoss.py                # Learned uncertainty-weighted multi-task loss
│   │   ├── train.py                  # Training + evaluation + MLflow logging
│   │   ├── mmoe.py                   # MMoE model class
│   │   ├── export_onnx.py            # ONNX export from MLflow registry
│   │   ├── sanity_check.py           # Isolated single-task MLP baseline tests
│   │   ├── SHAP.py                   # SHAP explainability for both task heads
│   │   └── inference/                # Production inference artifacts
│   │       ├── mmoe_production.onnx  # Exported ONNX model (20 → 2 outputs)
│   │       ├── feature_preprocessor.pkl  # Fitted sklearn ColumnTransformer
│   │       └── delivery_scaler.pkl   # Fitted StandardScaler for target inverse-transform
│   └── features/
│       ├── haversine.py              # Haversine distance feature
│       ├── cyclic.py                 # Cyclic features
│       ├── pipeline.py               # Feature pipeline
│       ├── build_features.py         # Feature builder
│       └── splits.py                 # Split data into train, val, test
│
├── reports/                          # Per-run CSVs + SHAP summary plots
│
└── EDA/
    ├── plots.py                      # Reusable Pydantic-configured plotting library
    └── 01_hypothesis_testing.ipynb   # Statistical hypothesis testing notebook
```

---

## ✅ Work Completed

### Phase 0 — Data Engineering (ETL) ✅

#### 1. Raw Data Ingestion

All **9 raw Olist CSV files** are ingested into DuckDB tables via `sql/build_db.sql`:

| Table         | Source File                        |
| ------------- | ---------------------------------- |
| `customers`   | `olist_customers_dataset.csv`      |
| `orders`      | `olist_orders_dataset.csv`         |
| `reviews`     | `olist_order_reviews_dataset.csv`  |
| `items`       | `olist_order_items_dataset.csv`    |
| `products`    | `olist_products_dataset.csv`       |
| `sellers`     | `olist_sellers_dataset.csv`        |
| `payments`    | `olist_order_payments_dataset.csv` |
| `geolocation` | `olist_geolocation_dataset.csv`    |

#### 2. Analytical Base Table (SQL View)

A single `analytical_base_table` view is built via **5 CTEs** that handle the one-to-many and duplicate-resolution problems in the raw data:

- **`aggregated_payments`** — Sums `payment_value` and takes `MAX(installments)` per order (handles multiple payment methods).
- **`distinct_geo`** — Averages duplicate lat/lng entries per zip code prefix.
- **`aggregated_items`** — Sums price and freight per order, resolves to a single seller/product.
- **`seller_stats`** — Computes historical order count per seller.
- **`seller_reviews`** — Computes historical average review score per seller.

The final `SELECT` joins all 8 tables, computes `delivery_days` (date diff between purchase and delivery), and filters to **delivered orders only**.

#### 3. Pydantic Data Contracts (`src/schemas.py`)

Every row of the analytical base table is validated against a strict `OlistAnalyticalRow` Pydantic model that enforces:

- **Type safety** — `datetime`, `str`, `float`, `int` types on all fields.
- **Business rules** — `review_score ∈ [1, 5]`, `price ≥ 0`, `freight_value ≥ 0`, `customer_state` is exactly 2 characters, etc.
- **NaN handling** — A custom `@field_validator` converts `float('nan')` → `None` across all optional fields, preventing silent data corruption.

The ETL pipeline (`src/etl.py`) runs every row through this contract and logs any schema violations via `loguru` before failing or passing the pipeline.

#### 4. Data Splits

The modelling-ready dataset has been split into **train / validation / test** Parquet files stored in `data/splits/`.

---

### Phase 1 — EDA & Hypothesis Testing ✅

#### Reusable Plotting Library (`EDA/plots.py`)

A production-grade plotting module built around a Pydantic-configured `Plotter` class:

- **8 plot methods** — `histogram`, `boxplot`, `scatter`, `countplot`, `barplot`, `heatmap`, `pairplot`, `lineplot`.
- **`PlotConfig`** — Pydantic model for validated, serializable plot settings (style, figsize, dpi, save directory, etc.).
- **Every method returns `(fig, ax)`** for further customization.
- **Auto-save support** — Optionally saves figures to a configurable directory.

#### Hypothesis Testing (`EDA/01_hypothesis_testing.ipynb`)

Statistical hypothesis testing to validate the core assumptions underpinning the MMoE architecture.

**Key findings:**

- **Mann-Whitney (p ≈ 0, r=0.386):** Low-rated orders take significantly longer to deliver — statistically justifies the MMoE architecture.
- **Spearman r=0.54 (p ≈ 0):** Strong positive correlation between geo distance and delivery time — validates `geo_distance_km` as a feature.
- **Kruskal-Wallis (H=795, p ≈ 0):** Higher freight quartiles associate with lower review scores (Q1: 4.31 vs Q4: 3.94) — validates `total_freight_value` as a satisfaction feature.
- **Chi-squared (p ≈ 0):** Negative review rate nearly doubles for late orders (14.8% → 27.9%) — confirms the intrinsic link between delivery performance and customer satisfaction.

---

## 🔑 Key Design Decisions

- **Time-based train/val/test split** — Random splitting on time-series order data causes leakage. Splits are chronological: train ends Apr 2018, val ends Jun 2018, test ends Aug 2018.
- **Outlier cap at p99 (46 days)** — Orders above the 99th percentile are fulfilment failures, not legitimate long-distance deliveries. 918 rows removed.
- **y=1 = satisfied (review >= 4)** — The majority class (~83.7%) is labelled positive. `WeightedRandomSampler` oversamples the minority (dissatisfied) class during training to address the ~5:1 class imbalance.
- **Delivery target normalization** — `delivery_days` is z-score normalized using the training set `StandardScaler`. The scaler is persisted as `delivery_scaler.pkl` for inverse-transform at evaluation and inference time.
- **Aggregated items CTE** — Direct `JOIN` on items inflates rows for multi-item orders. Items are aggregated per `order_id` before joining.

---

## Phase 2 - Feature Engineering

### Cyclic Encoding (`src/features/cyclic.py`)

Cyclic encoding is a feature encoding technique that converts periodic features (e.g. time of day, day of week) into a continuous space that captures the periodicity of the feature.

### Preprocessor Pipeline (`src/features/pipeline.py`)

The preprocessor pipeline is a `ColumnTransformer` that applies different transformations to different types of features:

- **Numeric features** — Imputed with median and scaled with StandardScaler.
- **Categorical features** — Imputed with a constant (`"missing_category"`) and ordinally encoded with unknown values mapped to -1.
- **Temporal features** — `order_purchase_timestamp` is cyclically encoded into 6 dimensions: `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos`, capturing intraday, weekly, and seasonal purchase patterns.

### Feature Building (`src/features/build_features.py`)

The feature building script loads the split Parquet data and applies the preprocessor pipeline to transform the features. It also processes the targets: review scores are binarized (>= 4 → satisfied), and delivery days are z-score normalized using a `StandardScaler` fitted on the training set only. The fitted scaler is persisted as `delivery_scaler.pkl` for inverse-transform at evaluation time.

### Preprocessed Data (`data/preprocessed/`)

The preprocessed data is saved in the `data/preprocessed/` directory as NumPy arrays. The final feature matrix has **20 columns**: 11 numeric + 3 ordinal categorical + 6 cyclic temporal. Arrays are saved as `train_features.npy`, `val_features.npy`, and `test_features.npy` for features, with corresponding target arrays for each task.

---

## Phase 3 - MMoE Model + MLflow ✅

### MMoE Model (`src/models/mmoe.py`)

The MMoE (Multi-gate Mixture-of-Experts) model is a neural network trained to simultaneously predict delivery time (regression) and customer satisfaction (binary classification). It uses independent gating networks per task, preventing negative transfer between the two learning objectives.

### MTLLoss Function (`src/models/MTLLoss.py`)

A custom multi-task loss that combines:

- **HuberLoss** for the delivery regression head (robust to outliers)
- **BCEWithLogitsLoss** for the satisfaction classification head

Task loss weights are **learned during training** alongside model parameters via uncertainty weighting, avoiding manual tuning.

### Dataset Class (`src/models/dataset.py`)

A custom PyTorch `Dataset` that loads the preprocessed `.npy` arrays for features, delivery day targets, and review score targets. Includes runtime NaN detection and replacement (from categorical features with missing values) to prevent gradient corruption during training.

### Training Infrastructure (`src/models/train.py`)

| Hyperparameter      | Value                              |
| ------------------- | ---------------------------------- |
| `input_dim`         | 20                                 |
| `num_experts`       | 5                                  |
| `hidden_dim`        | 64                                 |
| `num_hidden_layers` | 1                                  |
| `num_epochs`        | 10                                 |
| `batch_size`        | 32                                 |
| `learning_rate`     | 0.001 (model), 0.01 (loss weights) |
| `optimizer`         | Adam (separate param groups)       |
| `temperature`       | 0.3 (gating softmax)               |
| `sampler`           | WeightedRandomSampler              |

`WeightedRandomSampler` is used to address the ~5:1 class imbalance in the satisfaction task during training, oversampling the minority (negative review) class without artificially inflating the loss.

### Experiment Tracking

**37 experiments** were tracked end-to-end via MLflow, logging:

- All hyperparameters per run
- Per-epoch train & validation losses for both tasks
- Final test metrics (MAE, RMSE, Accuracy, Precision, Recall, F1, Balanced Accuracy, Confusion Matrix)
- Best model checkpoint registered in the **MLflow Model Registry**

---

### 🏆 Best Model — Run 12

After 37 experiments, **Run 12** achieved the best balanced accuracy on the satisfaction task while maintaining strong delivery regression results.

#### Final Test Metrics

| Metric                        | Value         |
| ----------------------------- | ------------- |
| **Delivery MAE**              | **4.24 days** |
| **Delivery RMSE**             | **5.26 days** |
| **Satisfaction Accuracy**     | 77.5%         |
| **Satisfaction Precision**    | 88.2%         |
| **Satisfaction Recall**       | 84.4%         |
| **Satisfaction F1**           | 85.9%         |
| **Satisfaction Balanced Acc** | **62.5%**     |

#### Confusion Matrix (Satisfaction — Test Set)

Here y=1 means **satisfied** (review score >= 4) and y=0 means **dissatisfied** (review score <= 3).

|                         | Predicted Dissatisfied | Predicted Satisfied |
| ----------------------- | ---------------------: | ------------------: |
| **Actual Dissatisfied** |               TN = 981 |          FP = 1,347 |
| **Actual Satisfied**    |             FN = 1,857 |         TP = 10,082 |

> The model correctly identifies **84.4%** of satisfied customers and **42.1%** of dissatisfied ones (TN rate). With an 83.7% majority class, the 62.5% balanced accuracy shows the model has learned genuine discrimination beyond the trivial "predict all satisfied" baseline (which scores 50% balanced accuracy).

#### Training Curve (Run 12)

| Epoch | Train Delivery Loss | Train Sat. Loss | Val Delivery Loss | Val Sat. Loss |
| :---: | :-----------------: | :-------------: | :---------------: | :-----------: |
|   0   |       0.1527        |     0.6220      |      0.1242       |    0.5898     |
|   2   |       0.1080        |     0.6001      |      0.1198       |    0.5917     |
|   5   |       0.1006        |     0.5951      |      0.1227       |    0.5537     |
|   9   |       0.0952        |     0.5915      |      0.1321       |    0.5646     |

Delivery loss converges steadily throughout training. Satisfaction loss plateaus near `0.59` on validation — consistent with the inherent noise in subjective customer sentiment labels.

#### Experiment Comparison (Selected Runs)

| Run    | Delivery MAE | Delivery RMSE | Balanced Acc |    F1     |
| ------ | :----------: | :-----------: | :----------: | :-------: |
| **12** |   **4.24**   |   **5.26**    |  **62.5%**   | **85.9%** |
| 19     |     3.36     |     4.65      |    57.5%     |   86.8%   |
| 20     |     3.74     |     4.80      |    57.7%     |   87.0%   |
| 35     |     3.48     |     4.59      |    59.7%     |   78.8%   |
| 36     |     4.18     |     5.08      |    61.1%     |   79.4%   |
| 37     |     3.79     |     4.81      |    60.6%     |   79.4%   |

Later runs (19–20) improved delivery MAE significantly (~3.4–3.7 days) but at the cost of satisfaction balanced accuracy (~57–58%). Runs 35–37 explored trading overall accuracy for better minority-class discrimination (TN rates up to 49.9%), pushing balanced accuracy back toward ~61%. Run 12 remains the best model by balanced accuracy — the primary metric for the imbalanced satisfaction task.

---

### 🔍 SHAP Explainability (`src/models/SHAP.py`)

A `SHAPExplainer` class wraps the frozen MMoE model and uses **KernelSHAP** to compute feature attributions on 300 held-out test samples (background of 100 training samples). Separate explainers are built for each task head, giving independent explanations for delivery and satisfaction predictions.

#### Delivery Time — SHAP Summary

![SHAP Summary — Delivery Time](reports/shap_summary_plot_delivery.png)

**Key findings:**

- **`geo_distance_km`** is the dominant driver — high distance (pink) pushes delivery time predictions strongly positive (up to +8 SHAP units), while short distances (blue) pull strongly negative (down to −6). The Haversine-engineered feature is by far the most valuable predictor.
- **`day_of_week_sin`** ranks 2nd — the day an order is placed meaningfully shifts delivery predictions, with blue (early-week) values clustering at −2 to −3. This likely reflects warehouse processing schedules and weekend cutoffs.
- **`seller_order_count`** ranks 3rd with a wide bimodal spread — high-volume sellers (pink) occasionally produce large positive SHAP values (+5 to +7), suggesting experienced sellers may operate from more remote logistics hubs with longer shipping routes.
- **`total_freight_value`** ranks 4th and shows the widest raw spread — high freight (pink) pushes delivery predictions strongly positive (up to +8), correlating with heavier, bulkier, or more remote shipments. This feature captures shipping complexity directly.
- **`hour_cos`** ranks 5th — purchase hour has moderate directional impact, likely capturing end-of-day cutoff effects (orders placed after cutoff → next-day processing → +1 day).
- **`seller_state`** shows a bimodal pattern — certain states (pink) consistently push predictions up by +2 to +5, while others push down. This captures Brazil's uneven logistics infrastructure across regions.
- **`product_volume_cm3`** shows a counterintuitive pattern — high volume (pink) pushes delivery predictions _negative_ (down to −7). This may reflect that large/heavy items use dedicated freight carriers with faster, more direct logistics than standard mail.
- Features below rank 7 (`seller_avg_review`, `month_sin`, `product_category_name`, etc.) have minimal individual impact on delivery predictions.

#### Customer Satisfaction — SHAP Summary

![SHAP Summary — Customer Satisfaction](reports/shap_summary_plot_satisfaction.png)

**Key findings:**

- **`seller_order_count`** is the single most impactful feature — high-volume sellers (pink) push predictions strongly toward satisfied (up to +8 SHAP units), while low-volume sellers (blue) push toward dissatisfied (down to −5). Seller experience is the strongest proxy for service quality.
- **`hour_cos`** and **`day_of_week_sin`** rank 2nd and 3rd — temporal purchase patterns have surprisingly high predictive power. This likely captures operational efficiency differences: orders placed during business hours may be processed faster and more carefully than off-hours/weekend orders.
- **`product_volume_cm3`** ranks 4th with bidirectional spread — larger products push predictions in both directions with high variance, reflecting category-dependent satisfaction patterns.
- **`seller_state`** ranks 5th with a bimodal pattern — some seller regions consistently produce higher satisfaction, possibly reflecting regional logistics infrastructure quality.
- **`total_freight_value`** ranks 6th — high freight (pink) pushes slightly toward satisfied, counterintuitively. This may reflect that customers paying premium shipping get faster, more reliable delivery.
- **`seller_avg_review`** ranks 7th and shows a counterintuitive pattern — high seller reputation (pink) pushes predictions _negative_ (down to −5). This is likely a suppression effect: after `seller_order_count` captures the primary seller signal, residual variation in avg_review may flag cases where even "good" sellers occasionally disappoint, and those complaints carry more predictive weight.
- **`month_sin`** and **`num_items`** have moderate but narrow impact — seasonal patterns and multi-item order complexity contribute marginally.
- **`geo_distance_km`** ranks surprisingly low (13th) for satisfaction despite being the #1 delivery driver — suggesting customers care about the _experience_ of delivery (seller responsiveness, packaging quality) more than absolute geographic distance.
- **`total_payment_value`**, **`product_photos_qty`**, **`product_description_length`**, **`customer_state`**, and **`hour_sin`** rank at the bottom with near-zero SHAP impact.

> **Note:** These SHAP plots were generated from a model that still included `delivery_lateness_days` as a feature. It ranked #11 for delivery and #14 for satisfaction with negligible impact in both — reinforcing the decision to remove it from the feature set due to target leakage concerns.

> **Architectural insight:** The two tasks share features (`geo_distance_km`, `seller_order_count`, temporal features) but prioritize them very differently. Delivery is dominated by geography and logistics features, while satisfaction is dominated by seller experience and temporal patterns. This divergence is precisely what MMoE's task-specific gating is designed to handle, allowing shared experts to learn common representations while the gates route relevant signals to each task head.

---

## Phase 4 — ONNX Export ✅

### Model Export (`src/models/export_onnx.py`)

The best model (Run 12) is loaded from the **MLflow Model Registry** and exported to ONNX format for production inference:

- **Input**: `[batch_size, 20]` float32 tensor
- **Outputs**: `delivery_days` (regression logit) and `satisfaction_logits` (classification logit)
- **Dynamic batch axis** allows single-request and batched inference
- Model size: ~320KB total (ONNX graph + external data)

The fitted `StandardScaler` (delivery target inverse-transform) and `ColumnTransformer` (feature preprocessing) are persisted as pickle files alongside the ONNX model in `src/models/inference/`.

---

## Phase 5 — FastAPI Serving ✅

### REST API (`src/serve/app.py`)

A production FastAPI endpoint that serves real-time predictions:

- **`POST /predict`** — Accepts order features as JSON, returns predicted delivery days and satisfaction score
- **Lifespan management** — ONNX session, feature scaler, and target scaler are loaded once at startup and cleared on shutdown
- **Haversine distance** computed server-side from customer/seller lat/lng coordinates
- **Pydantic validation** on all input fields with NaN-to-None sanitization
- **CORS middleware** enabled for cross-origin frontend requests

```
POST /predict → { "delivery_days": 12.3, "review_score": 1 }
```

Where `review_score = 1` means satisfied (predicted review >= 4 stars).

---

## Phase 6 — Deployment ✅

### Live Deployment

| Platform           | URL                                                                                  | Purpose                                    |
| ------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------ |
| **Render**         | [olist-ai-api.onrender.com](https://olist-ai-api.onrender.com)                       | FastAPI + ONNX Runtime serving             |
| **Portfolio Demo** | [ranvirsv.com/work/olist-mtl-pipeline](https://ranvirsv.com/work/olist-mtl-pipeline) | Interactive map-based demo calling the API |

---

## 🗺 Roadmap

- [x] **Phase 0 — Data Engineering** — DuckDB warehouse, Pydantic contracts, data splits
- [x] **Phase 1 — EDA & Hypothesis Testing** — Statistical validation of architecture assumptions
- [x] **Phase 2 — Feature Engineering** — Haversine distance, temporal features, encoding pipelines
- [x] **Phase 3 — MMoE Model + MLflow** — Multi-gate Mixture-of-Experts in PyTorch, 37-run experiment tracking, SHAP explainability
- [x] **Phase 4 — Model Export** — ONNX Runtime optimized inference
- [x] **Phase 5 — Serving** — FastAPI REST endpoint with Pydantic validation
- [x] **Phase 6 — Deployment** — Render cloud hosting, portfolio demo at [ranvirsv.com](https://ranvirsv.com/work/olist-mtl-pipeline)
