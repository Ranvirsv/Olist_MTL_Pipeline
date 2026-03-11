<div align="center">
  
# 📦 Olist Multi-Task Learning (MTL) Pipeline

**A Full-Stack Data Product for E-commerce Logistics and Customer Satisfaction**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5.0-yellow.svg)](https://duckdb.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com/)

</div>

---

## 🎯 The End Goal and The "Why"

### 🏢 The Business Objective

In e-commerce, **logistics and customer satisfaction are intrinsically linked**. A delayed shipment almost guarantees a negative review, regardless of product quality.

The goal of this project is to build a system that takes an order at the moment of purchase and simultaneously predicts two critical outcomes:

1. **Delivery Time (Regression)**: How many days will it take for the package to arrive?
2. **Customer Satisfaction (Classification)**: Will the user leave a positive 5-star review, or a 1-star complaint?

### 🧠 The Architectural "Why"

Standard models fail here. Treating these as two completely separate models ignores the heavy correlation between shipping delays and angry customers. Conversely, a rigid Shared-bottom model suffers from **negative transfer** because predicting numerical "days" and predicting human "sentiment" require fundamentally different mathematical representations.

**The Solution:** We are building a **Multi-gate Mixture-of-Experts (MMoE)**. This allows the neural network to share broad latent features (like geographic distance or freight cost) while utilizing independent gates to protect the task-specific heads from cross-contamination.

### ⚙️ The Engineering "Why"

A Jupyter notebook cannot run in production. This project demonstrates the jump from academic data science to a **Lead Machine Learning Engineer** role by integrating serverless SQL, strict data validation, version control, and high-performance serving to handle the entire lifecycle of enterprise data.

---

## 🛠 The Tech Stack

This stack replaces standard academic tools with modern, high-performance industry standards:

- **Environment & Dependencies**: `uv` _(Rust-based, lightning-fast reproducible virtual environments)_
- **Data Engineering (ETL)**: `DuckDB` _(Serverless analytical SQL)_ and `Pydantic` _(Strict Data Contracts)_
- **Data Versioning**: `DVC` _(Git for massive datasets)_
- **Statistical Research**: `scipy.stats` and `statsmodels` _(Hypothesis testing)_
- **Deep Learning**: `PyTorch` _(Custom MMoE architecture)_
- **Experiment Tracking**: `MLflow` _(Logging hyperparameters, gate temperatures, and loss curves)_
- **High-Performance Serving**: `ONNX Runtime` _(Optimized inference)_ and `FastAPI` _(REST API)_
- **CI/CD & UI**: `GitHub Actions` _(Automated testing)_, `Docker`, and `Streamlit` _(Interactive dashboard)_

---
