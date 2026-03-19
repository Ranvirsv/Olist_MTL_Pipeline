import torch
import numpy as np
import mlflow.pytorch
import mlflow
from dotenv import load_dotenv
import os
from loguru import logger
import joblib
import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, device, scaler, X_train, task="delivery"):
        self.model = model
        self.device = device
        self.scaler = scaler
        self.task = task
        if task == "delivery":
            self.explainer = shap.KernelExplainer(self._predict_delivery, X_train)
        elif task == "satisfaction":
            self.explainer = shap.KernelExplainer(self._predict_satisfaction, X_train)
        self.feature_names = [
        'total_freight_value', 'product_weight_g',
        'total_payment_value', 'max_installments', 'geo_distance_km',
        'delivery_lateness_days', 'product_description_length',
        'product_photos_qty', 'product_volume_cm3', 'num_items',
        'seller_order_count', 'seller_avg_review', 'product_category_name',
        'seller_state', 'customer_state',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
        'month_sin', 'month_cos'
    ]

    def _predict(self, X, delivery=False, satisfaction=False):
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            out_delivery, out_satisfaction = self.model(X)

            if delivery:
                return self.scaler.inverse_transform(out_delivery.detach().cpu().numpy().reshape(-1, 1)).flatten()
            if satisfaction:
                out_satisfaction = torch.sigmoid(out_satisfaction).detach().cpu().numpy().reshape(-1, 1).flatten()
                out_satisfaction = (out_satisfaction > 0.5).astype(int)
                return out_satisfaction

    def _predict_delivery(self, X):
        return self._predict(X, delivery=True)

    def _predict_satisfaction(self, X):
        return self._predict(X, satisfaction=True)

    def explain(self, X):
        shap_values = self.explainer.shap_values(X)
        summary_plot = shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        return summary_plot
        


def main():
    load_dotenv()

    ##=============================================================================
    ##                        LOAD X Train and Test
    ##=============================================================================
    root_dir = os.getenv('PROJECT_ROOT')
    X_train_path = f"{root_dir}/data/preprocessed/train_features.npy"
    X_test_path = f"{root_dir}/data/preprocessed/test_features.npy"
    scaler = joblib.load(f"{root_dir}/data/preprocessed/delivery_scaler.pkl")

    logger.info(f"Loading data")
    X_train = np.load(X_train_path)
    X_train_sample = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    X_test = np.load(X_test_path)
    X_test_sample = X_test[np.random.choice(X_test.shape[0], 300, replace=False)]

    ##=============================================================================
    ##                        LOAD MLFLOW
    ##=============================================================================
    ip_address = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(
        f"http://{ip_address}"
    )

    logger.info(f"MLFLOW Tracking URI: {mlflow.get_tracking_uri()}")
    
    ##=============================================================================
    ##                        LOAD MODEL
    ##=============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading Model")
    model_name = "MMoE Best Bal_Acc"
    model_version = 2
    model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_version}")
    model.to(device)
    model.eval()

    ##=============================================================================
    ##                        Delivery SHAP EXPLAINER
    ##=============================================================================
    shap.initjs()
    logger.info("Initializing SHAP Explainer")
    delivery_explainer = SHAPExplainer(model, device, scaler, X_train_sample, task="delivery")

    logger.info("Explaining SHAP values")
    delivery_summary_plot = delivery_explainer.explain(X_test_sample)

    logger.info("Saving SHAP summary plot")
    plt.savefig(f"{root_dir}/reports/shap_summary_plot_delivery.png")

    ##=============================================================================
    ##                        Satisfaction SHAP EXPLAINER
    ##=============================================================================
    logger.info("Initializing SHAP Explainer")
    satisfaction_explainer = SHAPExplainer(model, device, scaler, X_train_sample, task="satisfaction")
    
    logger.info("Explaining SHAP values")
    satisfaction_summary_plot = satisfaction_explainer.explain(X_test_sample)
    
    logger.info("Saving SHAP summary plot")
    plt.savefig(f"{root_dir}/reports/shap_summary_plot_satisfaction.png")

if __name__ == "__main__":
    main()
