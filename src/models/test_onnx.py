import os
import joblib
import numpy as np
import onnxruntime as rt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    accuracy_score, balanced_accuracy_score, 
    precision_score, recall_score, f1_score, confusion_matrix
)
from dotenv import load_dotenv

load_dotenv()

def main():
    base_dir = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    inference_dir = os.path.join(base_dir, "src", "models", "inference")
    data_dir = os.path.join(base_dir, "data", "preprocessed")
    
    # 1. Load ONNX Model & Scalers
    print("Loading ONNX Model...")
    model_path = os.path.join(inference_dir, "mmoe_production.onnx")
    ort_session = rt.InferenceSession(model_path)
    
    print("Loading Delivery Scaler...")
    delivery_scaler_path = os.path.join(inference_dir, "delivery_scaler.pkl")
    delivery_scaler = joblib.load(delivery_scaler_path)
    
    # 2. Load Test dataset
    print(f"Loading Test Data from {data_dir}...")
    X_test = np.load(os.path.join(data_dir, "test_features.npy")).astype(np.float32)
    y_test_delivery = np.load(os.path.join(data_dir, "test_delivery_days.npy")).astype(np.float32)
    y_test_review = np.load(os.path.join(data_dir, "test_review_score.npy")).astype(np.float32)
    
    # 3. Inference
    print("Running Inference over Test Set...")
    # Using batches to prevent memory issues on large test sets
    batch_size = 1000
    all_del_preds = []
    all_sat_preds = []
    
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        out_delivery, out_satisfaction = ort_session.run(None, {"input_features": X_batch})
        all_del_preds.append(out_delivery)
        all_sat_preds.append(out_satisfaction)
        
    out_delivery = np.concatenate(all_del_preds, axis=0)
    out_satisfaction = np.concatenate(all_sat_preds, axis=0)
    
    # 4. Process predictions
    # Transform delivery predictions back to original days
    out_delivery_orig = delivery_scaler.inverse_transform(out_delivery).flatten()
    y_test_delivery_orig = delivery_scaler.inverse_transform(y_test_delivery.reshape(-1, 1)).flatten()
    
    # Transform satisfaction logits to binary predictions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    out_satisfaction_probs = sigmoid(out_satisfaction)
    review_score_preds = (out_satisfaction_probs > 0.5).astype(int).flatten()
    y_test_review_int = y_test_review.astype(int).flatten()
    
    # 5. Compute Metrics
    print("\n--- TEST METRICS ---")
    
    # Delivery
    mae = mean_absolute_error(y_test_delivery_orig, out_delivery_orig)
    rmse = np.sqrt(mean_squared_error(y_test_delivery_orig, out_delivery_orig))
    print(f"Delivery MAE:  {mae:.4f} days")
    print(f"Delivery RMSE: {rmse:.4f} days")
    
    # Satisfaction
    acc = accuracy_score(y_test_review_int, review_score_preds)
    prec = precision_score(y_test_review_int, review_score_preds)
    rec = recall_score(y_test_review_int, review_score_preds)
    f1 = f1_score(y_test_review_int, review_score_preds)
    bal_acc = balanced_accuracy_score(y_test_review_int, review_score_preds)
    cm = confusion_matrix(y_test_review_int, review_score_preds)
    
    print(f"Satisfaction Accuracy:  {acc:.4f}")
    print(f"Satisfaction Precision: {prec:.4f}")
    print(f"Satisfaction Recall:    {rec:.4f}")
    print(f"Satisfaction F1 Score:  {f1:.4f}")
    print(f"Satisfaction Balanced Accuracy: {bal_acc:.4f}")
    print(f"Satisfaction Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
