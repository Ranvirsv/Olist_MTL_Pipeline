import torch
import mlflow.pytorch
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    ip_address = os.getenv("MLFLOW_TRACKING_URI")
    root_dir = os.getenv("PROJECT_ROOT")
    mlflow.set_tracking_uri(f"http://{ip_address}")

    # 1. Load your best model
    model = mlflow.pytorch.load_model("models:/MMoE Best Bal_Acc/2")
    model.to("cpu")
    model.eval()

    # 2. Create a dummy input tensor with the exact shape your model expects [Batch, Features]
    dummy_input = torch.randn(1, 20, dtype=torch.float32)

    # 3. Export to ONNX — save into inference/ alongside the scalers
    os.makedirs(f"{root_dir}/src/models/inference", exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input, 
        f"{root_dir}/src/models/inference/mmoe_production.onnx",
        input_names=["input_features"], 
        output_names=["delivery_days", "satisfaction_logits"],
        dynamic_axes={"input_features": {0: "batch_size"}} # Allows batch size to change
    )
    print("✅ Successfully exported to ONNX!")

if __name__ == "__main__":
    main()