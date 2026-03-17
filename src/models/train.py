import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv, find_dotenv, set_key
import os
from loguru import logger
from tqdm import tqdm
from dataset import OlistDataset
from mmoe import MMoE
from MTLLoss import MultiTaskLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, params, optimizer, mtl_loss, device='cpu', mlflow_exp_name="Olist_MTL"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.optimizer = optimizer
        self.mtl_loss = mtl_loss
        self.device = device
        self.mlflow_exp_name = mlflow_exp_name

        self.model.to(self.device)
        mlflow.set_experiment(self.mlflow_exp_name)

    def _train_one_epoch(self):
        self.model.train()
        self.mtl_loss.train()

        total_loss_delivery = 0
        total_loss_satisfaction = 0

        for X_batch, y_delivery_batch, y_satisfaction_batch in tqdm(self.train_loader):
            X_batch, y_delivery_batch, y_satisfaction_batch = X_batch.to(self.device), y_delivery_batch.to(self.device), y_satisfaction_batch.to(self.device)

            self.optimizer.zero_grad()

            out_delivery, out_satisfaction = self.model(X_batch)

            loss, loss_delivery, loss_satisfaction = self.mtl_loss(out_delivery, out_satisfaction, y_delivery_batch, y_satisfaction_batch)

            loss.backward()
            self.optimizer.step()

            total_loss_delivery += loss_delivery.item()
            total_loss_satisfaction += loss_satisfaction.item()

        avg_loss_delivery = total_loss_delivery / len(self.train_loader)
        avg_loss_satisfaction = total_loss_satisfaction / len(self.train_loader)

        return avg_loss_delivery, avg_loss_satisfaction
            

    def _validate(self):
        self.model.eval()
        self.mtl_loss.eval()

        with torch.no_grad():
            total_loss_delivery = 0
            total_loss_satisfaction = 0

            for X_batch, y_delivery_batch, y_satisfaction_batch in tqdm(self.val_loader):
                X_batch, y_delivery_batch, y_satisfaction_batch = X_batch.to(self.device), y_delivery_batch.to(self.device), y_satisfaction_batch.to(self.device)

                out_delivery, out_satisfaction = self.model(X_batch)

                loss, loss_delivery, loss_satisfaction = self.mtl_loss(out_delivery, out_satisfaction, y_delivery_batch, y_satisfaction_batch)
                

                total_loss_delivery += loss_delivery.item()
                total_loss_satisfaction += loss_satisfaction.item()

            avg_loss_delivery = total_loss_delivery / len(self.val_loader)
            avg_loss_satisfaction = total_loss_satisfaction / len(self.val_loader)

            return avg_loss_delivery, avg_loss_satisfaction

    def train(self):
        with mlflow.start_run(run_name=f"MMoE_Exp_{self.params['exp_num']}", nested=True):
            logger.info("MLFlow run started. Logging parameters")
            mlflow.log_params(self.params)

            best_val_loss = [float('inf'), float('inf')]

            for epoch in range(self.params['num_epochs']):
                logger.info(f"Epoch {epoch+1}/{self.params['num_epochs']}")
                avg_train_del_loss, avg_train_sat_loss = self._train_one_epoch()
                avg_val_del_loss, avg_val_sat_loss = self._validate()

                logger.info(f"EPOCH {epoch+1}/{self.params['num_epochs']} | "
                            f"Train Loss: [DEL: {avg_train_del_loss:.4f}, SAT: {avg_train_sat_loss:.4f}] | "
                            f"Val Loss: [DEL: {avg_val_del_loss:.4f}, SAT: {avg_val_sat_loss:.4f}]")

                current_val_loss = [avg_val_del_loss, avg_val_sat_loss]

                if current_val_loss[0] < best_val_loss[0] and current_val_loss[1] < best_val_loss[1]:
                    best_val_loss = current_val_loss
                    mlflow.pytorch.log_model(self.model, "best_mmoe_model", export_model=True)
                    logger.info("New best model saved!")

                mlflow.log_metrics({
                    'train_delivery_loss': avg_train_del_loss,
                    'train_satisfaction_loss': avg_train_sat_loss,
                    'val_delivery_loss': avg_val_del_loss,
                    'val_satisfaction_loss': avg_val_sat_loss
                }, step=epoch)
        
        logger.info("Training Complete! Model saved successfully!")
                

def main():
    env_path = find_dotenv()
    load_dotenv(env_path)
    
    root_dir = os.getenv("PROJECT_ROOT")
    ip_address = os.getenv("MLFLOW_TRACKING_URI")

    logger.info("Setting up MLFlow")
    mlflow.set_tracking_uri(
        f"http://{ip_address}"
    )

    experiment_num = os.getenv("EXP_NUM")
    set_key(env_path, "EXP_NUM", str(int(experiment_num) + 1))
    
    params = {
        'num_epochs': 10,
        'batch_size': 32,
        'input_dim': 14,
        'output_dim': 1,
        'learning_rate': 0.001,
        'num_experts': 5,
        'hidden_dim': 64,
        'num_hidden_layers': 3,
        'num_gate_hidden_layers': 0,
        'temperature': 1.0, 
        'exp_num': experiment_num
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = MMoE(input_dim=params['input_dim'], output_dim=params['output_dim'], num_experts=params['num_experts'], hidden_dim=params['hidden_dim'], num_hidden_layers=params['num_hidden_layers'], temperature=params['temperature'])
    model.to(device)

    del_gating_network = nn.Linear(params['input_dim'], params['num_experts']).to(device)
    sat_gating_network = nn.Sequential(nn.Linear(params['input_dim'], params['num_experts']), nn.Softmax(dim=1)).to(device)

    model.gate_delivery = del_gating_network
    model.gate_satisfaction = sat_gating_network

    mtl_loss = MultiTaskLoss(nn.HuberLoss(), nn.BCEWithLogitsLoss(), torch.tensor([True, False]))
    optimizer = optim.Adam((list(model.parameters()) + list(mtl_loss.parameters())), lr=params['learning_rate'])

    ## Log Model Architecture
    model_architecture_string = str(model)
    mlflow.log_text(model_architecture_string, "model_architecture.txt")
    loss_architecture_string = str(mtl_loss)
    mlflow.log_text(loss_architecture_string, "loss_architecture.txt")
    optimizer_architecture_string = str(optimizer)
    mlflow.log_text(optimizer_architecture_string, "optimizer_architecture.txt")

    train_dataset = OlistDataset(f"{root_dir}/data/preprocessed/train_features.npy", 
                                f"{root_dir}/data/preprocessed/train_delivery_days.npy", 
                                f"{root_dir}/data/preprocessed/train_review_score.npy")
    
    val_dataset = OlistDataset(f"{root_dir}/data/preprocessed/val_features.npy", 
                                f"{root_dir}/data/preprocessed/val_delivery_days.npy", 
                                f"{root_dir}/data/preprocessed/val_review_score.npy")

    logger.info("Creating DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    logger.info("Initializing Trainer")
    trainer = Trainer(model, train_loader, val_loader, params, optimizer, mtl_loss, device)
    trainer.train()

if __name__ == "__main__":
    main()
