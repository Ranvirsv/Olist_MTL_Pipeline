import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv, find_dotenv, set_key
import os
from loguru import logger
from tqdm import tqdm
from dataset import OlistDataset
from mmoe import MMoE
from MTLLoss import MultiTaskLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, params, optimizer, mtl_loss, device='cpu', mlflow_exp_name="Olist_MTL"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
            torch.nn.utils.clip_grad_norm_(self.mtl_loss.parameters(), max_norm=1.0)
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

    def evaluate(self, run_id):
        with mlflow.start_run(run_id=run_id):
            logger.info(f"MLFlow(exp_id: {self.params['exp_num']}) run started. Starting Evaluation")
            self.model.eval()

            delivery_scaler = joblib.load(self.params['delivery_scaler_path'])

            with torch.no_grad():
                batch_del_loss_mae = []
                batch_sat_acc = []
                batch_sat_prec = []
                batch_sat_rec = []
                batch_sat_f1 = []
                batch_sat_bal_acc = []
                all_targets, all_preds = [], []
                all_del_targets, all_del_preds = [], []


                for X_batch, y_delivery_batch, y_satisfaction_batch in tqdm(self.test_loader):
                    X_batch, y_delivery_batch, y_satisfaction_batch = X_batch.to(self.device), y_delivery_batch.to(self.device), y_satisfaction_batch.to(self.device)

                    out_delivery, out_satisfaction = self.model(X_batch)

                    out_delivery, out_satisfaction = out_delivery.view(-1), torch.sigmoid(out_satisfaction).view(-1)
                    out_satisfaction_labels = (out_satisfaction > 0.5).float()

                    ## Inverse-transform delivery predictions/targets back to original scale (days)
                    out_delivery_orig = delivery_scaler.inverse_transform(out_delivery.cpu().numpy().reshape(-1, 1)).flatten()
                    y_delivery_orig = delivery_scaler.inverse_transform(y_delivery_batch.cpu().numpy().reshape(-1, 1)).flatten()


                    batch_del_loss_mae.append(mean_absolute_error(y_delivery_orig, out_delivery_orig))
                    batch_sat_acc.append(accuracy_score(y_satisfaction_batch.cpu().numpy(), out_satisfaction_labels.cpu().numpy()))
                    batch_sat_prec.append(precision_score(y_satisfaction_batch.cpu().numpy(), out_satisfaction_labels.cpu().numpy()))
                    batch_sat_rec.append(recall_score(y_satisfaction_batch.cpu().numpy(), out_satisfaction_labels.cpu().numpy()))
                    batch_sat_f1.append(f1_score(y_satisfaction_batch.cpu().numpy(), out_satisfaction_labels.cpu().numpy()))
                    batch_sat_bal_acc.append(balanced_accuracy_score(y_satisfaction_batch.cpu().numpy(), out_satisfaction_labels.cpu().numpy()))
                    all_targets.extend(y_satisfaction_batch.cpu().numpy())
                    all_preds.extend(out_satisfaction_labels.cpu().numpy())
                    all_del_targets.extend(y_delivery_orig)
                    all_del_preds.extend(out_delivery_orig)

            avg_delivery_loss_rmse = np.sqrt(mean_squared_error(all_del_targets, all_del_preds))
            avg_delivery_loss_mae = np.mean(batch_del_loss_mae)
            avg_satisfaction_acc = np.mean(batch_sat_acc)
            avg_satisfaction_prec = np.mean(batch_sat_prec)
            avg_satisfaction_rec = np.mean(batch_sat_rec)
            avg_satisfaction_f1 = np.mean(batch_sat_f1)
            avg_satisfaction_bal_acc = np.mean(batch_sat_bal_acc)
            satisfaction_cm = confusion_matrix(all_targets, all_preds)

            mlflow.log_metrics({
                'delivery_loss_rmse': avg_delivery_loss_rmse,
                'delivery_loss_mae': avg_delivery_loss_mae,
                'satisfaction_acc': avg_satisfaction_acc,
                'satisfaction_prec': avg_satisfaction_prec,
                'satisfaction_rec': avg_satisfaction_rec,
                'satisfaction_f1': avg_satisfaction_f1,
                'satisfaction_bal_acc': avg_satisfaction_bal_acc
            })

        return avg_delivery_loss_rmse, avg_delivery_loss_mae, avg_satisfaction_acc, avg_satisfaction_prec, avg_satisfaction_rec, avg_satisfaction_f1, avg_satisfaction_bal_acc, satisfaction_cm

    def train(self):
        with mlflow.start_run(run_name=f"MMoE_Exp_{self.params['exp_num']}"):
            logger.info(f"MLFlow(exp_id: {self.params['exp_num']}) run started. Logging parameters")
            mlflow.log_params(self.params)

            best_val_loss = [float('inf'), float('inf')]

            for epoch in range(self.params['num_epochs']):
                if epoch == 3:
                    for param in self.mtl_loss.parameters():
                        param.requires_grad = False
                    for param in self.model.gate_delivery.parameters():
                        param.requires_grad = True
                    for param in self.model.gate_satisfaction.parameters():
                        param.requires_grad = True

                logger.info(f"Epoch {epoch+1}/{self.params['num_epochs']}")
                avg_train_del_loss, avg_train_sat_loss = self._train_one_epoch()
                avg_val_del_loss, avg_val_sat_loss = self._validate()

                logger.info(f"EPOCH {epoch+1}/{self.params['num_epochs']} | "
                            f"Train Loss: [DEL: {avg_train_del_loss:.4f}, SAT: {avg_train_sat_loss:.4f}] | "
                            f"Val Loss: [DEL: {avg_val_del_loss:.4f}, SAT: {avg_val_sat_loss:.4f}]")

                current_val_loss = [avg_val_del_loss, avg_val_sat_loss]

                if current_val_loss[0] + current_val_loss[1] <= best_val_loss[0] + best_val_loss[1]:
                    best_val_loss = current_val_loss
                    mlflow.pytorch.log_model(self.model, "best_mmoe_model")
                    logger.info("New best model saved!")

                mlflow.log_metrics({
                    'train_delivery_loss': avg_train_del_loss,
                    'train_satisfaction_loss': avg_train_sat_loss,
                    'val_delivery_loss': avg_val_del_loss,
                    'val_satisfaction_loss': avg_val_sat_loss
                }, step=epoch)

                with open(self.params['train_report_path'], 'a') as f:
                    f.write(f"{epoch},{avg_train_del_loss},{avg_train_sat_loss},{avg_val_del_loss},{avg_val_sat_loss}\n")
        
        logger.info("Training Complete! Model saved successfully!")
                

def main():
    ## _____________________________________________________________________
    ##                 MLFLOW SETUP
    ## _____________________________________________________________________
    env_path = find_dotenv()
    load_dotenv(env_path)
    
    root_dir = os.getenv("PROJECT_ROOT")
    ip_address = os.getenv("MLFLOW_TRACKING_URI")

    logger.info("Setting up MLFlow")
    mlflow.set_tracking_uri(
        f"http://{ip_address}"
    )

    experiment_num = os.getenv("EXP_NUM")

    report_path = f"{root_dir}/reports"
    os.makedirs(report_path, exist_ok=True)

    train_report_path = f"{report_path}/train_report_{experiment_num}.csv"
    test_report_path = f"{report_path}/test_report_{experiment_num}.csv"
    test_cm_path = f"{report_path}/test_cm_{experiment_num}.csv"

    with open(train_report_path, 'w') as f:
        f.write("epoch,train_delivery_loss,train_satisfaction_loss,val_delivery_loss,val_satisfaction_loss\n")
    with open(test_report_path, 'w') as f:
        f.write("test_delivery_loss_mae, test_delivery_loss_rmse,test_satisfaction_loss_acc,test_satisfaction_loss_prec,test_satisfaction_loss_rec,test_satisfaction_loss_f1,test_satisfaction_loss_bal_acc\n")
    with open(test_cm_path, 'w') as f:
        f.write("test_cm_tn,test_cm_fp,test_cm_fn,test_cm_tp\n")

    ## _____________________________________________________________________
    ##                 MODEL PARAMETERS
    ## _____________________________________________________________________
    
    params = {
        'num_epochs': 25,
        'batch_size': 256,
        'input_dim': 20,
        'output_dim': 1,
        'learning_rate': 0.001,
        'loss_learning_rate': 0.01,
        'num_experts': 5,
        'hidden_dim': 64,
        'num_hidden_layers': 2,
        'num_gate_hidden_layers': 0,
        'temperature': 1.0, 
        'exp_num': experiment_num,
        'train_report_path': train_report_path,
        'test_report_path': test_report_path,
        'delivery_scaler_path': f"{root_dir}/src/models/inference/delivery_scaler.pkl"
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ## _____________________________________________________________________
    ##                 MODEL INITIALIZATION
    ## _____________________________________________________________________

    model = MMoE(input_dim=params['input_dim'], output_dim=params['output_dim'], num_experts=params['num_experts'], hidden_dim=params['hidden_dim'], num_hidden_layers=params['num_hidden_layers'], temperature=params['temperature'])
    model.to(device)

    mtl_loss = MultiTaskLoss(nn.HuberLoss(), nn.BCEWithLogitsLoss().to(device), torch.tensor([True, False]))
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': params['learning_rate']},
        {'params': mtl_loss.parameters(), 'lr': params['loss_learning_rate']}
    ])

    ## _____________________________________________________________________
    ##                 DATASET AND DATALOADER
    ## _____________________________________________________________________
    
    train_dataset = OlistDataset(f"{root_dir}/data/preprocessed/train_features.npy", 
                                f"{root_dir}/data/preprocessed/train_delivery_days.npy", 
                                f"{root_dir}/data/preprocessed/train_review_score.npy")
    
    val_dataset = OlistDataset(f"{root_dir}/data/preprocessed/val_features.npy", 
                                f"{root_dir}/data/preprocessed/val_delivery_days.npy", 
                                f"{root_dir}/data/preprocessed/val_review_score.npy")

    test_dataset = OlistDataset(f"{root_dir}/data/preprocessed/test_features.npy", 
                                f"{root_dir}/data/preprocessed/test_delivery_days.npy", 
                                f"{root_dir}/data/preprocessed/test_review_score.npy")

    class_counts = torch.bincount(train_dataset[:][2].long())
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_dataset[:][2].long()]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    logger.info("Creating DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    ## _____________________________________________________________________
    ##                 TRAINER INITIALIZATION
    ## _____________________________________________________________________

    logger.info("Initializing Trainer")
    trainer = Trainer(model, train_loader, val_loader, test_loader, params, optimizer, mtl_loss, device)

    
    ## _____________________________________________________________________
    ##                 TRAINING
    ## _____________________________________________________________________
    trainer.train()

    ## _____________________________________________________________________
    ##                 MODEL REGISTRATION
    ## _____________________________________________________________________

    model_name = "MMoE Best"
    current_experiment = dict(mlflow.get_experiment_by_name(f"Olist_MTL"))
    experiemnt_id = current_experiment['experiment_id']
    df = mlflow.search_runs([experiemnt_id], order_by=["attribute.end_time DESC"])
    run_id = df.iloc[0]['run_id']
    model_registry_uri = f"runs:/{run_id}/best_mmoe_model"
    mlflow.register_model(model_uri=model_registry_uri, name=model_name)

    ## _____________________________________________________________________
    ##                 MODEL EVALUATION
    ## _____________________________________________________________________

    logger.info("Evaluating Model")
    model_version = os.getenv("MODEL_VERSION")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    trainer.model = model
    delivery_loss_rmse, delivery_loss_mae, satisfaction_acc, satisfaction_prec, satisfaction_rec, satisfaction_f1, satisfaction_bal_acc, satisfaction_cm = trainer.evaluate(run_id)

    ##_____________________________________________________________________
    ##                 METRICS LOGGING
    ##_____________________________________________________________________

    logger.info(f"Delivery Loss RMSE: {delivery_loss_rmse}")
    logger.info(f"Delivery Loss MAE: {delivery_loss_mae}")
    logger.info(f"Satisfaction Accuracy: {satisfaction_acc}")
    logger.info(f"Satisfaction Precision: {satisfaction_prec}")
    logger.info(f"Satisfaction Recall: {satisfaction_rec}")
    logger.info(f"Satisfaction F1 Score: {satisfaction_f1}")
    logger.info(f"Satisfaction Balanced Accuracy: {satisfaction_bal_acc}")
    logger.info(f"Satisfaction Confusion Matrix: {satisfaction_cm}")

    with open(f"{report_path}/test_report_{experiment_num}.csv", 'a') as f:
        f.write(f"{delivery_loss_mae},{delivery_loss_rmse},{satisfaction_acc},{satisfaction_prec},{satisfaction_rec},{satisfaction_f1},{satisfaction_bal_acc}\n")

    with open(f"{report_path}/test_cm_{experiment_num}.csv", 'a') as f:
        f.write(f"{satisfaction_cm[0,0]},{satisfaction_cm[0,1]},{satisfaction_cm[1,0]},{satisfaction_cm[1,1]}\n")

    ## _____________________________________________________________________
    ##                 EXPERIMENT VERSIONING
    ## _____________________________________________________________________

    set_key(env_path, "EXP_NUM", str(int(experiment_num) + 1))
    set_key(env_path, "MODEL_VERSION", str(int(model_version) + 1))

if __name__ == "__main__":
    main()
