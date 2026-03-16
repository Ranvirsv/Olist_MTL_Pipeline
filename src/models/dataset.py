import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger

class OlistDataset(Dataset):
    def __init__(self, x_path, y_delivery_path, y_statisfaction_path):
        logger.info(f"Loading data from {x_path}, {y_delivery_path}, {y_statisfaction_path}")
        self.x = np.load(x_path)
        self.y_delivery = np.load(y_delivery_path)
        self.y_statisfaction = np.load(y_statisfaction_path)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y_delivery = torch.tensor(self.y_delivery, dtype=torch.float32)
        self.y_statisfaction = torch.tensor(self.y_statisfaction, dtype=torch.long)

        logger.info(f"Data loaded: {self.x.shape}, {self.y_delivery.shape}, {self.y_statisfaction.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        features = self.x[idx]
        y_delivery = self.y_delivery[idx]
        y_statisfaction = self.y_statisfaction[idx]

        return features, y_delivery, y_statisfaction

if __name__ == "__main__":
    logger.info("Testing OlistDataset...")
    
    dataset = OlistDataset("data/preprocessed/train_features.npy", "data/preprocessed/train_delivery_days.npy", "data/preprocessed/train_review_score.npy")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    features, y_delivery, y_statisfaction = next(iter(dataloader))

    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Batch size: {dataloader.batch_size}")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Delivery target shape: {y_delivery.shape}")
    logger.info(f"Satisfaction target shape: {y_statisfaction.shape}")