import torch
import torch.nn as nn
from loguru import logger

class SoftmaxTemperature(nn.Module):
    def __init__(self, temperature=1.0, dim=1):
        super(SoftmaxTemperature, self).__init__()
        self.temperature = temperature
        self.dim = dim

    def forward(self, x):
        # Scale logits by temperature before softmax
        return torch.softmax(x / self.temperature, dim=self.dim)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=64, num_hidden_layers=0, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        self.gate = [
            nn.Linear(self.input_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
            SoftmaxTemperature(temperature=temperature)
        ]

        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            hidden_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(0.3)])
            
        self.gate[2:2] = hidden_layers

        self.gate = nn.Sequential(*self.gate)
        
    def forward(self, x):
        return self.gate(x)
        

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden_layers=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        self.expert = [
            nn.Linear(self.input_dim, self.hidden_dim), 
            nn.ReLU()
        ]

        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            hidden_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(0.3)])

        self.expert[2:2] = hidden_layers

        self.expert = nn.Sequential(*self.expert)
        
    def forward(self, x):
        expert_out = self.expert(x)
        return expert_out

class Tower(nn.Module):
    def __init__(self, output_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim    
        self.output_dim = output_dim
        
        self.tower = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
    def forward(self, x):
        tower_out = self.tower(x)
        return tower_out

class MMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim=64, num_hidden_layers=2, num_gate_hidden_layers=0, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_gate_hidden_layers = num_gate_hidden_layers
        
        self.gate_delivery = GatingNetwork(self.input_dim, self.num_experts, self.hidden_dim, self.num_gate_hidden_layers, temperature)
        self.gate_satisfaction = GatingNetwork(self.input_dim, self.num_experts, self.hidden_dim, self.num_gate_hidden_layers, temperature)
        
        self.experts = nn.ModuleList([
                ExpertNetwork(self.input_dim, self.output_dim, self.hidden_dim, self.num_hidden_layers) for _ in range(self.num_experts)
            ])

        self.delivery_head = Tower(self.output_dim, self.hidden_dim)
        self.satisfaction_head = Tower(self.output_dim, self.hidden_dim)
        
    def forward(self, x):
        expert_weight_delivery = self.gate_delivery(x)
        expert_weight_satisfaction = self.gate_satisfaction(x)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        shared_features_delivery = torch.sum(expert_weight_delivery.unsqueeze(-1) * expert_out, dim=1)
        shared_features_satisfaction = torch.sum(expert_weight_satisfaction.unsqueeze(-1) * expert_out, dim=1)

        out_delivery = self.delivery_head(shared_features_delivery)
        out_satisfaction = self.satisfaction_head(shared_features_satisfaction)

        return out_delivery, out_satisfaction


if __name__ == "__main__":
    logger.info("Testing MMoE model")

    batch_size = 32
    input_features = 21 ## acutal features in Olist Traning Data
    dummy_x = torch.randn(batch_size, input_features)

    logger.info(f"Input shape: {dummy_x.shape}")

    model = MMoE(input_dim=input_features, output_dim=1, num_experts=3, hidden_dim=64)
    
    out_del, out_sat = model(dummy_x)
    
    logger.info(f"Delivery Output shape: {out_del.shape} (Expected: [{batch_size}, 1])")
    logger.info(f"Satisfaction Output shape: {out_sat.shape} (Expected: [{batch_size}, 1])")
    logger.info("Forward pass successful!")


