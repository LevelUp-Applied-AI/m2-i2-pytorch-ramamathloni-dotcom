from trainer import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# --- 1. Housing Problem ---
df = pd.read_csv('data/housing.csv')
X = torch.tensor(((df.drop('price_jod', axis=1) - df.drop('price_jod', axis=1).mean()) / df.drop('price_jod', axis=1).std()).values, dtype=torch.float32)
y = torch.tensor(df[['price_jod']].values, dtype=torch.float32)
dataset = DataLoader(TensorDataset(X, y), batch_size=16)

model_h = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
config_h = {'epochs': 200, 'learning_rate': 0.001, 'checkpoint_path': 'housing_best.pt'}
trainer_h = Trainer(model_h, nn.MSELoss(), torch.optim.Adam(model_h.parameters(), lr=0.001), config_h)

print("Training Housing Model...")
trainer_h.train(dataset)
trainer_h.save_report("housing_report.json", "housing_loss.png")

# --- 2. Synthetic Problem (Sin Wave) ---
x_syn = torch.linspace(-5, 5, 100).view(-1, 1)
y_syn = torch.sin(x_syn) + torch.randn(x_syn.size()) * 0.1
syn_dataset = DataLoader(TensorDataset(x_syn, y_syn), batch_size=10)

model_s = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
config_s = {'epochs': 500, 'learning_rate': 0.01, 'checkpoint_path': 'syn_best.pt'}
trainer_s = Trainer(model_s, nn.MSELoss(), torch.optim.Adam(model_s.parameters(), lr=0.01), config_s)

print("\nTraining Synthetic Model...")
trainer_s.train(syn_dataset)
trainer_s.save_report("synthetic_report.json", "synthetic_loss.png")