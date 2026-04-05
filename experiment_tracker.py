import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from itertools import product

# 1. Define Parametric Model
class HousingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. Data Preparation (With 80/20 Split)
def prepare_data():
    df = pd.read_csv('data/housing.csv')
    X = df.drop('price_jod', axis=1)
    y = df[['price_jod']]
    
    # Normalization
    X = (X - X.mean()) / X.std()
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    
    # Fixed Seed Split
    torch.manual_seed(42)
    indices = torch.randperm(len(X_tensor))
    split = int(0.8 * len(X_tensor))
    
    X_train, X_test = X_tensor[indices[:split]], X_tensor[indices[split:]]
    y_train, y_test = y_tensor[indices[:split]], y_tensor[indices[split:]]
    
    return X_train, y_train, X_test, y_test

# 3. Training Function
def run_experiment(X_train, y_train, X_test, y_test, config):
    start_time = time.time()
    model = HousingModel(input_size=5, hidden_size=config['hidden_size'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        test_preds = model(X_test).numpy().flatten()
        test_actual = y_test.numpy().flatten()
        
        mae = np.mean(np.abs(test_actual - test_preds))
        ss_res = np.sum((test_actual - test_preds) ** 2)
        ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
    return {
        "config": config,
        "metrics": {
            "train_loss": train_loss,
            "test_mae": float(mae),
            "test_r2": float(r2),
            "time_s": float(training_time)
        }
    }

# 4. Main Execution Loop
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    
    # Grid Search Definition (30 combinations)
    lrs = [0.1, 0.01, 0.001]
    hiddens = [16, 32, 64, 128]
    epochs_list = [50, 100, 200]
    
    grid = list(product(lrs, hiddens, epochs_list))
    all_results = []
    
    print(f"Starting {len(grid)} experiments...")
    for lr, hidden, eps in grid:
        conf = {"lr": lr, "hidden_size": hidden, "epochs": eps}
        res = run_experiment(X_train, y_train, X_test, y_test, conf)
        all_results.append(res)
        print(f"Done: LR={lr}, Hidden={hidden}, Epochs={eps} -> MAE: {res['metrics']['test_mae']:.0f}")

    # Save to JSON
    with open('experiments.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # 5. Print Leaderboard
    sorted_res = sorted(all_results, key=lambda x: x['metrics']['test_mae'])
    print("\n" + "="*70)
    print(f"{'Rank':<5} | {'LR':<7} | {'Hidden':<6} | {'Epochs':<6} | {'Test MAE':<10} | {'R2':<6}")
    print("-" * 70)
    for i, r in enumerate(sorted_res[:10]):
        c, m = r['config'], r['metrics']
        print(f"{i+1:<5} | {c['lr']:<7} | {c['hidden_size']:<6} | {c['epochs']:<6} | {m['test_mae']:<10.0f} | {m['test_r2']:<6.3f}")

    # 6. Visualization
    plt.figure(figsize=(10,6))
    maes = [r['metrics']['test_mae'] for r in all_results]
    plt.scatter(range(len(maes)), maes)
    plt.axhline(y=10000, color='r', linestyle='--', label='Target (10k)')
    plt.title("Experiment MAE Distribution")
    plt.ylabel("MAE (JOD)")
    plt.legend()
    plt.savefig('experiment_summary.png')