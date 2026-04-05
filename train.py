import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Model Definition
class HousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

def main():
    # 2. Load Data
    df = pd.read_csv('data/housing.csv')
    X = df.drop('price_jod', axis=1)
    y = df[['price_jod']]

    # 3. Standardization
    X_mean, X_std = X.mean(), X.std()
    X_scaled = (X - X_mean) / X_std

    # 4. Convert to Tensors
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # 5. Train/Test Split (80/20) - Required by Challenge 1
    torch.manual_seed(42)
    indices = torch.randperm(len(X_tensor))
    split = int(0.8 * len(X_tensor))
    
    X_train, X_test = X_tensor[indices[:split]], X_tensor[indices[split:]]
    y_train, y_test = y_tensor[indices[:split]], y_tensor[indices[split:]]

    # 6. Training Setup
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_losses = []

    # 7. Training Loop
    for epoch in range(100):
        model.train()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 8. Metrics Calculation (Using NumPy as requested)
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).numpy().flatten()
        test_preds = model(X_test).numpy().flatten()
        y_train_np = y_train.numpy().flatten()
        y_test_np = y_test.numpy().flatten()

        def get_metrics(actual, pred):
            mae = np.mean(np.abs(actual - pred))
            ss_res = np.sum((actual - pred)**2)
            ss_tot = np.sum((actual - np.mean(actual))**2)
            r2 = 1 - (ss_res / ss_tot)
            return mae, r2

        tr_mae, tr_r2 = get_metrics(y_train_np, train_preds)
        te_mae, te_r2 = get_metrics(y_test_np, test_preds)

        print(f"\nTrain MAE: {tr_mae:.2f}, R2: {tr_r2:.4f}")
        print(f"Test MAE: {te_mae:.2f}, R2: {te_r2:.4f}")

    # 9. Save Visualizations
    plt.figure()
    plt.plot(train_losses)
    plt.savefig('loss_curve.png')
    
    plt.figure()
    plt.scatter(y_test_np, test_preds)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.savefig('predictions_plot.png')

if __name__ == "__main__":
    main()