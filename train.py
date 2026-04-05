import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def main():
    # Load data
    df = pd.read_csv('data/housing.csv')
    X = torch.tensor(df.drop('price_jod', axis=1).values, dtype=torch.float32)
    y = torch.tensor(df[['price_jod']].values, dtype=torch.float32)

    # Simple Model
    model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train for 100 epochs (The baseline)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model.pt')
    print("Baseline model trained and saved.")

if __name__ == "__main__":
    main()