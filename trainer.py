import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, criterion, optimizer, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.history = []

    def train(self, train_loader, val_loader=None):
        best_val_loss = float('inf')
        patience_counter = 0
        
        epochs = self.config.get('epochs', 100)
        early_stopping = self.config.get('early_stopping_patience', 10)
        checkpoint_path = self.config.get('checkpoint_path', 'best_model.pt')

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.history.append(avg_train_loss)

            # Validation Phase (Simple implementation)
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_loss = sum(self.criterion(self.model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
                
                # Early Stopping Logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), checkpoint_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % self.config.get('log_every_n_epochs', 10) == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

    def save_report(self, filename="report.json", plot_name="loss_curve.png"):
        # Save JSON
        report = {
            "config": self.config,
            "final_loss": self.history[-1],
            "history": self.history
        }
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save Plot
        plt.figure()
        plt.plot(self.history)
        plt.title("Training Loss")
        plt.savefig(plot_name)
        plt.close()