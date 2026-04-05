import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. تجهيز البيانات (نفس الخطوات السابقة)
    df = pd.read_csv('data/housing.csv')
    X = (df.drop('price_jod', axis=1) - df.drop('price_jod', axis=1).mean()) / df.drop('price_jod', axis=1).std()
    y = df[['price_jod']]
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # 2. تعريف الأوزان يدوياً (Random Initialization)
    # ملاحظة: requires_grad=False لأننا سنحسبها بأنفسنا
    torch.manual_seed(42)
    W1 = torch.randn(5, 32) * 0.01
    b1 = torch.zeros(1, 32)
    W2 = torch.randn(32, 1) * 0.01
    b2 = torch.zeros(1, 1)

    lr = 0.0001
    manual_losses = []

    print("Starting Manual Backpropagation...")
    for epoch in range(100):
        # --- FORWARD PASS ---
        # الطبقة الأولى: X @ W1 + b1
        z1 = X_tensor @ W1 + b1
        # ReLU: max(0, z1)
        h = z1.clone()
        h[h < 0] = 0 
        
        # الطبقة الثانية: h @ W2 + b2
        y_pred = h @ W2 + b2
        
        # حساب الـ MSE Loss يدوياً
        loss = torch.mean((y_pred - y_tensor)**2)
        manual_losses.append(loss.item())

        # --- BACKWARD PASS (الاشتقاق اليدوي - Chain Rule) ---
        N = y_tensor.size(0)
        # 1. مشتقة الـ Loss بالنسبة للتوقعات (dL/dy_pred)
        grad_y_pred = (2.0 / N) * (y_pred - y_tensor)
        
        # 2. مشتقات الطبقة الثانية
        grad_W2 = h.t() @ grad_y_pred
        grad_b2 = grad_y_pred.sum(dim=0, keepdim=True)
        
        # 3. العودة للطبقة الأولى (Backprop through ReLU)
        grad_h = grad_y_pred @ W2.t()
        grad_z1 = grad_h.clone()
        grad_z1[z1 <= 0] = 0 # مشتقة ReLU هي 0 للأرقام السالبة
        
        # 4. مشتقات الطبقة الأولى
        grad_W1 = X_tensor.t() @ grad_z1
        grad_b1 = grad_z1.sum(dim=0, keepdim=True)

        # --- UPDATE WEIGHTS (Gradient Descent) ---
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Manual Loss = {loss.item():.4f}")

    # حفظ رسمة الـ Loss للمقارنة
    plt.plot(manual_losses)
    plt.title("Manual Implementation Loss")
    plt.savefig('manual_loss_curve.png')
    print("\nDone! Saved manual_loss_curve.png")

if __name__ == "__main__":
    main()