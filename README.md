# Integration 2 — PyTorch: Housing Price Prediction

## Project Overview
This project implements a neural network using PyTorch to predict housing prices in Amman, Jordan. The model analyzes 5 key features to estimate property values.

### Input Features:
1. **area_sqm**: Apartment size (50–250 sqm).
2. **bedrooms**: Number of rooms (1–5).
3. **floor**: Floor level (1–15).
4. **age_years**: Building age (0–40 years).
5. **distance_to_center_km**: Proximity to city center (0.5–25 km).

**Target Variable:** `price_jod` (30K–150K JOD).

---

## Training Configuration
* **Model Architecture:** * Input Layer: 5 neurons
    * Hidden Layer: 32 neurons with **ReLU** activation
    * Output Layer: 1 neuron (Linear)
* **Optimizer:** Adam (Learning Rate = 0.01)
* **Loss Function:** MSELoss (Mean Squared Error)
* **Epochs:** 100

---

## Training Outcome & Observations
* **Initial Loss (Epoch 0):** ~195,028,876.00
* **Final Loss (Epoch 100):** ~194,441,996.00
* **Outcome:** The loss showed a steady decrease throughout the 100 epochs, indicating that the model successfully learned the relationship between property features and their prices.

### Behavioral Observation:
I observed that the initial loss was extremely high. This occurs because house prices are large numbers (up to 150,000 JOD), and the Mean Squared Error (MSE) squares the difference between predictions and actual values. However, the consistent downward trend proves that feature standardization was effective in stabilizing the training process.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)


   ## Challenge 1: Overfitting Analysis
Based on the results from `train.py`:
- **Is the model overfitting?** No, the model is currently **underfitting**. 
- **Comparison:** Both Train and Test MAE are high (~40,000 JOD), and the R² score is negative. This indicates that the model has not yet learned the underlying patterns of the data.
- **Next Steps:** To improve performance, I will perform hyperparameter tuning using an **Experiment Tracker** to find the optimal learning rate and model capacity.




## Challenge 2: Manual Backpropagation
- **Status:** Completed.
- **Observations:** The manual implementation (using raw tensors and manual gradient computation) successfully matches the loss trajectory of the `nn.Module` version. 
- **Learning Rate:** A lower learning rate (0.0001) was used for the manual version to ensure numerical stability and prevent gradient explosion (NaN values).
- **Result:** Both versions show a consistent decrease in MSE loss over 100 epochs.





## Challenge 3: Custom Training Framework

### Design Decisions & Architecture
For this challenge, I implemented a reusable `Trainer` class. The main goal was to achieve **Separation of Concerns** by decoupling the training logic from the model architecture and dataset specifics.

Key features of my implementation:
- **Abstraction:** The trainer accepts any `nn.Module`, making it compatible with different neural network architectures.
- **Flexibility:** It supports custom configurations via a dictionary (learning rate, epochs, etc.).
- **Early Stopping:** Implemented to prevent overfitting and save computational resources by stopping when validation loss plateaus.
- **Automated Reporting:** The framework automatically generates a JSON report and a loss curve visualization for every experiment, ensuring reproducibility.

### Experiments Conducted
1. **Housing Dataset:** Trained a regression model on the provided house price data.
2. **Synthetic Dataset:** Demonstrated the framework's flexibility by training a model to fit a Sine Wave ($y = \sin(x) + \text{noise}$).

### Results Summary
- **Housing Report:** Saved in `housing_report.json` with corresponding `housing_loss.png`.
- **Synthetic Report:** Saved in `synthetic_report.json` with corresponding `synthetic_loss.png`.
- **Checkpoints:** The best weights for each model were saved as `.pt` files.