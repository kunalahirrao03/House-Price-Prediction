import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("house_prices.csv")

# Features & target
X = df.drop("Price", axis=1).values
y = df["Price"].values

# ---------------- FEATURE SCALING ---------------- #
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  # Avoid division by zero

X_scaled = (X - X_mean) / X_std

# ---------------- TARGET SCALING ---------------- #
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Initialize parameters
n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0.0

# Cost function (MSE)
def cost_function(X, y, w, b):
    m = X.shape[0]
    return np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)

# Gradient computation
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    errors = np.dot(X, w) + b - y
    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db

# Gradient descent
def gradient_descent(X, y, w, b, alpha, num_iters):
    prev_cost = float("inf")
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 500 == 0:
            cost = cost_function(X, y, w, b)
            print(f"Iteration {i:>5}: Cost = {cost:.6f}")

            # ✅ Early stop if cost stops improving meaningfully
            if abs(prev_cost - cost) < 1e-9:
                print(f"   Converged early at iteration {i}")
                break
            prev_cost = cost

    return w, b

# ✅ Train — higher iterations, stable learning rate
w, b = gradient_descent(X_train, y_train, w, b, alpha=0.05, num_iters=5000)

print("\n✅ Training complete")

# ---------------- EVALUATE ---------------- #
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

y_pred_scaled = np.dot(X_test, w) + b
y_pred_actual = y_pred_scaled * y_std + y_mean
y_test_actual = y_test * y_std + y_mean

r2 = r2_score(y_test_actual, y_pred_actual)
mae = np.mean(np.abs(y_test_actual - y_pred_actual))

print(f"   R² Score : {r2:.4f}  (1.0 = perfect)")
print(f"   MAE      : ₹{mae:,.0f}")
print(f"   Pred range: ₹{y_pred_actual.min():,.0f} – ₹{y_pred_actual.max():,.0f}")

# ---------------- SAVE MODEL ---------------- #
with open("model.pkl", "wb") as f:
    pickle.dump((w, b, X_mean, X_std, y_mean, y_std), f)

print("\n✅ Model saved to model.pkl")