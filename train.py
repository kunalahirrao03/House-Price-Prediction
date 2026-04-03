import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Load dataset with explicit float64 dtypes to avoid ExtensionArray
df = pd.read_csv("app/house_prices.csv", dtype={
    'Area_sqft': 'float64', 
    'Bedrooms': 'float64', 
    'Bathrooms': 'float64', 
    'Age': 'float64', 
    'Location_Score': 'float64', 
    'Garage': 'float64', 
    'Price': 'float64'
})

# To numpy arrays explicitly
X = df.drop("Price", axis=1).to_numpy(dtype=float)
y = df["Price"].to_numpy(dtype=float)

print("Dtypes after load:")
print(df.dtypes)
print("\\nSample data:")
print(df.head())

# FEATURE SCALING - pure numpy
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1
X_scaled = (X - X_mean) / X_std

# TARGET SCALING - pure numpy
y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Training
n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0.0

def cost_function(X, y, w, b):
    m = X.shape[0]
    return np.sum((np.dot(X, w) + b - y)**2) / (2 * m)

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    errors = np.dot(X, w) + b - y
    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    prev_cost = float('inf')
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % 500 == 0:
            cost = cost_function(X, y, w, b)
            print(f"Iteration {i}: Cost = {cost:.6f}")
            if abs(prev_cost - cost) < 1e-9:
                print(f"Converged at iteration {i}")
                break
            prev_cost = cost
    return w, b

w, b = gradient_descent(X_train, y_train, w, b, 0.05, 5000)
print("\\nTraining complete")

# Evaluate
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

y_pred_scaled = np.dot(X_test, w) + b
y_pred = y_pred_scaled * y_std + y_mean
y_test_actual = y_test * y_std + y_mean

r2 = r2_score(y_test_actual, y_pred)
mae = np.mean(np.abs(y_test_actual - y_pred))

print(f"R² Score: {r2:.4f}")
print(f"MAE: ₹{mae:,.0f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((w, b, X_mean, X_std, y_mean, y_std), f)

print("Model saved to model.pkl")
