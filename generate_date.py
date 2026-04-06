import pandas as pd
import numpy as np
 
np.random.seed(42)
 
n = 1000
 
data = {
    "Area_sqft": np.random.randint(500, 3000, n),
    "Bedrooms": np.random.randint(1, 6, n),
    "Bathrooms": np.random.randint(1, 5, n),
    "Age": np.random.randint(0, 30, n),
    "Location_Score": np.random.randint(1, 11, n),
    "Garage": np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)
 
# Realistic Indian house pricing formula (values in ₹)
# Base: ~₹3,000–5,000 per sqft (mid-tier city)
# Bedrooms add a modest premium (not the main driver — area already captures size)
# Location score has a proportional multiplier on the base price
# Age depreciates value slightly
# Garage adds a flat premium
 
base_price_per_sqft = 4000  # ₹4,000/sqft baseline
 
df["Price"] = (
    df["Area_sqft"] * base_price_per_sqft                   
    + df["Bedrooms"] * 50_000                                
    + df["Bathrooms"] * 30_000                               
    - df["Age"] * 20_000                                     
    + df["Location_Score"] * df["Area_sqft"] * 200           
    + df["Garage"] * 100_000                                 
    + np.random.normal(0, 200_000, n)                        
).clip(lower=500_000)                                        
 
df.to_csv("house_prices.csv", index=False)
 
print("✅ Dataset created")
print(f"   Price range: ₹{df['Price'].min():,.0f} – ₹{df['Price'].max():,.0f}")
print(f"   Mean price:  ₹{df['Price'].mean():,.0f}")