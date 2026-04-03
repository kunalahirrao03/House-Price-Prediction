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