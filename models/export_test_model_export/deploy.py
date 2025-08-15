
# Model Deployment Script for export_test_model
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('export_test_model_random_forest.pkl')

# Example prediction function
def predict(data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    return model.predict(df)

# Example usage
# result = predict({"feature1": 1.0, "feature2": 2.0})
