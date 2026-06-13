import joblib
import numpy as np

model = joblib.load("../models/fraud_model.pkl")

def predict_transaction(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    return prediction[0], probability