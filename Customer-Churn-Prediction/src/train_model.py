# ==========================================
# CUSTOMER CHURN PREDICTION - PHASE 2
# MODEL TRAINING
# ==========================================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ----------------------------
# 1. Load Cleaned Dataset
# ----------------------------

print("Loading cleaned dataset...")

df = pd.read_csv("data/cleaned_churn_data.csv")

# ----------------------------
# 2. Split Features & Target
# ----------------------------

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ----------------------------
# 3. Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ----------------------------
# 4. Feature Scaling
# ----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 5. Train Random Forest
# ----------------------------

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Predictions
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# 7. Evaluation
# ----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 8. Save Model
# ----------------------------

pickle.dump(
    model,
    open("models/random_forest.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("\n✅ Model Saved Successfully!")
print("models/random_forest.pkl")
print("models/scaler.pkl")