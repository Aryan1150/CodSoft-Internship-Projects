# ==========================================
# CUSTOMER CHURN PREDICTION - PHASE 1
# Data Loading, Cleaning & Basic EDA
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Dataset
# ----------------------------
print("Loading Dataset...\n")

df = pd.read_csv("data/Churn_Modelling.csv")

# ----------------------------
# 2. Dataset Overview
# ----------------------------
print("First 5 Rows:")
print(df.head())

print("\n" + "="*50)

print("\nDataset Shape:")
print(df.shape)

print("\n" + "="*50)

print("\nColumn Names:")
print(df.columns.tolist())

print("\n" + "="*50)

# ----------------------------
# 3. Check Missing Values
# ----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

print("\n" + "="*50)

# ----------------------------
# 4. Basic Statistics
# ----------------------------
print("\nStatistical Summary:")
print(df.describe())

print("\n" + "="*50)

# ----------------------------
# 5. Churn Distribution
# ----------------------------
print("\nCustomer Churn Distribution:")
print(df["Exited"].value_counts())

# ----------------------------
# 6. Remove Unnecessary Columns
# ----------------------------
df = df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname"
    ]
)

print("\nColumns after dropping:")
print(df.columns.tolist())

# ----------------------------
# 7. Convert Categorical Data
# ----------------------------
df = pd.get_dummies(
    df,
    columns=["Gender", "Geography"],
    drop_first=True
)

print("\nDataset after Encoding:")
print(df.head())

print("\nEncoded Dataset Shape:")
print(df.shape)

# ----------------------------
# 8. Save Cleaned Dataset
# ----------------------------
df.to_csv(
    "data/cleaned_churn_data.csv",
    index=False
)

print("\n✅ Cleaned dataset saved successfully!")
print("Location: data/cleaned_churn_data.csv")

# ----------------------------
# 9. Visualization
# ----------------------------

# Churn Distribution
plt.figure(figsize=(6, 4))
df["Exited"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(6, 4))
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Balance Distribution
plt.figure(figsize=(6, 4))
df["Balance"].hist(bins=20)
plt.title("Balance Distribution")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\n🎉 Phase 1 Completed Successfully!")