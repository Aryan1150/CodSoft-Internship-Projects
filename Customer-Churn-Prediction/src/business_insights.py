# ==========================================
# CUSTOMER CHURN PREDICTION - PHASE 3
# BUSINESS INSIGHTS & FEATURE IMPORTANCE
# ==========================================

import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pd.read_csv("data/cleaned_churn_data.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# ------------------------------------------
# 2. Load Trained Model
# ------------------------------------------

print("\nLoading Random Forest Model...")

model = pickle.load(
    open("models/random_forest.pkl", "rb")
)

print("Model Loaded Successfully!")

# ------------------------------------------
# 3. Feature Importance Analysis
# ------------------------------------------

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

X = df.drop("Exited", axis=1)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# Save Feature Importance Table
feature_importance.to_csv(
    "data/feature_importance.csv",
    index=False
)

# Plot Feature Importance
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 4. Churn Distribution
# ------------------------------------------

print("\n" + "=" * 50)
print("CHURN DISTRIBUTION")
print("=" * 50)

print(df["Exited"].value_counts())

plt.figure(figsize=(6, 4))

df["Exited"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Exited")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 5. Average Age by Churn
# ------------------------------------------

print("\n" + "=" * 50)
print("AGE ANALYSIS")
print("=" * 50)

age_analysis = df.groupby(
    "Exited"
)["Age"].mean()

print(age_analysis)

plt.figure(figsize=(6, 4))

age_analysis.plot(
    kind="bar"
)

plt.title("Average Age by Churn Status")
plt.xlabel("Exited")
plt.ylabel("Average Age")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 6. Active Member Analysis
# ------------------------------------------

print("\n" + "=" * 50)
print("ACTIVE MEMBER ANALYSIS")
print("=" * 50)

active_analysis = pd.crosstab(
    df["IsActiveMember"],
    df["Exited"]
)

print(active_analysis)

active_analysis.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Active Member vs Churn")
plt.xlabel("Is Active Member")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 7. Number of Products Analysis
# ------------------------------------------

print("\n" + "=" * 50)
print("PRODUCT ANALYSIS")
print("=" * 50)

product_analysis = pd.crosstab(
    df["NumOfProducts"],
    df["Exited"]
)

print(product_analysis)

product_analysis.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Products vs Churn")
plt.xlabel("Number of Products")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 8. Balance Analysis
# ------------------------------------------

print("\n" + "=" * 50)
print("BALANCE ANALYSIS")
print("=" * 50)

balance_analysis = df.groupby(
    "Exited"
)["Balance"].mean()

print(balance_analysis)

plt.figure(figsize=(6, 4))

balance_analysis.plot(
    kind="bar"
)

plt.title("Average Balance by Churn")
plt.xlabel("Exited")
plt.ylabel("Average Balance")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 9. Salary Analysis
# ------------------------------------------

print("\n" + "=" * 50)
print("SALARY ANALYSIS")
print("=" * 50)

salary_analysis = df.groupby(
    "Exited"
)["EstimatedSalary"].mean()

print(salary_analysis)

plt.figure(figsize=(6, 4))

salary_analysis.plot(
    kind="bar"
)

plt.title("Average Salary by Churn")
plt.xlabel("Exited")
plt.ylabel("Average Salary")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 10. Business Insights Summary
# ------------------------------------------

print("\n" + "=" * 50)
print("BUSINESS INSIGHTS")
print("=" * 50)

top_feature = feature_importance.iloc[0]["Feature"]

print(f"\nMost Important Feature: {top_feature}")

print("\nKey Findings:")

print(
    """
1. Feature Importance identifies the strongest churn indicators.

2. Older customers tend to churn more frequently.

3. Inactive customers have higher churn rates.

4. Customers with fewer products are more likely to leave.

5. Balance and customer engagement strongly influence churn.

6. These insights can help banks improve customer retention.
"""
)

print("\nPhase 3 Completed Successfully!")