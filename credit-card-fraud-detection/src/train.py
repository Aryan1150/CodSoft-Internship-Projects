import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# LOAD DATASET
# ==========================================

dataset_path = "dataset"

files = os.listdir(dataset_path)
print("Files found:", files)

train_df = pd.read_csv(os.path.join(dataset_path, "fraudTrain.csv"))
test_df = pd.read_csv(os.path.join(dataset_path, "fraudTest.csv"))

# Merge datasets
df = pd.concat([train_df, test_df], ignore_index=True)

print("\nDataset Loaded Successfully!")
print("Shape:", df.shape)

# ==========================================
# OPTIONAL: USE SAMPLE FOR FASTER TRAINING
# ==========================================

print("\nTaking sample of 100,000 rows for faster training...")

df = df.sample(n=100000, random_state=42)

print("New Shape:", df.shape)

# ==========================================
# TARGET COLUMN
# ==========================================

target = "is_fraud"

# ==========================================
# DROP UNNECESSARY COLUMNS
# ==========================================

drop_cols = [
    "is_fraud",
    "Unnamed: 0",
    "trans_num",
    "first",
    "last",
    "street",
    "dob",
    "cc_num",
    "zip",
    "unix_time",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city"
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target]

# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numerical_cols
        )
    ]
)

# ==========================================
# MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Model...")
print("=" * 50)

# ==========================================
# TRAIN
# ==========================================

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ==========================================
# EVALUATION
# ==========================================

predictions = model.predict(X_test)

print("\nClassification Report")
print("=" * 50)
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print("=" * 50)
print(confusion_matrix(y_test, predictions))

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model.pkl")

print("\nModel Saved Successfully!")
print("Location: models/fraud_model.pkl")