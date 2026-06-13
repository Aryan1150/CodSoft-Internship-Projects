import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ==============================
# BASE DIRECTORY (IMPORTANT FIX)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path (safe + portable)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "spam.csv")

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# keep only useful columns (UCI dataset format)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset Loaded:", df.shape)

# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# ==============================
# PIPELINE (TF-IDF + MODEL)
# ==============================
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# ==============================
# TRAIN MODEL
# ==============================
model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "spam_model.pkl")
joblib.dump(model, model_path)

print("\nModel saved successfully at:", model_path)