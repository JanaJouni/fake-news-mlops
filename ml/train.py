from dotenv import load_dotenv
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample

# Load environment variables
load_dotenv()

# --------------------------
# Load data
# --------------------------
true_df = pd.read_csv("data/raw/True.csv")
fake_df = pd.read_csv("data/raw/Fake.csv")

true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

# --------------------------
# Balance the dataset
# --------------------------
n_samples = min(len(true_df), len(fake_df))

true_balanced = resample(true_df, n_samples=n_samples, random_state=42)
fake_balanced = resample(fake_df, n_samples=n_samples, random_state=42)

df = pd.concat([true_balanced, fake_balanced]).sample(frac=1, random_state=42)

print("Class distribution after balancing:")
print(df['label'].value_counts())

X = df["text"]
y = df["label"]

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# MLflow experiment
# --------------------------
dagshub.init(repo_owner='JanaJouni', repo_name='fake-news-mlops', mlflow=True)
mlflow.set_experiment("Fake-News-Detection")

with mlflow.start_run():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, pos_label="FAKE")

    print("Accuracy:", acc)
    print("F1-score:", f1)

    # --------------------------
    # Test a sample sentence
    # --------------------------
    sample_texts = [
        "monkey eats banana",
        "Government passes new law",
        "Scientists discover cure for disease"
    ]
    sample_preds = pipeline.predict(sample_texts)
    sample_probs = pipeline.predict_proba(sample_texts)

    for t, p, prob in zip(sample_texts, sample_preds, sample_probs):
        confidence = max(prob)
        print(f"Sample: '{t}' -> Predicted: {p}, Confidence: {confidence:.2f}")

    # --------------------------
    # Log parameters, metrics, model
    # --------------------------
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "1-2")
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="fake_news_detector"
    )

    print("Training complete and model logged to MLflow")
