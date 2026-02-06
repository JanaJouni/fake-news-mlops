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

load_dotenv()  # load env variables from .env file
# Load data
true_df = pd.read_csv("data/raw/True.csv")
fake_df = pd.read_csv("data/raw/Fake.csv")

true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
dagshub.init(repo_owner='JanaJouni', repo_name='fake-news-mlops', mlflow=True)
mlflow.set_experiment("Fake-News-Detection")

with mlflow.start_run():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, pos_label="FAKE")

    # Log params
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "1-2")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="fake_news_detector"
    )

    print("Training complete")
    print("Accuracy:", acc)
    print("F1-score:", f1)
