import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.keras
import dagshub

# -------------------------
# Load dataset
# -------------------------
true_df = pd.read_csv("data/raw/True.csv")
fake_df = pd.read_csv("data/raw/Fake.csv")

true_df["label"] = 1  # REAL
fake_df["label"] = 0  # FAKE

df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42)
X = df["text"].astype(str).values
y = df["label"].values

# -------------------------
# Tokenizer
# -------------------------
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, y, test_size=0.2, random_state=42
)

# -------------------------
# Build GRU model
# -------------------------
embedding_dim = 128

model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    GRU(128, return_sequences=False),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# -------------------------
# Train model
# -------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop]
)

# -------------------------
# Evaluate
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# -------------------------
# DagsHub + MLflow logging
# -------------------------
dagshub.init(repo_owner='JanaJouni', repo_name='fake-news-mlops', mlflow=True)
mlflow.set_experiment("Fake-News-GRU")

with mlflow.start_run():
    # Save tokenizer as artifact
    tokenizer_path = "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    mlflow.log_artifact(tokenizer_path, artifact_path="tokenizer")

    # Log GRU model
    mlflow.keras.log_model(
        model,
        artifact_path="gru_model",
        registered_model_name="fake_news_gru"
    )

    # Log metrics & params
    mlflow.log_param("max_words", max_words)
    mlflow.log_param("max_len", max_len)
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_metric("test_accuracy", acc)

print("Training complete. Model and tokenizer logged to DagsHub.")
