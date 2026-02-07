from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from backend.db import get_connection
from backend.model_loader import load_production_model

# --------------------------
# App setup
# --------------------------
app = FastAPI(title="Fake News Detection API (GRU)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# --------------------------
# Load GRU model + tokenizer
# --------------------------
try:
    model, tokenizer, max_len, model_version = load_production_model()
    logging.info("GRU model and tokenizer loaded from DagsHub")
except Exception as e:
    model = tokenizer = max_len = model_version = None
    logging.error(f"Failed to load GRU model: {e}")

# --------------------------
# Request schema
# --------------------------
class NewsRequest(BaseModel):
    text: str

# --------------------------
# Database helpers
# --------------------------
def save_prediction(text: str, label: str, confidence: float):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prediction_history (text, prediction, confidence) VALUES (%s,%s,%s)",
            (text, label, confidence)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to save prediction: {e}")

def delete_prediction(pred_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM prediction_history WHERE id=%s", (pred_id,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to delete prediction: {e}")

# --------------------------
# Health check
# --------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if model else "error",
        "model_loaded": model is not None,
        "model_version": model_version
    }

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict(request: NewsRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    seq = tokenizer.texts_to_sequences([request.text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    pred_prob = model.predict(pad)[0][0]
    label = "REAL" if pred_prob >= 0.5 else "FAKE"
    confidence = pred_prob if label == "REAL" else 1 - pred_prob

    save_prediction(text=request.text, label=label, confidence=float(confidence))

    return {"label": label, "confidence": round(float(confidence),4), "model_version": model_version}

# --------------------------
# History endpoint
# --------------------------
@app.get("/history", response_model=List[dict])
def get_history(limit: int = 10):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, text, prediction, confidence, created_at FROM prediction_history ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"id": r[0], "text": r[1], "label": r[2], "confidence": float(r[3]), "created_at": r[4]}
            for r in rows
        ]
    except Exception as e:
        logging.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

# --------------------------
# Delete endpoint
# --------------------------
@app.delete("/history/{pred_id}")
def delete_history(pred_id: int):
    try:
        delete_prediction(pred_id)
        return {"status": "deleted", "id": pred_id}
    except Exception as e:
        logging.error(f"Failed to delete prediction {pred_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete prediction")
