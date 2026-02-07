from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
from fastapi import Query

from backend.model_loader import load_production_model
from backend.db import get_connection

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# Load model at startup
# --------------------------------------------------
try:
    model, model_version = load_production_model()
except Exception as e:
    model = None
    model_version = None
    logging.error(f"Model loading failed: {e}")

# --------------------------------------------------
# Request schema
# --------------------------------------------------
class NewsRequest(BaseModel):
    text: str

# --------------------------------------------------
# Database helpers
# --------------------------------------------------
def save_prediction(text: str, label: str, confidence: float):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO prediction_history (text, prediction, confidence)
            VALUES (%s, %s, %s)
            """,
            (text, label, confidence)
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        # DB failure should NOT kill inference
        logging.error(f"Failed to save prediction: {e}")

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if model else "error",
        "model_loaded": model is not None,
        "model_version": model_version
    }

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(request: NewsRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # ML prediction
    label = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]
    confidence = round(float(max(probabilities)), 4)

    # Save to DB (non-blocking for user)
    save_prediction(
        text=request.text,
        label=label,
        confidence=confidence
    )

    return {
        "label": label,
        "confidence": confidence,
        "model_version": model_version
    }

# --------------------------------------------------
# History endpoint
# --------------------------------------------------
@app.get("/history", response_model=List[dict])
def get_history(limit: int = 10):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT text, prediction, confidence, created_at
            FROM prediction_history
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,)
        )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "text": r[0],
                "label": r[1],
                "confidence": r[2],
                "created_at": r[3]
            }
            for r in rows
        ]

    except Exception as e:
        logging.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")
    
    
# --------------------------------------------------
# delete endpoint
# --------------------------------------------------

@app.delete("/history")
def delete_history_item(text: str = Query(...)):
    """
    Delete a history record by exact text.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM prediction_history
            WHERE text = %s
            """,
            (text,)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "deleted_text": text}
    except Exception as e:
        logging.error(f"Failed to delete history item: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete history item")
