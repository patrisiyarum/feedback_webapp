"""
FastAPI Backend for FCR Feedback Categorization Model
This file should be deployed alongside your React frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT ops
import json
import base64
import numpy as np
from typing import List, Dict
import os
import psycopg2  # For PostgreSQL
import datetime  # For timestamps

app = FastAPI(title="FCR Feedback Categorization API")

# Configure CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and classes
model = None
main_classes = None
sub_classes = None

# Get Database URL from Environment
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Database Setup ---
def init_db():
    """Initializes the PostgreSQL database table if it doesn't exist."""
    if not DATABASE_URL:
        print("❌ ERROR: DATABASE_URL environment variable not set. Database logging will be disabled.")
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        # Create the table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            input_text TEXT,
            predicted_main_category VARCHAR(255),
            main_confidence REAL,
            predicted_sub_category VARCHAR(255),
            sub_confidence REAL
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            file_name VARCHAR(255) NOT NULL,
            file_data BYTEA NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            results JSONB NOT NULL DEFAULT '[]'::jsonb
        )
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads (created_at DESC)
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database table initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")

# Request/Response models
class PredictionRequest(BaseModel):
    text: str

class PredictionResult(BaseModel):
    label: str
    probability: float

class PredictionResponse(BaseModel):
    mainPredictions: List[PredictionResult]
    subPredictions: List[PredictionResult]

class BulkPredictionRequest(BaseModel):
    texts: List[str]

class BulkPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class UploadSaveRequest(BaseModel):
    file_name: str
    file_base64: str
    results: list

# Load model and classes on startup
@app.on_event("startup")
async def load_model_and_classes():
    global model, main_classes, sub_classes
    
    init_db()  # Initialize the PostgreSQL database
    
    try:
        # Adjust these paths based on where your files are located
        # Go one level UP (../) to find the files in the root directory
        model_path = os.getenv("MODEL_PATH", "../two_layer_categorization_model_fixed")
        main_classes_path = os.getenv("MAIN_CLASSES_PATH", "../main_category_classes.json")
        sub_classes_path = os.getenv("SUB_CLASSES_PATH", "../subcategory_classes.json")
        # Load model
        model = tf.saved_model.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        
        # Load class labels
        with open(main_classes_path) as f:
            main_classes = json.load(f)
        with open(sub_classes_path) as f:
            sub_classes = json.load(f)
        print(f"✅ Classes loaded: {len(main_classes)} main categories, {len(sub_classes)} subcategories")
        
    except Exception as e:
        print(f"❌ Error loading model or classes: {e}")
        # Note: In a real app, you might want to prevent the app from starting if the model fails
        # For now, we'll let it run so /health endpoint can report the error
        model = None 
        main_classes = []
        sub_classes = []


def predict_text(text: str) -> Dict:
    """Make prediction for a single text input AND log it to PostgreSQL"""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # Convert to tensor
        inputs = tf.constant([text])
        
        # Get predictions via the serving signature
        serve_fn = model.signatures["serving_default"]
        outputs = serve_fn(inputs)
        
        # Extract probabilities
        main_probs = outputs["main_category_output"][0].numpy() * 100
        sub_probs = outputs["subcategory_output"][0].numpy() * 100
        
        # Sort predictions by probability
        main_sorted = sorted(
            zip(main_classes, main_probs), 
            key=lambda x: x[1], 
            reverse=True
        )
        sub_sorted = sorted(
            zip(sub_classes, sub_probs), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # --- Database Logging ---
        if DATABASE_URL:
            try:
                top_main = main_sorted[0]
                top_sub = sub_sorted[0]

                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions (timestamp, input_text, predicted_main_category, main_confidence, predicted_sub_category, sub_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.datetime.now(datetime.timezone.utc), # Use timezone-aware datetime
                        text,
                        top_main[0],  # e.g., "Food Quality"
                        float(top_main[1]),  # e.g., 98.5
                        top_sub[0],   # e.g., "Taste Issues"
                        float(top_sub[1])    # e.g., 95.2
                    )
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"Database logging failed: {e}") # Log error but don't fail the request
        # --- End of Database Logging ---

        # Return the full prediction list as normal
        return {
            "mainPredictions": [
                {"label": label, "probability": float(prob)}
                for label, prob in main_sorted
            ],
            "subPredictions": [
                {"label": label, "probability": float(prob)}
                for label, prob in sub_sorted
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "FCR Feedback Categorization API",
        "version": "2.2"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "main_classes_count": len(main_classes) if main_classes else 0,
        "sub_classes_count": len(sub_classes) if sub_classes else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict category for a single text input"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    return predict_text(request.text)

@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(request: BulkPredictionRequest):
    """Predict categories for multiple text inputs"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    predictions = []
    for text in request.texts:
        try:
            # Handle empty strings from bulk list
            if not text or not text.strip():
                pred = {
                    "mainPredictions": [{"label": "No Input", "probability": 0.0}],
                    "subPredictions": [{"label": "No Input", "probability": 0.0}]
                }
            else:
                pred = predict_text(text)
            
            predictions.append(pred)
        except Exception as e:
            # Return error prediction for failed items
            predictions.append({
                "mainPredictions": [{"label": "Error", "probability": 0.0}],
                "subPredictions": [{"label": f"Processing Error: {e}", "probability": 0.0}]
            })
    
    return {"predictions": predictions}

@app.get("/categories")
async def get_categories():
    """Get available category labels"""
    if not main_classes or not sub_classes:
        raise HTTPException(status_code=503, detail="Classes not loaded")
    
    return {
        "mainCategories": main_classes,
        "subCategories": sub_classes
    }

# --- Upload Persistence Endpoints ---

@app.post("/uploads")
async def save_upload(request: UploadSaveRequest):
    """Save a processed upload (raw file + results) to the database."""
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        file_bytes = base64.b64decode(request.file_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 file data")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO uploads (created_at, file_name, file_data, row_count, results)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (
                datetime.datetime.now(datetime.timezone.utc),
                request.file_name,
                psycopg2.Binary(file_bytes),
                len(request.results),
                json.dumps(request.results),
            )
        )
        row = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "id": row[0],
            "created_at": row[1].isoformat(),
            "file_name": request.file_name,
            "row_count": len(request.results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")


@app.get("/uploads")
async def list_uploads():
    """List all past uploads (metadata only)."""
    if not DATABASE_URL:
        return {"uploads": []}

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, created_at, file_name, row_count FROM uploads ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "uploads": [
                {
                    "id": r[0],
                    "created_at": r[1].isoformat(),
                    "file_name": r[2],
                    "row_count": r[3],
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {e}")


@app.get("/uploads/{upload_id}")
async def get_upload(upload_id: int):
    """Get a specific upload's results (no file bytes)."""
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, created_at, file_name, row_count, results FROM uploads WHERE id = %s",
            (upload_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Upload not found")

        return {
            "id": row[0],
            "created_at": row[1].isoformat(),
            "file_name": row[2],
            "row_count": row[3],
            "results": row[4],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get upload: {e}")


@app.get("/uploads/{upload_id}/file")
async def download_upload_file(upload_id: int):
    """Download the original uploaded file."""
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_name, file_data FROM uploads WHERE id = %s",
            (upload_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Upload not found")

        return Response(
            content=bytes(row[1]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{row[0]}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {e}")


if __name__ == "__main__":
    import uvicorn
    # This part is for local debugging. Render uses the `uvicorn` command in your Start Command.
    print("--- Starting Uvicorn server for local development ---")
    print(f"--- DATABASE_URL set: {DATABASE_URL is not None} ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)

