from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT ops
from keras.layers import TFSMLayer
import json
import numpy as np
from typing import List, Dict
import os
import psycopg2  # <-- NEW: Import the PostgreSQL driver
import datetime  # <-- NEW: For timestamps

app = FastAPI(title="FCR Feedback Categorization API")

# ... (your CORS middleware setup) ...

# Global variables for model and classes
model = None
main_classes = None
sub_classes = None

# --- NEW: Get Database URL from Environment ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- NEW: Database Setup ---
def init_db():
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
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database table initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")

# ... (your Pydantic models: PredictionRequest, etc.) ...

# Load model and classes on startup
@app.on_event("startup")
async def load_model_and_classes():
    global model, main_classes, sub_classes
    
    if not DATABASE_URL:
        print("❌ ERROR: DATABASE_URL environment variable not set.")
    else:
        init_db()  # Initialize the PostgreSQL database
    
    try:
        # ... (your existing model and class loading code) ...
        print("✅ Model and classes loaded.")
    except Exception as e:
        print(f"❌ Error loading model or classes: {e}")

def predict_text(text: str) -> Dict:
    """Make prediction for a single text input AND log it to PostgreSQL"""
    try:
        # ... (your existing code to get predictions) ...
        inputs = tf.constant([text])
        outputs = model(inputs)
        main_probs = outputs["main_category_output"][0].numpy() * 100
        sub_probs = outputs["subcategory_output"][0].numpy() * 100
        
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
        
        # --- THIS IS THE MODIFIED PART ---
        top_main = main_sorted[0]
        top_sub = sub_sorted[0]

        # Log this prediction to the PostgreSQL database
        if DATABASE_URL:
            try:
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
        # --- END OF MODIFIED PART ---

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

# ... (rest of your FastAPI app: /health, /predict, /categories, etc.) ...
