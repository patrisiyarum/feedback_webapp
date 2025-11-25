"""
FastAPI Backend for FCR Feedback Categorization Model (Subcategory Only)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Required for BERT ops
import json
import numpy as np
from typing import List, Dict
import os
import psycopg2
import datetime
import re
import requests  # Required for downloading the model

app = FastAPI(title="FCR Feedback Categorization API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
model = None
sub_classes = None

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Constants & Preprocessing Logic ---
BEVERAGE_WORDS = ["water", "beverage", "drink", "juice", "soda", "coffee", "tea", "bottle", "bottles"]
SERVICE_ITEM_WORDS = [
    "utensil", "fork", "knife", "spoon", "napkin",
    "slipper", "amenity kit", "amenity bag", "pillow",
    "blanket", "glass", "tray setup", "tray"
]
CATERING_WORDS = [
    "catering", "kitchen", "boarded", "not boarded",
    "loaded", "never loaded", "improperly loaded",
    "missing meals", "no second meal", "not catered"
]
CREW_WORDS = [
    "flight attendant", "attendant", "fa", "cabin crew",
    "steward", "staff", "crew did not provide", "not offered"
]

def clean_text(t: str) -> str:
    t = str(t)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'other/comments:\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bY\b', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = t.strip().lower()
    return t

def keyword_features_from_text(t: str) -> List[float]:
    t = t.lower()
    has_beverage = int(any(w in t for w in BEVERAGE_WORDS))
    has_service_item = int(any(w in t for w in SERVICE_ITEM_WORDS))
    has_catering = int(any(w in t for w in CATERING_WORDS))
    has_crew = int(any(w in t for w in CREW_WORDS))
    return [float(has_beverage), float(has_service_item), float(has_catering), float(has_crew)]

# --- Google Drive Download Logic ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# --- Database Setup ---
def init_db():
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set. Logging disabled.")
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
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

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    text: str

class PredictionResult(BaseModel):
    label: str
    probability: float

class PredictionResponse(BaseModel):
    subPredictions: List[PredictionResult]

class BulkPredictionRequest(BaseModel):
    texts: List[str]

class BulkPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# --- Startup Event ---
@app.on_event("startup")
async def load_model_and_classes():
    global model, sub_classes
    
    init_db()
    
    # 1. Download Model if Missing
    model_path = os.getenv("MODEL_PATH", "subcategory_model_augmented.keras")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found at {model_path}. Downloading from Drive...")
        # YOUR FILE ID IS INSERTED HERE:
        FILE_ID = "1cw0XieJsrexcv3aPnjQDRccpoXdvhiCj" 
        try:
            download_file_from_google_drive(FILE_ID, model_path)
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

    try:
        # 2. Load Model
        # We must map 'KerasLayer' to the hub.KerasLayer class
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        print(f"✅ Model loaded from {model_path}")
        
        # 3. Load Classes
        sub_classes_path = os.getenv("SUB_CLASSES_PATH", "subcategory_classes.json")
        with open(sub_classes_path, 'r') as f:
            sub_classes = json.load(f)
        print(f"✅ Loaded {len(sub_classes)} subcategories")
        
    except Exception as e:
        print(f"❌ Error loading model/classes: {e}")
        model = None
        sub_classes = []

# --- Prediction Logic ---
def predict_text(text_input: str) -> Dict:
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        cleaned_text = clean_text(text_input)
        kw_feats = keyword_features_from_text(cleaned_text)
        
        input_text_tensor = np.array([cleaned_text])
        input_kw_tensor = np.array([kw_feats])
        
        predictions = model.predict([input_text_tensor, input_kw_tensor], verbose=0)
        probs = predictions[0] * 100 
        
        sub_predictions = []
        for i, prob in enumerate(probs):
            if i < len(sub_classes):
                sub_predictions.append({"label": sub_classes[i], "probability": float(prob)})
        
        sub_sorted = sorted(sub_predictions, key=lambda x: x['probability'], reverse=True)
        
        # Log to DB
        if DATABASE_URL:
            try:
                top_sub = sub_sorted[0]
                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        timestamp, input_text, 
                        predicted_main_category, main_confidence, 
                        predicted_sub_category, sub_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.datetime.now(datetime.timezone.utc),
                        text_input,
                        "N/A",
                        0.0,
                        top_sub["label"],
                        top_sub["probability"]
                    )
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"DB Log Error: {e}")
        
        return {"subPredictions": sub_sorted}

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Endpoints ---
@app.get("/")
async def root():
    return {"status": "online", "model": "Subcategory Classification Augmented"}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "sub_classes_count": len(sub_classes) if sub_classes else 0
    }

@app.get("/categories")
async def get_categories():
    if not sub_classes:
        raise HTTPException(status_code=503, detail="Classes not loaded")
    return {"subCategories": sub_classes}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    return predict_text(request.text)

@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(request: BulkPredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for t in request.texts:
        if not t or not t.strip():
            results.append({"subPredictions": []})
        else:
            try:
                results.append(predict_text(t))
            except:
                results.append({"subPredictions": [{"label": "Error", "probability": 0.0}]})
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
