"""
FastAPI Backend for FCR Feedback Categorization Model (Subcategory Only)
NO DATABASE VERSION
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
import re
import gdown  # Required for downloading the model

app = FastAPI(title="FCR Feedback Categorization API")

# --- CORS Configuration ---
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
    """Cleans text to match training data distribution."""
    t = str(t)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'other/comments:\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bY\b', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = t.strip().lower()
    return t

def keyword_features_from_text(t: str) -> List[float]:
    """Extracts the 4 keyword features required by the model."""
    t = t.lower()
    has_beverage = int(any(w in t for w in BEVERAGE_WORDS))
    has_service_item = int(any(w in t for w in SERVICE_ITEM_WORDS))
    has_catering = int(any(w in t for w in CATERING_WORDS))
    has_crew = int(any(w in t for w in CREW_WORDS))
    return [float(has_beverage), float(has_service_item), float(has_catering), float(has_crew)]

# --- Robust Model Download Logic ---
def download_model_if_missing(model_path):
    """Downloads model from Google Drive if not found locally."""
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found at {model_path}. Downloading from Drive...")
        
        # FILE ID from your specific Google Drive link
        file_id = "1cw0XieJsrexcv3aPnjQDRccpoXdvhiCj"
        
        try:
            gdown.download(id=file_id, output=model_path, quiet=False)
            
            if os.path.exists(model_path) and os.path.getsize(model_path) < 100000:
                print("❌ Error: Downloaded file is too small (likely an error page). Deleting...")
                os.remove(model_path)
            else:
                print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

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
    
    # 1. Download Model if Missing
    model_path = os.getenv("MODEL_PATH", "subcategory_model_augmented.keras")
    download_model_if_missing(model_path)
    
    try:
        # 2. Load Model
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
        print(f"❌ Error loading model or classes: {e}")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"DEBUG: File size is {size_mb:.2f} MB")
        model = None
        sub_classes = []

# --- Prediction Logic ---
def predict_text(text_input: str) -> Dict:
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        cleaned_text = clean_text(text_input)
        kw_feats = keyword_features_from_text(cleaned_text)
        
        # Prepare Inputs
        input_text_tensor = np.array([cleaned_text])
        input_kw_tensor = np.array([kw_feats])
        
        # Predict
        predictions = model.predict([input_text_tensor, input_kw_tensor], verbose=0)
        
        # Format Results
        probs = predictions[0] * 100 
        sub_predictions = []
        for i, prob in enumerate(probs):
            if i < len(sub_classes):
                sub_predictions.append({"label": sub_classes[i], "probability": float(prob)})
        
        sub_sorted = sorted(sub_predictions, key=lambda x: x['probability'], reverse=True)
        
        # Note: Database Logging removed
        
        return {"subPredictions": sub_sorted}

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "model": "Subcategory Classification Augmented (8 Labels) - No DB"}

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
