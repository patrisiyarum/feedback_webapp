from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_text  # Registers the necessary ops for the model
import tensorflow_hub as hub
from keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

# ============================================================
# 1️⃣ Initialize the FastAPI App
# ============================================================
app = FastAPI(
    title="Feedback Categorization API",
    description="An API to classify user feedback into main and sub-categories.",
    version="2.0.0"
)

# ============================================================
# 2️⃣ Add CORS Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Replace '*' with your Vercel frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3️⃣ Define Model + Class Paths
# ============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "two_layer_categorization_model_fixed")
MAIN_CLASSES_PATH = os.path.join(BASE_DIR, "..", "main_category_classes.json")
SUB_CLASSES_PATH = os.path.join(BASE_DIR, "..", "subcategory_classes.json")

model = None
input_key = "text"  # Default, will auto-detect
main_category_classes = []
subcategory_classes = []

# ============================================================
# 4️⃣ Load Model + JSON Class Files
# ============================================================
@app.on_event("startup")
def load_model_and_classes():
    global model, main_category_classes, subcategory_classes, input_key

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")

        model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
        print("✅ Model loaded successfully from:", MODEL_PATH)
        print("🔍 Model signature info:", model.signatures)

        # Detect the correct input key (text vs inputs)
        sig = model.signatures.get("serving_default")
        if sig:
            for key in sig.structured_input_signature[1].keys():
                input_key = key
                break
        print(f"✅ Detected input key: '{input_key}'")

        # Load category label files
        with open(MAIN_CLASSES_PATH, "r") as f:
            main_category_classes[:] = json.load(f)
            print(f"✅ Loaded {len(main_category_classes)} main categories.")

        with open(SUB_CLASSES_PATH, "r") as f:
            subcategory_classes[:] = json.load(f)
            print(f"✅ Loaded {len(subcategory_classes)} subcategories.")

    except Exception as e:
        print("❌ Error during model/class load:", e)

# ============================================================
# 5️⃣ API Schemas
# ============================================================
class PredictionRequest(BaseModel):
    text: str

# ============================================================
# 6️⃣ Health Check Endpoint
# ============================================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "unavailable",
        "model_loaded": model is not None,
        "input_key": input_key
    }

# ============================================================
# 7️⃣ Prediction Endpoint
# ============================================================
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or not main_category_classes or not subcategory_classes:
        raise HTTPException(status_code=503, detail="Model or class data not available.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Build proper TensorFlow input
        input_tensor = tf.constant([request.text])
        print(f"🧠 Running model using key '{input_key}'")

        # Dynamically call model with correct input key
        if input_key == "inputs":
            predictions = model(inputs=input_tensor)
        else:
            predictions = model(text=input_tensor)

        # Handle possible dict outputs
        if isinstance(predictions, dict):
            outputs = list(predictions.values())
        else:
            outputs = predictions

        main_preds = outputs[0][0].numpy()
        sub_preds = outputs[1][0].numpy()

        # Convert to sorted lists of label-prob pairs
        main_results = [
            {"label": main_category_classes[i], "probability": float(main_preds[i]) * 100}
            for i in range(len(main_preds))
        ]
        sub_results = [
            {"label": subcategory_classes[i], "probability": float(sub_preds[i]) * 100}
            for i in range(len(sub_preds))
        ]

        main_results.sort(key=lambda x: x["probability"], reverse=True)
        sub_results.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "mainPredictions": main_results,
            "subPredictions": sub_results
        }

    except Exception as e:
        print("❌ Prediction failed:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ============================================================
# 8️⃣ Root Endpoint
# ============================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "Feedback Categorization API is running."}
