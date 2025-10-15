# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_text  # This is crucial! It registers the necessary ops for the model.
import tensorflow_hub as hub
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="Feedback Categorization API",
    description="An API to classify user feedback into main and sub-categories.",
    version="1.0.0"
)

# --- 2. Add CORS Middleware ---
# This allows your frontend (running on a different URL) to safely communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT: For production, change "*" to your frontend's specific URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load Model and Class Files on Startup ---
# We define paths relative to this 'main.py' file.
# Assumes 'api' folder is at the same level as the model and json files.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'two_layer_categorization_model_fixed')
MAIN_CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'main_category_classes.json')
SUB_CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'subcategory_classes.json')

model = None
main_category_classes = []
subcategory_classes = []

@app.on_event("startup")
def load_model_and_classes():
    """
    This function runs once when the API server starts.
    It loads the TensorFlow model and the category class files into memory.
    """
    global model, main_category_classes, subcategory_classes
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ TensorFlow model loaded successfully.")
        else:
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")
            
        with open(MAIN_CLASSES_PATH, 'r') as f:
            main_category_classes = json.load(f)
            print(f"✅ Loaded {len(main_category_classes)} main categories.")
            
        with open(SUB_CLASSES_PATH, 'r') as f:
            subcategory_classes = json.load(f)
            print(f"✅ Loaded {len(subcategory_classes)} sub-categories.")
            
    except Exception as e:
        print(f"❌ Critical Error: Could not load model or class files. API will not be functional.")
        print(f"Error details: {e}")
        # In a real-world scenario, you might want the app to exit if the model fails to load.

# --- 4. Define API Data Schemas (using Pydantic) ---
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    main_category: str
    sub_category: str
    main_category_confidence: float
    sub_category_confidence: float

# --- 5. Create API Endpoints ---
@app.get("/")
def read_root():
    """A simple 'health check' endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Feedback Categorization API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts a text input and returns the predicted main and sub-category with confidence scores.
    """
    if model is None or not main_category_classes or not subcategory_classes:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please check server logs for more details."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # The model expects a list of strings and returns a list of predictions
        predictions = model.predict([request.text])
        
        # The output of your model is a list containing two arrays:
        # predictions[0] is for the main category
        # predictions[1] is for the sub-category
        main_category_preds = predictions[0][0]
        sub_category_preds = predictions[1][0]

        # Get the index of the highest probability for each category
        main_pred_index = np.argmax(main_category_preds)
        sub_pred_index = np.argmax(sub_category_preds)

        # Get the confidence scores (the actual probability value)
        main_confidence = float(main_category_preds[main_pred_index])
        sub_confidence = float(sub_category_preds[sub_pred_index])

        # Map the index back to the human-readable class name
        main_category_result = main_category_classes[main_pred_index]
        sub_category_result = subcategory_classes[sub_pred_index]

        return PredictionResponse(
            main_category=main_category_result,
            sub_category=sub_category_result,
            main_category_confidence=main_confidence,
            sub_category_confidence=sub_confidence
        )
    except Exception as e:
        # Catch any errors during prediction
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
