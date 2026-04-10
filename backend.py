import os
import time
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Configure TensorFlow logging
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZES = {
    "ResNet50": (64, 64),
    "VGG16": (150, 150),
    "Xception": (64, 64),
}
CLASSES = ["COVID-19", "NORMAL"]

# Global dictionary to hold models
models = {}

# =============================================================================
# MODEL LOADING (LIFESPAN EVENT)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Starting up Backend API...")
    print("[*] Loading COVID-19 Models...")
    
    for name in MODEL_SIZES:
        model_path = os.path.join(BASE_DIR, f"{name}_Model.keras")
        if not os.path.exists(model_path):
            print(f"[ERROR] Could not find model file: {model_path}")
            continue
        try:
            models[name] = tf.keras.models.load_model(model_path)
            print(f"  --> {name} loaded successfully (Size: {MODEL_SIZES[name]}).")
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            
    if not models:
        print("[WARNING] No models were loaded successfully!")
    else:
        print("[*] All available models loaded and ready for inference.")
        
    yield
    
    print("[*] Shutting down Backend API...")
    models.clear()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="COVID-19 AI Diagnostics API",
    description="Backend API for multi-model COVID-19 detection from Chest X-Rays.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the web UI static files
# Ensure the directory exists before mounting
WEB_UI_DIR = os.path.join(BASE_DIR, "web_ui")
if os.path.exists(WEB_UI_DIR):
    app.mount("/ui", StaticFiles(directory=WEB_UI_DIR), name="ui")

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_image_bytes(image_bytes: bytes, target_size: tuple) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        return np.expand_dims(arr, axis=0) / 255.0
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
async def read_index():
    """Serve the professional web UI as the home page."""
    index_path = os.path.join(WEB_UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "COVID-19 Diagnostics API is running. UI folder not found."}

@app.get("/health")
async def health_check():
    """Simple endpoint to verify server status and loaded models."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "message": "COVID-19 Diagnostics API is running."
    }

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    """Accepts an X-Ray image upload and returns predictions from all models."""
    
    if not models:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file.")
        
    results = {}
    
    # Run predictions across available models
    for name, model in models.items():
        size = MODEL_SIZES[name]
        try:
            arr = preprocess_image_bytes(image_bytes, size)
            
            t0 = time.time()
            pred = model.predict(arr, verbose=0)
            elapsed_ms = (time.time() - t0) * 1000
            
            idx = int(np.argmax(pred[0]))
            conf = float(pred[0][idx]) * 100
            
            results[name] = {
                "label": CLASSES[idx],
                "confidence_percent": round(conf, 2),
                "raw_scores": [round(float(p)*100, 2) for p in pred[0]],
                "time_ms": round(elapsed_ms, 2)
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    # Determine final verdict using majority vote
    valid_preds = [data["label"] for name, data in results.items() if "label" in data]
    if valid_preds:
        final_verdict = max(set(valid_preds), key=valid_preds.count)
        avg_confidence = sum([data["confidence_percent"] for data in results.values() if "confidence_percent" in data]) / len(valid_preds)
    else:
        final_verdict = "ERROR"
        avg_confidence = 0.0

    return {
        "filename": file.filename,
        "final_verdict": final_verdict,
        "average_confidence_percent": round(avg_confidence, 2),
        "model_results": results
    }

if __name__ == "__main__":
    import uvicorn
    # Local development server execution
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
