from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import io
from PIL import Image
import numpy as np
import time
from src.inference.wrapper import InferenceEngine

# Global variable for the engine to be loaded on startup
inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine
    print("[INFO] Initializing system lifespan...")
    try:
        inference_engine = InferenceEngine()
        print(
            f"[INFO] Engine loaded successfully. Model: {inference_engine.config.infrastructure.model_save_path}"
        )
    except Exception as e:
        print(f"[ERROR] Engine initialization failed: {str(e)}")
    yield
    print("[INFO] Shutting down service.")


app = FastAPI(
    title="Baseer AI Backend",
    description="High-fidelity inference service for automated object recognition.",
    version="1.0",
    contact={
        "name": "Muhammad Hamza Saqib",
        "url": "https://github.com/ladla20",
    },
    lifespan=lifespan,
)

from fastapi.responses import JSONResponse, RedirectResponse


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Service status endpoint."""
    return {"status": "healthy", "model_loaded": inference_engine is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Asynchronous prediction endpoint.
    Accepts image file, sanitizes, and returns predictions.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. Resize to CIFAR-10 dimensions (32x32)
        image = image.resize((32, 32))

        # 3. Convert to float tensor in range [0, 1] — shape (H, W, C)
        img_array = np.array(image).astype(np.float32) / 255.0

        # 4. Transpose to (C, H, W) — required by PyTorch models
        img_array = img_array.transpose(2, 0, 1)

        # 5. CRITICAL: Apply CIFAR-10 normalization (must match training pipeline)
        # These are the exact constants used during training in dataloader.py
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std

        # 6. Run Secure Inference
        start_time = time.time()
        result = inference_engine.predict(img_array)
        latency = (time.time() - start_time) * 1000  # ms

        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 4),
            "latency_ms": round(latency, 2),
            "engine_status": result["status"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
