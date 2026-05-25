from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from contextlib import asynccontextmanager
import io
from PIL import Image
import numpy as np
import time
from src.inference.wrapper import InferenceEngine

# Global variable for the engine
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


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": inference_engine is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Lean prediction endpoint. Delegates preprocessing and inference
    to the specialized InferenceEngine.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. Read raw image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_img = np.array(image)

        # 2. Run Engine Inference
        start_time = time.time()
        result = inference_engine.predict(raw_img)
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
