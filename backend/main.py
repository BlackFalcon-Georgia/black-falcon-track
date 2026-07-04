"""
BLACK FALCON - Visual Tracking Backend
========================================
FastAPI service that receives an image frame, runs object detection,
and returns bounding boxes for all detected objects. The client
(web or mobile) handles selection + tracking logic on top of this.

Two detection modes:
  /detect        -> general-purpose (COCO classes: person, car, etc.)
                     used for the police/car-tracking use case.
  /detect_drone  -> our own fine-tuned model, trained specifically to
                     recognize "drone" as its own class.

Endpoints:
  GET  /health           -> simple health check
  POST /detect           -> upload a JPEG frame, get back general detections
  POST /detect_drone     -> upload a JPEG frame, get back drone detections
"""

import io
import time
from typing import List

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Black Falcon Detection API")

# Allow the web demo (and later the mobile app) to call this API
# from any origin during development. Tighten this before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
# We lazy-load the model on first request so the server boots quickly
# on platforms like Render (which have startup time limits).
_model = None
_drone_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        # yolov8n.pt = "nano" version: small, fast, good for a free-tier
        # server with no GPU. Ultralytics will auto-download the weights
        # the first time this runs.
        _model = YOLO("yolov8n.pt")
    return _model


def get_drone_model():
    global _drone_model
    if _drone_model is None:
        from ultralytics import YOLO
        # Our own fine-tuned model — trained specifically to recognize
        # "drone" as its own class (see /models/drone_best.pt).
        import os
        weights_path = os.path.join(os.path.dirname(__file__), "models", "drone_best.pt")
        _drone_model = YOLO(weights_path)
    return _drone_model


class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    # A lightweight "visual signature" for this object (color histogram),
    # used client-side for re-identification when the object briefly leaves
    # and re-enters the frame. Deliberately NOT a deep-learning embedding —
    # this keeps memory usage on the free-tier server near zero.
    embedding: List[float] = []


def compute_embedding(image: Image.Image, x1: float, y1: float, x2: float, y2: float) -> List[float]:
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(image.width, int(x2)), min(image.height, int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return [0.0] * 24
    crop = image.crop((ix1, iy1, ix2, iy2)).resize((32, 32)).convert("RGB")
    arr = np.asarray(crop).astype(np.float32) / 255.0
    features: List[float] = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        features.extend(hist.tolist())
    return features


class DetectResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float
    image_width: int
    image_height: int


def run_detection(model, image: Image.Image, min_conf: float = 0.40) -> tuple[List[Detection], float]:
    start = time.time()
    results = model.predict(image, verbose=False)[0]
    elapsed_ms = (time.time() - start) * 1000

    detections: List[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        # Filter out low-confidence noise
        if conf < min_conf:
            continue

        # Filter out implausibly huge boxes (YOLO sometimes emits a
        # near-full-frame low-quality box) — real tracked objects rarely
        # fill more than ~85% of the frame in this use case.
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        frame_area = image.width * image.height
        if frame_area > 0 and (box_area / frame_area) > 0.85:
            continue

        detections.append(
            Detection(
                class_name=class_name,
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                embedding=compute_embedding(image, x1, y1, x2, y2),
            )
        )
    return detections, elapsed_ms


@app.get("/health")
def health():
    return {"status": "ok", "service": "black-falcon-detection"}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    """General-purpose detection (COCO classes: person, car, etc.) —
    used for the police/car-tracking use case."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = get_model()
    detections, elapsed_ms = run_detection(model, image)

    return DetectResponse(
        detections=detections,
        inference_ms=elapsed_ms,
        image_width=image.width,
        image_height=image.height,
    )


@app.post("/detect_drone", response_model=DetectResponse)
async def detect_drone(file: UploadFile = File(...)):
    """Drone-specific detection, using our own fine-tuned model
    (trained on ~1000 drone images — see /models/drone_best.pt).

    NOTE: on the free-tier server, running this alongside heavy use of
    /detect may increase memory pressure since two models can end up
    loaded at once. If you see instability, avoid hitting both endpoints
    in the same session until the instance is upgraded.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = get_drone_model()
    detections, elapsed_ms = run_detection(model, image, min_conf=0.35)

    return DetectResponse(
        detections=detections,
        inference_ms=elapsed_ms,
        image_width=image.width,
        image_height=image.height,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
