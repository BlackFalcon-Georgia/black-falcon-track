"""
BLACK FALCON - Visual Tracking Backend
========================================
FastAPI service that receives an image frame, runs object detection,
and returns bounding boxes for all detected objects. The client
(web or mobile) handles selection + tracking logic on top of this.

Phase 1: General object detection (pretrained COCO model via YOLOv8n).
  NOTE: The pretrained model detects general classes (person, car, truck,
  airplane, bird, etc.) — not a drone-specific class, since public
  pretrained models don't ship with a "drone" category. This endpoint
  is the foundation; Phase 2 will add a fine-tuned drone detector.

Endpoints:
  GET  /health           -> simple health check
  POST /detect           -> upload a JPEG frame, get back detections
"""

import io
import time
from typing import List

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


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        # yolov8n.pt = "nano" version: small, fast, good for a free-tier
        # server with no GPU. Ultralytics will auto-download the weights
        # the first time this runs.
        _model = YOLO("yolov8n.pt")
    return _model


class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class DetectResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float
    image_width: int
    image_height: int


@app.get("/health")
def health():
    return {"status": "ok", "service": "black-falcon-detection"}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = get_model()

    start = time.time()
    results = model.predict(image, verbose=False)[0]
    elapsed_ms = (time.time() - start) * 1000

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        # Filter out low-confidence noise
        if conf < 0.40:
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
            )
        )

    return DetectResponse(
        detections=detections,
        inference_ms=elapsed_ms,
        image_width=image.width,
        image_height=image.height,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
