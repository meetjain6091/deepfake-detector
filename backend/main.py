import json
from pathlib import Path

from database import DetectionLog, SessionLocal
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model_loader import build_and_load_model
from pipeline import process_image_pipeline
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

model = None

app = FastAPI(
    title="Dual-Paradigm Deepfake Detector API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/predict")
@app.post("/api/detect")
async def detect_deepfake(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    global model
    # Lazy-load the model on the first request to prevent startup OOM crashes
    if model is None:
        print("Lazy-loading EfficientNetB4 Dual-Paradigm Model on first request...")
        model = build_and_load_model(MODELS_DIR)
        print("Model loaded successfully.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a valid image."
        )

    try:
        contents = await file.read()
        model_input, metadata, images_b64 = process_image_pipeline(contents)

        prediction = model.predict(model_input, verbose=0)
        fake_prob = float(prediction[0][0])
        real_prob = float(1.0 - fake_prob)

        threshold = 0.50
        label = "FAKE" if fake_prob >= threshold else "REAL"
        confidence = fake_prob if label == "FAKE" else real_prob

        # Database Logging
        db_log = DetectionLog(
            filename=file.filename,
            prediction_label=label,
            confidence=confidence,
            fake_probability=fake_prob,
            real_probability=real_prob,
            num_faces_detected=metadata["num_faces"],
            bbox_coordinates=json.dumps(metadata["bbox"]),
            rotation_angle=metadata["angle"],
            original_image_b64=images_b64["original"],
            bbox_image_b64=images_b64["bounding_box"],
            aligned_face_b64=images_b64["aligned_crop"],
            resized_crop_b64=images_b64["resized_224"],
        )
        db.add(db_log)
        db.commit()
        db.refresh(db_log)

        return {
            "status": "success",
            "log_id": db_log.id,
            "filename": file.filename,
            "results": {
                "label": label,
                "confidence_percentage": round(confidence * 100, 2),
                "probabilities": {
                    "fake": round(fake_prob, 4),
                    "real": round(real_prob, 4),
                },
            },
            "intermediate_stages": {
                "metadata": metadata,
                "visualizations": images_b64,
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")


@app.get("/api/history")
def get_detection_history(limit: int = 10, db: Session = Depends(get_db)):
    logs = (
        db.query(DetectionLog)
        .order_by(DetectionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "filename": log.filename,
            "label": log.prediction_label,
            "confidence": round(log.confidence * 100, 2),
            "timestamp": log.timestamp.isoformat(),
        }
        for log in logs
    ]