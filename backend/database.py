from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./deepfake_detection.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Classification Output
    prediction_label = Column(String, nullable=False)  # FAKE / REAL
    confidence = Column(Float, nullable=False)
    fake_probability = Column(Float, nullable=False)
    real_probability = Column(Float, nullable=False)

    # Pipeline Statistics
    num_faces_detected = Column(Integer, nullable=False)
    bbox_coordinates = Column(String, nullable=True)  # JSON String e.g. "[x, y, w, h]"
    rotation_angle = Column(Float, nullable=True)

    # Base64 Intermediate Images
    original_image_b64 = Column(Text, nullable=False)
    bbox_image_b64 = Column(Text, nullable=True)
    aligned_face_b64 = Column(Text, nullable=True)
    resized_crop_b64 = Column(Text, nullable=True)


Base.metadata.create_all(bind=engine)