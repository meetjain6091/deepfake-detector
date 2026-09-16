import base64
import io
import cv2
from mtcnn import MTCNN
import numpy as np
from PIL import Image

detector = None


def get_detector():
    global detector
    if detector is None:
        detector = MTCNN()
    return detector


def convert_np_to_b64(img_np_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def process_image_pipeline(image_bytes: bytes):
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_rgb = np.array(pil_img.convert("RGB"))
    except Exception as e:
        raise ValueError(f"Unable to decode input image: {e}")

    mtcnn_detector = get_detector()
    results = mtcnn_detector.detect_faces(img_rgb)

    if len(results) == 0:
        raise ValueError("No face detected in the uploaded image.")

    result = results[0]
    x, y, w, h = result["box"]
    x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

    # 1. Bounding Box & Eye Visualizations
    vis_bbox_img = img_rgb.copy()
    cv2.rectangle(vis_bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    keypoints = result["keypoints"]
    left_eye = np.array(keypoints["left_eye"])
    right_eye = np.array(keypoints["right_eye"])

    cv2.circle(vis_bbox_img, tuple(left_eye), 4, (255, 0, 0), -1)
    cv2.circle(vis_bbox_img, tuple(right_eye), 4, (0, 0, 255), -1)
    cv2.line(vis_bbox_img, tuple(left_eye), tuple(right_eye), (255, 255, 0), 2)

    # 2. Eye Alignment & Rotation Calculation
    delta_y = float(right_eye[1] - left_eye[1])
    delta_x = float(right_eye[0] - left_eye[0])
    angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2),
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(
        img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0])
    )

    # 3. Crop & Resizing
    x2 = min(x + w, aligned_face.shape[1])
    y2 = min(y + h, aligned_face.shape[0])
    aligned_face_crop = aligned_face[y:y2, x:x2]

    if aligned_face_crop.size == 0:
        raise ValueError("Face crop failed after alignment.")

    resized_face = cv2.resize(aligned_face_crop, (224, 224))

    # Matching inference notebook setup: float32 scaled tensor (0-255)
    model_input = np.expand_dims(resized_face.astype(np.float32), axis=0)

    # Base64 Image Conversion
    b64_original = convert_np_to_b64(img_rgb)
    b64_bbox = convert_np_to_b64(vis_bbox_img)
    b64_aligned = convert_np_to_b64(aligned_face_crop)
    b64_resized = convert_np_to_b64(resized_face)

    metadata = {
        "num_faces": len(results),
        "bbox": [int(x), int(y), int(w), int(h)],
        "left_eye": left_eye.tolist(),
        "right_eye": right_eye.tolist(),
        "angle": round(angle, 3),
    }

    images_b64 = {
        "original": f"data:image/png;base64,{b64_original}",
        "bounding_box": f"data:image/png;base64,{b64_bbox}",
        "aligned_crop": f"data:image/png;base64,{b64_aligned}",
        "resized_224": f"data:image/png;base64,{b64_resized}",
    }

    return model_input, metadata, images_b64