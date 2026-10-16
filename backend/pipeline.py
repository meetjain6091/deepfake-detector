import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path

# Initialize OpenCV Haar Cascade safely using stable cv2 data path
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_image_pipeline(image_bytes: bytes):
    # 1. Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_np = np.array(image)
    h, w, _ = original_np.shape

    # Convert to grayscale and BGR for OpenCV processing
    img_cv2 = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # 2. Detect face using lightweight Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        raise ValueError("No face detected in the uploaded image. Please upload a clear facial photo.")

    # Select the largest face detected and cast to standard Python ints
    x, y, w_box, h_box = max(faces, key=lambda f: f[2] * f[3])
    x, y, w_box, h_box = int(x), int(y), int(w_box), int(h_box)
    x2 = min(w, x + w_box)
    y2 = min(h, y + h_box)

    # Approximate eye locations for alignment visualization compatibility
    left_eye = (int(x + w_box * 0.3), int(y + h_box * 0.4))
    right_eye = (int(x + w_box * 0.7), int(y + h_box * 0.4))
    keypoints = {'left_eye': left_eye, 'right_eye': right_eye}

    # 3. Affine Alignment approximation with explicit Python types
    d_x = right_eye[0] - left_eye[0]
    d_y = right_eye[1] - left_eye[1]
    angle = float(np.degrees(np.arctan2(d_y, d_x)))

    center = (int((x + x2) / 2), int((y + y2) / 2))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_cv2 = cv2.warpAffine(img_cv2, M, (w, h), flags=cv2.INTER_CUBIC)

    # Crop the face region with padding
    pad = int(0.2 * max(w_box, h_box))
    crop_x1 = max(0, x - pad)
    crop_y1 = max(0, y - pad)
    crop_x2 = min(w, x2 + pad)
    crop_y2 = min(h, y2 + pad)

    face_crop = aligned_cv2[crop_y1:crop_y2, crop_x1:crop_x2]
    if face_crop.size == 0:
        face_crop = img_cv2[y:y2, x:x2]

    # 4. Resize to 224x224 for EfficientNetB4
    resized_face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

    # Prepare model input tensor (normalized to [0, 1])
    model_input = np.expand_dims(resized_rgb.astype(np.float32) / 255.0, axis=0)

    # Generate visualizations for the 2x2 UI matrix
    _, orig_encoded = cv2.imencode('.jpg', img_cv2)
    orig_b64 = base64.b64encode(orig_encoded).decode('utf-8')

    bbox_img = img_cv2.copy()
    cv2.rectangle(bbox_img, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.circle(bbox_img, left_eye, 4, (0, 0, 255), -1)
    cv2.circle(bbox_img, right_eye, 4, (0, 0, 255), -1)
    _, bbox_encoded = cv2.imencode('.jpg', bbox_img)
    bbox_b64 = base64.b64encode(bbox_encoded).decode('utf-8')

    _, aligned_encoded = cv2.imencode('.jpg', face_crop)
    aligned_b64 = base64.b64encode(aligned_encoded).decode('utf-8')

    _, tensor_encoded = cv2.imencode('.jpg', resized_face)
    tensor_b64 = base64.b64encode(tensor_encoded).decode('utf-8')

    metadata = {
        "num_faces": len(faces),
        "bbox": [int(x), int(y), int(w_box), int(h_box)],
        "angle": round(float(angle), 2)
    }

    images_b64 = {
        "original": f"data:image/jpeg;base64,{orig_b64}",
        "bounding_box": f"data:image/jpeg;base64,{bbox_b64}",
        "aligned_crop": f"data:image/jpeg;base64,{aligned_b64}",
        "resized_224": f"data:image/jpeg;base64,{tensor_b64}"
    }

    return model_input, metadata, images_b64