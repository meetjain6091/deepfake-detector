import cv2
import numpy as np
from PIL import Image
import io
import base64
from mtcnn import MTCNN

# Initialize MTCNN globally once to prevent memory leaks from repeated instantiations
detector = None

def get_mtcnn_detector():
    global detector
    if detector is None:
        detector = MTCNN()
    return detector

def process_image_pipeline(image_bytes: bytes):
    # 1. Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_np = np.array(image)
    h, w, _ = original_np.shape

    # Convert to BGR for OpenCV processing
    img_cv2 = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

    # 2. Detect face using cached MTCNN detector
    mtcnn = get_mtcnn_detector()
    faces = mtcnn.detect_faces(img_cv2)

    if not faces:
        raise ValueError("No face detected in the uploaded image. Please upload a clear facial photo.")

    # Select the face with the highest confidence
    best_face = max(faces, key=lambda x: x['confidence'])
    x, y, w_box, h_box = best_face['box']

    # Ensure bounding box coordinates are within image boundaries
    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x + w_box)
    y2 = min(h, y + h_box)

    # Extract keypoints (eyes, nose, mouth)
    keypoints = best_face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # 3. Affine Alignment based on eye coordinates
    d_x = right_eye[0] - left_eye[0]
    d_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(d_y, d_x))

    center = ((x + x2) // 2, (y + y2) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_cv2 = cv2.warpAffine(img_cv2, M, (w, h), flags=cv2.INTER_CUBIC)

    # Crop the aligned face region with padding
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
    # Stage 1: Original
    _, orig_encoded = cv2.imencode('.jpg', img_cv2)
    orig_b64 = base64.b64encode(orig_encoded).decode('utf-8')

    # Stage 2: Bounding Box & Landmarks
    bbox_img = img_cv2.copy()
    cv2.rectangle(bbox_img, (x, y), (x2, y2), (0, 255, 0), 2)
    for key, point in keypoints.items():
        cv2.circle(bbox_img, point, 4, (0, 0, 255), -1)
    _, bbox_encoded = cv2.imencode('.jpg', bbox_img)
    bbox_b64 = base64.b64encode(bbox_encoded).decode('utf-8')

    # Stage 3: Aligned Crop
    _, aligned_encoded = cv2.imencode('.jpg', face_crop)
    aligned_b64 = base64.b64encode(aligned_encoded).decode('utf-8')

    # Stage 4: Resized 224x224 Tensor view
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