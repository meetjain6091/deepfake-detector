import os
import urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB4

INPUT_SHAPE = (224, 224, 3)
QUANTUM_DIM = 64

# Direct download link generated from your Google Drive file ID
DEFAULT_CLOUD_WEIGHTS_URL = os.environ.get(
    "WEIGHTS_URL",
    "https://drive.google.com/uc?export=download&id=1QUp62FZYCpnHkI_WCbHs3QJAMKV7suBD"
)


class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_conv = layers.Conv2D(
            filters=1, kernel_size=1, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        attention_map = self.attention_conv(inputs)
        return inputs * attention_map


class QuantumInspiredTransform(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        initializer = np.random.normal(size=(self.dim, self.dim)).astype(
            np.float32
        )
        q, _ = np.linalg.qr(initializer)
        q = q.astype(np.float32)
        self.unitary = self.add_weight(
            name="unitary_transform",
            shape=(self.dim, self.dim),
            initializer=tf.keras.initializers.Constant(q),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.unitary, transpose_b=True)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


def normalize_quantum_state(x):
    return tf.math.l2_normalize(x, axis=-1)


def measurement_probabilities(x):
    probabilities = tf.square(x)
    return probabilities / (
        tf.reduce_sum(probabilities, axis=-1, keepdims=True) + 1e-8
    )


def download_file_from_google_drive(url, destination):
    """Helper to download files from Google Drive handling large file confirmation tokens."""
    import http.cookiejar

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)

    print(f"Requesting download from cloud URL...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            # Check if Google Drive returned a virus-warning HTML confirmation page instead of the binary file
            if b'download-warning' in content:
                # Extract confirmation token for large files
                import re
                token_match = re.search(r'confirm=([0-9A-Za-z_]+)', content.decode('utf-8', errors='ignore'))
                if token_match:
                    token = token_match.group(1)
                    confirm_url = f"{url}&confirm={token}"
                    req_confirm = urllib.request.Request(confirm_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req_confirm) as confirm_response:
                        with open(destination, 'wb') as f:
                            f.write(confirm_response.read())
                    return
            with open(destination, 'wb') as f:
                f.write(content)
    except Exception as e:
        raise RuntimeError(f"Cloud download failed: {e}")


def build_and_load_model(models_dir: Path) -> Model:
    # 1. EfficientNet Backbone & Shared Encoder Architecture
    efficientnet_backbone = EfficientNetB4(
        weights=None, include_top=False, input_shape=INPUT_SHAPE
    )

    input_layer = layers.Input(shape=INPUT_SHAPE, name="face_input")
    features = efficientnet_backbone(input_layer)
    refined_features = SpatialAttention(name="spatial_attention")(features)
    pooled_features = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(refined_features)
    embedding = layers.Dense(500, activation="relu", name="embedding_dense")(
        pooled_features
    )
    embedding = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1), name="embedding_normalization"
    )(embedding)

    siamese_shared_encoder = Model(
        inputs=input_layer, outputs=embedding, name="Siamese_Shared_Encoder"
    )

    # 2. Quantum-Inspired Encoder Architecture
    quantum_input = layers.Input(shape=(500,), name="classical_embedding")
    x = layers.Dense(
        QUANTUM_DIM, activation=None, name="quantum_projection"
    )(quantum_input)
    x = layers.Lambda(normalize_quantum_state, name="state_normalization")(x)
    x = QuantumInspiredTransform(
        dim=QUANTUM_DIM, name="quantum_inspired_transform"
    )(x)
    x = layers.Lambda(
        measurement_probabilities, name="measurement_probabilities"
    )(x)

    quantum_encoder = Model(
        inputs=quantum_input, outputs=x, name="Quantum_Inspired_Encoder"
    )

    # 3. Concatenation and Dual-Paradigm Classification Head
    final_input = layers.Input(shape=INPUT_SHAPE, name="final_face_input")
    classical_features = siamese_shared_encoder(final_input)
    quantum_features = quantum_encoder(classical_features)

    fused_features = layers.Concatenate(name="feature_fusion")(
        [classical_features, quantum_features]
    )

    fusion_dense = layers.Dense(256, activation="relu", name="fusion_dense_256")(
        fused_features
    )
    fusion_dense = layers.Dense(64, activation="relu", name="fusion_dense_64")(
        fusion_dense
    )
    output = layers.Dense(1, activation="sigmoid", name="deepfake_output")(
        fusion_dense
    )

    final_model = Model(
        inputs=final_input, outputs=output, name="Dual_Paradigm_Deepfake_Detector"
    )
    final_model.build((None, 224, 224, 3))

    # Path Resolution for Weights File
    weights_path = models_dir / "dual_paradigm_deepfake_model.weights.h5"
    if not weights_path.exists():
        weights_path = models_dir.parent / "dual_paradigm_deepfake_model.weights.h5"

    # If weights are missing locally (Render deployment), download them automatically
    if not weights_path.exists():
        print(f"Weights file not found locally at {weights_path}. Downloading from Google Drive...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_file_from_google_drive(DEFAULT_CLOUD_WEIGHTS_URL, weights_path)
            print("Model weights downloaded and saved successfully.")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download weights from cloud: {e}. Expected path: {weights_path}"
            )

    print(f"Loading weights into reconstructed architecture from: {weights_path}")
    final_model.load_weights(weights_path)
    return final_model