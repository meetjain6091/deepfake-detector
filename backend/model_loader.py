import os
import gc
from pathlib import Path
import numpy as np
import tensorflow as tf

# Force single-threading to reduce memory overhead on Render
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Enable global mixed precision so weights and activations use half the memory
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB4

INPUT_SHAPE = (224, 224, 3)
QUANTUM_DIM = 64


class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_conv = layers.Conv2D(
            filters=1, kernel_size=1, padding="same", activation="sigmoid", dtype='float16'
        )

    def call(self, inputs):
        attention_map = self.attention_conv(inputs)
        attention_map = tf.cast(attention_map, inputs.dtype)
        return inputs * attention_map


class QuantumInspiredTransform(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        initializer = np.random.normal(size=(self.dim, self.dim)).astype(np.float32)
        q, _ = np.linalg.qr(initializer)
        q = q.astype(np.float16)  # Use float16 for weights to save RAM
        self.unitary = self.add_weight(
            name="unitary_transform",
            shape=(self.dim, self.dim),
            initializer=tf.keras.initializers.Constant(q),
            trainable=True,
            dtype='float16'
        )
        super().build(input_shape)

    def call(self, inputs):
        unitary = tf.cast(self.unitary, inputs.dtype)
        return tf.matmul(inputs, unitary, transpose_b=True)

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


def build_and_load_model(models_dir: Path) -> Model:
    # Clear any lingering TF sessions to free up RAM
    tf.keras.backend.clear_session()
    gc.collect()

    efficientnet_backbone = EfficientNetB4(
        weights=None, include_top=False, input_shape=INPUT_SHAPE, dtype='float16'
    )
    efficientnet_backbone.trainable = False

    input_layer = layers.Input(shape=INPUT_SHAPE, name="face_input", dtype='float16')
    features = efficientnet_backbone(input_layer)
    refined_features = SpatialAttention(name="spatial_attention")(features)
    pooled_features = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(refined_features)
    embedding = layers.Dense(500, activation="relu", name="embedding_dense", dtype='float16')(
        pooled_features
    )
    embedding = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1), name="embedding_normalization"
    )(embedding)

    siamese_shared_encoder = Model(
        inputs=input_layer, outputs=embedding, name="Siamese_Shared_Encoder"
    )

    quantum_input = layers.Input(shape=(500,), name="classical_embedding", dtype='float16')
    x = layers.Dense(
        QUANTUM_DIM, activation=None, name="quantum_projection", dtype='float16'
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

    final_input = layers.Input(shape=INPUT_SHAPE, name="final_face_input", dtype='float16')
    classical_features = siamese_shared_encoder(final_input)
    quantum_features = quantum_encoder(classical_features)

    fused_features = layers.Concatenate(name="feature_fusion")(
        [classical_features, quantum_features]
    )

    fusion_dense = layers.Dense(256, activation="relu", name="fusion_dense_256", dtype='float16')(
        fused_features
    )
    fusion_dense = layers.Dense(64, activation="relu", name="fusion_dense_64", dtype='float16')(
        fusion_dense
    )
    # Output must be float32 for numerical stability with sigmoid
    output = layers.Dense(1, activation="sigmoid", name="deepfake_output", dtype='float32')(
        fusion_dense
    )

    final_model = Model(
        inputs=final_input, outputs=output, name="Dual_Paradigm_Deepfake_Detector"
    )
    final_model.build((None, 224, 224, 3))

    weights_path = models_dir / "dual_paradigm_deepfake_model.weights.h5"
    if not weights_path.exists():
        weights_path = models_dir.parent / "dual_paradigm_deepfake_model.weights.h5"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file missing at {weights_path}")

    print(f"Loading memory-optimized EfficientNetB4 weights from: {weights_path}")
    final_model.load_weights(weights_path)

    # Aggressive memory cleanup after loading weights
    gc.collect()

    return final_model