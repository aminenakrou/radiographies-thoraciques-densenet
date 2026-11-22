"""Grad-CAM : quelles zones de la radiographie ont poussé le modèle vers une pathologie donnée."""

import numpy as np
import tensorflow as tf

from .features import backbone, load_image


def build_full_model(W: np.ndarray, b: np.ndarray) -> tf.keras.Model:
    """DenseNet121 (sans tête) + moyenne globale + couche Dense reprenant les régressions logistiques."""
    base = backbone(pooling=None)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    dense = tf.keras.layers.Dense(W.shape[1], name="logits")
    out = dense(x)
    model = tf.keras.Model(base.input, [base.output, out])
    dense.set_weights([W, b])
    return model


def heatmap(model: tf.keras.Model, path: str, class_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Carte Grad-CAM (224 × 224, entre 0 et 1) et image d'origine redimensionnée."""
    x = load_image(path)[None, ...]
    with tf.GradientTape() as tape:
        conv, logits = model(x, training=False)
        target = logits[:, class_index]
    grads = tape.gradient(target, conv)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = tf.image.resize(cam[..., None], (224, 224)).numpy()[..., 0]
    cam = cam / cam.max() if cam.max() > 0 else cam
    raw = tf.image.resize(tf.io.decode_png(tf.io.read_file(path), channels=1), (224, 224)).numpy()[..., 0]
    return cam, raw
