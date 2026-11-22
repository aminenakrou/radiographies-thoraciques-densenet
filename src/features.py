"""Extraction de caractéristiques avec DenseNet121 pré-entraîné sur ImageNet (transfert d'apprentissage)."""

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = 224
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def load_image(path: str) -> np.ndarray:
    """Radiographie en niveaux de gris -> 224 × 224 × 3, normalisée comme pour l'entraînement ImageNet de DenseNet."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.keras.applications.densenet.preprocess_input(img).numpy()


def backbone(pooling: str | None = "avg") -> tf.keras.Model:
    return tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, pooling=pooling,
                                             input_shape=(IMG_SIZE, IMG_SIZE, 3))


def embeddings(paths: list[str], batch: int = 32) -> np.ndarray:
    """Vecteurs de 1 024 caractéristiques par image, mis en cache sur le disque."""
    key = hashlib.md5("|".join(paths).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"emb_{key}.npy"
    if cache.exists():
        return np.load(cache)
    model = backbone()
    out = []
    for i in range(0, len(paths), batch):
        x = np.stack([load_image(p) for p in paths[i:i + batch]])
        out.append(model.predict(x, verbose=0))
    emb = np.concatenate(out).astype(np.float32)
    np.save(cache, emb)
    return emb
