"""Données : radiographies thoraciques NIH ChestX-ray8 (images du domaine public, NIH Clinical Center).

Les images et les étiquettes ne sont pas versionnées. Deux façons de fournir les étiquettes :
- un CSV au format du projet : colonne Image, colonne PatientId et une colonne 0/1 par pathologie ;
- le fichier officiel du NIH, Data_Entry_2017.csv, converti par `from_nih_data_entry`.

Le chemin des images et des étiquettes se règle par variables d'environnement :
    CXR_IMAGES  (dossier des fichiers .png)
    CXR_LABELS  (un ou plusieurs CSV séparés par « ; »)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis",
    "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
NOMS_FR = {
    "Atelectasis": "Atélectasie", "Cardiomegaly": "Cardiomégalie", "Consolidation": "Condensation",
    "Edema": "Œdème", "Effusion": "Épanchement pleural", "Emphysema": "Emphysème", "Fibrosis": "Fibrose",
    "Hernia": "Hernie", "Infiltration": "Infiltration", "Mass": "Masse", "Nodule": "Nodule",
    "Pleural_Thickening": "Épaississement pleural", "Pneumonia": "Pneumonie", "Pneumothorax": "Pneumothorax",
}
SEED = 2026


def from_nih_data_entry(path: str) -> pd.DataFrame:
    """Convertit Data_Entry_2017.csv (NIH) : « Finding Labels » = pathologies séparées par « | »."""
    raw = pd.read_csv(path)
    out = pd.DataFrame({"Image": raw["Image Index"], "PatientId": raw["Patient ID"]})
    labels = raw["Finding Labels"].fillna("")
    for p in PATHOLOGIES:
        out[p] = labels.str.split("|").apply(lambda xs, p=p: int(p in xs))
    return out


def load_labels(label_paths: list[str], image_dir: str) -> pd.DataFrame:
    """Réunit les fichiers d'étiquettes, retire les doublons et garde les images présentes sur le disque."""
    frames = []
    for p in label_paths:
        df = pd.read_csv(p)
        if "Finding Labels" in df.columns:
            df = from_nih_data_entry(p)
        frames.append(df[["Image", "PatientId"] + PATHOLOGIES])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("Image").reset_index(drop=True)
    present = set(os.listdir(image_dir))
    df = df[df["Image"].isin(present)].reset_index(drop=True)
    df[PATHOLOGIES] = df[PATHOLOGIES].astype(int)
    return df


def config() -> tuple[list[str], str]:
    images = os.environ.get("CXR_IMAGES")
    labels = os.environ.get("CXR_LABELS")
    if not images or not labels:
        raise SystemExit("Renseignez CXR_IMAGES (dossier des images) et CXR_LABELS (CSV d'étiquettes), voir README.")
    return labels.split(";"), images


def patient_split(df: pd.DataFrame, test: float = 0.2, valid: float = 0.15, seed: int = SEED):
    """Découpage par patient : aucun patient n'apparaît dans deux ensembles (pas de fuite)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    trval_idx, test_idx = next(gss.split(df, groups=df["PatientId"]))
    trval = df.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=valid / (1 - test), random_state=seed)
    tr_idx, va_idx = next(gss2.split(trval, groups=trval["PatientId"]))
    return (trval.iloc[tr_idx].reset_index(drop=True), trval.iloc[va_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


def check_no_overlap(*parts: pd.DataFrame) -> bool:
    sets = [set(p["PatientId"]) for p in parts]
    return all(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def image_paths(df: pd.DataFrame, image_dir: str) -> list[str]:
    return [str(Path(image_dir) / f) for f in df["Image"]]


def prevalence(df: pd.DataFrame) -> pd.Series:
    return df[PATHOLOGIES].mean()


def positives(df: pd.DataFrame) -> pd.Series:
    return df[PATHOLOGIES].sum()


__all__ = ["PATHOLOGIES", "NOMS_FR", "load_labels", "from_nih_data_entry", "patient_split", "check_no_overlap",
           "image_paths", "config", "prevalence", "positives", "np"]
