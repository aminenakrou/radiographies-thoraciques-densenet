"""Évaluation d'un test diagnostique : AUC avec IC bootstrap, seuil, sensibilité, spécificité, VPP, VPN, calibration."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .data import NOMS_FR, PATHOLOGIES


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 2026) -> tuple[float, float, float]:
    """AUC et intervalle de confiance à 95 % par bootstrap (rééchantillonnage des patients du test)."""
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(y, s)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            draws.append(roc_auc_score(y[i], s[i]))
    return auc, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Seuil qui maximise sensibilité + spécificité − 1 (indice de Youden), choisi sur la validation."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def confusion_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {"VP": tp, "VN": tn, "FP": fp, "FN": fn,
            "sensibilite": div(tp, tp + fn), "specificite": div(tn, tn + fp),
            "VPP": div(tp, tp + fp), "VPN": div(tn, tn + fn)}


def report(Y_te: np.ndarray, P_te: np.ndarray, Y_va: np.ndarray, P_va: np.ndarray, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for k, p in enumerate(PATHOLOGIES):
        y, s = Y_te[:, k], P_te[:, k]
        if not 0 < y.sum() < len(y):
            continue
        auc, lo, hi = bootstrap_auc(y, s, n_boot)
        thr = youden_threshold(Y_va[:, k], P_va[:, k]) if 0 < Y_va[:, k].sum() < len(Y_va) else 0.5
        rows.append({"pathologie": NOMS_FR[p], "cas_positifs_test": int(y.sum()), "prevalence": float(y.mean()),
                     "AUC": auc, "IC95_bas": lo, "IC95_haut": hi, "seuil": thr,
                     **confusion_metrics(y, (s >= thr).astype(int))})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Fréquence observée par tranche de probabilité prédite (diagramme de fiabilité), tous labels confondus."""
    df = pd.DataFrame({"y": y.ravel(), "p": p.ravel()})
    df["tranche"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("tranche", observed=True)
    return pd.DataFrame({"proba_moyenne": g["p"].mean(), "frequence_observee": g["y"].mean(), "n": g["y"].size()})


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))
