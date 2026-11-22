"""Une régression logistique pondérée par pathologie, sur les caractéristiques DenseNet, puis calibration de Platt.

La pondération des classes (équivalent d'une perte pondérée) compense la rareté des cas positifs,
mais déforme les probabilités : la calibration de Platt, apprise sur la validation, les corrige.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import PATHOLOGIES

# Grille élargie vers une régularisation plus forte : la meilleure AUC de validation (0,715 contre 0,703)
# est obtenue avec C ≤ 0,001 pour la plupart des pathologies (≈ 1 000 images pour 1 024 caractéristiques).
GRILLE_C = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)


class MultiLabelClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models, self.platt, self.C = {}, {}, {}

    def fit(self, X_tr, Y_tr, X_va, Y_va):
        """Pour chaque pathologie : choix de la régularisation C sur la validation (AUC), puis calibration."""
        Xs_tr = self.scaler.fit_transform(X_tr)
        Xs_va = self.scaler.transform(X_va)
        for k, p in enumerate(PATHOLOGIES):
            y_tr, y_va = Y_tr[:, k], Y_va[:, k]
            best = None
            for C in GRILLE_C:
                m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xs_tr, y_tr)
                auc = roc_auc_score(y_va, m.decision_function(Xs_va)) if 0 < y_va.sum() < len(y_va) else 0.5
                if best is None or auc > best[0]:
                    best = (auc, C, m)
            _, self.C[p], self.models[p] = best
            score_va = self.models[p].decision_function(Xs_va).reshape(-1, 1)
            self.platt[p] = LogisticRegression(C=1e6, max_iter=1000).fit(score_va, y_va) if 0 < y_va.sum() < len(y_va) else None
        return self

    def scores(self, X) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.column_stack([self.models[p].decision_function(Xs) for p in PATHOLOGIES])

    def proba_brute(self, X) -> np.ndarray:
        """Probabilités avant calibration (biaisées vers le haut par la pondération des classes)."""
        Xs = self.scaler.transform(X)
        return np.column_stack([self.models[p].predict_proba(Xs)[:, 1] for p in PATHOLOGIES])

    def proba(self, X) -> np.ndarray:
        """Probabilités calibrées (Platt)."""
        s = self.scores(X)
        cols = []
        for k, p in enumerate(PATHOLOGIES):
            cal = self.platt[p]
            cols.append(cal.predict_proba(s[:, [k]])[:, 1] if cal is not None else 1 / (1 + np.exp(-s[:, k])))
        return np.column_stack(cols)

    def dense_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Poids équivalents d'une couche Dense sur les caractéristiques brutes (pour Grad-CAM)."""
        mean, std = self.scaler.mean_, self.scaler.scale_
        W = np.column_stack([self.models[p].coef_[0] / std for p in PATHOLOGIES])
        b = np.array([self.models[p].intercept_[0] - np.sum(self.models[p].coef_[0] * mean / std) for p in PATHOLOGIES])
        return W.astype(np.float32), b.astype(np.float32)
