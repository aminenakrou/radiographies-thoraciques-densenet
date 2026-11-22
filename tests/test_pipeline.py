"""Tests sans images ni réseau de neurones : étiquettes, découpage par patient, métriques, classifieur."""

import numpy as np
import pandas as pd
import pytest

from src.data import PATHOLOGIES, check_no_overlap, from_nih_data_entry, patient_split
from src.evaluation import bootstrap_auc, brier, confusion_metrics, youden_threshold
from src.model import MultiLabelClassifier


def test_conversion_fichier_officiel_nih(tmp_path):
    p = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame({
        "Image Index": ["a.png", "b.png", "c.png"],
        "Finding Labels": ["Effusion|Mass", "No Finding", "Hernia"],
        "Patient ID": [1, 1, 2],
    }).to_csv(p, index=False)
    df = from_nih_data_entry(str(p))
    assert df.loc[0, "Effusion"] == 1 and df.loc[0, "Mass"] == 1 and df.loc[0, "Hernia"] == 0
    assert df.loc[1, PATHOLOGIES].sum() == 0
    assert df.loc[2, "Hernia"] == 1


def test_decoupage_sans_patient_commun():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Image": [f"{i}.png" for i in range(600)], "PatientId": rng.integers(0, 200, 600)})
    for p in PATHOLOGIES:
        df[p] = rng.integers(0, 2, 600)
    tr, va, te = patient_split(df)
    assert len(tr) + len(va) + len(te) == 600
    assert check_no_overlap(tr, va, te)


def test_metriques_confusion():
    y = np.array([1, 1, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0])
    m = confusion_metrics(y, pred)
    assert (m["VP"], m["FN"], m["FP"], m["VN"]) == (1, 1, 1, 2)
    assert m["sensibilite"] == pytest.approx(0.5)
    assert m["specificite"] == pytest.approx(2 / 3)
    assert m["VPP"] == pytest.approx(0.5)
    assert m["VPN"] == pytest.approx(2 / 3)


def test_auc_bootstrap_et_seuil():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(0, 0.8, 400)
    auc, lo, hi = bootstrap_auc(y, s, n_boot=200)
    assert lo < auc < hi and auc > 0.7
    t = youden_threshold(y, s)
    assert s.min() <= t <= s.max() or np.isinf(t)


def test_classifieur_et_calibration():
    rng = np.random.default_rng(2)
    n, d = 400, 20
    X = rng.normal(size=(n, d))
    Y = (X[:, : len(PATHOLOGIES)] + rng.normal(0, 1, (n, len(PATHOLOGIES))) > 1.2).astype(int)
    clf = MultiLabelClassifier().fit(X[:300], Y[:300], X[300:], Y[300:])
    P = clf.proba(X[300:])
    assert P.shape == (100, len(PATHOLOGIES))
    assert ((P >= 0) & (P <= 1)).all()
    # la pondération des classes surestime les probabilités ; la calibration doit réduire l'erreur de Brier
    assert brier(Y[300:], P) < brier(Y[300:], clf.proba_brute(X[300:]))
    W, b = clf.dense_weights()
    logits = X[300:] @ W + b
    np.testing.assert_allclose(logits, clf.scores(X[300:]), rtol=1e-4, atol=1e-4)
