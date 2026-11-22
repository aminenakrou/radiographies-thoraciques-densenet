"""Pipeline complet : caractéristiques DenseNet121 -> classifieurs par pathologie -> évaluation -> Grad-CAM.

Utilisation (voir README pour les données) :
    set CXR_IMAGES=chemin\\vers\\images
    set CXR_LABELS=chemin\\vers\\etiquettes.csv
    python main.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.data import NOMS_FR, PATHOLOGIES, check_no_overlap, config, image_paths, load_labels, patient_split  # noqa: E402
from src.evaluation import brier, calibration_table, report  # noqa: E402
from src.features import embeddings  # noqa: E402
from src.gradcam import build_full_model, heatmap  # noqa: E402
from src.model import MultiLabelClassifier  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
BLUE, ORANGE, GREY = "#1f5f8b", "#d9822b", "#9aa5b1"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_auc(r):
    r = r.sort_values("AUC")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(r["AUC"], r["pathologie"], xerr=[r["AUC"] - r["IC95_bas"], r["IC95_haut"] - r["AUC"]],
                fmt="o", color=BLUE, ecolor=GREY, capsize=3, lw=1.5)
    for y, (auc, n) in enumerate(zip(r["AUC"], r["cas_positifs_test"])):
        ax.text(1.005, y, f"{auc:.2f}  (n+ = {n})", va="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=ORANGE, ls="--", lw=1.2, label="Hasard (AUC 0,5)")
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC sur le jeu de test (IC à 95 % par bootstrap)")
    ax.set_title("Performance par pathologie", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "auc_par_pathologie.png")


def fig_roc(Y, P, r):
    top = r.head(4)["pathologie"].tolist()
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, color in zip(top, [BLUE, ORANGE, "#3a8a5c", "#7b5ea7"]):
        k = [NOMS_FR[p] for p in PATHOLOGIES].index(name)
        fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
        auc = float(r.loc[r["pathologie"] == name, "AUC"].iloc[0])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--")
    ax.set_xlabel("1 − spécificité")
    ax.set_ylabel("Sensibilité")
    ax.set_title("Courbes ROC (4 meilleures pathologies)", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "courbes_roc.png")


def fig_calibration(Y, P_raw, P_cal):
    fig, ax = plt.subplots(figsize=(6, 6))
    for P, label, color in [(P_raw, "Avant calibration (classes pondérées)", ORANGE), (P_cal, "Après calibration de Platt", BLUE)]:
        t = calibration_table(Y, P)
        t = t[t["n"] >= 20]  # tranches trop peu peuplées : fréquence observée non fiable
        ax.plot(t["proba_moyenne"], t["frequence_observee"], marker="o", color=color, lw=2, label=f"{label} — Brier {brier(Y, P):.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", label="Calibration parfaite")
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Fréquence observée")
    ax.set_title("Calibration des probabilités (test, 14 pathologies)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "calibration.png")


def fig_gradcam(model, te, paths, P, n=4):
    """Pour quelques vrais positifs bien classés : radiographie et carte Grad-CAM de la pathologie."""
    choix = []
    for p in ["Effusion", "Cardiomegaly", "Edema", "Emphysema", "Pneumothorax", "Atelectasis"]:
        k = PATHOLOGIES.index(p)
        pos = np.where(te[p].to_numpy() == 1)[0]
        if len(pos):
            i = pos[np.argmax(P[pos, k])]
            choix.append((i, k))
        if len(choix) == n:
            break
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    for col, (i, k) in enumerate(choix):
        cam, raw = heatmap(model, paths[i], k)
        axes[0, col].imshow(raw, cmap="gray")
        axes[0, col].set_title(f"{NOMS_FR[PATHOLOGIES[k]]}\nproba {P[i, k]:.2f}", fontsize=10)
        axes[1, col].imshow(raw, cmap="gray")
        axes[1, col].imshow(cam, cmap="jet", alpha=0.4)
        for a in axes[:, col]:
            a.axis("off")
    fig.suptitle("Grad-CAM : zones qui ont le plus pesé dans la prédiction (images du jeu de test)", fontweight="bold")
    fig.tight_layout()
    save(fig, "gradcam.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    label_files, image_dir = config()
    df = load_labels(label_files, image_dir)
    tr, va, te = patient_split(df)
    assert check_no_overlap(tr, va, te), "un patient apparaît dans deux ensembles"
    print(f"{len(df)} radiographies, {df['PatientId'].nunique()} patients | train {len(tr)}, validation {len(va)}, test {len(te)}")

    E = embeddings(image_paths(df, image_dir))
    idx = {f: i for i, f in enumerate(df["Image"])}
    X = lambda d: E[[idx[f] for f in d["Image"]]]  # noqa: E731
    Y = lambda d: d[PATHOLOGIES].to_numpy()  # noqa: E731

    clf = MultiLabelClassifier().fit(X(tr), Y(tr), X(va), Y(va))
    P_va, P_te, P_te_raw = clf.proba(X(va)), clf.proba(X(te)), clf.proba_brute(X(te))
    r = report(Y(te), P_te, Y(va), P_va)
    r.to_csv(OUT / "resultats_par_pathologie.csv", index=False, float_format="%.4f")

    fig_auc(r)
    fig_roc(Y(te), P_te, r)
    fig_calibration(Y(te), P_te_raw, P_te)
    W, b = clf.dense_weights()
    fig_gradcam(build_full_model(W, b), te, image_paths(te, image_dir), P_te)

    resume = {
        "radiographies": len(df), "patients": int(df["PatientId"].nunique()),
        "decoupage": {"train": len(tr), "validation": len(va), "test": len(te)},
        "auc_moyenne": float(r["AUC"].mean()), "auc_mediane": float(r["AUC"].median()),
        "brier_avant_calibration": brier(Y(te), P_te_raw), "brier_apres_calibration": brier(Y(te), P_te),
        "regularisation_C": clf.C,
    }
    with open(OUT / "resume.json", "w", encoding="utf-8") as fh:
        json.dump(resume, fh, indent=2, ensure_ascii=False)
    cols = ["pathologie", "cas_positifs_test", "AUC", "IC95_bas", "IC95_haut", "sensibilite", "specificite", "VPP", "VPN"]
    print(r[cols].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in resume.items() if k != "regularisation_C"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
