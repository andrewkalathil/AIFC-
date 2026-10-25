"""
baselines.py - simple baselines on the same patient-level splits as AIFC_v2.py (sklearn only, no TensorFlow).

    python baselines.py --features results_v3/features.csv --out baselines_out --seeds 5

Models (all evaluated on the unseen test patients, tau chosen on the validation patients by F1):
  Mean_only_LR   logistic regression on the Mean feature alone (a soft intensity threshold)
  LR_5feat       logistic regression on the 5 standardized features
  HistGB_5feat   gradient boosting on the 5 standardized features
  Mean_rank_AUC  AUC of the raw Mean feature as a score (no model at all)
Same seeds, same scaler and clipping as AIFC_v2.py, so rows are directly comparable with Table 1.
"""
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import AIFC_v2 as A

FEATS = A.FEATURES


def run(df, seeds):
    rows = []
    for seed in range(seeds):
        tr, va, te = A.patient_split(df, seed)
        sc = StandardScaler().fit(df.loc[tr, FEATS])
        f = lambda idx: np.clip(sc.transform(df.loc[idx, FEATS]), -5, 5)
        Xtr, Xva, Xte = f(tr), f(va), f(te)
        ytr, yva, yte = (df.loc[i, "Label"].values for i in (tr, va, te))
        models = {
            "Mean_only_LR": (LogisticRegression(class_weight="balanced", max_iter=1000), [0]),
            "LR_5feat": (LogisticRegression(class_weight="balanced", max_iter=1000), list(range(5))),
            "HistGB_5feat": (HistGradientBoostingClassifier(class_weight="balanced", random_state=seed), list(range(5))),
        }
        for name, (model, cols) in models.items():
            model.fit(Xtr[:, cols], ytr)
            pv, pt = model.predict_proba(Xva[:, cols])[:, 1], model.predict_proba(Xte[:, cols])[:, 1]
            tau, _ = A.choose_thresholds(yva, pv, None, taus=np.arange(0.05, 0.996, 0.005))
            m = A.metrics(yte, A.predict_labels(pt, None, tau, 1.0), pt)
            m.update(Variant=name, Seed=seed, tau=tau)
            rows.append(m)
        s = Xte[:, 0]                                  # raw Mean as a score
        rows.append(dict(Variant="Mean_rank_AUC", Seed=seed, AUC=float(roc_auc_score(yte, s))))
        print(f"seed {seed}: " + "  ".join(f"{r['Variant']}={r['AUC']:.4f}" for r in rows[-4:]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results_v2/features.csv")
    ap.add_argument("--out", default="baselines_final")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.features)
    print(f"features file: {os.path.abspath(a.features)}")
    print(f"{len(df):,} patches, {df.PatientID.nunique()} patients, "
          f"{int(df.Label.sum()):,} tumor patches ({df.Label.mean():.1%})")
    print("(the final run should show 211,835 patches and 17,509 tumor patches)\n")
    R = run(df, a.seeds)
    R["n_patches"] = len(df)
    R.to_csv(os.path.join(a.out, "baselines_runs.csv"), index=False)
    cols = ["Recall", "Specificity", "Precision", "F1_Dice", "BalancedAcc", "AUC"]
    g = R.groupby("Variant")[cols]
    S = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    S.to_csv(os.path.join(a.out, "baselines_summary.csv"))
    print("\n=== mean ± SD over seeds (test patients) ===")
    print(S.to_string())


if __name__ == "__main__":
    main()
