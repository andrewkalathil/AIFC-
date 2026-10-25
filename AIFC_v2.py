"""
AIFC v2 - Atanassov intuitionistic fuzzy classifier for BraTS patch classification.

Changes relative to the original script
---------------------------------------
1. Patient-level train / validation / test split (GroupShuffleSplit). No patient
   contributes patches to more than one split.
2. Intensity z-scoring is done per VOLUME inside the brain mask (not per slice
   with min-max), and slices are chosen WITHOUT looking at the ground truth
   (evenly spaced over the brain, so tumour-free slices are included).
3. Features: mean, variance, skewness, excess kurtosis, and the centre-referenced
   maximum deviation D = max_k |x_k - x_c|  (= Hausdorff distance between the
   patch intensity set and the singleton {x_c}).
4. The AIFS membership layer is INSIDE the network and trainable:
     mean, variance      -> sigmoid membership
     skewness, kurtosis  -> two-sided Gaussian membership
     non-membership      -> Sugeno generator  nu = (1-mu)/(1+lambda*mu)
     lambda              -> softplus(lam0 + lam1 * D)   (D drives hesitation)
     aggregation         -> intuitionistic fuzzy weighted average (softmax weights)
   so hesitation pi varies from patch to patch and the pi-cap is meaningful.
5. Scaler and SMOTE are fitted on the training patients only. The decision
   threshold tau and hesitation cap eta are chosen on the VALIDATION patients only.
6. Several random seeds; results are reported as mean +/- SD. Variants:
     MLP_raw     plain MLP on the 5 standardized features (no AIFS layer)
     AIFS_full   AIFS layer, lambda driven by D
     AIFS_noD    AIFS layer, lambda constant (lam1 frozen at 0)
   plus optional drop-one-feature ablations (--ablate).
7. ROC-AUC is computed once per model from the continuous output, so it no longer
   changes with tau or eta.
8. Error-detection analysis: AUROC, risk-coverage curve and error capture of four uncertainty scores
   (hesitation pi, spread of the feature memberships, distance to the threshold, entropy); outputs
   error_detection_runs.csv, error_detection_summary.csv, figure_error_detection.png.

Usage
-----
  python AIFC_v2.py --data <path to MICCAI_BraTS2020_TrainingData> --out results_v2
Options: --seeds 5 --patch 11 --slices 6 --epochs 60 --reuse --ablate
Requires: numpy pandas scipy scikit-learn nibabel imbalanced-learn tensorflow matplotlib
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = ["Mean", "Variance", "Skewness", "Kurtosis", "CenterDev"]
TFEAT = 4  # first four features are the moments; the fifth (D) drives hesitation


# ---------------------------------------------------------------------------
# Data loading and feature extraction
# ---------------------------------------------------------------------------
def safe_load_nifti(path):
    import nibabel as nib
    try:
        return np.nan_to_num(nib.load(path).get_fdata().astype(np.float32))
    except Exception as e:  # noqa
        print(f"Skipping {path}: {e}")
        return None


def zscore_volume(vol):
    """z-score using brain voxels (non-zero) of the whole volume."""
    brain = vol != 0
    if brain.sum() < 100:
        return None, brain
    m, s = vol[brain].mean(), vol[brain].std()
    out = np.zeros_like(vol, dtype=np.float32)
    out[brain] = (vol[brain] - m) / (s + 1e-8)
    return out, brain


def pick_slices(brain, k, min_pixels=500):
    """k evenly spaced slices between the 20th and 80th percentile of brain slices.
    Chosen from the image only, never from the ground-truth mask."""
    counts = brain.sum(axis=(0, 1))
    valid = np.where(counts >= min_pixels)[0]
    if len(valid) == 0:
        return []
    lo = valid[int(0.2 * (len(valid) - 1))]
    hi = valid[int(0.8 * (len(valid) - 1))]
    return sorted(set(int(i) for i in np.linspace(lo, hi, k).round()))


def patch_features(P):
    """P: (n, p*p) patch intensities -> (n, 5) feature matrix."""
    n, m = P.shape
    c = P[:, m // 2]                                   # central pixel (odd patch size)
    return np.column_stack([
        P.mean(axis=1),
        P.var(axis=1),
        skew(P, axis=1),
        kurtosis(P, axis=1),                           # excess kurtosis
        np.abs(P - c[:, None]).max(axis=1),            # D = Hausdorff({patch},{x_c})
    ])


def extract_patches(img, mask, brain, p, min_brain_frac=0.5, std_min=0.02):
    """Non-overlapping p x p patches of one slice. Returns (features, labels, ys, xs)."""
    assert p % 2 == 1, "patch size must be odd so every patch has a centre pixel"
    H, W = img.shape
    nh, nw = H // p, W // p

    def tile(a):
        a = a[:nh * p, :nw * p]
        return a.reshape(nh, p, nw, p).transpose(0, 2, 1, 3).reshape(nh * nw, p * p)

    P, M, B = tile(img), tile(mask.astype(np.float32)), tile(brain.astype(np.float32))
    ys = np.repeat(np.arange(nh) * p, nw)
    xs = np.tile(np.arange(nw) * p, nh)
    keep = (B.mean(axis=1) >= min_brain_frac) & (P.std(axis=1) >= std_min)
    if keep.sum() == 0:
        return np.empty((0, 5)), np.empty(0, int), ys[:0], xs[:0]
    labels = (M[keep].mean(axis=1) > 0.5).astype(int)   # majority-of-patch label
    return patch_features(P[keep]), labels, ys[keep], xs[keep]


def process_patient(pid, flair_path, seg_path, p, n_slices, min_brain_frac=0.9):
    img = safe_load_nifti(flair_path)
    seg = safe_load_nifti(seg_path)
    if img is None or seg is None:
        print(f"  SKIPPED {pid}: could not read the NIfTI files (OneDrive online-only? corrupt file?)")
        return []
    if img.shape != seg.shape:
        print(f"  SKIPPED {pid}: FLAIR shape {img.shape} != seg shape {seg.shape}")
        return []
    z, brain = zscore_volume(img)
    if z is None:
        print(f"  SKIPPED {pid}: volume is empty (fewer than 100 non-zero voxels)")
        return []
    rows = []
    for s in pick_slices(brain, n_slices):
        F, lab, ys, xs = extract_patches(z[:, :, s], seg[:, :, s] > 0, brain[:, :, s], p, min_brain_frac)
        for i in range(len(lab)):
            rows.append([pid, s, int(ys[i]), int(xs[i]), *F[i], int(lab[i])])
    return rows


def build_dataset(patients, p, n_slices, min_brain_frac=0.9):
    cols = ["PatientID", "Slice", "Y", "X", *FEATURES, "Label"]
    rows = []
    for k, (pid, fl, sg) in enumerate(patients):
        print(f"[{k + 1}/{len(patients)}] {pid}")
        rows.extend(process_patient(pid, fl, sg, p, n_slices, min_brain_frac))
    if not rows:
        raise RuntimeError("Patients were found but no patches were extracted - "
                           "read the 'SKIPPED' messages printed above for the reason.")
    return pd.DataFrame(rows, columns=cols)


def discover_patients(root):
    """Find every folder under root that holds both a FLAIR and a seg NIfTI file."""
    out = []
    for folder, _, files in sorted(os.walk(root)):
        fl = sg = None
        for f in files:
            low = f.lower()
            if low.endswith((".nii", ".nii.gz")):
                if "flair" in low:
                    fl = os.path.join(folder, f)
                elif "seg" in low:
                    sg = os.path.join(folder, f)
        if fl and sg:
            out.append((os.path.basename(folder), fl, sg))
    return out


def diagnose_folder(root):
    """Explain why no patients were found."""
    print(f"\nNo patient with both a FLAIR and a seg NIfTI file was found under:\n  {root}")
    if not os.path.exists(root):
        print("This folder does not exist. Check the path (OneDrive folder name, '(1)' suffix, spelling).")
        return
    from collections import Counter
    ext, examples = Counter(), []
    for folder, _, files in os.walk(root):
        for f in files:
            e = ".nii.gz" if f.lower().endswith(".nii.gz") else os.path.splitext(f)[1].lower()
            ext[e] += 1
            if len(examples) < 8:
                examples.append(os.path.join(folder, f))
    print("File types found in that folder tree:", dict(ext) or "none (folder is empty)")
    print("First files:", *examples, sep="\n  ")
    if ext.get(".h5"):
        print("\nThese are .h5 slice files, not the NIfTI volumes this script needs. "
              "Point --data at the folder holding BraTS20_Training_001, BraTS20_Training_002, ... "
              "(each with *_flair.nii and *_seg.nii files).")
    elif not (ext.get(".nii") or ext.get(".nii.gz")):
        print("\nNo .nii / .nii.gz files here. Point --data at the folder with the BraTS patient folders.")
    else:
        print("\nNIfTI files exist but not as FLAIR+seg pairs in the same folder. "
              "File names must contain 'flair' and 'seg'.")


# ---------------------------------------------------------------------------
# Splitting, thresholds, metrics (pure numpy / sklearn)
# ---------------------------------------------------------------------------
def patient_split(df, seed, test=0.20, val=0.15):
    """Split by PatientID so no patient appears in two splits."""
    g1 = GroupShuffleSplit(1, test_size=test, random_state=seed)
    trv, te = next(g1.split(df, groups=df["PatientID"]))
    d2 = df.iloc[trv]
    g2 = GroupShuffleSplit(1, test_size=val / (1 - test), random_state=seed)
    tr, va = next(g2.split(d2, groups=d2["PatientID"]))
    return d2.iloc[tr].index.values, d2.iloc[va].index.values, df.iloc[te].index.values


def confusion(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return (int((y & pred).sum()), int((~y & pred).sum()),
            int((~y & ~pred).sum()), int((y & ~pred).sum()))


def metrics(y, pred, prob):
    tp, fp, tn, fn = confusion(y, pred)
    rec = tp / (tp + fn) if tp + fn else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, Recall=rec, Specificity=spe,
                Precision=pre, F1_Dice=f1, BalancedAcc=(rec + spe) / 2,
                AUC=float(roc_auc_score(y, prob)))


def choose_thresholds(y, prob, pi, taus=None, etas=None):
    """Pick (tau, eta) maximising F1 on VALIDATION data. pi=None -> no hesitation cap."""
    taus = np.arange(0.30, 0.991, 0.01) if taus is None else taus
    if pi is None:
        etas = [1.0]
        pi = np.zeros_like(prob)
    elif etas is None:
        qs = np.unique(np.quantile(pi, [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]))
        etas = qs
    best, bt, be = -1.0, 0.5, 1.0
    for tau in taus:
        for eta in etas:
            tp, fp, tn, fn = confusion(y, (prob >= tau) & (pi <= eta))
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best:
                best, bt, be = f1, float(tau), float(eta)
    return bt, be


def predict_labels(prob, pi, tau, eta):
    pi = np.zeros_like(prob) if pi is None else pi
    return ((prob >= tau) & (pi <= eta)).astype(int)


# ---------------------------------------------------------------------------
# AIFS layer and models (TensorFlow imported lazily)
# ---------------------------------------------------------------------------
SP_INV_1 = 0.5413248546129181  # softplus^-1(1)


def build_model(tf, n_in, use_aifs, hesitation_from_D=True):
    K = tf.keras

    class AIFSLayer(K.layers.Layer):
        """Input (B,5): [mean, var, skew, kurt, D] standardized.
        Output (B,15): mu_j(4), nu_j(4), pi_j(4), mu, nu, pi (aggregated)."""

        def __init__(self, hesitation_from_D=True, **kw):
            super().__init__(**kw)
            self.hfd = hesitation_from_D

        def build(self, input_shape):
            c0 = K.initializers.Constant(SP_INV_1)
            self.c = self.add_weight(name="c", shape=(4,), initializer="zeros")
            self.a = self.add_weight(name="a", shape=(2,), initializer="ones")
            self.sL = self.add_weight(name="sL", shape=(2,), initializer=c0)
            self.sR = self.add_weight(name="sR", shape=(2,), initializer=c0)
            self.lam0 = self.add_weight(name="lam0", shape=(), initializer=c0)
            self.lam1 = self.add_weight(name="lam1", shape=(),
                                        initializer=K.initializers.Constant(0.5 if self.hfd else 0.0),
                                        trainable=self.hfd)
            self.w_logits = self.add_weight(name="w_logits", shape=(4,), initializer="zeros")
            super().build(input_shape)

        def call(self, x):
            eps = 1e-4
            xm, D = x[:, :4], x[:, 4:5]
            mu_s = tf.sigmoid(self.a * (xm[:, :2] - self.c[:2]))
            s = tf.where(xm[:, 2:] < self.c[2:], tf.nn.softplus(self.sL),
                         tf.nn.softplus(self.sR)) + 1e-3
            mu_g = tf.exp(-tf.square(xm[:, 2:] - self.c[2:]) / (2.0 * tf.square(s)))
            mu = tf.clip_by_value(tf.concat([mu_s, mu_g], axis=-1), eps, 1.0 - eps)
            lam = tf.nn.softplus(self.lam0 + self.lam1 * D) + 1e-3        # (B,1)
            nu = (1.0 - mu) / (1.0 + lam * mu)
            pi = tf.nn.relu(1.0 - mu - nu)
            w = tf.nn.softmax(self.w_logits)
            mu_a = 1.0 - tf.exp(tf.reduce_sum(w * tf.math.log(1.0 - mu), axis=-1, keepdims=True))
            nu_a = tf.exp(tf.reduce_sum(w * tf.math.log(nu), axis=-1, keepdims=True))
            pi_a = tf.nn.relu(1.0 - mu_a - nu_a)
            return tf.concat([mu, nu, pi, mu_a, nu_a, pi_a], axis=-1)

    inp = K.Input(shape=(n_in,))
    if use_aifs:
        layer = AIFSLayer(hesitation_from_D, name="aifs")
        z = layer(inp)
        pi_out = K.layers.Lambda(lambda t: t[:, -1:], name="pi_agg")(z)
    else:
        z, pi_out = inp, None
    h = K.layers.Dense(32, activation="relu")(z)
    h = K.layers.Dropout(0.3)(h)
    h = K.layers.Dense(16, activation="relu")(h)
    prob = K.layers.Dense(1, activation="sigmoid", name="prob")(h)
    train_model = K.Model(inp, prob)
    aux_model = K.Model(inp, [prob, pi_out]) if use_aifs else None
    return train_model, aux_model


def learned_params(model):
    try:
        L = model.get_layer("aifs")
    except ValueError:
        return None
    sp = lambda v: np.log1p(np.exp(v))
    w = np.exp(L.w_logits.numpy()); w /= w.sum()
    return dict(
        centers=dict(zip(FEATURES[:4], L.c.numpy().round(4).tolist())),
        slope_mean_var=L.a.numpy().round(4).tolist(),
        width_left_skew_kurt=sp(L.sL.numpy()).round(4).tolist(),
        width_right_skew_kurt=sp(L.sR.numpy()).round(4).tolist(),
        lambda0=float(L.lam0.numpy()), lambda1=float(L.lam1.numpy()),
        feature_weights=dict(zip(FEATURES[:4], w.round(4).tolist())),
    )


def fit_predict(tf, cfg, Xtr, ytr, Xva, yva, Xte, seed, epochs):
    tf.keras.utils.set_random_seed(seed)
    train_m, aux_m = build_model(tf, Xtr.shape[1], cfg["use_aifs"], cfg.get("hfd", True))
    train_m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                    metrics=[tf.keras.metrics.AUC(name="auc")])
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                          restore_best_weights=True)
    train_m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=128,
                callbacks=[cb], verbose=0)

    def run(X):
        if aux_m is None:
            return train_m.predict(X, batch_size=4096, verbose=0).ravel(), None
        p, pi = aux_m.predict(X, batch_size=4096, verbose=0)
        return p.ravel(), pi.ravel()

    feat_m = None
    if cfg["use_aifs"]:
        try:   # model returning the 15 layer outputs: mu_j, nu_j, pi_j (4 each), then aggregated mu, nu, pi
            feat_m = tf.keras.Model(train_m.input, train_m.get_layer("aifs").output)
        except Exception as e:  # noqa
            print("  note: could not build the feature-level output model:", e)

    def run_full(X):
        """(probability, aggregated pi, layer outputs z) - z is None when unavailable."""
        p = train_m.predict(X, batch_size=4096, verbose=0).ravel()
        if not cfg["use_aifs"]:
            return p, None, None
        z = feat_m.predict(X, batch_size=4096, verbose=0) if feat_m is not None else None
        pi = z[:, -1] if z is not None else run(X)[1]
        return p, pi, z

    run.full = run_full
    return train_m, run


# ---------------------------------------------------------------------------
# Hesitation as an error detector
# ---------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def uncertainty_scores(p, tau, pi=None, z=None):
    """Scores where HIGHER means MORE uncertain.
    margin_to_tau   minus the distance of the logit from the decision threshold (confidence baseline)
    entropy         binary predictive entropy (confidence baseline)
    hesitation_pi   aggregated intuitionistic hesitation of the AIFS layer
    mu_spread       SD of the four feature-level memberships (disagreement among the features)"""
    p = np.asarray(p, float)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"margin_to_tau": -np.abs(_logit(p) - _logit(tau)),
           "entropy": -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))}
    if pi is not None:
        out["hesitation_pi"] = np.asarray(pi, float)
    if z is not None:
        out["mu_spread"] = np.std(np.asarray(z)[:, :4], axis=1)
    return out


def risk_coverage_curve(err, u):
    """Error rate among the k most certain patches (lowest u first), for k = 1..n."""
    order = np.argsort(u, kind="mergesort")
    err = np.asarray(err)[order]
    n = len(err)
    return np.arange(1, n + 1) / n, np.cumsum(err) / np.arange(1, n + 1)


def error_detection(y, pred, scores, fracs=(0.05, 0.10, 0.20), coverage=0.90, xmean=None):
    """How well does each uncertainty score find the classifier's errors?
    AUROC_error          AUROC of the score for 'this patch is misclassified' (0.5 = no information,
                         below 0.5 means errors get LOWER uncertainty)
    capture@x%           share of all errors inside the x% most uncertain patches (random = x%)
    AURC_gain            area under the risk-coverage curve, scaled: 0 = random, 1 = perfect ordering
    F1_at_90cov / F1_full  F1 after rejecting the 10% most uncertain patches, and without rejection
    AUROC_FP_among_pred_pos  same as AUROC_error but only among patches predicted tumor"""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    err = (y != pred).astype(int)
    n, E = len(y), int(err.sum())

    def f1(a, b):
        tp, fp, tn, fn = confusion(a, b)
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    orc = np.sort(err)
    oracle = float((np.cumsum(orc) / np.arange(1, n + 1)).mean())
    out = {}
    for name, u in scores.items():
        u = np.asarray(u, float)
        desc = np.argsort(-u, kind="mergesort")           # most uncertain first
        r = {"AUROC_error": float(roc_auc_score(err, u)) if 0 < E < n else np.nan}
        for f in fracs:
            k = int(np.ceil(f * n))
            r[f"capture@{int(round(f * 100))}%"] = float(err[desc[:k]].sum() / E) if E else np.nan
        _, risk = risk_coverage_curve(err, u)
        base = float(err.mean())
        r["AURC_gain"] = float((base - risk.mean()) / (base - oracle)) if base > oracle else np.nan
        keep = desc[::-1][: int(np.floor(coverage * n))]
        r["F1_at_90cov"], r["F1_full"] = f1(y[keep], pred[keep]), f1(y, pred)
        pos = pred == 1
        r["AUROC_FP_among_pred_pos"] = (float(roc_auc_score(err[pos], u[pos]))
                                        if pos.sum() > 0 and 0 < err[pos].sum() < pos.sum() else np.nan)
        if xmean is not None:
            r["Spearman_with_Mean"] = float(pd.Series(u).corr(pd.Series(np.asarray(xmean)), method="spearman"))
        out[name] = r
    return out


def plot_error_detection(state, E, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    err = (np.asarray(state["y"]) != np.asarray(state["pred"])).astype(int)
    for name, u in state["scores"].items():
        c, r = risk_coverage_curve(err, u)
        m = c >= 0.5
        ax[0].plot(c[m], r[m], label=name)
    c, r = risk_coverage_curve(err, np.sort(err).astype(float))
    m = c >= 0.5
    ax[0].plot(c[m], r[m], "k:", label="perfect ordering")
    ax[0].axhline(err.mean(), color="grey", ls="--", label="random rejection")
    ax[0].set_xlabel("coverage (share of most certain patches kept)")
    ax[0].set_ylabel("error rate among kept patches")
    ax[0].set_title("Risk-coverage curve (AIFS_full, seed 0)")
    ax[0].legend(fontsize=8)
    g = E[E.Variant == "AIFS_full"].groupby("Score")["capture@10%"].agg(["mean", "std"])
    ax[1].bar(g.index, g["mean"], yerr=g["std"].fillna(0), capsize=3)
    ax[1].axhline(0.10, color="grey", ls="--", label="random (10%)")
    ax[1].set_ylabel("share of all errors in the 10% most uncertain patches")
    ax[1].set_title("Error capture at 10% review (mean ± SD over seeds)")
    ax[1].tick_params(axis="x", rotation=20)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def run_experiments(df, args):
    import tensorflow as tf
    from imblearn.over_sampling import SMOTE

    variants = {
        "MLP_raw": dict(use_aifs=False),
        "AIFS_full": dict(use_aifs=True, hfd=True),
        "AIFS_noD": dict(use_aifs=True, hfd=False),
    }
    drop_cols = list(range(5)) if args.ablate else []
    runs, params_log, fig_state = [], {}, None
    err_rows, err_state = [], None

    for seed in range(args.seeds):
        tr, va, te = patient_split(df, seed)
        sc = StandardScaler().fit(df.loc[tr, FEATURES])
        f = lambda idx: np.clip(sc.transform(df.loc[idx, FEATURES]), -5, 5)
        Xtr, Xva, Xte = f(tr), f(va), f(te)
        ytr, yva, yte = (df.loc[i, "Label"].values for i in (tr, va, te))
        print(f"\nseed {seed}: patients train/val/test = "
              f"{df.loc[tr,'PatientID'].nunique()}/{df.loc[va,'PatientID'].nunique()}/"
              f"{df.loc[te,'PatientID'].nunique()}; positives {ytr.mean():.3f}/{yva.mean():.3f}/{yte.mean():.3f}")

        def with_smote(X, y):
            return SMOTE(random_state=seed).fit_resample(X, y)

        jobs = [(n, c, None) for n, c in variants.items()]
        jobs += [(f"AIFS_full-drop_{FEATURES[j]}", variants["AIFS_full"], j) for j in drop_cols]

        for name, cfg, drop in jobs:
            A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
            if drop is not None:      # remove a feature by setting it to its training mean
                A[:, drop] = B[:, drop] = C[:, drop] = 0.0
            Xs, ys = with_smote(A, ytr)
            model, run = fit_predict(tf, cfg, Xs, ys, B, yva, C, seed, args.epochs)
            pv, piv = run(B)
            pt, pit = run(C)
            tau, eta = choose_thresholds(yva, pv, piv)
            m = metrics(yte, predict_labels(pt, pit, tau, eta), pt)
            m.update(Variant=name, Seed=seed, tau=tau, eta=eta,
                     pi_mean=float(pit.mean()) if pit is not None else np.nan,
                     pi_max=float(pit.max()) if pit is not None else np.nan)
            runs.append(m)
            if pit is not None:        # same model, no hesitation cap (tau re-chosen on validation)
                tau2, _ = choose_thresholds(yva, pv, None)
                m2 = metrics(yte, predict_labels(pt, None, tau2, 1.0), pt)
                m2.update(Variant=name + "_nocap", Seed=seed, tau=tau2, eta=1.0,
                          pi_mean=m["pi_mean"], pi_max=m["pi_max"])
                runs.append(m2)
            if name in ("MLP_raw", "AIFS_full"):     # is hesitation useful for finding errors?
                tau_e, _ = choose_thresholds(yva, pv, None)           # no hesitation cap: no circularity
                pred_e = (pt >= tau_e).astype(int)
                _, pi_e, z_e = run.full(C)
                sc_e = uncertainty_scores(pt, tau_e, pi_e, z_e)
                for sname, r in error_detection(yte, pred_e, sc_e, xmean=C[:, 0]).items():
                    err_rows.append(dict(Variant=name, Seed=seed, Score=sname, **r))
                if name == "AIFS_full" and seed == 0:
                    err_state = dict(y=yte, pred=pred_e, scores=sc_e)
            if name == "AIFS_full":
                params_log[seed] = learned_params(model)
                if seed == 0:
                    fig_state = dict(run=run, tau=tau, eta=eta, te=te, sc=sc, pi_test=pit, prob_test=pt)
            print(f"  {name:28s} AUC={m['AUC']:.4f} Rec={m['Recall']:.3f} "
                  f"Spe={m['Specificity']:.3f} F1={m['F1_Dice']:.3f} tau={tau:.2f} eta={eta:.3f}")

    R = pd.DataFrame(runs)
    R.to_csv(os.path.join(args.out, "results_runs.csv"), index=False)
    cols = ["Recall", "Specificity", "Precision", "F1_Dice", "BalancedAcc", "AUC"]
    g = R.groupby("Variant")[cols]
    S = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    S.to_csv(os.path.join(args.out, "results_summary.csv"))
    json.dump(params_log, open(os.path.join(args.out, "learned_aifs_params.json"), "w"), indent=2)
    print("\n=== mean ± SD over seeds (test patients) ===")
    print(S.to_string())
    if err_rows:
        E = pd.DataFrame(err_rows)
        E.to_csv(os.path.join(args.out, "error_detection_runs.csv"), index=False)
        mc = [c for c in E.columns if c not in ("Variant", "Seed", "Score")]
        ge = E.groupby(["Variant", "Score"])[mc]
        ES = (ge.mean().round(3).astype(str) + " ± " + ge.std().round(3).astype(str))
        ES.to_csv(os.path.join(args.out, "error_detection_summary.csv"))
        print("\n=== can uncertainty find the errors? (test patients, mean ± SD over seeds) ===")
        print(ES[["AUROC_error", "capture@5%", "capture@10%", "capture@20%", "AURC_gain",
                  "AUROC_FP_among_pred_pos", "Spearman_with_Mean"]].to_string())
        print("random rejection: AUROC 0.5, capture@10% = 0.10, AURC_gain 0")
        if err_state is not None:
            plot_error_detection(err_state, E, os.path.join(args.out, "figure_error_detection.png"))
            print("figure saved:", os.path.join(args.out, "figure_error_detection.png"))
    pairs = [("AIFS_full", "MLP_raw"), ("AIFS_full", "AIFS_noD"), ("AIFS_noD", "MLP_raw"),
             ("AIFS_full", "AIFS_full_nocap")]
    PT = paired_table(R, pairs, cols)
    PT.round(4).to_csv(os.path.join(args.out, "paired_differences.csv"), index=False)
    print("\n=== paired differences over seeds (mean diff, SD, seeds where first is better) ===")
    print(PT.round(4).to_string(index=False))
    if fig_state is not None:
        d = df.loc[fig_state["te"]].copy()
        d["pi"] = fig_state["pi_test"]
        d["region"] = region_of_patches(d, args.patch)
        RG = d.groupby("region").agg(n=("pi", "size"), pi_mean=("pi", "mean"), pi_median=("pi", "median"),
                                      CenterDev_mean=("CenterDev", "mean"), tumor_frac=("Label", "mean"))
        RG.round(4).to_csv(os.path.join(args.out, "hesitation_by_region.csv"))
        print("\n=== aggregated hesitation by region (seed 0, test patients) ===")
        print(RG.round(4).to_string())
        if args.data:
            make_figure(df, fig_state, args)
    print("\nAll outputs were saved in:", os.path.abspath(args.out))
    print("Files:", ", ".join(sorted(os.listdir(args.out))))
    return R


def region_of_patches(d, p):
    """Label each patch of one dataframe (PatientID, Slice, Y, X, Label) by where it sits:
    tumor_core, tumor_rim (tumor patch touching a non-tumor patch), healthy_rim (non-tumor touching
    tumor), brain_edge (non-tumor next to a missing/outside patch, no tumor neighbour) or healthy_interior."""
    lab = {(a, b, y, x): l for a, b, y, x, l in zip(d.PatientID, d.Slice, d.Y, d.X, d.Label)}
    offs = [(dy * p, dx * p) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    out = []
    for a, b, y, x, l in zip(d.PatientID, d.Slice, d.Y, d.X, d.Label):
        nb = [lab.get((a, b, y + dy, x + dx)) for dy, dx in offs]
        has_t, has_h, has_missing = (1 in nb), (0 in nb), (None in nb)
        if l == 1:
            out.append("tumor_rim" if has_h else "tumor_core")
        elif has_t:
            out.append("healthy_rim")
        elif has_missing:
            out.append("brain_edge")
        else:
            out.append("healthy_interior")
    return np.array(out)


def paired_table(R, pairs, cols):
    rows = []
    for a, b in pairs:
        if a not in set(R.Variant) or b not in set(R.Variant):
            continue
        A = R[R.Variant == a].set_index("Seed")[cols]
        B = R[R.Variant == b].set_index("Seed")[cols]
        D = (A - B).dropna()
        for c in cols:
            d = D[c].values
            rows.append(dict(Comparison=f"{a} - {b}", Metric=c, MeanDiff=d.mean(), SD=d.std(ddof=1),
                             Wins=f"{int((d > 0).sum())}/{len(d)}"))
    return pd.DataFrame(rows)


def make_figure(df, state, args):
    """Unseen test patients only. Panels: FLAIR | ground truth | TP/FP/FN overlay | hesitation map.
    The slice shown is the one with most tumour patches (for display only, not used in training)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pats = {pid: (fl, sg) for pid, fl, sg in discover_patients(args.data)}
    test_pids = df.loc[state["te"], "PatientID"].unique()[:3]
    fig, axes = plt.subplots(len(test_pids), 4, figsize=(16, 4 * len(test_pids)), squeeze=False)
    p = args.patch
    for r, pid in enumerate(test_pids):
        d = df[df.PatientID == pid]
        s = int(d.groupby("Slice")["Label"].sum().idxmax())
        d = d[d.Slice == s]
        img = safe_load_nifti(pats[pid][0])[:, :, s]
        gt = safe_load_nifti(pats[pid][1])[:, :, s] > 0
        X = np.clip(state["sc"].transform(d[FEATURES]), -5, 5)
        prob, pi = state["run"](X)
        pred = predict_labels(prob, pi, state["tau"], state["eta"])
        ov = np.zeros((*img.shape, 3), np.uint8)
        hm = np.zeros(img.shape, np.float32)
        for (y, x, lab, pr, h) in zip(d.Y, d.X, d.Label, pred, pi if pi is not None else np.zeros(len(d))):
            col = [0, 255, 0] if lab and pr else [255, 0, 0] if (not lab and pr) else [0, 100, 255] if lab else None
            if col:
                ov[y:y + p, x:x + p] = col
            hm[y:y + p, x:x + p] = h
        axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_title(f"FLAIR ({pid}, slice {s})")
        axes[r, 1].imshow(img, cmap="gray"); axes[r, 1].imshow(np.ma.masked_where(~gt, gt), cmap="autumn", alpha=0.6)
        axes[r, 1].set_title("Ground truth")
        axes[r, 2].imshow(img, cmap="gray"); axes[r, 2].imshow(ov, alpha=0.7)
        axes[r, 2].set_title("Green=TP  Red=FP  Blue=FN")
        im = axes[r, 3].imshow(img, cmap="gray"); axes[r, 3].imshow(np.ma.masked_where(hm == 0, hm), cmap="viridis", alpha=0.8)
        axes[r, 3].set_title("Hesitation pi (aggregated)")
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout()
    path = os.path.join(args.out, "figure_unseen_patients.png")
    plt.savefig(path, dpi=150)
    print("figure saved:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="folder containing the BraTS patient folders (searched recursively)")
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--patch", type=int, default=11)
    ap.add_argument("--slices", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-brain-frac", dest="min_brain_frac", type=float, default=0.9,
                    help="keep patches with at least this fraction of brain pixels (0.9 avoids skull-edge patches)")
    ap.add_argument("--reuse", action="store_true", help="reuse features.csv if present")
    ap.add_argument("--ablate", action="store_true", help="add drop-one-feature runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv = os.path.join(args.out, "features.csv")
    if args.reuse and os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        if not args.data:
            sys.exit("--data is required unless --reuse finds features.csv")
        patients = discover_patients(args.data)
        print(f"Found {len(patients)} patients with FLAIR + seg under: {args.data}")
        if not patients:
            diagnose_folder(args.data)
            sys.exit(1)
        df = build_dataset(patients, args.patch, args.slices, args.min_brain_frac)
        df.to_csv(csv, index=False)
    print(df["Label"].value_counts(), "patients:", df["PatientID"].nunique())
    run_experiments(df, args)


if __name__ == "__main__":
    main()
