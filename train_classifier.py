#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the bundled cough classifier and feature scaler.

This script rebuilds the two artifacts under ``models/`` from scratch:

    models/cough_classifier              -> xgboost.XGBClassifier (binary:logistic)
    models/cough_classification_scaler   -> sklearn StandardScaler (68 features)

It mirrors the *exact* 68-feature vector and ordering used at inference time by
``src.DSP.classify_cough`` (EEPD, ZCR, RMSP, DF, spectral_features, SF_SSTD,
SSL_SD, MFCC, CF, LGTH, PSD), fits a StandardScaler, then trains an
XGBClassifier using the hyperparameters recovered from the original pickle:

    n_estimators=100, max_depth=5, learning_rate=0.1,
    gamma=0.17610092774182107, min_child_weight=1,
    subsample=0.5535958564598562, colsample_bytree=1,
    reg_alpha=0, reg_lambda=0.7654873377280748,
    scale_pos_weight=1, base_score=0.5, objective='binary:logistic'

NOTE ON LABELS
--------------
The original training labels are NOT shipped in this repo. The COUGHVID dataset
(https://doi.org/10.5281/zenodo.4048311) ships ``metadata_compiled.csv`` with
per-expert quality columns. The default labelling here treats a recording as a
cough (positive, 1) unless an expert marked it ``no_cough`` (negative, 0), and
only keeps expert-reviewed rows. This is a reasonable reconstruction of the
"is this a usable cough recording" task the model performs; adjust to match your
own ground truth via ``--labels-csv``/``--label-col`` if you have it.

USAGE
-----
    # COUGHVID reconstruction (derive labels from expert quality columns):
    python3 train_classifier.py --coughvid-dir /path/to/coughvid_wav/

    # Generic: your own labels CSV (binary 0/1 in --label-col, file id in --id-col)
    python3 train_classifier.py --data-dir wavs/ --labels-csv labels.csv \
        --id-col uuid --label-col is_cough
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from src.DSP import preprocess_cough
from src.feature_class import features

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_kwargs):
        return it

# ---- These MUST match src/DSP.py:classify_cough exactly to reproduce the model
FREQ_CUTS = [(0, 200), (300, 425), (500, 650), (950, 1150),
             (1400, 1800), (2300, 2400), (2850, 2950), (3800, 3900)]
FEATURE_FCT_LIST = ['EEPD', 'ZCR', 'RMSP', 'DF', 'spectral_features',
                    'SF_SSTD', 'SSL_SD', 'MFCC', 'CF', 'LGTH', 'PSD']
N_FEATURES = 68

# Hyperparameters recovered from the original models/cough_classifier pickle.
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    gamma=0.17610092774182107,
    min_child_weight=1,
    subsample=0.5535958564598562,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=0.7654873377280748,
    scale_pos_weight=1,
    base_score=0.5,
    objective='binary:logistic',
    booster='gbtree',
)


def extract_features(x, fs):
    """Return the 68-dim feature vector for one signal, in the model's order.

    This replicates the body of ``src.DSP.classify_cough`` (minus the
    scaler/model steps) so that training features are byte-for-byte comparable
    to what the deployed classifier sees.
    """
    x = np.asarray(x)
    if x.size == 0 or np.max(np.abs(x)) == 0:
        return None
    x, fs = preprocess_cough(x, fs)
    data = (fs, x)
    obj = features(FREQ_CUTS)
    vec = []
    for feature in FEATURE_FCT_LIST:
        feature_values, _ = getattr(obj, feature)(data)
        for value in feature_values:
            vec.append(value[0] if isinstance(value, np.ndarray) else value)
    return np.asarray(vec, dtype=np.float64)


def derive_coughvid_labels(meta, id_col='uuid'):
    """Build a binary label from COUGHVID expert quality columns.

    Positive (1) = expert reviewed and did NOT mark ``no_cough``.
    Negative (0) = expert marked ``no_cough``.
    Rows without any expert quality label are dropped.
    """
    quality_cols = [c for c in meta.columns if c.startswith('quality_')]
    if not quality_cols:
        sys.exit("No 'quality_*' columns found; pass --labels-csv instead.")
    reviewed = meta[quality_cols].notna().any(axis=1)
    meta = meta[reviewed].copy()
    is_no_cough = meta[quality_cols].apply(
        lambda row: (row.dropna() == 'no_cough').any(), axis=1)
    meta['__label__'] = (~is_no_cough).astype(int)
    return meta[[id_col, '__label__']].rename(columns={'__label__': 'label'})


def build_dataset(data_dir, labels_df, id_col):
    """Extract features for every labelled file that exists on disk."""
    data_dir = Path(data_dir)
    X, y, used = [], [], 0
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                       desc='extracting features'):
        wav = data_dir / f"{row[id_col]}.wav"
        if not wav.is_file():
            continue
        try:
            fs, sig = wavfile.read(str(wav))
        except Exception as exc:  # unreadable/corrupt wav
            print(f"  skip {wav.name}: {exc}", file=sys.stderr)
            continue
        feats = extract_features(sig, fs)
        if feats is None or feats.shape[0] != N_FEATURES:
            continue
        if not np.all(np.isfinite(feats)):
            continue
        X.append(feats)
        y.append(int(row['label']))
        used += 1
    if used == 0:
        sys.exit("No usable (wav found + features extracted) samples. "
                 "Check --data-dir and that filenames match the id column.")
    return np.vstack(X), np.asarray(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coughvid-dir',
                    help="COUGHVID folder containing .wav files AND "
                         "metadata_compiled.csv; labels derived automatically.")
    ap.add_argument('--data-dir', help="Folder of .wav files (generic mode).")
    ap.add_argument('--labels-csv', help="CSV with id + binary label columns.")
    ap.add_argument('--id-col', default='uuid',
                    help="Column holding the file id / basename (default: uuid).")
    ap.add_argument('--label-col', default='label',
                    help="Binary 0/1 label column in --labels-csv (default: label).")
    ap.add_argument('--out-model', default='models/cough_classifier')
    ap.add_argument('--out-scaler', default='models/cough_classification_scaler')
    ap.add_argument('--features-cache',
                    help="Optional .npz path to cache/reuse extracted X,y.")
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # ---- Resolve data source + labels -----------------------------------
    if args.features_cache and Path(args.features_cache).is_file():
        print(f"Loading cached features from {args.features_cache}")
        cache = np.load(args.features_cache)
        X, y = cache['X'], cache['y']
    else:
        if args.coughvid_dir:
            data_dir = args.coughvid_dir
            meta_path = Path(args.coughvid_dir) / 'metadata_compiled.csv'
            if not meta_path.is_file():
                sys.exit(f"metadata_compiled.csv not found in {args.coughvid_dir}")
            meta = pd.read_csv(meta_path)
            labels_df = derive_coughvid_labels(meta, id_col=args.id_col)
        elif args.data_dir and args.labels_csv:
            data_dir = args.data_dir
            labels_df = pd.read_csv(args.labels_csv).rename(
                columns={args.label_col: 'label'})
        else:
            ap.error("Provide --coughvid-dir, or both --data-dir and --labels-csv.")

        print(f"Labelled rows: {len(labels_df)} "
              f"(positives={int(labels_df['label'].sum())})")
        X, y = build_dataset(data_dir, labels_df, args.id_col)
        if args.features_cache:
            np.savez_compressed(args.features_cache, X=X, y=y)
            print(f"Cached features -> {args.features_cache}")

    print(f"Dataset: X={X.shape}, positives={int(y.sum())}/{len(y)}")
    assert X.shape[1] == N_FEATURES, f"expected {N_FEATURES} features, got {X.shape[1]}"

    # ---- Train/test split -----------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from xgboost import XGBClassifier

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    # ---- Fit scaler ON TRAIN, then transform -----------------------------
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ---- Train classifier ------------------------------------------------
    clf = XGBClassifier(random_state=args.seed, n_jobs=-1,
                        eval_metric='logloss', **XGB_PARAMS)
    clf.fit(X_tr_s, y_tr)

    # ---- Evaluate --------------------------------------------------------
    proba = clf.predict_proba(X_te_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n=== Held-out evaluation ===")
    print(f"ROC-AUC : {roc_auc_score(y_te, proba):.4f}")
    print(f"Accuracy: {accuracy_score(y_te, pred):.4f}")
    print(classification_report(y_te, pred, digits=3))

    # ---- Persist (pickle, matching the original artifact format) ---------
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    with open(args.out_model, 'wb') as f:
        pickle.dump(clf, f)
    # Also drop a modern, version-proof JSON next to the pickle.
    clf.get_booster().save_model(args.out_model + '.json')
    print(f"\nSaved:\n  {args.out_scaler}\n  {args.out_model}"
          f"\n  {args.out_model}.json (modern XGBoost format)")


if __name__ == '__main__':
    main()
