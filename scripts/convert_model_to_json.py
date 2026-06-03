#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-save the legacy pickled cough classifier as a modern XGBoost JSON model.

The bundled ``models/cough_classifier`` was pickled with scikit-learn 0.22.1 and
an old XGBoost (pre-JSON binary booster format). Modern XGBoost (>=1.6) cannot
unpickle it:

    XGBoostError: Check failed: header[0] == '{' ...

This script bridges the gap WITHOUT requiring you to manually manage a second
environment. It uses ``uv`` to spin up a throwaway interpreter pinned to the
*old* xgboost/scikit-learn/numpy, loads the legacy pickle there, and exports:

    models/cough_classifier_migrated.json      <- portable booster (runtime model)
    models/cough_classifier_migrated_meta.json <- wrapper attrs (classes_, params)
    models/cough_classification_scaler.json    <- scaler mean_/scale_/var_

It then re-loads the JSON in THIS (modern) environment and verifies that
``predict_proba`` matches the legacy model on random inputs.

USAGE
-----
    python3 convert_model_to_json.py
    python3 convert_model_to_json.py --model models/cough_classifier --tol 1e-5

Requires: ``uv`` on PATH (https://docs.astral.sh/uv/). Modern xgboost in the
current env is used only for verification; pass --no-verify to skip it.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Legacy stack that can unpickle the original artifacts. xgboost 1.0.2 still
# writes the old binary booster but can EXPORT modern JSON via save_model(.json).
LEGACY_PINS = [
    "--python", "3.8",
    "--with", "numpy==1.21.6",
    "--with", "scipy==1.7.3",
    "--with", "scikit-learn==0.22.2.post1",
    "--with", "xgboost==1.0.2",
]

# Runs inside the legacy uv environment. Reads paths from argv.
LEGACY_EXPORT_SRC = r'''
import json, pickle, sys
import numpy as np
import xgboost

model_path, scaler_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

with open(model_path, "rb") as f:
    clf = pickle.load(f)

booster = clf.get_booster()
booster.save_model(f"{out_dir}/cough_classifier_migrated.json")

def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

meta = {
    "objective": getattr(clf, "objective", None),
    "n_classes_": jsonable(getattr(clf, "n_classes_", None)),
    "classes_": jsonable(getattr(clf, "classes_", None)),
    "n_features": jsonable(getattr(clf, "n_features_in_", None)),
    "params": {k: jsonable(v) for k, v in clf.get_xgb_params().items()},
    "legacy_xgboost_version": xgboost.__version__,
}

# Deterministic probe matrix so the modern side can reproduce predictions.
rng = np.random.RandomState(0)
n_feat = meta["n_features"] or booster.num_features() or 68
probe = rng.rand(8, int(n_feat)).astype(np.float32)
meta["probe_X"] = probe.tolist()
meta["probe_proba"] = clf.predict_proba(probe)[:, 1].tolist()

with open(f"{out_dir}/cough_classifier_migrated_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Scaler -> plain JSON (loads fine in modern sklearn but JSON is version-proof).
try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    sc = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": getattr(scaler, "var_", None).tolist() if getattr(scaler, "var_", None) is not None else None,
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
        "n_features": int(len(scaler.mean_)),
    }
    with open(f"{out_dir}/cough_classification_scaler.json", "w") as f:
        json.dump(sc, f, indent=2)
    print("SCALER_OK")
except Exception as exc:
    print(f"SCALER_SKIP {exc}")

print("EXPORT_OK")
'''


def export_with_legacy_env(model_path, scaler_path, out_dir):
    """Run the legacy exporter inside a pinned uv environment."""
    if shutil.which("uv") is None:
        sys.exit("uv not found on PATH. Install it: "
                 "https://docs.astral.sh/uv/getting-started/installation/")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(LEGACY_EXPORT_SRC)
        script = tf.name
    cmd = ["uv", "run", *LEGACY_PINS, "python", script,
           str(model_path), str(scaler_path), str(out_dir)]
    print("Spawning legacy environment (xgboost==1.0.2, sklearn==0.22.2)...")
    print("  " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    Path(script).unlink(missing_ok=True)
    if res.stdout:
        print(res.stdout, end="")
    if "EXPORT_OK" not in res.stdout:
        print(res.stderr, file=sys.stderr)
        sys.exit(f"Legacy export failed (exit {res.returncode}).")


def verify_modern(out_dir, tol):
    """Reload the JSON in the current env and compare predictions."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not importable here; skipping verification.")
        return
    meta = json.loads((out_dir / "cough_classifier_migrated_meta.json").read_text())
    clf = XGBClassifier()
    clf.load_model(str(out_dir / "cough_classifier_migrated.json"))

    probe = np.asarray(meta["probe_X"], dtype=np.float32)
    expected = np.asarray(meta["probe_proba"], dtype=np.float64)

    booster = clf.get_booster()
    import xgboost as xgb
    got = booster.predict(xgb.DMatrix(probe))  # raw binary:logistic prob of class 1
    got = np.asarray(got, dtype=np.float64)

    max_err = float(np.max(np.abs(got - expected)))
    print(f"\nVerification: max |Δ predict_proba| = {max_err:.3e} (tol {tol:.1e})")
    if max_err <= tol:
        print("MATCH — modern JSON reproduces the legacy model.")
    else:
        print("MISMATCH — predictions differ beyond tolerance!", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/cough_classifier")
    ap.add_argument("--scaler", default="models/cough_classification_scaler")
    ap.add_argument("--out-dir", default="models",
                    help="Where to write the .json artifacts (default: models/).")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="Max allowed predict_proba difference in verification.")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    scaler_path = Path(args.scaler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        sys.exit(f"Model not found: {model_path}")

    export_with_legacy_env(model_path, scaler_path, out_dir)
    print(f"\nWrote:\n  {out_dir/'cough_classifier_migrated.json'}"
          f"\n  {out_dir/'cough_classifier_migrated_meta.json'}"
          f"\n  {out_dir/'cough_classification_scaler.json'}")

    if not args.no_verify:
        verify_modern(out_dir, args.tol)

    print("\nLoad it in modern XGBoost with:")
    print("    from xgboost import XGBClassifier")
    print(f"    clf = XGBClassifier(); clf.load_model('{out_dir/'cough_classifier_migrated.json'}')")


if __name__ == "__main__":
    main()
