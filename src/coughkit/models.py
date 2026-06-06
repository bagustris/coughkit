"""Loading helpers for the bundled cough classifier and feature scaler.

The model artifacts live in the repository's top-level ``models/`` directory.
This is resolved relative to the installed package (works for editable
``pip install -e .`` installs) and can be overridden with the
``COUGHKIT_MODEL_DIR`` environment variable for wheel installs or custom
locations.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier


def _default_model_dir():
    env = os.environ.get("COUGHKIT_MODEL_DIR")
    if env:
        return Path(env)
    # src/coughkit/models.py -> parents[2] == repository root
    return Path(__file__).resolve().parents[2] / "models"


MODEL_DIR = _default_model_dir()
MIGRATED_MODEL_PATH = MODEL_DIR / "cough_classifier_migrated.json"
LEGACY_MODEL_PATH = MODEL_DIR / "cough_classifier"
SCALER_PATH = MODEL_DIR / "cough_classification_scaler"
SCALER_JSON_PATH = MODEL_DIR / "cough_classification_scaler.json"


class NumpyStandardScaler:
    """Small runtime equivalent of sklearn.preprocessing.StandardScaler."""

    def __init__(self, mean, scale, var=None, with_mean=True, with_std=True,
                 n_features=None):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.var_ = None if var is None else np.asarray(var, dtype=np.float64)
        self.with_mean = with_mean
        self.with_std = with_std
        self.n_features_in_ = (
            int(n_features) if n_features is not None else len(self.mean_)
        )

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("Expected a 2D array for scaler.transform().")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {x.shape[1]}."
            )

        transformed = x.copy()
        if self.with_mean:
            transformed -= self.mean_
        if self.with_std:
            transformed /= self.scale_
        return transformed


def load_cough_classifier():
    """Load the cough classifier, preferring the migrated XGBoost JSON model.

    The original bundled ``models/cough_classifier`` file is a Python pickle
    containing an XGBoost 0.90 in-memory Booster serialization. Modern XGBoost
    cannot unpickle that format directly, so the migrated JSON model produced by
    ``scripts/convert_model_to_json.py`` is the stable representation.
    """
    if MIGRATED_MODEL_PATH.is_file():
        model = XGBClassifier()
        model.load_model(str(MIGRATED_MODEL_PATH))
        return model

    try:
        with LEGACY_MODEL_PATH.open("rb") as model_file:
            return pickle.load(model_file)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the legacy pickled XGBoost model. The bundled "
            "models/cough_classifier file was serialized with XGBoost 0.90 and "
            "is not loadable by modern XGBoost. Regenerate "
            "models/cough_classifier_migrated.json with "
            "scripts/convert_model_to_json.py, then try again."
        ) from exc


def load_scaler():
    """Load the StandardScaler used to scale the 68-feature vector."""
    if SCALER_JSON_PATH.is_file():
        with SCALER_JSON_PATH.open() as scaler_file:
            scaler = json.load(scaler_file)
        return NumpyStandardScaler(
            mean=scaler["mean_"],
            scale=scaler["scale_"],
            var=scaler.get("var_"),
            with_mean=scaler.get("with_mean", True),
            with_std=scaler.get("with_std", True),
            n_features=scaler.get("n_features"),
        )

    with SCALER_PATH.open("rb") as scaler_file:
        return pickle.load(scaler_file)
