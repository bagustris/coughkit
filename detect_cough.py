#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python wrapper to detect cough."""

import argparse
import pickle
from pathlib import Path

from scipy.io import wavfile
from xgboost import XGBClassifier

from src.DSP import classify_cough


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MIGRATED_MODEL_PATH = MODEL_DIR / 'cough_classifier_migrated.json'
LEGACY_MODEL_PATH = MODEL_DIR / 'cough_classifier'
SCALER_PATH = MODEL_DIR / 'cough_classification_scaler'


def load_cough_classifier():
    """Load the cough classifier with support for the migrated XGBoost model.

    The original bundled ``models/cough_classifier`` file is a Python pickle
    containing an XGBoost 0.90 in-memory Booster serialization. XGBoost 1.7+
    cannot unpickle that format directly. The migrated JSON model is the stable
    model-IO representation exported from the legacy pickle.
    """
    if MIGRATED_MODEL_PATH.is_file():
        model = XGBClassifier()
        model.load_model(str(MIGRATED_MODEL_PATH))
        return model

    try:
        with LEGACY_MODEL_PATH.open('rb') as model_file:
            return pickle.load(model_file)
    except Exception as exc:
        raise RuntimeError(
            'Failed to load the legacy pickled XGBoost model. The bundled '
            'models/cough_classifier file was serialized with XGBoost 0.90 and '
            'is not loadable by modern XGBoost. Restore or generate '
            'models/cough_classifier_migrated.json using a legacy XGBoost '
            'environment, then run detection again.'
        ) from exc


def main(input_file):
    """
    Detect cough in a given audio file
    Inputs:
        input_file: (str) path to audio file
    Outputs:
        result: (float) probability that a given file is a cough
    """
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f'Input audio file not found: {input_path}')

    model = load_cough_classifier()
    with SCALER_PATH.open('rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    fs, x = wavfile.read(str(input_path))
    prob = classify_cough(x, fs, model, scaler)
    print(f"{input_path} has probability of cough: {prob}")
    return prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Path to input audio file',
                        required=True)
    args = parser.parse_args()
    main(args.input)

