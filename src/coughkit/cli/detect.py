#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect whether an audio file contains a cough (returns a probability)."""

import argparse
from pathlib import Path

from scipy.io import wavfile

from coughkit.dsp import classify_cough
from coughkit.models import load_cough_classifier, load_scaler


def detect(input_file):
    """Return the probability that *input_file* contains a cough sound."""
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f'Input audio file not found: {input_path}')

    model = load_cough_classifier()
    scaler = load_scaler()

    fs, x = wavfile.read(str(input_path))
    prob = classify_cough(x, fs, model, scaler)
    print(f"{input_path} has probability of cough: {prob}")
    return prob


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input audio file')
    args = parser.parse_args(argv)
    detect(args.input)


if __name__ == '__main__':
    main()
