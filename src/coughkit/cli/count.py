#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count cough events in an audio file or live microphone recording.

Pipeline: segment high-energy regions → classify each segment → count those
above a probability threshold.
"""

import argparse
from pathlib import Path

import librosa

from coughkit.audio_io import record_mic
from coughkit.dsp import classify_cough
from coughkit.models import load_cough_classifier, load_scaler
from coughkit.segmentation import segment_cough

DEFAULT_THRESHOLD = 0.5


def _load_file(input_file, fs_out=16000):
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    x, fs = librosa.load(str(input_path), sr=fs_out)
    return x, fs


def count(input_file=None, use_mic=False, duration=None, fs_out=16000,
          threshold=DEFAULT_THRESHOLD, verbose=False):
    """Return the number of segments classified as a cough above *threshold*."""
    if use_mic:
        x, fs = record_mic(duration=duration, fs=fs_out)
    else:
        x, fs = _load_file(input_file, fs_out=fs_out)

    model = load_cough_classifier()
    scaler = load_scaler()

    segments, _ = segment_cough(x, fs, cough_padding=0.2)

    cough_count = 0
    for i, seg in enumerate(segments):
        prob = classify_cough(seg, fs, model, scaler)
        is_cough = prob >= threshold
        if is_cough:
            cough_count += 1
        if verbose:
            label = "cough" if is_cough else "not cough"
            print(f"  segment {i+1}: {len(seg)/fs:.2f}s  p={prob:.3f}  [{label}]")

    source = "microphone" if use_mic else input_file
    print(f"{source}: {cough_count} cough(s) detected")
    return cough_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('-i', '--input', metavar='FILE',
                        help='Path to input audio file')
    source.add_argument('-m', '--mic', action='store_true',
                        help='Record from the default microphone (stop with Ctrl+C)')

    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='Max recording duration in seconds for --mic '
                             '(default: unlimited, stop with Ctrl+C)')
    parser.add_argument('-fs', '--fs_out', type=int, default=16000,
                        help='Sampling rate for loading/recording (default: 16000)')
    parser.add_argument('-t', '--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Probability threshold for classifying a segment as a '
                             f'cough (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print per-segment probability and classification')

    args = parser.parse_args(argv)
    count(input_file=args.input, use_mic=args.mic, duration=args.duration,
          fs_out=args.fs_out, threshold=args.threshold, verbose=args.verbose)


if __name__ == '__main__':
    main()
