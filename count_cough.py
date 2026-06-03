#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count the number of cough events in an audio file or live microphone recording.

Pipeline: segment high-energy regions → classify each segment → count those
above a probability threshold.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import librosa

from src.segmentation import segment_cough
from src.DSP import classify_cough
from detect_cough import load_cough_classifier, SCALER_PATH

DEFAULT_THRESHOLD = 0.5


def _require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        sys.exit("sounddevice is not installed. Run: pip install sounddevice")


def record_mic(duration=None, fs=16000, chunk_size=0.1):
    """Record from the default microphone until Ctrl+C or *duration* seconds.

    Returns a float32 numpy array normalised to [-1, 1].
    """
    sd = _require_sounddevice()

    chunks = []
    limit_samples = int(duration * fs) if duration else None

    print("Recording… (press Ctrl+C to stop)" +
          (f" [max {duration}s]" if duration else ""))
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            blocksize=int(fs * chunk_size)) as mic:
            while True:
                audio, _ = mic.read(int(fs * chunk_size))
                chunks.append(audio.flatten())
                if limit_samples and sum(len(c) for c in chunks) >= limit_samples:
                    break
    except KeyboardInterrupt:
        print()

    if not chunks:
        return np.zeros(1, dtype=np.float32), fs

    x = np.concatenate(chunks)
    if limit_samples:
        x = x[:limit_samples]
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return x, fs


def count_from_file(input_file, fs_out=16000):
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    x, fs = librosa.load(str(input_path), sr=fs_out)
    return x, fs


def main(input_file=None, use_mic=False, duration=None, fs_out=16000,
         threshold=DEFAULT_THRESHOLD, verbose=False):
    if use_mic:
        x, fs = record_mic(duration=duration, fs=fs_out)
    else:
        x, fs = count_from_file(input_file, fs_out=fs_out)

    model = load_cough_classifier()
    with SCALER_PATH.open('rb') as f:
        scaler = pickle.load(f)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count cough events in an audio file or live microphone recording.')

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

    args = parser.parse_args()
    main(input_file=args.input, use_mic=args.mic, duration=args.duration,
         fs_out=args.fs_out, threshold=args.threshold, verbose=args.verbose)
