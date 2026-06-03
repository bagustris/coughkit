#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segment a recording into individual cough .wav files."""

import argparse
import os

import librosa
import soundfile as sf

from coughkit.segmentation import segment_cough


def segment(input_file, dir_output='./', fs_out=16000):
    """Write one .wav per detected cough segment; return the output paths."""
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Input audio file not found: {input_file}')

    os.makedirs(dir_output, exist_ok=True)

    x, fs = librosa.load(input_file, sr=fs_out)
    cough_segments, _ = segment_cough(x, fs, cough_padding=0)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    paths = []
    for i in range(len(cough_segments)):
        output_path = os.path.join(dir_output, f'{base_name}-{i}.wav')
        sf.write(output_path, cough_segments[i], fs)
        print(f"Write to {output_path}")
        paths.append(output_path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-o', '--output_dir', help='output directory',
                        default='./', type=str)
    parser.add_argument('-fs', '--fs_out', help='output sampling rate',
                        default=16000, type=int)
    args = parser.parse_args(argv)
    segment(args.input_file, args.output_dir, args.fs_out)


if __name__ == '__main__':
    main()
