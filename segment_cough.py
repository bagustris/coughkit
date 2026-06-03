#!/usr/bin/env python3
""" program to segment cough from audio files """
# bagus@ep.its.ac.id

import argparse
import os

import librosa
import soundfile as sf

from src.segmentation import segment_cough

def main(input_file, dir_output='./', fs_out=16000):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Input audio file not found: {input_file}')

    os.makedirs(dir_output, exist_ok=True)

    x, fs = librosa.load(input_file, sr=fs_out)
    cough_segments, _ = segment_cough(x, fs, cough_padding=0)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(0, len(cough_segments)):
        output_path = os.path.join(dir_output, f'{base_name}-{i}.wav')
        sf.write(output_path, cough_segments[i], fs)
        print(f"Write to {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segment cough from audio files')
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-o', '--output_dir', help='output directory', 
                        default='./', type=str)
    parser.add_argument('-fs', '--fs_out', help='output sampling rate', 
                        default=16000, type=int)
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.fs_out)