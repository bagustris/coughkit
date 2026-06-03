"""Audio I/O helpers: dataset conversion and microphone capture."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd


def convert_files(folder):
    """Convert files from .webm and .ogg to .wav.

    folder: path to the COUGHVID database and ``metadata_compiled.csv``.
    Requires ffmpeg on PATH.
    """
    df = pd.read_csv(folder + 'metadata_compiled.csv')
    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if (counter % 1000 == 0):
            print("Finished {0}/{1}".format(counter, len(names_to_convert)))
        if os.path.isfile(folder + name + '.webm'):
            subprocess.call(["ffmpeg", "-i", folder + name + ".webm", folder + name + ".wav"])
        elif os.path.isfile(folder + name + '.ogg'):
            subprocess.call(["ffmpeg", "-i", folder + name + ".ogg", folder + name + ".wav"])
        else:
            print("Error: No file name {0}".format(name))


def _require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        sys.exit("sounddevice is not installed. Run: pip install sounddevice")


def record_mic(duration=None, fs=16000, chunk_size=0.1):
    """Record from the default microphone until Ctrl+C or *duration* seconds.

    Returns a float32 numpy array normalised to [-1, 1] and the sample rate.
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
