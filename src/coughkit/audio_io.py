"""Audio I/O helpers: dataset conversion and microphone capture."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np


def _require(module_name, install_hint):
    """Import an optional dependency, exiting with an actionable message if absent."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        sys.exit(f"{module_name} is not installed. Run: pip install {install_hint}")


def convert_files(folder):
    """Convert files from .webm and .ogg to .wav.

    folder: path to the COUGHVID database and ``metadata_compiled.csv``.
    Requires ffmpeg on PATH and the optional ``train`` extra (pandas):
    ``pip install coughkit[train]``.
    """
    pd = _require("pandas", "coughkit[train]")
    folder_path = Path(folder)

    df = pd.read_csv(folder_path / "metadata_compiled.csv")
    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if counter % 1000 == 0:
            print("Finished {0}/{1}".format(counter, len(names_to_convert)))

        wav_path = folder_path / f"{name}.wav"
        source_path = next(
            (folder_path / f"{name}{suffix}"
             for suffix in (".webm", ".ogg")
             if (folder_path / f"{name}{suffix}").is_file()),
            None,
        )
        if source_path is None:
            print("Error: No file name {0}".format(name))
            continue

        subprocess.run(["ffmpeg", "-i", str(source_path), str(wav_path)],
                       check=True)


def record_mic(duration=None, fs=16000, chunk_size=0.1):
    """Record from the default microphone until Ctrl+C or *duration* seconds.

    Returns a float32 numpy array normalised to [-1, 1] and the sample rate.
    """
    sd = _require("sounddevice", "sounddevice")

    chunks = []
    sample_count = 0
    limit_samples = int(duration * fs) if duration else None
    block_samples = int(fs * chunk_size)

    print("Recording… (press Ctrl+C to stop)" +
          (f" [max {duration}s]" if duration else ""))
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            blocksize=block_samples) as mic:
            while True:
                audio, _ = mic.read(block_samples)
                chunk = audio.flatten()
                chunks.append(chunk)
                sample_count += len(chunk)
                if limit_samples and sample_count >= limit_samples:
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
