from pathlib import Path

import librosa
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "data" / "sample_recordings"
FS = 16000  # common test sampling rate


# ── Synthetic signals ──────────────────────────────────────────────────────

@pytest.fixture
def silence():
    """One second of silence."""
    return np.zeros(FS, dtype=np.float32), FS


@pytest.fixture
def sine_1khz():
    """One second of a 1 kHz sine wave, normalised to [-1, 1]."""
    t = np.linspace(0, 1, FS, endpoint=False)
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    return x, FS


@pytest.fixture
def impulse_train():
    """Short bursts of noise separated by silence — mimics multiple coughs."""
    rng = np.random.RandomState(42)
    x = np.zeros(FS * 5, dtype=np.float32)
    for center in [0.5, 1.5, 2.5, 3.5]:          # 4 bursts
        s = int(center * FS)
        burst = rng.randn(int(0.3 * FS)).astype(np.float32)
        x[s:s + len(burst)] += burst * 0.9
    x /= np.max(np.abs(x))
    return x, FS


# ── Real sample audio ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cough_wav():
    x, fs = librosa.load(str(SAMPLES / "cough.wav"), sr=FS)
    return x, fs


@pytest.fixture(scope="session")
def not_cough_wav():
    x, fs = librosa.load(str(SAMPLES / "not_cough.wav"), sr=FS)
    return x, fs


@pytest.fixture(scope="session")
def artif_cough_wav():
    x, fs = librosa.load(str(SAMPLES / "artif-cough.wav"), sr=FS)
    return x, fs


# ── Model / scaler ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def classifier():
    from coughkit.models import load_cough_classifier
    return load_cough_classifier()


@pytest.fixture(scope="session")
def scaler():
    from coughkit.models import load_scaler
    return load_scaler()
