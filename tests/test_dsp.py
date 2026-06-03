import numpy as np
import pytest

from coughkit.dsp import preprocess_cough, classify_cough


class TestPreprocessCough:
    def test_output_dtype_is_float32(self, sine_1khz):
        x, fs = sine_1khz
        x_out, _ = preprocess_cough(x, fs)
        assert x_out.dtype == np.float32

    def test_downsamples_to_12khz(self, sine_1khz):
        x, fs = sine_1khz
        _, fs_out = preprocess_cough(x, fs)
        assert fs_out == 12000

    def test_normalises_to_unit_range(self, sine_1khz):
        x, fs = sine_1khz
        x_out, _ = preprocess_cough(x * 500, fs)   # large amplitude input
        # filtering/resampling can slightly overshoot; allow 1% headroom
        assert np.max(np.abs(x_out)) <= 1.01

    def test_converts_stereo_to_mono(self):
        rng = np.random.RandomState(0)
        stereo = rng.randn(16000, 2).astype(np.float32)
        x_out, _ = preprocess_cough(stereo, 16000)
        assert x_out.ndim == 1

    def test_already_at_target_rate_unchanged_length(self):
        x = np.random.randn(12000).astype(np.float32)
        x_out, fs_out = preprocess_cough(x, 12000)
        assert fs_out == 12000
        assert len(x_out) == len(x)


class TestClassifyCough:
    def test_returns_float_in_unit_interval(self, cough_wav, classifier, scaler):
        x, fs = cough_wav
        p = classify_cough(x, fs, classifier, scaler)
        assert isinstance(float(p), float)
        assert 0.0 <= float(p) <= 1.0

    def test_cough_file_high_probability(self, cough_wav, classifier, scaler):
        x, fs = cough_wav
        assert classify_cough(x, fs, classifier, scaler) >= 0.5

    def test_not_cough_file_low_probability(self, not_cough_wav, classifier, scaler):
        x, fs = not_cough_wav
        assert classify_cough(x, fs, classifier, scaler) < 0.5

    def test_silence_returns_zero(self, classifier, scaler):
        p = classify_cough(np.zeros(16000, dtype=np.float32), 16000, classifier, scaler)
        assert p == 0

    def test_empty_returns_zero(self, classifier, scaler):
        p = classify_cough(np.array([]), 16000, classifier, scaler)
        assert p == 0
