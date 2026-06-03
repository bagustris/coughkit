import numpy as np
import pytest

from coughkit.segmentation import segment_cough, compute_SNR


class TestSegmentCough:
    def test_empty_signal_returns_empty(self):
        segs, mask = segment_cough(np.array([]), 16000)
        assert segs == []
        assert len(mask) == 0

    def test_silence_returns_no_segments(self, silence):
        x, fs = silence
        segs, mask = segment_cough(x, fs)
        assert segs == []
        assert not np.any(mask)

    def test_cough_file_returns_at_least_one_segment(self, cough_wav):
        x, fs = cough_wav
        segs, _ = segment_cough(x, fs)
        assert len(segs) >= 1

    def test_impulse_train_detects_multiple_bursts(self, impulse_train):
        x, fs = impulse_train
        segs, _ = segment_cough(x, fs)
        assert len(segs) >= 2

    def test_mask_length_matches_signal(self, cough_wav):
        x, fs = cough_wav
        _, mask = segment_cough(x, fs)
        assert len(mask) == len(x)

    def test_mask_dtype_is_bool(self, cough_wav):
        x, fs = cough_wav
        _, mask = segment_cough(x, fs)
        assert mask.dtype == bool

    def test_segments_are_numpy_arrays(self, impulse_train):
        x, fs = impulse_train
        segs, _ = segment_cough(x, fs)
        for seg in segs:
            assert isinstance(seg, np.ndarray)

    def test_segment_minimum_length_respected(self, impulse_train):
        x, fs = impulse_train
        min_cough_len = 0.2
        segs, _ = segment_cough(x, fs, min_cough_len=min_cough_len)
        for seg in segs:
            assert len(seg) / fs >= min_cough_len - 0.05  # allow for padding


class TestComputeSNR:
    def test_silence_returns_zero_snr(self, silence):
        x, fs = silence
        assert compute_SNR(x, fs) == 0.0

    def test_cough_returns_positive_snr(self, cough_wav):
        x, fs = cough_wav
        snr = compute_SNR(x, fs)
        assert snr > 0

    def test_snr_is_finite(self, cough_wav):
        x, fs = cough_wav
        assert np.isfinite(compute_SNR(x, fs))
