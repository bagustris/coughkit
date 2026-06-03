import numpy as np
import pytest

from coughkit.features import features

FREQ_CUTS = [(0, 200), (300, 425), (500, 650), (950, 1150),
             (1400, 1800), (2300, 2400), (2850, 2950), (3800, 3900)]

# Expected feature counts (must match coughkit.dsp.classify_cough ordering)
EXPECTED_COUNTS = {
    "EEPD": 19,
    "ZCR": 1,
    "RMSP": 1,
    "DF": 1,
    "spectral_features": 6,
    "SF_SSTD": 2,
    "SSL_SD": 2,
    "MFCC": 26,
    "CF": 1,
    "LGTH": 1,
    "PSD": len(FREQ_CUTS),
}


@pytest.fixture
def feat_obj():
    return features(FREQ_CUTS)


@pytest.fixture
def data_1s(sine_1khz):
    return sine_1khz  # (x, fs) already in data tuple form expected by features


def _data(x, fs):
    return (fs, x)


class TestFeatureCounts:
    """Each feature function returns the advertised number of values and matching names."""

    @pytest.mark.parametrize("fn,expected", EXPECTED_COUNTS.items())
    def test_output_length(self, feat_obj, sine_1khz, fn, expected):
        x, fs = sine_1khz
        vals, names = getattr(feat_obj, fn)(_data(x, fs))
        # flatten to a scalar list the way classify_cough does
        scalars = []
        for v in vals:
            scalars.append(v[0] if isinstance(v, np.ndarray) else v)
        assert len(scalars) == expected, f"{fn}: got {len(scalars)}, want {expected}"
        assert len(names) == expected, f"{fn} names: got {len(names)}, want {expected}"

    def test_total_feature_count_is_68(self, feat_obj, sine_1khz):
        x, fs = sine_1khz
        vec = []
        for fn in EXPECTED_COUNTS:
            vals, _ = getattr(feat_obj, fn)(_data(x, fs))
            for v in vals:
                vec.append(v[0] if isinstance(v, np.ndarray) else v)
        assert len(vec) == 68


class TestFeatureValues:
    def test_zcr_silence_is_zero(self, feat_obj, silence):
        x, fs = silence
        vals, _ = feat_obj.ZCR(_data(x, fs))
        assert np.ravel(vals)[0] == pytest.approx(0.0)

    def test_rmsp_silence_is_zero(self, feat_obj, silence):
        x, fs = silence
        vals, _ = feat_obj.RMSP(_data(x, fs))
        assert np.ravel(vals)[0] == pytest.approx(0.0)

    def test_lgth_matches_duration(self, feat_obj, sine_1khz):
        x, fs = sine_1khz
        vals, _ = feat_obj.LGTH(_data(x, fs))
        assert np.ravel(vals)[0] == pytest.approx(len(x) / fs, rel=1e-3)

    def test_psd_bands_sum_to_one_or_less(self, feat_obj, sine_1khz):
        x, fs = sine_1khz
        vals, _ = feat_obj.PSD(_data(x, fs))
        assert np.sum(vals) <= 1.0 + 1e-6

    def test_cf_silence_is_zero(self, feat_obj, silence):
        x, fs = silence
        vals, _ = feat_obj.CF(_data(x, fs))
        assert np.ravel(vals)[0] == pytest.approx(0.0)

    def test_mfcc_returns_finite_values(self, feat_obj, sine_1khz):
        x, fs = sine_1khz
        vals, _ = feat_obj.MFCC(_data(x, fs))
        assert np.all(np.isfinite(vals))

    def test_spectral_features_silence_all_zero(self, feat_obj, silence):
        x, fs = silence
        vals, _ = feat_obj.spectral_features(_data(x, fs))
        assert np.all(vals == 0.0)
