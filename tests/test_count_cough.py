from pathlib import Path

import pytest

from coughkit.cli.count import count

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_recordings"


class TestCount:
    def test_cough_file_returns_one(self):
        assert count(input_file=str(SAMPLES / "cough.wav")) == 1

    def test_not_cough_file_returns_zero(self):
        assert count(input_file=str(SAMPLES / "not_cough.wav")) == 0

    def test_artif_cough_returns_one(self):
        assert count(input_file=str(SAMPLES / "artif-cough.wav")) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            count(input_file="nonexistent.wav")

    def test_custom_threshold_zero_accepts_all_segments(self):
        # With threshold=0 every segment passes the classifier; must be >= 1
        assert count(input_file=str(SAMPLES / "cough.wav"), threshold=0.0) >= 1

    def test_custom_threshold_one_rejects_all_segments(self):
        # Nothing ever reaches p=1.0
        assert count(input_file=str(SAMPLES / "cough.wav"), threshold=1.0) == 0

    def test_verbose_does_not_raise(self, capsys):
        count(input_file=str(SAMPLES / "cough.wav"), verbose=True)
        out = capsys.readouterr().out
        assert "p=" in out
