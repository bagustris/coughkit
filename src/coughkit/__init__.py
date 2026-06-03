"""coughkit — detect, segment, and count cough sounds in audio recordings."""

from .dsp import classify_cough, preprocess_cough
from .segmentation import segment_cough, compute_SNR
from .models import load_cough_classifier, load_scaler

__version__ = '0.1.0'

__all__ = [
    'classify_cough',
    'preprocess_cough',
    'segment_cough',
    'compute_SNR',
    'load_cough_classifier',
    'load_scaler',
]
