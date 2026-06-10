"""Cough segmentation and SNR.

The hysteresis-comparator segmenter and the mask-based SNR formula originally
lived here. They now live in the shared ``audiokit`` package as
``energy_vad`` and ``compute_snr``; the functions below are thin wrappers that
preserve coughkit's historical API:

* ``segment_cough`` returns the cough audio sub-arrays (not just indices), as
  callers (``cli/segment.py``, ``cli/count.py``) expect.
* ``compute_SNR`` keeps its original name and signature.
"""

import numpy as np
from audiokit import compute_snr as _compute_snr
from audiokit import energy_vad


def segment_cough(
    x,
    fs,
    cough_padding=0.2,
    min_cough_len=0.2,
    th_l_multiplier=0.1,
    th_h_multiplier=2,
):
    """Segment a recording into individual coughs using a hysteresis comparator.

    Inputs:
    * x (np.array): cough signal
    * fs (float): sampling frequency in Hz
    * cough_padding (float): seconds added to each side of a detected cough
    * min_cough_len (float): minimum length of a segment considered a cough
    * th_l_multiplier (float): low hysteresis threshold as a multiple of RMS
    * th_h_multiplier (float): high hysteresis threshold as a multiple of RMS

    Outputs:
    * cough_segments (list of np.array): one signal array per detected cough
    * cough_mask (np.array of bool): True where a cough is in progress

    Detection is delegated to ``audiokit.energy_vad``; the sample-index
    segments it returns are sliced from ``x`` to reproduce the original
    array-returning behaviour.
    """
    x = np.asarray(x)
    segments, cough_mask = energy_vad(
        x,
        fs,
        padding=cough_padding,
        min_len=min_cough_len,
        low_mult=th_l_multiplier,
        high_mult=th_h_multiplier,
    )
    cough_segments = [x[start:end] for start, end in segments]
    return cough_segments, cough_mask


def compute_SNR(x, fs):
    """Compute the Signal-to-Noise ratio of signal ``x`` (sample rate ``fs``).

    Delegates to ``audiokit.compute_snr`` (mask-based RMS signal/noise using
    the same hysteresis segmentation).
    """
    return _compute_snr(x, fs)
