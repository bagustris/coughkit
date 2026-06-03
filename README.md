# Detect and Segment Cough
This repository hosts code and models to **detect** and **segment** cough sounds. Detecting a cough returns the probability that an audio file contains cough sounds. Segmenting coughs writes separate WAV files for individual detected cough events in an input recording.


# Input-output

- Input: audio files (`.wav`) that may contain one or more cough sounds.
- Output:
  - Detection: cough probability for the input file.
  - Segmentation: new WAV files containing individual cough segments.

# **Supported Python Version and Model** (IMPORTANT!)

The current `requirements.txt` targets modern Python versions, including Python 3.13, and uses CPU-only XGBoost via `xgboost-cpu`.

The original bundled classifier pickle (`models/cough_classifier`) was produced with XGBoost 0.90, which cannot be unpickled by modern XGBoost. Runtime detection now loads `models/cough_classifier_migrated.json`, a stable model-IO export migrated from that legacy pickle. The scaler (`models/cough_classification_scaler`) is still an older `scikit-learn` pickle, so Python may warn when loading it with newer packages. Treat model outputs from a newer runtime as compatibility-sensitive and validate them against known recordings before production use.

# Installation

Install the package (and its dependencies) into a virtual environment with pip or uv. An editable install (`-e`) is recommended so the bundled models in `models/` resolve correctly:

```
# Inference-only (minimal dependencies):
pip install -e .
# or
uv pip install -e .

# With development tools (pytest, tqdm):
pip install -e '.[dev]'

# With training tools (pandas, required for scripts/train_classifier.py):
pip install -e '.[train]'
```

This installs the `coughkit` package plus three console commands: `cough-detect`, `cough-segment`, and `cough-count`. The core dependencies use `xgboost-cpu`, so GPU-only packages such as `nvidia-nccl-cu12` are not required for CPU inference.

## API/Command Line Usage

```
# Detect cough (probability that the file contains a cough):
cough-detect -i input_file.wav

# Segment coughs into a chosen output directory:
cough-segment -i input_file.wav -o output_segments/

# Count cough events in a file:
cough-count -i input_file.wav

# Count cough events from the microphone (stop with Ctrl+C):
cough-count -m
```

Each command is also runnable as a module, e.g. `python -m coughkit.cli.detect -i input_file.wav`.

## Python Usage
```
from coughkit import classify_cough, load_cough_classifier, load_scaler
from scipy.io import wavfile

input_file = './data/sample_recordings/cough.wav'

model = load_cough_classifier()
scaler = load_scaler()

fs, x = wavfile.read(input_file)
prob = classify_cough(x, fs, model, scaler)
print(f"{input_file} has probability of cough: {prob}")
```
# Example
```
# detect cough:
cough-detect -i data/sample_recordings/cough.wav
data/sample_recordings/cough.wav has probability of cough: 0.995194137096405

# segment cough
cough-segment -i data/sample_recordings/cough.wav -o output_segments/
Write to output_segments/cough-0.wav
Write to output_segments/cough-1.wav
```

You may see an `InconsistentVersionWarning` from `scikit-learn` when loading the legacy scaler. This warning is expected with modern environments and does not prevent detection from running.

# Overview: 

In the wake of the COVID-19 pandemic, mass coronavirus testing has proven essential to governments in monitoring the spread of the disease, isolating infected individuals, and effectively “flattening the curve” of infections over time. However, this oropharyngeal swab test is physically invasive and must be performed by a trained clinician. This requires patients to travel to a laboratory facility to get tested, thereby potentially infecting others along the way. Ideally, testing would be performed noninvasively at no cost, and administered at the homes of potential patients to minimize contamination risk.

The World Health Organization (WHO) has reported that 67.7% of COVID-19 patients exhibit a “dry cough,” which may be audibly different from coughs caused by other pathologies. Such cough sounds analysis has proven successful in diagnosing respiratory conditions like pertussis, asthma, and pneumonia.

At the Embedded Systems Laboratory (ESL) at EPFL, we have developed the COUGHVID database, which is an extensive dataset of COVID-19 cough sounds from around the world, partially validated by expert pulmonologists. We contribute our data, signal preprocessing source code, cough classification algorithm, and feature extraction methods to assist the global research community in developing algorithms to automatically screen for COVID-19 based on cough sounds.

# Data access

The COUGHVID dataset can be downloaded from the following Zenodo link: https://doi.org/10.5281/zenodo.4048311

## Notebooks
The `coughvid_classification_example.ipynb` notebook illustrates the usage of the cough classifier model for removing unwanted recordings from a cough database.

The `segmentation_and_SNR_example.ipynb` notebook is an example of how to use the automatic cough segmentation and SNR estimation algorithm.

## Source code

The reusable library lives in `src/coughkit/`:

- `coughkit/dsp.py` — signal preprocessing plus `classify_cough` (filter → 68-feature extraction → scale → XGBoost probability).
- `coughkit/features.py` — the `features` class computing all audio features used for cough classification.
- `coughkit/segmentation.py` — `segment_cough` (hysteresis comparator) and `compute_SNR`.
- `coughkit/models.py` — loaders for the bundled classifier and scaler.
- `coughkit/audio_io.py` — COUGHVID `.webm`/`.ogg` → `.wav` conversion (requires ffmpeg) and microphone capture.
- `coughkit/cli/` — the `cough-detect`, `cough-segment`, and `cough-count` entry points.

Offline tooling lives in `scripts/`:

- `scripts/train_classifier.py` — reproduce the classifier + scaler from the COUGHVID dataset.
- `scripts/convert_model_to_json.py` — migrate the legacy XGBoost 0.90 pickle to modern JSON.

## Models

- `models/cough_classifier_migrated.json`: modern XGBoost model loaded at runtime.
- `models/cough_classifier`: original legacy XGBoost 0.90 pickle, kept for provenance/backward compatibility.
- `models/cough_classification_scaler`: feature scaler used before classification.

Use `coughkit.load_cough_classifier()` instead of loading `models/cough_classifier` directly. Loading the legacy pickle with modern XGBoost will fail.


# Citation 

When using this resource, please cite the following publication: 


1. Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Sci Data* **8,** 156 (2021). https://doi.org/10.1038/s41597-021-00937-4 

2. B. T. Atmaja, Zanjabila, Suyanto, and A. Sasou, “Comparing hysteresis comparator and RMS threshold methods for automatic single cough segmentations,” Int. J. Inf. Technol., no. 0123456789, Dec. 2023, doi: 10.1007/s41870-023-01626-8.

# Reference  
1. https://c4science.ch/diffusion/10770/  (original repo forked from)

# Changelog  
- 21/03/2022: Submit paper to interspeech about cough segmentation, commit: f330c2fb90431c736ee495b668ac0b0e0994b0cf  
- 08/02/2022: Rename repo from `detect-cough` to `detect-segment-cough`
- 03/02/2022: Initial version, forked from EPFL's original repo
