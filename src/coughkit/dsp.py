import numpy as np
import librosa
from scipy.signal import butter,filtfilt

from .features import features

MIN_FEATURE_SAMPLES = 2048

# Frequency bands and feature-function order. This exact ordering is
# load-bearing: the bundled scaler and classifier were trained on the resulting
# 68-feature vector. scripts/train_classifier.py mirrors these constants.
FREQ_CUTS = [(0, 200), (300, 425), (500, 650), (950, 1150),
             (1400, 1800), (2300, 2400), (2850, 2950), (3800, 3900)]
FEATURE_FCT_LIST = ['EEPD', 'ZCR', 'RMSP', 'DF', 'spectral_features',
                    'SF_SSTD', 'SSL_SD', 'MFCC', 'CF', 'LGTH', 'PSD']


def classify_cough(x, fs, model, scaler):
    """Classify whether an inputted signal is a cough or not using filtering, feature extraction, and ML classification
    Inputs: 
        x: (float array) raw cough signal
        fs: (int) sampling rate of raw signal
        model: cough classification ML model loaded from file
    Outputs:
        result: (float) probability that a given file is a cough 
    """
    x = np.asarray(x)
    if x.size == 0 or np.max(np.abs(x)) == 0:
        return 0

    x,fs = preprocess_cough(x,fs)
    if x.size < MIN_FEATURE_SAMPLES:
        return 0

    data = (fs,x)
    feature_values_vec = []
    obj = features(FREQ_CUTS)
    for feature in FEATURE_FCT_LIST:
        feature_values, feature_names = getattr(obj,feature)(data)
        for value in feature_values:
            if isinstance(value,np.ndarray):
                feature_values_vec.append(value[0])
            else:
                feature_values_vec.append(value)
    feature_values_scaled = scaler.transform(np.array(feature_values_vec).reshape(1,-1))
    result = model.predict_proba(feature_values_scaled)[:,1]
    return result[0]

def preprocess_cough(x,fs, cutoff = 6000, normalize = True, filter_ = True, downsample = True):
    """
    Normalize, lowpass filter, and downsample cough samples in a given data folder 
    
    Inputs: x*: (float array) time series cough signal
    fs*: (int) sampling frequency of the cough signal in Hz
    cutoff: (int) cutoff frequency of lowpass filter
    normalize: (bool) normailzation on or off
    filter: (bool) filtering on or off
    downsample: (bool) downsampling on or off
    *: mandatory input
    
    Outputs: x: (float32 array) new preprocessed cough signal
    fs: (int) new sampling frequency
    """
    
    fs_downsample = cutoff*2
    x = np.asarray(x)
    
    #Preprocess Data
    if len(x.shape)>1:
        x = np.mean(x,axis=1)                          # Convert to mono
    x = x.astype(np.float32, copy=False)
    if normalize:
        x = x/(np.max(np.abs(x))+1e-17)                # Norm to range between -1 to 1
    if filter_ and fs > fs_downsample and x.size > 15:
        b, a = butter(4, fs_downsample/fs, btype='lowpass') # 4th order butter lowpass filter
        x = filtfilt(b, a, x)
    if downsample and fs != fs_downsample:
        x = librosa.resample(x, orig_sr=fs, target_sr=fs_downsample)
    
    fs_new = fs_downsample if downsample else fs

    return np.float32(x), fs_new
