import librosa
import os
import sys
sys.path.append('../src')
from segmentation import segment_cough
import soundfile as sf

# input dir
positif = '/home/bagus/github/virufy-data/clinical/original/pos/'
negatif = '/home/bagus/github/virufy-data/clinical/original/neg/'

# output dir
seg_pos = '/home/bagus/github/virufy-data/clinical/segment/pos/'
seg_neg = '/home/bagus/github/virufy-data/clinical/segment/neg/'


# segments cough DIR 
def segment_dir(dir_input, dir_output):
    os.makedirs(dir_output, exist_ok=True)
    for i in os.listdir(dir_input):
        printf('Segmenting... ' + i)
        x, fs = librosa.load(dir_input + i, sr=16000)
        x = librosa.util.normalize(x)
        cough_segments, cough_mask = segment_cough(x, fs, cough_padding=0)
        for j in range(0,len(cough_segments)):
            sf.write(dir_output + i[:-4] + '-' + str(j) + '.wav', cough_segments[j], fs)
            print(f"Write {dir_output + i[:-4] + '-' + str(j) + '.wav'})
            

# process positive and negative files
segment_dir(positif, seg_pos)
segment_dir(negatif, seg_neg)