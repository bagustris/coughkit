# simple cough segmentation on virufy data
# reference code: 
# https://github.com/zanjabil2502/Tugas_Akhir/blob/main/code/Automatic%20Segmentation%20for%20Cough.ipynb
import os
import librosa as lb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import soundfile as sf
import argparse

def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]

def simple_segment(audio_input,threshold):
    Scaler = MinMaxScaler(feature_range=(0,1))
    wav, fs = lb.load(audio_input, sr=None)
    wav = lb.util.normalize(wav)
    rms = lb.feature.rms(y=wav)
    times = lb.times_like(rms, sr=fs)
    rms = rms.reshape(rms.shape[1],rms.shape[0])
    rms_norm = Scaler.fit_transform(rms)
    
    index = 0
    time_cut = []
    start = []
    end = []
    for value in rms_norm:
        if value > threshold:
            time_cut.append(times[index-3])
            try:
                if rms_norm[index+1] < threshold:
                    time_cut.append(times[index+3])

                    start.append(time_cut[0])
                    end_index = len(time_cut)
                    end.append(time_cut[end_index-1])
                    time_cut = []
            except:
                time_cut.append(times[index])

                start.append(time_cut[0])
                end_index = len(time_cut)
                end.append(time_cut[end_index-1])
                time_cut = []

        index=index+1 
    return start, end, wav, fs

def make_dir(dir_output):
    types = ['pos','neg']
    if os.path.exists(dir_output)==False:
        os.mkdir(dir_output)
        for i in types:
            paths = os.path.join(dir_output,i)
            os.mkdir(paths)

def segment_dir(dir_input, dir_output,threshold):
    for folder in os.listdir(dir_input):
        paths = os.path.join(dir_input,folder)
        #print(paths)
        for file in os.listdir(paths):
            if file.endswith('.mp3'):
                filename = os.path.join(paths,file)
                print('Segmenting... ' + file)
                start, end, wav, fs = simple_segment(filename,threshold)
                #print(start,end)
                for num in range(len(start)):
                    #print(num)
                    sliced_data = slice_data(start=start[num], end=end[num], raw_data=wav, sample_rate=fs)
                    fdr = folder
                    fn = file.split('.')[0]+'_'+str(num)+'.wav'
                    path = os.path.join(dir_output,fdr,fn)
                    duration = lb.get_duration(y=sliced_data,sr=fs)
                    if duration >= 0.3 and duration <=3:
                        sf.write(file=path, data=sliced_data, samplerate=fs)
                        #print('Segmented ' + file)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dir_input', type=str, required=True,
                        help="path of directory input")
    parser.add_argument('-o', '--dir_output', type=str, required=True,
                        help="path of directory output")
    parser.add_argument('-t', '--threshold', type=float, required=True,
                        help="threshold for segmentation")
    args = parser.parse_args()
    
    dir_input = args.dir_input
    dir_output = args.dir_output
    threshold = args.threshold
    
    make_dir(dir_output)
    segment_dir(dir_input,dir_output,threshold)