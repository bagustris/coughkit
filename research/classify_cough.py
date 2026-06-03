import os
import pandas as pd
import librosa as lb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import soundfile as sf
import argparse
import math
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score, accuracy_score, cohen_kappa_score

Scaler = MinMaxScaler(feature_range=(0,1))

def duration(dir_input):
    times = []
    for folder in os.listdir(dir_input):
        paths = os.path.join(dir_input,folder)
        for file in os.listdir(paths):
            filename = os.path.join(paths,file)
            wav,sr = lb.load(filename,sr=None)
            len_duration = lb.get_duration(y=wav,sr=sr)
            times.append(len_duration)

    max_duration = math.ceil(max(times))
    
    return max_duration
    
def append_data(csv_path, dir_input, len_duration):
    data = []
    label = []
    category = {'cough':1,'other':0}
    
    df_data = pd.read_csv(csv_path)
    
    for index,row in df_data.iterrows():
        name = row['filename']
        types = row['label']
        folder = name.split('-')[0]
        filename = os.path.join(dir_input,folder,name)
        print('Reading '+name+' ...')
        wav,sr = lb.load(filename,sr=None)
        wav = lb.util.normalize(wav)
        pad = lb.util.pad_center(data=wav,size=(sr*len_duration))
        rms = lb.feature.rms(y=pad)
        rms = rms.reshape(rms.shape[1],rms.shape[0])
        rms_norm = Scaler.fit_transform(rms)

        data.append(rms_norm)
        label.append(category[types])

    data = np.array(data)
    data = data.reshape(data.shape[0],data.shape[1])
    
    label = np.array(label)
    
    return data,label
    
def run_program(data,label):
    print('Processing...')
    model = GaussianMixture(n_components=2, random_state=42)
    model = model.fit(data)
        
    pred = model.predict(data)
    
    matrix_index = ['other', 'cough']
    
    UAR = balanced_accuracy_score(label, pred)
    accuracy = accuracy_score(label,pred)
    roc_auc = roc_auc_score(label,pred,average ='macro')
    kappa = cohen_kappa_score(pred,label)
    
    print('='*60)
    print('-'*25+'RESULT'+'-'*25)
    print('='*60)
    print('UAR: ', UAR)
    print('Accuracy: ', accuracy)
    print('ROC AUC Score: ', roc_auc)
    print('Cohen Kappa Score: ', kappa)
    print('='*60)
    print(classification_report(label, pred, target_names=matrix_index))
    
def run_all(csv_path,dir_input):
    len_dur = duration(dir_input)
    data,label = append_data(csv_path,dir_input,len_dur)
    run_program(data,label)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_path', type=str, required=True,
                        help="path of csv file")
    parser.add_argument('-i', '--dir_input', type=str, required=True,
                        help="path of directory input")
    args = parser.parse_args()
    
    run_all(args.csv_path,args.dir_input)
    
    