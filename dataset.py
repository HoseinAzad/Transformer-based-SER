import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
import torch
import librosa
import os, fnmatch
import os.path
import os

def get_files(path, extention):
    files = []
    for root, dirNames, fileNames in os.walk(path):
        for fileName in fnmatch.filter(fileNames, '*' + extention):
            files.append(os.path.join(root, fileName))
    return files


def name_to_label(file_name):
    name_to_label = {'A':'anger' , 'H':'happiness' , 'N':'neutral' ,
                    'S':'sadness' , 'W':'surprise' , 'F':'fear' }
    for key in name_to_label:
        if key in file_name:
            return name_to_label[key] 


def load_data(path, extention='.wav'):
    if not os.path.isdir(path):
        print('Dataset path does not exist ')
        quit()
    else:
        # Retrieve all files in chosen path with the specific extension 
        audios_path = get_files(path, extention)
        # Get the file name as its label
        labels = [name_to_label(os.path.basename(path).replace(extention, '')) for path in audios_path]
        if len(audios_path) == 0:
            print('There is no sample in dataset')
            quit()
        else:
            # Encode labels to digits
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
            df = pd.DataFrame({"path": audios_path, "label": labels})

            return df, le


def split_data(dataframe, train_size=0.85, stratify=None):
    train_data, val_data = train_test_split(dataframe, train_size = train_size, stratify = stratify, shuffle=True  )
    return train_data, val_data

class SpeechDataset(Dataset):

    def __init__(self, data , feature_extractor):
        self.data = data
        self.feature_extractor = feature_extractor
        self.tsr = 16e3 # target sampling rate

    def __getitem__(self, item):
        path = self.data.path.values[item]
        label = self.data.label.values[item]
        speech = self.file_to_array(path, self.tsr)
        input = self.feature_extractor(speech , sampling_rate = self.tsr).input_values[0]
        return input, label

    def file_to_array(self, path, sampling_rate):
        array, sr = librosa.load(path, sr= sampling_rate)
        return array

    def __len__(self):
        return len(self.data)



def collate_fn_padd(batch , feature_extractor):
    batch = np.array(batch, dtype=object)
    inputs = batch[:, 0]
    labels = batch[:, 1]
    input_features = [{"input_values": feature} for feature in inputs]
    padded_inputs = feature_extractor.pad(input_features, padding=True, return_tensors="pt")
    labels = np.vstack(labels).astype(float)
    labels = torch.from_numpy(labels).squeeze().type(torch.LongTensor)
    return padded_inputs['input_values'], labels



def get_data_loaders(train_data, val_data , train_bs  , fe):
    collate_fn = lambda batch: collate_fn_padd(batch, feature_extractor = fe )
    train_dl = DataLoader(SpeechDataset(train_data , fe ) , batch_size = train_bs, collate_fn = collate_fn, shuffle=True)
    val_dl = DataLoader(SpeechDataset(val_data , fe ) , batch_size = 2, collate_fn = collate_fn, shuffle=True)
    
    return train_dl, val_dl
