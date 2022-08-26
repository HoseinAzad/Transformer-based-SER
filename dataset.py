import fnmatch
import os
import os.path
import numpy as np
import pandas as pd
import torch
import tensorflow
import torchaudio
from audiomentations import Compose, Shift
from datasets import load_dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor


def get_all_files(path, extension):
    files = []
    for root, dirNames, fileNames in os.walk(path):
        for fileName in fnmatch.filter(fileNames, '*' + extension):
            files.append(os.path.join(root, fileName))

    return files


def name_to_label(name):
    names_dict = {'A': 'anger', 'H': 'happiness', 'N': 'neutral', 'S': 'sadness', 'W': 'surprise', 'F': 'fear'}
    for key in names_dict:
        if key in name:
            return names_dict[key]


def load_data(path, extension='.wav'):
    """
    Goal : collect all data samples (audio files) with specified data type (extension) that exist in given path
        and also label samples
    Output : A pandas dataframe that includes data sample path and its label
    """
    if not os.path.isdir(path):
        print('Dataset path does not exist ')
        quit()
    else:
        audios = get_all_files(path, extension)
        # Get the file name as its label
        labels = [name_to_label(os.path.basename(path).replace(extension, '')) for path in audios]
        if len(audios) == 0:
            print('There is no sample in dataset')
            quit()
        else:
            # Encode labels to digits
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
            # Create pandas dataframe
            df = pd.DataFrame({"path": audios, "label": labels})

            return df, le


def split_data(dataframe, train_size=0.85, stratify=None):
    train_df, val_df = train_test_split(dataframe, train_size=train_size, stratify=stratify, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def make_dataset(df, train_size):
    """
    Goal : Split data
    Output : A Huggingface dataset that includes train and validation sets
    """
    # Split data to train and validation sets  with maintaining classes distribution
    train_df, val_df = split_data(df, train_size=train_size, stratify=df['label'])
    # Duplicate each data sample in the train set (in order to perform data augmentation methods)
    # --> Also you can use the below code to do random over-sampling in minority classes :
    # </> train_df = random_over_sampling(train_df)
    train_df = pd.concat([train_df, train_df], axis=0)
    # Save data to csv files
    train_csv_path = "train.csv"
    val_csv_path = "validation.csv"
    train_df.to_csv(train_csv_path, sep="\t", encoding="utf-8", index=False)
    val_df.to_csv(val_csv_path, sep="\t", encoding="utf-8", index=False)
    # Create an (huggingface) dataset
    data_files = {"train": train_csv_path, "validation": val_csv_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )

    return dataset


def random_over_sampling(data):
    over_sampler = RandomOverSampler(random_state=42, sampling_strategy={1: 152, 2: 804, 5: 762})
    x_train, y_train = over_sampler.fit_resample(data['path'].values.reshape(-1, 1), data['label'].values)
    return pd.DataFrame({"path": x_train.squeeze(), "label": y_train})


def get_feature_extractor(sampling_rate=16e3):
    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sampling_rate,
                                    padding_value=0.0, do_normalize=True, return_attention_mask=False)


def file_to_array(path, target_sampling_rate):
    array, sampling_rate = torchaudio.load(path)
    # "facebook/hubert-base-ls960" pre-trained on 16kHz sampled speech audios. So we resample all audio files to make
    #   sure the model input sampling rate is also 16k .
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    return resampler(array).squeeze().numpy()


class ShEMODataset(Dataset):

    def __init__(self, dataset, train):
        self.dataset = dataset
        self.feature_extractor = get_feature_extractor()
        self.train = train

    def __getitem__(self, item):
        path = self.dataset[item]['path']
        label = self.dataset[item]['label']
        sr = self.feature_extractor.sampling_rate
        # Convert audio file to speech (1d) array
        speech = file_to_array(path, sr)
        # audio time shift augmentation
        # --> The idea is very simple. It just shifts audio to left/right with a random second
        augment = Compose([
            Shift(min_fraction=-1, max_fraction=1, p=0.5),
            # You can shift the sudio pitch too, with below code :
            # </> PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        ])
        # Do augmentation only on train set samples.
        if self.train:
            speech = augment(samples=speech, sample_rate=sr)

        input_values = self.feature_extractor(speech,
                                              sampling_rate=sr,
                                              # truncation=True,
                                              # max_length=int(sr * 5) # means 5 second
                                              ).input_values[0]

        return input_values, label

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    """
    Goal: Make the length of all speech arrays (that exist in a batch)
        equal to the longest array ( by padding : adding zeros to the end of arrays).
    Output: A batch of data (Model input and true label ) with specified batch size
    """
    batch = np.array(batch, dtype=object)
    input_values = batch[:, 0]
    labels = batch[:, 1]

    input_features = [{"input_values": feature} for feature in input_values]
    padded_inputs = get_feature_extractor().pad(input_features, padding=True, return_tensors="pt")

    labels = np.vstack(labels).astype(np.float)
    labels = torch.from_numpy(labels).squeeze().type(torch.LongTensor)

    return padded_inputs['input_values'], labels


def get_data_loaders(dataset, train_bs, val_bs):
    train_dl = DataLoader(ShEMODataset(dataset['train'], train=True), batch_size=train_bs,
                          collate_fn=collate_fn, shuffle=True)
    val_dl = DataLoader(ShEMODataset(dataset['validation'], train=False), batch_size=val_bs,
                        collate_fn=collate_fn, shuffle=True)

    return train_dl, val_dl
