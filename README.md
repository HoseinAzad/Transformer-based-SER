# Transformer-based model for Speech Emotion Recognition(SER) - Implemented by Pytorch

## Overview:
There are two classes of features to solve the emotion recognition problem from speech: lexical features (the vocabulary used) and acoustic features (sound properties). We could also use both to solve the problem. But note that using lexical or linguistic features would require having a transcript of the speech; in other words, it requires an additional step for text extraction from speech (speech recognition). Hence, in this project, we only use acoustic features.<br>

Further, there are two approaches to representing emotions:
* Dimensional Representation: Representing emotions with dimensions such as Valence (on a negative to positive scale), Activation or Energy (on a low to high scale),
and Dominance (on an active to passive scale).
* Discrete Classification: Classifying emotions in discrete labels like anger, happiness, etc.

Dimensional Representation is more elaborate and gives more information. However, due to the lack of annotated audio data in a dimensional format, we used a discrete classification approach in this project.

## Model
The model comprises two main parts: a pre-trained speech model based on transformer architecture to extract features, named [Hubert](https://arxiv.org/abs/2106.07447), and accepts a float array corresponding to the raw waveform of the speech signal. And a classifier head that takes the Hubert output and contains two linear layers and a tanh activation function.<br>
Note that loading the Hubert is performed with the help of the AutoModel class (from Huggingface ) and just by changing the model_checkpoint variable (in config.py ), you could use other architectures like Wav2vec. (for more information, read this Huggingface [document](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)).



## Dataset
I used the [ShEMO](https://arxiv.org/abs/1906.01155) (Sharif Emotional Speech Database) to train and evaluate the model.
The database includes 3000 semi-natural utterances, equivalent to 3 hours and 25 minutes of speech data extracted from online radio plays.
As you can see in the bar chart, the dataset is very imbalanced which makes classifying harder, especially in minority classes.
So we used data augmentation methods to improve the performance and accuracy of the model.
<p align="left">
 <img src="https://github.com/hoseinAzdmlki/SER/blob/master/ims/dataset_class_distribution.png" width="600"height="400" class="centerImage" >
</p>


