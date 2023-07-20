# Speech Emotion Recognition(SER) - with huggingface

## Overview:
To solve the emotion recognition problem with speech, two classes of features are lexical features (the vocabulary used) and acoustic features (sound properties).
We could also use both to solve the problem. But note that using lexical features would require having a transcript of the speech; in other words, it requires an 
additional step for text extraction from speech (speech recognition). Hence, we choose to analyze the acoustic features in this work.<br>

Further, there are two approaches to representing emotions:
* Dimensional Representation: Representing emotions with dimensions such as Valence (on a negative to positive scale), Activation or Energy (on a low to high scale),
and Dominance (on an active to passive scale).
* Discrete Classification: Classifying emotions in discrete labels like anger, happiness and etc.

Dimensional Representation is more elaborate and gives more information. But due to the lack of annotated audio data in a dimensional format, we used discrete classification approach in this project.

## Dataset
We used the [ShEMO](http://saliency.mit.edu/results_cat2000.html) (Sharif Emotional Speech Database) dataset in this project to train and evaluate the model.
The database includes 3000 semi-natural utterances, equivalent to 3 hours and 25 minutes of speech data extracted from online radio plays.

 <img src="https://github.com/hoseinAzdmlki/SER/blob/master/ims/dataset_class_distribution.png" 
 width="500"
 height="400" 
 class="centerImage" 
 background="WITH">
 
As you can see in the bar chart, the dataset is very imbalanced, which makes classifying harder for the model, especially in minority classes.
So we used data augmentation methods to improve the performance and accuracy of the model.
 


