# Transformer-based model for Speech Emotion Recognition(SER) - implemented by Pytorch

## Overview:
There are two classes of features to solve the emotion recognition problem from speech: lexical features (the vocabulary used) and acoustic features (sound properties). We could also use both to solve the problem. But note that using lexical or linguistic features would require having a transcript of the speech; in other words, it requires an additional step for text extraction from speech (speech recognition). Hence, in this project, we only use acoustic features.<br>

Further, there are two approaches to representing emotions:
* Dimensional Representation: Representing emotions with dimensions such as Valence (on a negative to positive scale), Activation or Energy (on a low to high scale),
and Dominance (on an active to passive scale).
* Discrete Classification: Classifying emotions in discrete labels like anger, happiness and etc.

Dimensional Representation is more elaborate and gives more information. But due to the lack of annotated audio data in a dimensional format, we used discrete classification approach in this project.

## Model
/***/



## Dataset
/***/


