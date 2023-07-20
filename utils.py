import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def data_distribution(df, classes_names):
    classes = np.unique(df['label'].values) # class
    values = [df['label'].values.tolist().count(class_) for class_ in classes] # frequency
    plt.figure( figsize=(10 , 6)  , dpi=100 )
    plt.bar(classes , values)
    plt.xticks(classes, classes_names , size=10)
    plt.xlabel('Class', size=12)
    plt.ylabel('Frequency', size=12)
    plt.title('Class Distribution of Dataset', size=13)
    plt.show()


def plot_training(loss_list, metric_list, title):
    # %matplotlib inline
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5) )
    fig.subplots_adjust(wspace=.2)
    plotLoss(ax1, np.array(loss_list), title)
    plotAccuracy(ax2, np.array(metric_list), title)
    plt.show()


def plotLoss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label="Train_loss")
    ax.plot(loss_list[:, 1], label="Validation_loss")
    ax.set_title("Loss Curves - " + title, fontsize=12)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})
    ax.grid()


def plotAccuracy(ax, metric_list, title):
    ax.plot(metric_list[:], label="validation_Accuracy")
    ax.set_title("Accuracy Curve - " + title, fontsize=12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})
    ax.grid()


def report(labels, preds, encoder):
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    # Decode labels (ids to class name)
    preds = encoder.inverse_transform(preds)
    labels = encoder.inverse_transform(labels)
    # Calculate accuracy for each class
    class_accuracies = []
    for class_ in encoder.classes_:
        class_acc = np.mean(preds[labels == class_] == class_)
        class_accuracies.append(class_acc)

    print( list(zip(encoder.classes_,class_accuracies)))
    print(classification_report(labels, preds, labels = encoder.classes_))
    plot_cnf_matrix(cm , encoder.classes_)


def plot_cnf_matrix(cm , classes):
    cm_df = pd.DataFrame(cm,classes,classes)                      
    plt.figure(figsize=(10,10))  
    sns.heatmap(cm_df , annot=True , cmap='Blues', fmt='g')
