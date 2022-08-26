import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch


def plot_dist(df, classes_names):
    classes = np.unique(df['label'].values)
    counts = [df['label'].values.tolist().count(class_) for class_ in classes]
    df = pd.DataFrame({'classes': classes, 'count': counts})

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='classes', y='count', data=df)
    ax.set_xticklabels(classes_names, size=12)
    ax.set_xlabel('Class', size=13)
    ax.set_ylabel('Frequency', size=13)
    ax.set_title('Class Distribution of Dataset', size=15)
    ax.bar_label(ax.containers[0])
    plt.show()


def plot_training(loss_list, metric_list, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.subplots_adjust(wspace=.2)
    plotLoss(ax1, np.array(loss_list), title)
    plotAccuracy(ax2, np.array(metric_list), title)
    plt.show()


def plotLoss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label="Train_loss")
    ax.plot(loss_list[:, 1], label="Validation_Loss")
    ax.set_title("Loss Curves - " + title, fontsize=12)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})


def plotAccuracy(ax, metric_list, title):
    ax.plot(metric_list[:], label="Validation_Accuracy")
    ax.set_title("Accuracy Curve - " + title, fontsize=12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})


def plot_cnf_matrix(cm, classes):
    cm_df = pd.DataFrame(cm, classes, classes)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')


def report(labels, preds, encoder):
    # confusion matrix
    cm = confusion_matrix(labels, preds)
    # decode labels (ids to class name)
    preds = encoder.inverse_transform(preds)
    labels = encoder.inverse_transform(labels)
    # calculate accuracy for each class
    class_accuracies = []
    for class_ in encoder.classes_:
        class_acc = np.mean(preds[labels == class_] == class_)
        class_accuracies.append(class_acc)

    print(list(zip(encoder.classes_, class_accuracies)))
    print(classification_report(labels, preds, labels=encoder.classes_))
    plot_cnf_matrix(cm, encoder.classes_)


def collect(outputs, labels, predictions, true_labels):
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    if len(predictions) == 0:
        predictions = preds
        true_labels = labels
    else:
        predictions = np.concatenate((predictions, preds))
        true_labels = np.concatenate((true_labels, labels))

    return predictions, true_labels
