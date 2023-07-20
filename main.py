
from transformers import Wav2Vec2FeatureExtractor
from model import get_model
from dataset import load_data , split_data , get_data_loaders
from utils import data_distribution
import torch
from torch import nn
import numpy as np
import random
from config import *


def get_classes_weight(labels):
    classes_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                       classes=np.unique(labels),
                                                       y=np.array(labels))
    return torch.tensor(classes_weight, dtype=torch.float)


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


def train(model, dataloader, optimizer, criterion, epoch, device):
    # put the model on train mode
    model.train()
    losses, predictions, true_labels = [], [], []

    for iter, (inputs, labels) in enumerate(dataloader):


        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Collect predictions and true labels
        predictions, true_labels = collect(outputs, labels, predictions, true_labels)

        if iter % round((len(dataloader) / 5)) == 0:
            print(f'\r[Epoch][Batch] = [{epoch + 1}][{iter}] -> Loss = {np.mean(losses):.4f} ')

    return np.mean(losses), accuracy_score(true_labels, predictions), predictions , true_labels


def evaluate(model, dataloader, criterion, device):
    # put the model on evaluation mode
    model.eval()
    losses, predictions, true_labels = [], [], []

    for iter, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Collect predictions and true labels
        predictions, true_labels = collect(outputs, labels, predictions, true_labels)

    return np.mean(losses), accuracy_score(true_labels, predictions) , predictions , true_labels



def trainModel(data_path, check_point, lr, epocks, weight_decay, sch_gamma, sch_step,
               title='', train_bs=2  , plot_data_dist=False):
    

    # load data
    df, label_encoder = load_data(data_path)
    num_classes = len(label_encoder.classes_)
    print('Data loaded successfully') ; print('-' * 50)

    # Plot data distribution
    if plot_data_dist : data_distribution(df, label_encoder.classes_)

    # Split data to trian and validation sets
    train_data, val_data = split_data(df, stratify = df['label'])
    print('Number of train samples =', len(train_data) )
    print('Number of test samples =', len(val_data) ) ; print('-' * 50)

    print('Loading FeatureExtractor ...', end ='')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(check_point)
    print('\rFeatureExtractor loaded successfully') ; print('-' * 50)

    # Create data loaders
    train_dataloader, val_dataloader = get_data_loaders(train_data , val_data , train_bs, feature_extractor)
    print('Number of train batches =', len(train_dataloader))
    print('Number of validaion batches =', len(val_dataloader) ) ; print('-' * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'is available' ) ; print('-' * 50)

    print('Loading model ...', end ='')
    model = get_model(check_point, num_classes, device)
    print('\rModel loaded successfully') ; print('-' * 50)

    # Determine the type of : optimizer, scheduling and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    # fc_weights = get_classes_weight(train_data.label.values).to(device)
    # criterion = nn.CrossEntropyLoss(weight=fc_weights)
    criterion = nn.CrossEntropyLoss()

    print('Start Training ....',  end ='' )
    best_acc = 0 ; loss_list, acc_list = [], []
    for epock in range(epocks):
        train_loss, trian_acc , _ , _ = train(model, train_dataloader, optimizer, criterion, epock, device)
        val_loss , val_acc , _ , _ = evaluate(model, val_dataloader, criterion, device)
        # scheduler.step()
        loss_list.append([train_loss, val_loss])
        acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, 'best-model.pt')
        # print(f'\tTrain -> Loss = {train_loss:.4f} /  accuracy = {trian_acc:.4f}')
        # print(f'\tValidation -> Loss = {val_loss:.4f} /  accuracy = {val_acc:.4f}')
        plot_training(np.array(loss_list), np.array(acc_list), title)

    best_model = torch.load('best-model.pt')
    test_loss, test_acc, test_preds, test_labels = evaluate(best_model, val_dataloader , criterion, device)
    print('-' * 30, '\nBest model on validation set -> Loss =', test_loss, f'Accuracy = {test_acc * 100:.2f} %')
    report(test_labels, test_preds, label_encoder)



if __name__ == '__main__':
    random_seed=3 
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    trainModel(DATASET_PATH, HUBERT, LR, EPOCHS, WEIGHT_DECAY, SCH_GAMMA, SCH_STEP)
