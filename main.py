from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import torch
import numpy as np
from torch import nn
from config import *
from dataset import load_data, make_dataset, get_data_loaders
from model import get_model
from utils import *


def get_classes_weight(labels):
    classes_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels),
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
            print(f'[Epoch][Batch] = [{epoch + 1}][{iter}] -> Loss = {np.mean(losses):.4f} ')

    return np.mean(losses), accuracy_score(true_labels, predictions), predictions, true_labels


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

    return np.mean(losses), accuracy_score(true_labels, predictions), predictions, true_labels


def trainModel(data_path, check_point, lr, epochs, weight_decay, sch_gamma, sch_step, train_size, title='', train_bs=2,
               val_bs=2):
    # load data
    df, label_encoder = load_data(data_path)
    print('-' * 70)
    print('Data loaded successfully')
    print('-' * 70)

    # Plot data class distribution
    # </> plot_dist(df, label_encoder.classes_)

    # create dataset
    dataset = make_dataset(df, train_size)
    print('-' * 70)
    print('Number of Train samples =', len(dataset['train']))
    print('Number of Validation samples =', len(dataset['validation']))

    # instantiate data loaders
    train_dataloader, val_dataloader = get_data_loaders(dataset, train_bs, val_bs)
    print('-' * 40)
    print('Number of Train batches =', len(train_dataloader), '| batch size =', train_bs)
    print('Number of Validation batches =', len(val_dataloader), '| batch size =', val_bs)

    # Specify processor device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-' * 40)
    print(device, 'is available')

    # Instantiate model
    num_classes = len(label_encoder.classes_)
    model = get_model(check_point, num_classes, device)
    print('-' * 40)
    print('Model loaded successfully')

    # Determine the type of : optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    fc_weights = get_classes_weight(dataset['train'][:]['label']).to(device)
    criterion = nn.CrossEntropyLoss(weight=fc_weights)

    best_acc = 0
    loss_list, acc_list = [], []

    print('-' * 40)
    print('Start Training ....\n')
    for epoch in range(epochs):

        train_loss, train_acc, _, _ = train(model, train_dataloader, optimizer, criterion, epoch, device)
        val_loss, val_acc, _, _ = evaluate(model, val_dataloader, criterion, device)
        scheduler.step()

        loss_list.append([train_loss, val_loss])
        acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, 'best-model.pt')

        print(f'\tTrain -> Loss = {train_loss:.4f} /  accuracy = {train_acc:.4f}')
        print(f'\tValidation -> Loss = {val_loss:.4f} /  accuracy = {val_acc:.4f}')

    plot_training(np.array(loss_list), np.array(acc_list), title)

    best_model = torch.load('best-model.pt')

    val_loss, val_acc, val_preds, val_labels = evaluate(best_model, val_dataloader, criterion, device)
    print('-' * 30, '\nBest result on validation set -> Loss =', val_loss, f'Accuracy = {val_acc * 100:.2f} %')
    report(val_labels, val_preds, label_encoder)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    trainModel(data_path,
               model_checkpoint,
               lr,  # learning rate
               epochs,
               wd,  # weight decay
               sc_g,  # Scheduler gamma
               sc_s,  # Scheduler step
               train_size,
               train_bs=bs,
               )
