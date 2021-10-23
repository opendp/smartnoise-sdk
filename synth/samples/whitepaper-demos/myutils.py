import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda'):
    '''
    Runs training loop for classification problems. Returns Keras-style
    per-epoch history of loss and accuracy over training and validation data.

    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Search space optimizer (e.g. Adam)
    loss_fn :
        Loss function (e.g. nn.CrossEntropyLoss())
    train_dl : 
        Iterable dataloader for training data.
    val_dl :
        Iterable dataloader for validation data.
    epochs : int
        Number of epochs to run
    device : string
        Specifies 'cuda' or 'cpu'

    Returns
    -------
    Dictionary
        Similar to Keras' fit(), the output dictionary contains per-epoch
        history of training loss, training accuracy, validation loss, and
        validation accuracy.
    '''

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
              (epoch+1, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

def plot_learning_curve(history):
    plt.rcParams.update(plt.rcParamsDefault)
    
    epochs = range(1, len(history['acc']) + 1)

    plt.figure(figsize=(9,9))
    
    plt.subplot(2, 1, 1)

    plt.plot(history['loss'], 'red', label='Training loss')
    plt.plot(history['val_loss'], 'green', label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(2, 1, 2)

    plt.plot(history['acc'], 'orange', label='Training accuracy')
    plt.plot(history['val_acc'], 'blue', label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and validation accuracy')

    plt.ylabel('Accuracy score')   
    plt.grid()
    
    plt.tight_layout()

    plt.show()
    return

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def print_metrics(y_test, y_pred, y_pred_prob, labels):

    accuracy_sc = round(accuracy_score(y_test, y_pred), 3)
    roc_auc_sc = round(roc_auc_score(y_test, y_pred_prob), 3)

    print(classification_report(y_test, y_pred, target_names = labels))

    print('Accuracy score:', accuracy_sc, ' --- ', 'ROC AUC score:', roc_auc_sc)

def predict_loader (model, test_loader, device):
    
    lbllist = torch.zeros(0,dtype=torch.long, device = device)
    predlist = torch.zeros(0,dtype=torch.long, device = device)
    problist = torch.zeros(0, dtype = torch.float, device = device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
                inputs = data.to(device)
                classes = target.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim = 1)
                probs = F.softmax(outputs, dim = 1)
            
                # Append batch prediction results
                lbllist = torch.cat([lbllist, classes.view(-1)])
                predlist = torch.cat([predlist, preds.view(-1)])
                problist = torch.cat([problist, probs])
    return lbllist, predlist, problist

def dptrain(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda', target_delta=3000): #=len(training_loader.dataset.samples)):
    '''
    Runs training loop for classification problems. Returns Keras-style
    per-epoch history of loss and accuracy over training and validation data.

    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Search space optimizer (e.g. Adam)
    loss_fn :
        Loss function (e.g. nn.CrossEntropyLoss())
    train_dl : 
        Iterable dataloader for training data.
    val_dl :
        Iterable dataloader for validation data.
    epochs : int
        Number of epochs to run
    device : string
        Specifies 'cuda' or 'cpu'

    Returns
    -------
    Dictionary
        Similar to Keras' fit(), the output dictionary contains per-epoch
        history of training loss, training accuracy, validation loss, and
        validation accuracy.
    '''

    print('dptrain() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['epsilon'] = []

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]
            
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta)

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f - epsilon: %5.2f, best alpha: %5.2f' % \
              (epoch+1, epochs, train_loss, train_acc, val_loss, val_acc, epsilon, best_alpha))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epsilon'].append(epsilon)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history
