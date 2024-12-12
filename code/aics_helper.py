## helper functions for order
# Imports
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train(model, data, epochs=5, learn_rate=0.001, momentum=0.9):
    # own function, structure base from cifar10_tutorial.ipynb, but modified
    # everthing but the running_loss usage
    criterion = nn.BCELoss()  # match loss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
    n_batches=len(data)//data.batch_size
    for epoch in range(epochs):
        total_loss = 0
        running_loss = 0
        for i, batch in tqdm(enumerate(data), total=n_batches):
            # reset gradient
            optimizer.zero_grad()
            # Extract input and labels, make predictions
            X, y = batch
            output = model(X)

            # calculate loss, backpropagata & update gradient
            loss = criterion(output, y.reshape(-1,1))  # reshape for single dim
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % (n_batches//4) == (n_batches//4)-1:
                print(f'[{epoch+1}, {i+1:5d}] avg loss: {running_loss/(n_batches//4):.3f}',
                      end='\r')  # kept from class notebook, udpated with n_batches
                running_loss = 0.0

        print(f'\repoch: {epoch}\ttotal loss: {total_loss}\tavg loss: {total_loss/n_batches}')

    print('Finished Training')

    return model


def test(model, testdata):
    gold, preds = list(), list()
    n_batches = len(testdata)//testdata.batch_size  # new for my tqdm
    with torch.no_grad():
    ## adapted ##
        for data in tqdm(testdata, total=n_batches):
            X, y_true = data
            # calculate outputs by running images through the network
            output = model(X).round().squeeze() 
            # Store true and predicted class labels
            gold.extend(list(np.array(y_true)))
            preds.extend(list(np.array(output)))

    return gold, preds

def score_preds(gold, y_pred):
    print('Performance')
    print(f'Accuracy: {accuracy_score(gold, y_pred):.2%}',
        f'Precison: {precision_score(gold, y_pred):.2%}',
        f'Recall: {recall_score(gold, y_pred):.2%}',
        f'F1-Score: {f1_score(gold, y_pred):.2%}',
        sep='\n')