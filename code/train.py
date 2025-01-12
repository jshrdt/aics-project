# Train script
# Imports
import argparse

import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from classes import CI_LOADER, CIFAKE_CNN, get_files

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', help='size of train data batches',
                    default=32)
parser.add_argument('-ep', '--epochs', help='number of epochs to train on',
                    default=5)
parser.add_argument('-mf', '--modelfile', help='set filename to save model to (in ../models{modelname}.pth)',
                    default=False)
# add lr, momentum?
args, unk = parser.parse_known_args()

with open('config.json') as f:
    config = json.load(f)


def train_model(model: CIFAKE_CNN, data: CI_LOADER, epochs: int = 5,
          learn_rate: float = 0.001, momentum: float = 0.9,
          log: bool = False) -> CIFAKE_CNN:
    """Train and return CNN for binary image classification.

    Args:
        model (CIFAKE_CNN): CNN model structure
        data (list[tuple]): List of (filename, class label) items as str
        epochs (int, optional): Number of epochs. Defaults to 5.
        learn_rate (float, optional): Optimizer learn rate. Defaults to 0.001.
        momentum (float, optional): Momentum for optimizer. Defaults to 0.9.
        log (bool, optional): If true, print current avg loss at 25%
            increments per epoch. Defaults to False.

    Returns:
        CIFAKE_CNN: CNN model for binary image classification
    """
    # structure base from cifar10_tutorial.ipynb; modified
    # everthing but: running_loss usage, lr, momentum
    criterion = nn.BCELoss()  # match loss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)

    for epoch in range(epochs):
        total_loss = 0
        running_loss = 0
        for i, batch in tqdm(enumerate(data), total=len(data.batches)):
            # reset gradient
            optimizer.zero_grad()

            # Extract input and labels, make predictions
            X, y = batch
            output = model(X)

            # calculate & store loss
            loss = criterion(output, y.reshape(-1, 1))  # reshape for single dim
            running_loss += loss.item()
            total_loss += loss.item()

            # backpropagate & update gradient
            loss.backward()
            optimizer.step()

            # print current avg loss every quarter epoch
            if log and i % (len(data.batches)//4) == (len(data.batches)//4)-1:
                print(f'[{epoch+1}, {i+1:5d}] avg loss: {running_loss/(len(data.batches)//4):.3f}',
                      end='\r')  # kept from class notebook, udpated with len(data.batches)
                running_loss = 0.0

        print(f'\repoch: {epoch}\ttotal loss: {total_loss}\tavg loss: {total_loss/len(data.batches)}')

    print('Finished Training')

    return model


if __name__=='__main__':
    print('Running...')
    from tqdm import tqdm
    # Get files & create loader
    train_files = get_files(config['CIFAKE_dir'])['train']
    traindata = CI_LOADER(train_files, batch_size=int(args.batch_size))
    # Initiate & train model
    model = CIFAKE_CNN()
    model = train_model(model, traindata, epochs=int(args.epochs))

    # Optional: save to file
    if args.modelfile:
        torch.save(model.state_dict(), f"../models/{args.modelfile}.pth")
        print('Model saved to', f"../models/{args.modelfile}.pth")
    print('\n-- Finished --')