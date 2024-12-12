# Train script
# Imports
import argparse
import os

import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from classes import CIFAKE_loader, CIFAKE_CNN

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', help='size of train data batches',
                    default=32)
parser.add_argument('-ep', '--epochs', help='number of epochs to train on',
                    default=5)
parser.add_argument('-save', '--save_model', help='number of epochs to train on',
                    action='store_true', default=False)
args, unk = parser.parse_known_args()

with open('config.json') as f:
    config = json.load(f)


def get_files(cifake_dir: str):
    """Get train/test files from cifake_dir as dict, specific to cifake dir"""
    collect = dict()
    for root, dirs, files in os.walk(cifake_dir):
        if len(files)>1:
            subdir = root.split('/')[-2]
            subclass = root.split('/')[-1]
            collect[subdir] = (collect.get(subdir, list())
                               + [(os.path.join(root, fname), subclass)
                                  for fname in files])
    return collect


def train(model: CIFAKE_CNN, data: list[tuple], epochs: int = 5,
          learn_rate: float = 0.001, momentum: float = 0.9) -> CIFAKE_CNN:
    """Train and return CNN for binary image classification.

    Args:
        model (CIFAKE_CNN): CNN model structure
        data (list[tuple]): List of (filename, class label) items as str
        epochs (int, optional): Number of epochs. Defaults to 5.
        learn_rate (float, optional): Optimizer learn rate. Defaults to 0.001.
        momentum (float, optional): Momentum for optimizer. Defaults to 0.9.

    Returns:
        CIFAKE_CNN: CNN model for binary image classification
    """
    # own function, structure base from cifar10_tutorial.ipynb, but modified
    # everthing but the running_loss usage
    criterion = nn.BCELoss()  # match loss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
    n_batches=len(data)//data.batch_size  # determine nr of batches for pretty tqdm

    for epoch in range(epochs):
        total_loss = 0
        running_loss = 0
        for i, batch in tqdm(enumerate(data), total=n_batches):
            # reset gradient
            optimizer.zero_grad()
            # Extract input and labels, make predictions
            X, y = batch
            output = model(X)

            # calculate & store loss
            loss = criterion(output, y.reshape(-1,1))  # reshape for single dim
            running_loss += loss.item()
            total_loss += loss.item()

            # backpropagate & update gradient
            loss.backward()
            optimizer.step()

            # print statistics
            if i % (n_batches//4) == (n_batches//4)-1:
                print(f'[{epoch+1}, {i+1:5d}] avg loss: {running_loss/(n_batches//4):.3f}',
                      end='\r')  # kept from class notebook, udpated with n_batches
                running_loss = 0.0

        print(f'\repoch: {epoch}\ttotal loss: {total_loss}\tavg loss: {total_loss/n_batches}')

    print('Finished Training')

    return model


if __name__=='__main__':
    # Get files & create loader
    train_files = get_files(config['CIFAKE_dir'])['train']
    traindata = CIFAKE_loader(train_files, batch_size=args.batch_size)
    # Initiate & train model
    model = CIFAKE_CNN()
    model = train(model, traindata, epochs=args.epochs)

    # Optional: save to file
    if args.save_model:
        torch.save(model.state_dict(), config['model'])
