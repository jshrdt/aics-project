## helper functions for testing ##
# Imports
import numpy as np

from tqdm.notebook import tqdm
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from classes import CIFAKE_CNN, CIFAKE_loader

def test_model(model: CIFAKE_CNN, testdata: CIFAKE_loader) -> tuple[list]:
    """Test binary CNN model on test dataloader, return y_true and y_pred."""
    gold, preds = list(), list()
    n_batches = len(testdata)//testdata.batch_size
    with torch.no_grad():
        for data in tqdm(testdata, total=n_batches):
            X, y_true = data
            # calculate outputs by running images through the network
            output = model(X).squeeze()
            # Store true and predicted class labels
            gold.extend(list(np.array(y_true)))
            preds.extend(list(np.array(output)))

    return gold, preds


def score_preds(gold: list, preds: list, thresh: float = 0.5,
                verbose: bool = False) -> tuple[float]:
    """Make binary label decision for y_pred based on threshold, 
    calculate and return evaluation metrics with y_true."""

    y_pred = [1 if pred >= thresh else 0 for pred in preds]
    acc = accuracy_score(gold, y_pred)
    prec = precision_score(gold, y_pred)
    rec = recall_score(gold, y_pred)
    f1  = f1_score(gold, y_pred)

    if verbose:
        print('Performance')
        print(f'Accuracy: {acc:.2%}',
              f'Precison: {prec:.2%}',
              f'Recall: {rec:.2%}',
              f'F1-Score: {f1:.2%}',
              sep='\n')

    return acc, prec, rec, f1