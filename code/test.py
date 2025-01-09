## helper functions for testing ##
# Imports
print('Running...')
import argparse
import numpy as np
import pandas as pd
import json

from tqdm.notebook import tqdm
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from classes import CIFAKE_CNN, CIFAKE_loader
from train import get_files

parser = argparse.ArgumentParser()
parser.add_argument('-mf', '--modelfile', help='set filename to load model from (in ../models{modelname}.pth)',
                    default='base_model')
parser.add_argument('-thr', '--decision_thresh', help='set decision threshold for evaluation',
                    default=0.5)

args, unk = parser.parse_known_args()

with open('config.json') as f:
    config = json.load(f)


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
        print(f'\nPerformance (n={len(gold)} test imgs, decision threshold={thresh})')
        print(f'Accuracy:\t{acc:.2%}',
              f'Precison:\t{prec:.2%}',
              f'Recall: \t{rec:.2%}',
              f'F1-Score:\t{f1:.2%}',
              sep='\n')

    return acc, prec, rec, f1


def test_thresh_size(gold, preds):
    """Calculate and return evaluation metrics for a given set of gold
    labels and predictions for decision threshold values 0.1->0.9.
    """
    data = {}
    for i in range(1, 10):
        thresh = round(i/10, 2)
        data[thresh] = {metric: score_preds(gold, preds, thresh=thresh)[i]
                        for i, metric in enumerate(['Accuracy', 'Precision',
                                                    'Recall', 'F1-Score'])}
        df = pd.DataFrame(data.values(), index=data.keys()).mul(100)
    return df

if __name__=='__main__':
    from tqdm import tqdm
    # Fetch test files & get test data
    testfiles = get_files(config['CIFAKE_dir'])['test']
    testdata = CIFAKE_loader(testfiles, batch_size=32)
    # Load the model to test
    model = CIFAKE_CNN()
    model.load_state_dict(torch.load(f"../models/{args.modelfile}.pth"))
    # Make predictions & evaluate
    gold, preds = test_model(model, testdata)
    score_preds(gold, preds, thresh=float(args.decision_thresh), verbose=True)
    print('\n-- Finished --')
