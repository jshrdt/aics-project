## helper functions for evaluation ##
# Imports
import argparse

import json
import torch
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

from classes import CIFAKE_CNN, CI_LOADER, get_files

parser = argparse.ArgumentParser()
parser.add_argument('-mf', '--modelfile', help='set filename to load model from (in ../models{modelname}.pth)',
                    default='base_model')
parser.add_argument('-thr', '--decision_thresh', help='set decision threshold for evaluation',
                    default=0.5)

args, unk = parser.parse_known_args()

with open('./config.json') as f:
    config = json.load(f)


def test_model(model: CIFAKE_CNN, testdata: CI_LOADER) -> tuple[list]:
    """Test binary CNN model on test dataloader, return y_true and y_pred."""
    gold, preds = list(), list()
    model.eval()
    with torch.no_grad():
        for data in tqdm(testdata, total=len(testdata.batches)):
            X, y_true = data
            # calculate outputs by running images through the network
            output = model(X).squeeze()
            # Store true and predicted class labels
            gold.extend(list(np.array(y_true)))
            preds.extend(list(np.array(output)))

    return gold, preds


def score_preds(gold: list, preds: list, thresh: float = 0.5,
                per_class: bool = False, verbose: bool = False) -> tuple[float]:
    """Make binary label decision for y_pred based on threshold, 
    calculate and return evaluation metrics with y_true."""

    labs, avg = ([0,1], None) if per_class else (None, 'binary')

    y_pred = [1 if pred >= thresh else 0 for pred in preds]
    acc = accuracy_score(gold, y_pred)
    prec = precision_score(gold, y_pred, labels=labs, average=avg, zero_division=0.0)
    rec = recall_score(gold, y_pred, labels=labs, average=avg, zero_division=0.0)
    f1  = f1_score(gold, y_pred, labels=labs, average=avg)

    if per_class:
        # Add per-class measures & averages
        eval = pd.DataFrame([np.append(m, sum(m)/len(m)) for m in [prec,rec,f1]],
                            index=["precision", "recall", "f1-score"],
                            columns=['Fake', 'Real', 'Average'])
    else:
        eval = pd.DataFrame([prec,rec,f1],
                            index=["precision", "recall", "f1-score"],
                            columns=['score'])

    if verbose:
        print(f'\nPerformance (n={len(gold)} test imgs, decision threshold={thresh})')
        print(f'Overall accuracy: {acc:.2%}\n', eval, sep='\n')

    return acc, prec, rec, f1, eval


def score_content_preds(gold: list, preds: list, class_dict: dict,
                        thresh: float = 0.5, ) -> tuple[float]:
    """Scoring function for content labels from CIFAR100."""
    # make real/fake decision
    binary_preds = [1 if pred >= thresh else 0 for pred in preds]

    # here, store og content label as prediction, if 'real' was predicted,
    # to get performance per content class
    gold = [x[1] for x in gold] # content labels
    y_pred = [g if pred==1 else -1 for pred, g in zip(binary_preds, gold)]

    # get labels as idx & string for df later
    labs = list(set([int(x) for x in gold]))
    cols = [class_dict[x] if x!=-1 else 'FAKE' for x in labs]

    # Score predicitons as if content labels were predicted
    acc = accuracy_score(gold, y_pred)
    rec = recall_score(gold, y_pred, labels=labs, average=None, zero_division=0.0)
    eval = pd.DataFrame([rec], index=["recall"], columns=cols).transpose()

    print(f'\nPerformance (n={len(gold)} test imgs, decision threshold={thresh})')
    print(f'Overall accuracy: {acc:.2%}\n', eval, sep='\n')

    return acc, eval


def test_thresh_size(gold, preds):
    """Calculate and return evaluation metrics for a given set of gold
    labels and predictions for decision threshold values 0.1->0.9.
    """
    data = {}
    # Get prediction scores for each decision threshold
    for i in range(1, 10):
        thresh = round(i/10, 2)
        data[thresh] = {metric: score_preds(gold, preds, thresh=thresh)[i]
                        for i, metric in enumerate(['Accuracy', 'Precision',
                                                    'Recall', 'F1-Score'])}
        df = pd.DataFrame(data.values(), index=data.keys()).mul(100)
    return df


def visualise(thresh_data_df, title='Effect of decision threshold size on evaluation metrics'):
    """Pd df plottin func."""
    fig = thresh_data_df.plot(ylim=(0,100))
    fig.set_ylabel('Score %')
    fig.set_xlabel('Decision threshold value')
    fig.title.set_text(title)
    fig.legend(loc='lower center')
    plt.show()


if __name__=='__main__':
    print('Running...')
    from tqdm import tqdm
    # Fetch test files & get test data
    testdata = CI_LOADER(get_files(config['CIFAKE_dir'])['test'])
    # Load the model to test
    model = CIFAKE_CNN()
    model.load_state_dict(torch.load(f"../models/{args.modelfile}.pth"))
    # Make predictions & evaluate
    gold, preds = test_model(model, testdata)
    score_preds(gold, preds, thresh=float(args.decision_thresh), verbose=True)
    print('\n-- Finished --')
