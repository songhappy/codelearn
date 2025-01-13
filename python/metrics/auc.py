import numpy as np
import torchmetrics
from bigdl.orca.learn.pytorch.pytorch_metrics import PytorchMetric
from matplotlib import pyplot
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

import torch


class AUC(PytorchMetric):
    def __init__(self,
                 reorder: bool = True,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False):
        self.internalauc = torchmetrics.AUC(reorder, compute_on_step, dist_sync_on_step)

    def __call__(self, preds, targets):
        self.internalauc.update(preds, targets)

    def compute(self):
        return self.internalauc.compute()


def sk_auc():
    # generate 2 class dataset
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
    # split into train/test sets
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


if __name__ == '__main__':
    prediction = [0.3, 0.4, 0.2, 0.5, 0.6, 0.7, 0.8]
    pred_labels = [1, 0, 1, 1, 0, 1, 1]
    gr = [0, 1, 0, 1, 1, 1, 1.0]
    arr = list(zip(prediction, gr))
    auc = calc_auc(arr)
    fpr, tpr, _ = roc_curve(gr, prediction)
    # pyplot.plot(fpr, tpr, marker='.', label='Sample Data')
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()

    print(auc)

    print("*******")
    pytorchAUC = AUC(reorder=True)
    preds = torch.from_numpy(np.array(prediction))
    target = torch.from_numpy(np.array(gr))

    sci_auc = roc_auc_score(target, preds)
    # pytorchAUC(preds, target)
    # print(pytorchAUC.compute())

    auc = torchmetrics.functional.auc(preds.reshape(7, 1), target.reshape(7, 1), reorder=True)
    print("*******")
    print(sci_auc)
    print(auc)

    pytorchauc = torchmetrics.AUROC()
    pytorchauc(preds, target.to(torch.int))
    print("pytorch", pytorchauc.compute())

    print("*******")
    acc = accuracy_score(gr, pred_labels)
    pytorchacc = torchmetrics.Accuracy()
    pytorchacc(torch.from_numpy(np.array(pred_labels)), target.to(torch.int))
    print(pytorchacc.compute())
    print(acc)
