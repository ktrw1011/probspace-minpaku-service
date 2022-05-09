import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def multiclass_metric(name:str, target:np.ndarray, pred:np.ndarray):
    if pred.ndim != 1:
        pred = np.argmax(pred, axis=1)
        
    if name == 'acc':
        return accuracy_score(target, pred)
    elif name== 'f1':
        return f1_score(target, pred)
    elif name == 'precision':
        return precision_score(target, pred)
    elif name == 'recall':
        return recall_score(target, pred)
    else:
        ValueError