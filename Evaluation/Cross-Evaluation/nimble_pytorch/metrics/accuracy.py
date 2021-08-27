import torch, torch.nn as nn, torch.nn.functional as F
from . import BaseMetric

class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y, **kwargs):
        c_pred = y_pred.argmax(-1)
        acc = (c_pred == y).sum() / y.shape[0] * 100.
        return {'acc' : acc}