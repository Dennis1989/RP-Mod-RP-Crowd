import torch, torch.nn as nn, torch.nn.functional as F
from . import BaseMetric

class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y, **kwargs):
        c_pred = y_pred.argmax(-1)
        acc = (c_pred == y).sum() / y.shape[0] * 100.
        return {'acc' : acc}

class MacroAccuracy(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y, **kwargs):
        c_pred = y_pred.argmax(-1)
        correct_preds = (c_pred == y)
        
        accs = []
        for cls in y.unique():
            clss = (cls == y)
            cls_acc = (correct_preds & clss).sum() / (clss.sum() + 1e-8) * 100.
            accs.append(cls_acc)
        return {'macro_acc' : torch.stack(accs).mean()}
    
class MultiLabelAccuracy(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, labels, **kwargs):
        acc = (preds.round() == labels).float().mean() * 100.
        return {'m_acc' : acc}