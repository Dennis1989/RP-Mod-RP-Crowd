import torch.nn as nn

class BaseMetric(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y, **kwargs):
        return {'base' : 0.}