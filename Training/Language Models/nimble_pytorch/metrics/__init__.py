import torch.nn as nn
import math

class BaseMetric(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y, **kwargs):
        return {'base' : 0.}
    
    
class EntropyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, y, eps=1e-8, **kwargs):
        preds = preds.abs()
        log_dim = math.log2(preds.shape[1]) + eps
        entropy = (-preds * (preds + eps).log2()).sum(1) / log_dim
        return {'Entropy' : entropy.mean()}