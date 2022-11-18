import torch, torch.nn as nn, torch.nn.functional as F

class LogSoftmax(nn.Module):
    def __init__(self, dim=-1, keys=('prediction_logits', 'seq_relationship_logits')):
        super().__init__()
        self.keys = keys
        self.dim = dim
        
    def forward(self, **output):
        for k in self.keys:
            if k in output:
                output[k] = F.log_softmax(output[k], dim=self.dim)
        return output
    
    
class Softmax(nn.Module):
    def __init__(self, dim=-1, keys=('prediction_logits', 'seq_relationship_logits')):
        super().__init__()
        self.keys = keys
        self.dim = dim
        
    def forward(self, **output):
        for k in self.keys:
            if k in output:
                output[k] = F.softmax(output[k], dim=self.dim)
        return output