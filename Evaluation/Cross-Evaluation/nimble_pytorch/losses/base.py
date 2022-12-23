### Credits: https://github.com/wangleiofficial/label-smoothing-pytorch/blob/main/label_smoothing.py

import torch, torch.nn as nn, torch.nn.functional as F
import math

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.modules.loss._Loss):
    def __init__(self, epsilon: float = 0.1, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_preds, target, *args, **kwargs):
        log_preds = y_preds.log()
        n = log_preds.size()[-1]
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, weight=self.weight, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
    
class WeightedMultiLabelCrossEntropyLoss(nn.modules.loss._Loss):
    def __init__(self, weight, label_smoothing=0., self_assigned=False, reduction='mean'):
        super().__init__()
        self.weight = torch.tensor(weight)
        self.label_smoothing = label_smoothing
        self.self_assigned = self_assigned
        self.reduction = reduction
        
    def forward(self, probs, targets, *args, **kwargs):
        weight = self.weight.to(targets.device)
        if len(targets.shape) >= 2 and not self.self_assigned:
            targets = (targets * weight.unsqueeze(0)).argmax(-1)
        elif len(targets.shape) >= 2 and self.self_assigned:
            targets = (targets * probs).argmax(-1)
        return F.cross_entropy(probs, targets, weight=weight, reduction=self.reduction, 
                               label_smoothing=self.label_smoothing)
    
class LabelSmoothingNLLLoss(nn.modules.loss._Loss):
    def __init__(self, epsilon: float = 0.1, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, log_preds, target, *args, **kwargs):
        n = log_preds.size()[-1]
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, weight=self.weight, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

    
class PenalizedNLLLoss(nn.modules.loss._Loss):
    def __init__(self, reduction=torch.mean):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_preds, targets, penalties, weight=None, *args, **kwargs):
        preds = log_preds.argmax(-1, keepdim=True)
        penalty = penalties.gather(1, preds).squeeze()
        loss_nll = F.nll_loss(log_preds, targets, weight=weight, reduction='none')
        return self.reduction(loss_nll * penalty)
    
class BiLinearPenalizedNLLLoss(nn.modules.loss._Loss):
    def __init__(self, alpha=.9, reduction=torch.mean):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, log_preds, targets, penalties, weight=None, *args, **kwargs):
        preds = log_preds.exp()
        penalty = self.alpha * (preds * penalties).sum(-1)
        loss_nll = (1. - self.alpha) * F.nll_loss(log_preds, targets, weight=weight, reduction='none')
        return self.reduction(loss_nll + penalty)
    

class BiLinearPenalizedBCELoss(nn.modules.loss._Loss):
    def __init__(self, alpha=.9, reduction=torch.mean):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, preds, targets, penalties, weight=None, *args, **kwargs):
        p_weights = (targets - preds).abs() * penalties
        loss_bce = F.binary_cross_entropy(preds, targets, weight=weight, reduction='none')
        return self.reduction((1. - self.alpha) * loss_bce + self.alpha * p_weights)
    
class LinearPenalizedLoss(nn.modules.loss._Loss):
    def __init__(self, entropy_coeff=.001, scaler=1., reduction=torch.mean): # 36731
        super().__init__()
        self.reduction = reduction
        self.entropy_coeff = entropy_coeff
        self.scaler = scaler

    def forward(self, preds, targets, penalties, eps=1e-8, *args, **kwargs):
        preds = preds.abs()
        log_dim = math.log2(preds.shape[1]) + eps
        entropy = (-preds * (preds + eps).log2()).sum(1) / log_dim
        
        er_preds = (targets - preds).abs()
        p_weights = (er_preds * penalties / self.scaler).sum(1)
        return self.reduction(p_weights - self.entropy_coeff * entropy)
    
class LinearPenalizedLoss2(nn.modules.loss._Loss):
    def __init__(self, entropy_coeff=.001, reduction=torch.mean):
        super().__init__()
        self.reduction = reduction
        self.entropy_coeff = entropy_coeff

    def forward(self, preds, targets, penalties, eps=1e-8, *args, **kwargs):
        preds = preds.abs()
        log_dim = math.log2(preds.shape[1]) + eps
        entropy = (-preds * (preds + eps).log2()).sum(1) / log_dim
        
        er_preds = (targets - preds).abs()
        p_weights = (er_preds * penalties / 36731).sum(1)
        return self.reduction(p_weights - self.entropy_coeff * entropy)
    
    
class BiLinearPenalizedBCELogitsLoss(nn.modules.loss._Loss):
    def __init__(self, alpha=.9, reduction=torch.mean):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, targets, penalties, weight=None, *args, **kwargs):
        p_weights = F.softmax(logits, dim=-1) * penalties
        loss_bce = F.binary_cross_entropy(torch.sigmoid(logits), targets, weight=weight, reduction='none')
        return self.reduction((1. - self.alpha) * loss_bce + self.alpha * p_weights)
    

class BilinearLoss(nn.modules.loss._Loss):
    def __init__(self, A, alpha=.9, weight=None, log=False, weighted_ce=False):
        super().__init__()
        self.A = A
        self.aplha = alpha
        self.weight = weight
        self.log = log
        self.weighted_ce = weighted_ce

    def forward(self, preds, target_score, *args, **kwargs):
        target = target_score.argmin(-1)

        if self.log:
            l_preds = torch.log(1. - preds + 1e-16)
            #loss_bl = -1. * self.aplha * torch.einsum('bp, bp -> b', self.A[target,:], l_preds).mean()
            loss_bl = -1. * self.aplha * (self.A[target,:] * l_preds).sum(-1)
        else:
            #loss_bl = self.aplha * torch.einsum('bp, bp -> b', self.A[target,:], preds).mean()
            loss_bl = self.aplha * (self.A[target,:] * preds).sum(-1)

        loss_ce = (1.-self.aplha) * F.cross_entropy(preds, target, weight=self.weight, reduction='none')
        if self.weighted_ce:
            weights = (target_score[:,0] - target_score[:,1]).abs().to(preds.device)
            loss_ce = weights * loss_ce

        return (loss_ce + loss_bl).mean()
