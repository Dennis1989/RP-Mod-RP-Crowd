### Credits: https://github.com/wangleiofficial/label-smoothing-pytorch/blob/main/label_smoothing.py

import torch, torch.nn as nn, torch.nn.functional as F

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
