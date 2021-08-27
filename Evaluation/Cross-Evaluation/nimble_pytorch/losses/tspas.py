import torch, torch.nn as nn, torch.nn.functional as F

class Par10Weighted(nn.modules.loss._Loss):
    def __init__(self, weight=None, log1p=False):
        super().__init__()
        self.weight = weight
        self.log1p = log1p

    def forward(self, log_preds, target_scores, num_town=None, num_town_eps=1e4, **kwargs):
        with torch.no_grad():
            weights = (target_scores[:,0] - target_scores[:,1]).abs().to(log_preds.device)
            if self.log1p: weights = torch.log(1. + weights)
            target = target_scores.argmin(-1).to(log_preds.device)
            num_town = num_town.to(log_preds.device) if num_town is not None else None

        loss = F.nll_loss(log_preds, target, weight=self.weight, reduction='none')
        loss = num_town_eps * weights * loss / num_town if num_town is not None else weights * loss
        return loss.mean()

    
class Par10Loss(nn.modules.loss._Loss):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta

    def forward(self, log_preds, target_scores, **kwargs):
        with torch.no_grad():
            vbs_scores = target_scores.min(-1)[0].to(log_preds.device)
            weights = (target_scores[:,0] - target_scores[:,1]).abs().to(log_preds.device)

        preds = log_preds.exp()
        preds_score = (preds * target_scores).sum(-1)
        loss = F.smooth_l1_loss(preds_score, vbs_scores, reduction='none', beta=self.beta)
        loss = weights * loss
        return loss.mean()