import torch.nn as nn

class BaseLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y, **kwargs):
        pass
    
    
class L1Loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class MSELoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class CTCLoss(nn.CTCLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class NLLLoss(nn.NLLLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class PoissonNLLLoss(nn.PoissonNLLLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
#class GaussianNLLLoss(nn.GaussianNLLLoss):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
        
#    def forward(self, y_pred, y, **kwargs):
#        return super().forward(y_pred, y)
    
class KLDivLoss(nn.KLDivLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class BCELoss(nn.BCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class MarginRankingLoss(nn.MarginRankingLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class HingeEmbeddingLoss(nn.HingeEmbeddingLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class MultiLabelMarginLoss(nn.MultiLabelMarginLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
#class HuberLoss(nn.HuberLoss):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
        
#    def forward(self, y_pred, y, **kwargs):
#        return super().forward(y_pred, y)
    
class SmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class SoftMarginLoss(nn.SoftMarginLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class MultiLabelSoftMarginLoss(nn.MultiLabelSoftMarginLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class CosineEmbeddingLoss(nn.CosineEmbeddingLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class MultiMarginLoss(nn.MultiMarginLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class TripletMarginLoss(nn.TripletMarginLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
class TripletMarginWithDistanceLoss(nn.TripletMarginWithDistanceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, y_pred, y, **kwargs):
        return super().forward(y_pred, y)
    
