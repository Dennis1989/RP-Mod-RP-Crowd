import torch.nn as nn
import torch.nn.functional as F

class HLoss(nn.modules.loss._Loss):
    def __init__(self, key='loss'):
        self.name = 'Transformers.HLoss'
        super().__init__()
        self.key = key
        
    def forward(self, output, batch, **kwargs):
        return output[self.key]


class NLLLoss(nn.modules.loss._Loss):
    def __init__(self, logits_key='logits', label_key='labels'):
        super().__init__()
        self.name = 'Transformers.NLLLoss'
        self.logits_key = logits_key
        self.label_key = label_key
        
    def forward(self, output, batch, **kwargs):
        return F.nll_loss(output[self.logits_key], batch[self.label_key].squeeze(-1))
    
    
class DoubleLoss(nn.modules.loss._Loss):
    def __init__(self, alpha, use_log=False,
                 lm_inp_key='labels', cls_inp_key='next_sentence_label',
                 lm_out_key='prediction_logits', cls_out_key='seq_relationship_logits'):
        super().__init__()
        self.alpha = alpha
        self.name = 'Transformers.DoubleLoss'
        
        self.mlm_inp_key = lm_inp_key
        self.mlm_out_key = lm_out_key
        self.cls_inp_key = cls_inp_key
        self.cls_out_key = cls_out_key
        
        self.loss = F.nll_loss if use_log else F.cross_entropy
        
    def forward(self, output, batch, **kwargs):
        cls_loss = self.loss(output[self.cls_out_key], batch[self.cls_inp_key].squeeze(-1))
        mlm_loss = self.loss(output[self.mlm_out_key].transpose(1,-1), 
                                   batch[self.mlm_inp_key], ignore_index=-100).nan_to_num() # Due to changes in PyTorch 1.12
        return self.alpha * mlm_loss + (1-self.alpha) * cls_loss