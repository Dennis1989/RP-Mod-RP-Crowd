import torch, torch.nn as nn, torch.nn.functional as F
from . import BaseMetric
import numpy as np
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

class Accuracy(BaseMetric):
    def __init__(self, key_output='seq_relationship_logits', key_input='next_sentence_label'):
        super().__init__()
        self.key_input = key_input
        self.key_output = key_output
        
    def forward(self, bert_output, bert_input, **kwargs):
        preds_labels = bert_output[self.key_output]
        preds_labels = preds_labels.argmax(-1)
        labels = bert_input[self.key_input]
        
        acc = (preds_labels == labels).sum() / labels.shape[0] * 100.
        return {'hate_acc' : acc}
    
    
class AreaUnderCurve(BaseMetric):
    def __init__(self, use_log=False, key_output='seq_relationship_logits', key_input='next_sentence_label'):
        super().__init__()
        self.key_input = key_input
        self.key_output = key_output
        self.use_log = use_log
        
    def forward(self, bert_output, bert_input, **kwargs):
        preds = bert_output[self.key_output].exp() if self.use_log else bert_output[self.key_output]
        preds = preds[:,1].cpu().numpy()
        labels = bert_input[self.key_input]
        labels = labels.cpu().numpy()
        try:
            auc = sk_roc_auc_score(labels, preds)
        except:
            auc = (np.round(preds) == labels).sum() / labels.shape[0] * 100.
        return {'hate_auc' : auc}