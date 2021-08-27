import torch, torch.nn as nn, torch.nn.functional as F
from . import BaseMetric
import csv
import numpy as np
from bitstring import BitArray

class RelativeERT(BaseMetric):
    def __init__(self, ert_file, ert_key='ert', index_col=('fid', 'dim', 'iid'),
                 active_modules=['cma_active', 'cma_elitist', 'cma_mirrored', 'cma_orthogonal', 'cma_sequential', 
                                 'cma_threshold', 'cma_tpa', 'cma_selection', 'cma_weights_option',
                                 'cma_base_sampler', 'cma_ipop']):
        super().__init__()
        self.erts = {}
        self.name = 'RelativeERTMetric'
        
        with open(ert_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = '_'.join([row[k] for k in index_col])
                config = '_'.join([v for k, v in row.items() if k in active_modules])
                ert = float(row[ert_key])
                
                if key not in self.erts:
                    self.erts[key] = {}
                
                self.erts[key][config] = ert
         
    
    def forward(self, log_preds, target, function_keys=None, vbs_erts=1., **kwargs):
        rel_ert = -1.
        
        if function_keys is not None and isinstance(function_keys, list):
            pred_configs = log_preds.argmax(-2).cpu().tolist()
            
            predicted_erts = []
            for p_config, f_key in zip(pred_configs, function_keys):
                p_config = '_'.join([str(int(p)) for p in p_config])
                
                if f_key in self.erts and p_config in self.erts[f_key]:
                    predicted_erts.append(self.erts[f_key][p_config])
                else:
                    predicted_erts.append(1.)
                    print('### Warning: func {} with config {} could not be found!'.format(f_key, p_config))
                    
            predicted_erts = torch.Tensor(predicted_erts).to(vbs_erts.device)
            rel_ert = (predicted_erts / vbs_erts).mean().cpu().item()
        
        return {'rERT' : rel_ert}
    
    
class RelativeSingleERTMetric(BaseMetric):
    def __init__(self, ert_file, ert_key='ert', index_col=('fid', 'dim', 'iid'),
                 active_modules=['cma_active', 'cma_elitist', 'cma_mirrored', 'cma_orthogonal', 'cma_sequential', 
                                 'cma_threshold', 'cma_tpa', 'cma_selection', 'cma_weights_option',
                                 'cma_base_sampler', 'cma_ipop']):
        super().__init__()
        self.erts = {}
        self.name = 'RelativeSingleERTMetric'
        
        with open(ert_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = '_'.join([row[k] for k in index_col])
                config = BitArray([int(v) for k, v in row.items() if k in active_modules]).uint
                ert = float(row[ert_key])
                
                if key not in self.erts:
                    self.erts[key] = {}
                
                self.erts[key][config] = ert
         
    
    def forward(self, log_preds, target, function_keys=None, vbs_erts=1., **kwargs):
        rel_ert = -1.
        
        if function_keys is not None and isinstance(function_keys, list):
            pred_configs = log_preds.argmax(-1).cpu().tolist()
            
            predicted_erts = []
            for p_config, f_key in zip(pred_configs, function_keys):
                if f_key in self.erts and p_config in self.erts[f_key]:
                    predicted_erts.append(self.erts[f_key][p_config])
                else:
                    predicted_erts.append(1.)
                    print('### Warning: func {} with config {} could not be found!'.format(f_key, p_config))
                    
            predicted_erts = torch.Tensor(predicted_erts).to(vbs_erts.device)
            rel_ert = (predicted_erts / vbs_erts).mean().cpu().item()
        
        return {'rERT' : rel_ert}