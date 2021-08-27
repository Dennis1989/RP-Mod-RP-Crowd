import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
import mlflow.pyfunc as pyfunc

class PyTorchBertWrapper(pyfunc.PythonModel):
    def __init__(self, model, tokenizer, class_names=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.class_names = class_names
        
    def _tokenize(self, comment):
        return self.tokenizer(comment, truncation=True, padding=True)
    
    def predict(self, context=None, comments=[], **kwargs):
        self.model = self.model.eval()
        with torch.no_grad():
            encodings = self._tokenize(comments)
            items = {key: torch.tensor(val) for key, val in encodings.items()}
            for key in items.keys():
                kwargs[key] = items[key]
                
            preds = np.exp(self.model(kwargs).numpy())
            
            if self.class_names is not None:
                return pd.DataFrame(preds, columns=self.class_names)
            else:
                return pd.DataFrame(preds)
     
    
class PyTorchBertMlflowWrapper(pyfunc.PythonModel): 
    def __init__(self, model, tokenizer, class_names=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.class_names = class_names
        
    def _tokenize(self, comment):
        return self.tokenizer(comment, truncation=True, padding=True)
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            encodings = self._tokenize(data)
            items = {key: torch.tensor(val) for key, val in encodings.items()}
            kwargs = {}
            for key in items.keys():
                kwargs[key] = items[key]  
                    
            preds = np.exp(self.model(kwargs).numpy())
            
            if self.class_names is not None:
                return pd.DataFrame(preds, columns=self.class_names)
            else:
                return pd.DataFrame(preds)