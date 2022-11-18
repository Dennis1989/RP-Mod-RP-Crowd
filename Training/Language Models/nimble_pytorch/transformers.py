from .models import NimbleSequential
import torch, torch.nn as nn, torch.nn.functional as F

class NimbleBert(NimbleSequential):
    def __init__(self, bert, *layers, name=None, **kwargs):
        super().__init__(name=name)
        self.sequential = nn.ModuleList([bert, *layers])
            
            
    def forward(self, inp):
        for mod in self.sequential:
            inp = mod(**inp)
        return inp
    
    
    def predict(self, inp):
        self.eval()
        with torch.no_grad():
            return self.forward(inp)
        
    
    def _on_train_epoch_start(self, **kwargs):
        self.train_history[self.current_epoch] = {'loss' : []}
        return kwargs
    
    def _on_train_epoch_end(self, **kwargs):
        kwargs['warmup_scheduler'] = None
        if 'gradually_unfreeze' in kwargs and (self.current_epoch % kwargs['gradually_unfreeze']) == 0:
            self.unfreeze(last_froozen_layer_only=True)
            self.log_train(__unfreeze=True)
        else:
            self.log_train(__unfreeze=False)
        
        return kwargs
            
    
    def _train_step(self, batch, y, batch_kwargs={}, **kwargs):
        output = self.forward(batch)
        for m in self.metrics:
            with torch.no_grad():
                self.log_train(**m(output, batch, **batch_kwargs, **kwargs))

        return self.loss(output, batch, **batch_kwargs, **kwargs)
        
    
    def _valid_step(self, batch, y, batch_kwargs={}, **kwargs):
        output = self.forward(batch)
        for m in self.metrics:
            self.log_valid(**m(output, batch, **batch_kwargs, **kwargs))

        return self.loss(output, batch, **batch_kwargs, **kwargs)
    
    
    def freeze(self):
        layers = [self.sequential[0].bert.embeddings] 
        layers += list(self.sequential[0].bert.encoder.layer)
        layers += [self.sequential[0].bert.pooler]
        
        for layer in layers:
            for params in layer.parameters():
                params.requires_grad = False
    
    
    def unfreeze(self, last_froozen_layer_only=False):
        if not last_froozen_layer_only:
            layer = self.sequential
        else:
            layers = [self.sequential[0].bert.embeddings] 
            layers += list(self.sequential[0].bert.encoder.layer)
            layers += [self.sequential[0].bert.pooler]
            
            for layer in layers[::-1]: # Iterate reverse and find params where 
                if not all([p.requires_grad for p in layer.parameters()]):
                    break
                    
        for param in layer.parameters():
                param.requires_grad = True
                