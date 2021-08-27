import torch, torch.nn as nn, torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)
    

class Permute(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)

    
class InputDistributor(nn.Module):
    def __init__(self, *blocks, operator=None, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.operator = operator
        self.kwargs = kwargs
        
    def forward(self, x):
        if not isinstance(x, list) and len(x) == len(self.blocks):
            raise ValueError('Input must be of list and of same length!')
        
        x_r = []
        for i, block in enumerate(self.blocks):
            if block is not None:
                x_r.append(block.forward(x[i]))
            else:
                x_r.append(x[i])
        if self.operator is not None:
            return self.operator(x_r, **self.kwargs)
        else:
            return x_r

    
class SelfMultiHeadAttention(nn.Module):
    def __init__(self, *args, input_shape='LNE', operator=torch.add,**kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(*args, **kwargs)
        self.operator = operator
        
        if input_shape not in ['LNE', 'NLE', 'NEL']:
            raise ValueError('input_shape must be in [LNE, NLE, NEL],')
            
        if input_shape == 'NLE':
            self.permute_shape_in, self.permute_shape_out = (1,0,2), (1,0,2)
        elif input_shape == 'NEL':
            self.permute_shape_in, self.permute_shape_out = (2,0,1), (1,2,0)
        else:
            self.permute_shape_in, self.permute_shape_out = (0,1,2), (0,1,2)
    
    def forward(self, x):
        x_r = x.permute(*self.permute_shape_in)
        x_r = self.mha(x_r, x_r, x_r)[0]
        x_r = x_r.permute(*self.permute_shape_out)
        return self.operator(x, x_r)
    

class SequentialCat(nn.Module):
    def __init__(self, *layers, dim=1, concat_input=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = 1
        self.concat_input = concat_input
    
    def forward(self, x):
        out = [x]
        for layer in self.layers:
            out.append(layer(out[-1]))
        if self.concat_input:
            return torch.cat(out, dim=self.dim)
        else:
            return torch.cat(out[1:], dim=self.dim)
        
        
class SequentialShortcut(nn.Module):
    def __init__(self, *layers, operator=torch.add):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.operator = operator
    
    def forward(self, x):
        x_r = x
        for layer in self.layers:
            x_r = layer(x_r)
        return self.operator(x, x_r)

    
class GlobalMaxPoolingLayer1d(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.max(self.dim)[0]
        return x
    
    
class GlobalMaxPoolingLayer2d(nn.Module):
    def __init__(self, dim=(3,2)):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.max(self.dim[0])[0]
        x = x.max(self.dim[1])[0]
        return x
    
    
class GlobalMaxPoolingLayer3d(nn.Module):
    def __init__(self, dim=(4,3,2)):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.max(self.dim[0])[0]
        x = x.max(self.dim[1])[0]
        x = x.max(self.dim[2])[0]
        return x