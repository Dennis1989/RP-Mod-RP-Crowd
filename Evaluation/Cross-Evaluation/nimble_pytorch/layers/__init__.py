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

        
class SelfAttentionPointCloud(nn.Module):
    def __init__(self, input_dim, embed_dim, kv_dim=None, dropout=0.1, bias=True, add_bias_kv=False):
        super().__init__()
        kv_dim = embed_dim if kv_dim is None else kv_dim
        self.dropout = dropout
        
        self.q_proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(input_dim, kv_dim, bias=add_bias_kv)
        self.v_proj = nn.Linear(input_dim, kv_dim, bias=add_bias_kv)
        
        self.x_proj = nn.Linear(input_dim, embed_dim, bias=bias) if input_dim != embed_dim else None
            
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        
        for layer in self.out_proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    
    def forward(self, x, attn_mask=None):
        # Avoid dropout in model.eval() mode!
        dropout = self.dropout if self.training else 0.
        
        # convert mask to float (copied from PyTorch repository)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        # Execute input projection and receive q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.x_proj(x) if self.x_proj is not None else x
        
        # Execute Scaled Dot Attention
        attn_output, _ = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout)
        
        # Execute output projection with (x - attention)
        attn_output = self.out_proj(x - attn_output)
        
        # Add x and attention
        return (x + attn_output)
    

class SequentialCat(nn.Module):
    def __init__(self, *layers, dim=1, concat_input=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = dim
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