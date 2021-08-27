import torch, torch.nn as nn, torch.nn.functional as F

class GRUHead(nn.Module):
    def __init__(self, features, modules, states=2, num_gru_layers=1, dropout=0., bidirectional=False):
        super().__init__()
        directions = 2 if bidirectional else 1
        self.rnn_block = nn.GRU(features, features // directions, 
                                num_layers=num_gru_layers, 
                                dropout=dropout, 
                                bidirectional=bidirectional)
        
        self.linears = nn.ModuleList([nn.Linear(features, states) for _ in range(modules)])
        self.dropout = nn.Dropout(dropout)
        self.modules = modules
        
    def forward(self, x):
        x = x.repeat(self.modules,1,1) ## Append Modules Dim
        x, _ = self.rnn_block(x) ## Apply GRU Layer(s)
        x = torch.split(x, 1) ## Split tensor into modules
        x = [o.squeeze(0) for o in x] ## Remove outer Dim
        x = [self.dropout(o) for o in x] ## Apply Dropout to final layers
        x = [l(o) for l, o in zip(self.linears, x)] ## Apply Linear Layers
        return torch.stack(x, -1) ## Stack and return
    
    
class GRUHeadV2(nn.Module):
    def __init__(self, features, modules, states=2, num_gru_layers=1, dropout=0., bidirectional=False):
        super().__init__()
        directions = 2 if bidirectional else 1
        self.rnn_block = nn.GRU(features, features // directions, 
                                num_layers=num_gru_layers, 
                                dropout=dropout, 
                                bidirectional=bidirectional)
        
        self.conv1d = nn.Conv1d(features, states, 1)
        self.dropout = nn.Dropout(dropout)
        self.modules = modules
        
    def forward(self, x):
        x = x.repeat(self.modules,1,1) ## Append Modules Dim
        x, _ = self.rnn_block(x) ## Apply GRU Layer(s)
        x = self.dropout(x.permute(1,2,0)) ## Move Modules Dim last and apply dropout
        x = self.conv1d(x)
        return x
    
class GRUHeadV3(nn.Module):
    def __init__(self, features, modules, states=2, num_gru_layers=1, dropout=0., bidirectional=False):
        super().__init__()
        directions = 2 if bidirectional else 1
        self.rnn_block = nn.GRU(features, features // directions, 
                                num_layers=num_gru_layers, 
                                dropout=dropout, 
                                bidirectional=bidirectional)
        
        self.conv1d = nn.Conv1d(features, states, 1)
        self.dropout = nn.Dropout(dropout)
        self.modules = modules
        
    def forward(self, x):
        x = x.repeat(self.modules,1,1) ## Append Modules Dim
        x1, _ = self.rnn_block(x) ## Apply GRU Layer(s)
        x, x1 = x.permute(1,2,0), x1.permute(1,2,0) ## Move Modules Dim last
        x = x * F.softmax(x1, dim=-1)
        x = F.relu(self.dropout(x)) ## Apply dropout
        x = self.conv1d(x)
        return x