import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        w = F.relu(self.w)
        return x[0]*w[0] + x[1]*w[1]

class ACON(nn.Module):
    def __init__(self):
        super().__init__()
        self.α = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        
    def forward(self, x):
        α = F.relu(self.α)
        x = α[0] * x * torch.sigmoid(x * α[1])
        return x

class Mg(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_in, d_out//4, kernel_size = 2**(k+2)+1, padding = 2**(k+1), stride = 2, bias = False),
                nn.BatchNorm1d(d_out//4), ACON(),  
                nn.Conv1d(d_out//4, d_out//4, kernel_size = 2**(k+2)+1, padding = 2**(k+1), stride = 2, bias = False),
                nn.BatchNorm1d(d_out//4), ACON(), 
            ) for k in range(4)
        ])
        
        self.add = Add()
        
        self.idx = nn.Sequential(
            nn.Conv1d(d_in, d_out, kernel_size = 1, padding = 0, stride = 1, bias = False),
            nn.BatchNorm1d(d_out), ACON(), nn.MaxPool1d(4, 4),
        )
        
        self.point = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size = 1, padding = 0, stride = 1, bias = False),
            nn.BatchNorm1d(d_out), ACON(), 
        )
    
    def forward(self, x):
        
        i = self.idx(x)
        
        xes = []
        for conv in self.convs:
            xes.append(conv(x))
        x = torch.cat(xes, dim = 1)
        
        x = self.point(x)
        
        x = self.add([x, i])
        
        return x
    
class MgNet(nn.Module):
    def __init__(self, Stages, Proximal_categories, Distal_categories):
        super().__init__()
        
        '''
        Stages: In the original MgNet experiment, the Stages was set to [4, 16, 64, 128] and the default single channel data input, which can be modified according to actual needs.
        '''
        
        self.stages = nn.Sequential(
            OrderedDict([
                ('Stage1', Mg(1, Stages[0]) ),
                ('Stage2', Mg(Stages[0], Stages[1]) ),
                ('Stage3', Mg(Stages[1], Stages[2]) ),
                ('Stage4', Mg(Stages[2], Stages[3]) ),
            ])
        )
        
        self.FC = nn.Sequential(nn.Linear(Stages[3], Stages[3] * 2), nn.GELU())
        
        self.ProximalPredictor = nn.Sequential(OrderedDict([
            ('Proximal' + 'Dropout', nn.Dropout()),
            ('Proximal' + 'Result', nn.Linear(Stages[3] * 2, Proximal_categories))
        ]))
        
        self.DistalPredictor = nn.Sequential(OrderedDict([
            ('Distal' + 'Dropout', nn.Dropout()),
            ('Distal' + 'Result', nn.Linear(Stages[3] * 2, Distal_categories))
        ]))
        
        self.w = nn.Parameter(torch.as_tensor([1, 0], dtype=torch.float32), requires_grad=True)
        
        self.zero_last_layer_weight()
        
    def zero_last_layer_weight(self):
        self.ProximalPredictor.ProximalResult.weight.data = torch.zeros_like(self.ProximalPredictor.ProximalResult.weight)
        self.ProximalPredictor.ProximalResult.bias.data = torch.zeros_like(self.ProximalPredictor.ProximalResult.bias)
        self.DistalPredictor.DistalResult.weight.data = torch.zeros_like(self.DistalPredictor.DistalResult.weight)
        self.DistalPredictor.DistalResult.bias.data = torch.zeros_like(self.DistalPredictor.DistalResult.bias)
        
    def forward(self, x):
        
        x = self.stages(x)
            
        mu = x.mean(dim = 2)
        ma = x.max(dim = 2)[0]
        
        x = self.w[0] * mu + self.w[1] * ma
        
        x = self.FC(x)
        
        return self.ProximalPredictor(x), self.DistalPredictor(x)