import torch
from torch.nn import Module

class CustomReLU(Module):
    def __init__(self, inplace=False):
        super(CustomReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        if not self.inplace:
            return torch.where(x > 0, x, torch.zeros_like(x))
        else:
            return x.relu_()

class CustomGeLU(Module):
    def __init__(self):
        super(CustomGeLU, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class CustomSigmoid(Module):
    def __init__(self):
        super(CustomSigmoid, self).__init__()
    
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class CustomSin(Module):
    def __init__(self):
        super(CustomSin, self).__init__()
    
    def forward(self, x):
        return torch.sin_(x)

        