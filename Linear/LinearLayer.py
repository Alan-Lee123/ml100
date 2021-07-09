import torch
from torch.nn import Module, Parameter

class CustomLinear(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(CustomLinear,self).__init__()
        self.weight = Parameter(torch.empty((in_channels, out_channels)), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None
        
        self._init_weights()
    
    def forward(self, x):
        if self.bias is not None:
            return torch.matmul(x, self.weight) + self.bias
        else:
            return torch.matmul(x, self.weight)
    
    def _init_weights(self):
        bound = 1 / torch.sqrt(torch.tensor(self.weight.shape[0]))
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)

