import torch
from torch import Tensor
from torch.nn import Module, Parameter

class BatchNormalization2D(Module):
    def __init__(self, number_features, momentum=0.1, eps=1e-5):
        super(BatchNormalization2D, self).__init__()
        self.weight = Parameter(torch.ones(number_features), requires_grad=True)
        self.bias = Parameter(torch.zeros(number_features), requires_grad=True)
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(number_features))
        self.register_buffer('running_var', torch.ones(number_features))
    
    def forward(self, x):
        def expand(input):
            return input[None, :, None, None].expand_as(x)

        if self.train:
            mean = torch.mean(x, dim=(0,2,3))
            var = torch.var(x, dim=(0,2,3))
            x = (x - expand(mean)) / torch.sqrt(self.eps + expand(var))
            self.running_mean = self.running_mean * (1 - self.momentum) + self.momentum * mean
            self.running_var = self.running_var * (1 - self.momentum) + self.momentum * var
        else:
            x = (x - expand(self.running_mean)) / \
                torch.sqrt(self.eps + expand(self.running_var))
        return x * expand(self.weight) + expand(self.bias)

