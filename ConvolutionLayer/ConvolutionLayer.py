import torch
from torch.nn import Module, Parameter, functional
import numpy as np

class CustomConvolution2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConvolution2d, self).__init__()
        self.stride = stride
        self.padding = padding
        if type(kernel_size) is tuple or type(kernel_size) is list:
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]
        else:
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        
        self.out_channels = out_channels
        self.weight = Parameter(torch.empty((out_channels, in_channels, self.kernel_height, self.kernel_width)), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None
    
    def img2col(self, x):
        N, C, H, W = x.shape
        out_height = (H - self.kernel_height) // self.stride + 1
        out_width = (W - self.kernel_width) // self.stride + 1
        i0 = np.repeat(np.arange(self.kernel_height), self.kernel_width)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(self.kernel_width), self.kernel_height * C)
        j1 = self.stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.broadcast_to(np.repeat(np.arange(C), self.kernel_height * self.kernel_width).reshape(-1, 1), (C * self.kernel_height * self.kernel_width, out_height * out_width))
        cols = x[:, k, i, j].permute(0, 2, 1)
        return cols, out_height, out_width
    
    def forward(self, x):
        # x shape is [N, C, H, W]
        N = x.shape[0]
        x_padded = functional.pad(x, (self.padding,self.padding, self.padding, self.padding))
        cols, out_height, out_width = self.img2col(x_padded)
        conv_feature = torch.matmul(cols, self.weight.view(self.out_channels, -1).T)
        if self.bias is not None:
            conv_feature = conv_feature + self.bias
        feature_map = conv_feature.view(N, out_height, out_width, self.out_channels).permute(0, 3, 1, 2)
        return feature_map

