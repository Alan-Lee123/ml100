import torch
from torch.nn import Module, BatchNorm2d, Sequential, ReLU, Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d, MaxPool2d

from ConvolutionLayer.ConvolutionLayer import CustomConvolution2d

def conv3x3(in_channels, out_channels, stride):
    return CustomConvolution2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def make_conv_layer(in_channels, out_channels, kernel_size, padding):
    return Sequential(
        CustomConvolution2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        BatchNorm2d(out_channels),
        ReLU(inplace=True)
    )

class Resnet_Basic_Block(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Resnet_Basic_Block, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu2 = ReLU(inplace=True)
        if in_channels != out_channels or stride != 1:
            self.downsample = Sequential(
                CustomConvolution2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.bn1(y1)
        y3 = self.relu1(y2)
        y4 = self.conv2(y3)
        y5 = self.bn2(y4)
        if self.downsample != None:
            identity = self.downsample(x)
        else:
            identity = x
        y6 = y5 + identity
        y7 = self.relu2(y6)
        return y7

class Resnet18(Module):
    def __init__(self, num_label):
        super(Resnet18, self).__init__()
        self.conv1 = make_conv_layer(3, 64, 3, 1)
        self.pool1 = MaxPool2d(2, 2)
        self.block1 = Resnet_Basic_Block(64, 64, stride=1)
        self.block2 = Resnet_Basic_Block(64, 128, stride=2)
        self.block3 = Resnet_Basic_Block(128, 256, stride=2)
        self.block4 = Resnet_Basic_Block(256, 512, stride=2)
        self.pool5 = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, num_label)
        self.apply(self._init_weights)
    
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.pool1(y1)
        y3 = self.block1(y2)
        y4 = self.block2(y3)
        y5 = self.block3(y4)
        resnet18_feature = self.block4(y5)
        y6 = self.pool5(resnet18_feature)
        y7 = torch.flatten(y6, 1)
        out = self.fc(y7)
        return out
    
    def _init_weights(self, m):
        if isinstance(m, CustomConvolution2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out' ,nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            print('init: ', m)
        if isinstance(m, BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
            print('init: ', m)
        if isinstance(m, Resnet_Basic_Block):
            torch.nn.init.constant_(m.bn2.weight, 0)
            print("init: zero out last batchnorm's weight: ", m)


