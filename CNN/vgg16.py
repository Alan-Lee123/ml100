import torch
from torch.nn import Module

class VGG16(Module):
    def __init__(self, num_label):
        super(VGG16, self).__init__()
        self.conv_block1 = self.make_conv_block([3, 64], [64, 64])
        self.conv_block2 = self.make_conv_block([64, 128], [128, 128])
        self.conv_block3 = self.make_conv_block([128, 256, 256], [256, 256, 256])
        self.conv_block4 = self.make_conv_block([256, 512, 512], [512, 512, 512])
        self.conv_block5 = self.make_conv_block([512, 512, 512], [512, 512, 512])
        # self.fc1 = self.make_fc_layer(512, 512)
        self.fc2 = torch.nn.Linear(512, num_label)
        self.apply(self._init_weights)

    def make_conv_layer(self, in_channels, out_channels, kernel_size, padding_size):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
    
    def make_fc_layer(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.Dropout2d(),
            torch.nn.ReLU()
        )
    
    def make_conv_block(self, in_channel_list, out_channel_list):
        n = len (in_channel_list)
        conv_layers = [self.make_conv_layer(in_channel_list[k], out_channel_list[k], 3, 1) 
            for k in range(n)]
        pooling_layer = torch.nn.MaxPool2d(2, 2)
        return torch.nn.Sequential(*conv_layers, pooling_layer)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            print('init: ', m)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)
            print('init: ', m)
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
            print('init: ', m)

    
    def forward(self, x):
        y1 = self.conv_block1(x)
        y2 = self.conv_block2(y1)
        y3 = self.conv_block3(y2)
        y4 = self.conv_block4(y3)
        vgg_feature = self.conv_block5(y4)
        y5 = vgg_feature.view(vgg_feature.shape[0], -1)
        # y6 = self.fc1(y5)
        out = self.fc2(y5)
        return out
    





