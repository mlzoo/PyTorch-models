import torch
import torch.nn as nn

import torchvision

class PlainBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(PlainBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
        
class PlainNet(nn.Module):
    def __init__(self, depth=18, num_classes=10):
        super(PlainNet, self).__init__()
        
        layers = {
            18 : [2, 2, 2, 2],
            34 : [3, 4, 6, 3]
        }
        
        assert depth in [18, 34], 'depth must be in [18, 34, 50, 101, 152, 200]'
        
        self.input_channels = 64
        self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, layers[depth][0])
        self.layer2 = self._make_layer(64, 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[depth][3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, input_channels, output_channels, blocks, stride=1):

        layers = []
        layers.append(PlainBlock(input_channels, output_channels, stride=stride))
        
        
        for i in range(blocks):
            layers.append(PlainBlock(output_channels, output_channels))
        return nn.Sequential(*layers)
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
#         print(x.shape)
        
        x = self.maxpool(x)
#         print(x.shape)
        
        x = self.layer1(x)
#         print(x.shape)
        
        x = self.layer2(x)
#         print(x.shape)

        x = self.layer3(x)
#         print(x.shape)

        x = self.layer4(x)
#         print(x.shape)
        
        x = self.avgpool(x)
#         print(x.shape)

        x = x.view(x.size(0), -1)
#         print(x.shape)

        x = self.fc(x)
        
        return x
