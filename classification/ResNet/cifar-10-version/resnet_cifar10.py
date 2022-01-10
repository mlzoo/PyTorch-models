import torch
import torch.nn as nn

import torchvision

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, projection=None):
        super(BasicBlock, self).__init__()
        
        
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.projection = projection
        
        
    def forward(self, x):
        input = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.projection:
            input = self.projection(input)
        
        x = self.relu(x + input)
        
        return x
        
class ResNetCifar10(nn.Module):
    def __init__(self, depth=18, num_classes=10):
        super(ResNetCifar10, self).__init__()
        
        layers = {
            18 : [2, 2, 2, 2],
            34 : [3, 4, 6, 3],
        }
        
        assert depth in [18, 34], 'depth must be in [18, 34]'
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, 16, 2)
        self.layer2 = self._make_layer(16, 32, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(32, 64, layers[depth][2], stride=2)
        
        self.avgpool = nn.AvgPool2d(8)
        
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, input_channels, output_channels, blocks, stride=1):
        
        projection = None
        
        if stride != 1 or input_channels != output_channels:
            projection = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        
        layers = []
        layers.append(BasicBlock(input_channels, output_channels, stride=stride, projection=projection))
        
        for i in range(blocks):
            layers.append(BasicBlock(output_channels, output_channels))
            
            
        return nn.Sequential(*layers)
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
#         print(x.shape)
        
        x = self.layer1(x)
#         print(x.shape)
        
        x = self.layer2(x)
#         print(x.shape)

        x = self.layer3(x)
#         print(x.shape)

        x = self.avgpool(x)
#         print(x.shape)

        x = x.view(x.size(0), -1)
#         print(x.shape)

        x = self.fc(x)
        
        return x
