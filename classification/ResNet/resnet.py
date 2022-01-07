import math
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(input_channels, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channels, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x
        # print('bottle x:', x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('bottle conv1:', out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print('bottle conv2:', out.shape)
        
        out = self.conv3(out)
        out = self.bn3(out)
        # print('bottle conv3:', out.shape)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            # print('bottle downsample:', residual.shape)
        out += residual
        out = self.relu(out)
        # print('bottle f(x)+x:', out.shape)
        return out

class ResNet(nn.Module):
    def __init__(self, depth=18, num_classes=10, bottleneck=False):
        super(ResNet, self).__init__()
        
        blocks = {
            18: BasicBlock, 
            34: BasicBlock, 
            50: Bottleneck, 
            101: Bottleneck, 
            152: Bottleneck, 
            200: Bottleneck
                 }
        layers = {
            18: [2, 2, 2, 2], 
            34: [3, 4, 6, 3], 
            50: [3, 4, 6, 3], 
            101: [3, 4, 23, 3], 
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
        }
        assert depth in blocks.keys(), 'depth must be in [18, 34, 50, 101, 152, 200]'
        
        
        # 第一层CNN，7x7卷积，64个kernel, stride=2, padding=3
        # 输出 224x224，输出112x112 计算公式: (224 - 7 + 2 * 3) / 2 + 1 = 112
        
        self.input_channels = 64
        self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4个block
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        
        # average pooling
        self.avgpool = nn.AvgPool2d(7)
        
        # 全连接层
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
        
        # 初始化CNN和BN的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # 下采样
        downsample = None
        # 如果 stride != 1(X和Y的size不一样)
        # 或者输入X的channel数不等于输出Y的channel数
        # 那么做一个1x1卷积，让他们相等
        if stride != 1 or self.input_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        # 第一层单独写，因为可能遇到 stride=2，size减半的情况，需要通过1x1卷积修正
        layers = []
        layers.append(block(self.input_channels, planes, stride, downsample))
        self.input_channels = planes * block.expansion
        
        # 后面blocks - 1层
        for i in range(1, blocks):
            layers.append(block(self.input_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        # print('conv 7x7:', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        # print('max pool:',x.shape)
        
        x = self.layer1(x)
        # print('layer1:', x.shape)
        
        x = self.layer2(x)
        # print('layer2:', x.shape)
        
        x = self.layer3(x)
        # print('layer3:', x.shape)
        
        x = self.layer4(x)
        # print('layer4:', x.shape)
        
        x = self.avgpool(x)
        # print('avg pool:', x.shape)
        
        x = x.view(x.size(0), -1)
        # print('flatten:', x.shape)
        
        x = self.fc(x)
        # print('fc:', x.shape)

        return x

def ResNet18(**kwargs):
    return ResNet(depth=18,**kwargs)

def ResNet34(**kwargs):
    return ResNet(depth=34,**kwargs)

def ResNet50(**kwargs):
    return ResNet(depth=50,**kwargs)

def ResNet101(**kwargs):
    return ResNet(depth=101,**kwargs)

def ResNet152(**kwargs):
    return ResNet(depth=152,**kwargs)

def ResNet200(**kwargs):
    return ResNet(depth=200,**kwargs)