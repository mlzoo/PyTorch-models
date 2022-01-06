from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torchvision
from torchvision import transforms

from alexnet import AlexNet

def load_checkpoint(model, checkpoint_PATH, optimizer, device='cuda'):
    print('Loading model...')
    model_CKPT = torch.load(checkpoint_PATH) # 加载模型
    
    model.load_state_dict(model_CKPT['state_dict']) # 替换权重
    model = model.to(device)
    
    print('Loading optimizer...')
    optimizer.load_state_dict(model_CKPT['optimizer']) # 替换optimizer中的权重
    
    print('Loading succeed')
    return model, optimizer

model = AlexNet(num_classes=10)# .to(device)
optimizer = optim.Adam(params = model.parameters(),lr=0.001)

model, optimizer = load_checkpoint(model, './alexnet--epoch-1--accuracy-xxx.pth.tar', optimizer)

# 推理
print(model(torch.rand([4,3,224,224], device='cuda')))