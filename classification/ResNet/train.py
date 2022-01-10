from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import tz, zoneinfo

import torch
from torch import nn, optim
import torchvision
from torchvision import transforms

from resnet import *

#################
#################
### Get time
tz_sh = tz.gettz('Asia/Shanghai')
now_sh = datetime.now(tz=tz_sh)
print(now_sh)

depth=50
model_name = 'resnet'

#################
#################
### train model
batch_size = 32

# 数据增强
transform = transforms.Compose([
        transforms.Resize((224, 224)), # 转换为 224 x 224
        transforms.RandomResizedCrop(224), # 随机剪裁
        transforms.RandomHorizontalFlip(), # 水平翻转
        transforms.ToTensor(), # 转换为Tensor
    ])


# 内置 Cifar-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./cifar10', 
                                        train=True,
                                        download=True, 
                                        transform=transform
                                       )
trainset.data, trainset.targets = trainset.data[:40000], trainset.targets[:40000]

valset = torchvision.datasets.CIFAR10(root='./cifar10', 
                                        train=True,
                                        download=True, 
                                        transform=transform
                                       )

valset.data, valset.targets = valset.data[40000:], valset.targets[40000:]

testset = torchvision.datasets.CIFAR10(root='./cifar10', 
                                       train=False,
                                       download=True, 
                                       transform=transform
                                      )

# 创建DataLoader
train_loader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=True, 
                                          num_workers=4
                                         )

val_loader = torch.utils.data.DataLoader(valset, 
                                          batch_size=batch_size,
                                          shuffle=True, 
                                          num_workers=4
                                         )

test_loader = torch.utils.data.DataLoader(testset, 
                                         batch_size=batch_size,
                                         shuffle=False, 
                                         num_workers=4
                                        )

# 类别名
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet(num_classes=10, depth=depth).to(device)

model.train()

optimizer = optim.Adam(params = model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 100

epoch_losses = []
epoch_accuracies = []

epoch_val_losses = []
epoch_val_accuracies = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in tqdm(train_loader):
        # 获取 X 和 y，并转换为 gpu
        data = data.to(device)
        label = label.to(device)
        
        # 预测(前向传播)
        output = model(data)
        # 计算 loss
        loss = criterion(output, label)
        
        # 清空之前的梯度
        optimizer.zero_grad()
        # 反向传播梯度
        loss.backward()
        # 根据反向传播的梯度，更新权重
        optimizer.step()
        
        # 概率最大的类别的 index == label 为1，最终取 mean
        acc = ((output.argmax(dim=1) == label).float().mean())
        
        # 平均 accuracy
        epoch_accuracy += acc/len(train_loader)
        # 平均 loss
        epoch_loss += loss/len(train_loader)
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy, epoch_loss))
    
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_accuracy)
    print('Saving model:')

    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in tqdm(val_loader):
            data = data.to(device)
            label = label.to(device)
            
            val_output = model(data)
            val_loss = criterion(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)
            
        epoch_val_losses.append(epoch_val_loss)
        epoch_val_accuracies.append(epoch_val_accuracy)
        print('Epoch : {}, val accuracy : {}, val loss : {}'.format(epoch+1, epoch_val_accuracy, epoch_val_loss))
# # 需要保存模型，可以取消注释
#     file_name = '{}-{}-{}--epoch-{}-acc-{:.2f}.pth.tar'.format(model_name,depth,now_sh,epoch + 1, epoch_val_accuracy.item())
#     torch.save({'epoch': epoch + 1, 
#             'state_dict': model.state_dict(), 
#             'best_loss': min(epoch_accuracies).item(),
#             'optimizer': optimizer.state_dict()
#            },
#            file_name)
#     print('Model saved at', file_name)
    
    print('Drawing')
    plt.figure()
    plt.plot([item.item() for item in epoch_accuracies],'ro-',color='red', label='train acc')
    plt.plot([item.item() for item in epoch_val_accuracies],'ro-',color='blue', label='val acc')
    plt.legend()
    plt.savefig('{}-{}-{}-accuracy.jpg'.format(model_name, depth, now_sh))
    plt.show()

    plt.figure()
    plt.plot([item.item() for item in epoch_losses],'ro-',color='red', label='train loss')
    plt.plot([item.item() for item in epoch_val_losses],'ro-',color='blue', label='val loss')
    plt.legend()
    plt.savefig('{}-{}-{}-loss.jpg'.format(model_name, depth, now_sh))
    plt.show()
    
    f = '{}-{}-{}-train-loss.txt'.format(model_name, depth, now_sh)
    with open(f, 'a') as file:   
        file.write(str(epoch_loss.item()) + "\n")

    f = '{}-{}-{}-val-loss.txt'.format(model_name, depth, now_sh)
    with open(f, 'a') as file:  
        file.write(str(epoch_val_loss.item()) + "\n")
    
    f = '{}-{}-{}-train-acc.txt'.format(model_name, depth, now_sh)
    with open(f, 'a') as file:   
        file.write(str(epoch_accuracy.item()) + "\n")

    f = '{}-{}-{}-val-acc.txt'.format(model_name, depth, now_sh)
    with open(f, 'a') as file:  
        file.write(str(epoch_val_accuracy.item()) + "\n")
