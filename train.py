# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models import *

import os
import argparse

#from models import *
from utils import progress_bar

torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='res18', help='choose from res18(def), res34, res50, vgg, densenet, mobilenet, shufflenet, efficientnet, resnext')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--epoch', default=50, type=int, help='learning rate')
parser.add_argument('--bs', default=128, type=int, help='batchsize')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# you might need to change lr along with bs for good accuracy.
bs = int(args.bs)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
class mymodel(nn.Module):
    def __init__(self, basemodel):
        super(mymodel, self).__init__()
        self.features = basemodel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        if model_name == "res34" or model_name == "res18":
            num_ch = 512
        else:
            num_ch = 2048
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(num_ch, 10, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)#.squeeze(2).squeeze(2)
        x = self.fc1(x)
        return x
    
model_name = args.net
if args.net=='res18':
    basemodel = resnet18(pretrained=True)
    basemodel = nn.Sequential(*list(basemodel.children())[1:-2])    
    net = mymodel(basemodel) 
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    basemodel = resnet34(pretrained=True)
    basemodel = nn.Sequential(*list(basemodel.children())[1:-2])
    net = mymodel(basemodel)
elif args.net=='res50':
    basemodel = resnet50(pretrained=True)
    basemodel = nn.Sequential(*list(basemodel.children())[1:-2])
    net = mymodel(basemodel)
elif args.net=='res101':
    basemodel = resnet101(pretrained=True)
    basemodel = nn.Sequential(*list(basemodel.children())[1:-2])
    net = mymodel(basemodel)
# net = PreActResNet18()
# net = GoogLeNet()
elif args.net=='densenet': 
    net = DenseNet121()
elif args.net=='resnext':
    net = ResNeXt29_4x64d()
# net = MobileNet()
elif args.net=='mobilenet': 
    net = MobileNetV2()
# net = DPN92()
elif args.net=='shufflenet':
    net = ShuffleNetv2()
elif args.net=='efficientnet':
    net = Efficientnet()
else:
    print("{} not found").format(args.net)

net = net.to(device)
print(net)

if args.fp16:
    from fp16util import network_to_half
    net = network_to_half(net)

if device == 'cuda':
    # net = torch.nn.DataParallel(net) # make parallel
    """ can't use dataparallel for onnx..
    see https://github.com/pytorch/pytorch/issues/13397 """
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # set input output as float and send to device.
        inputs, targets = inputs.float().to(device), targets.long().to(device)
        optimizer.zero_grad()
        outputs = net(inputs).squeeze(2).squeeze(2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).squeeze(2).squeeze(2)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not args.fp16:
            torch.save(state, './checkpoint/'+args.net+'-ckpt.t7')
        else:
            torch.save(state, './checkpoint/'+args.net+'-'+'FP16-ckpt.t7')
        best_acc = acc

list_loss = []

for epoch in range(start_epoch, start_epoch+int(args.epoch)):
    trainloss = train(epoch)
    test(epoch)
    
    list_loss.append(trainloss)
    print(list_loss)
print(list_loss)
