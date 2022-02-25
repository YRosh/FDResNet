import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, models

import torchvision
import torchvision.transforms as transforms

import cv2
import os
import sys
import argparse
import PIL
from models import *
from utils import progress_bar
import time
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--testset', default=2, type=str, help='testset')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),  # ,, PIL.Image.BICUBIC
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = datasets.ImageFolder(root='path', transform=transform_train)
# testset = datasets.ImageFolder(root='path', transform=transform_test)

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=1)

# net = FG_ResNet101()
net = ResNet101()


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/google-net-run-1.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(best_acc, start_epoch)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
schedular = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[120, 170], gamma=0.1, last_epoch=-1)
# Training
# print(net.module.layer1[0].sigma_h)


def write_sigma(e):
    file1 = open("./logs/cifar100-l7-h7-1.txt", "a")
    file1.write("Epoch: {}\nLayer 1:\n".format(e))
    for i, block in enumerate(net.module.layer1, 0):
        file1.write("{} -- {} -- {}\n".format(i,
                    block.sigma_h.tolist(), block.sigma_l.tolist()))
    file1.write("Layer 2:\n")
    for i, block in enumerate(net.module.layer2, 0):
        file1.write("{} -- {} -- {}\n".format(i,
                    block.sigma_h.tolist(), block.sigma_l.tolist()))
    file1.write("Layer 3:\n")
    for i, block in enumerate(net.module.layer3, 0):
        file1.write("{} -- {} -- {}\n".format(i,
                    block.sigma_h.tolist(), block.sigma_l.tolist()))
    file1.write("Layer 4:\n")
    for i, block in enumerate(net.module.layer4, 0):
        file1.write("{} -- {} -- {}\n".format(i,
                    block.sigma_h.tolist(), block.sigma_l.tolist()))
    file1.write("\n")
    file1.close()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader, 0):
        stime = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx < 5 and epoch == 0:
            print("\r{}/{} ----- {}".format(batch_idx,
                  len(trainloader), round(time.time()-stime, 3)))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # write_sigma(epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
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
        torch.save(state, './checkpoint/google-net-run-1.pth')
        best_acc = acc

    return acc


for epoch in range(0, 200):
    train(epoch)
    schedular.step()
    acc = test(epoch)
    print(acc, best_acc)
