import os
import time
import torch
import argparse
import datetime
import torch.nn as nn
import torch.optim as optim
from model import pyramidnet
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--num_worker', type=int, default=4, help='')
args = parser.parse_args()


def main(root="/mnt/4T/Data/data/CIFAR"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Tools.print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root=root, train=True, download=True, transform=transforms_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    Tools.print('==> Making model..')

    net = pyramidnet()
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    Tools.print('The number of parameters of model is {}'.format(num_params))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for _ in range(10):
        Tools.print("epoch {}".format(_))
        train(net, criterion, optimizer, train_loader, device)
    pass


def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total

        batch_time = time.time() - start

        if batch_idx % 20 == 0:
            Tools.print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc, batch_time))
            pass

        pass

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    Tools.print("Training time {}, Acc {}".format(elapse_time, 100 * correct / total))
    pass


"""
2020-06-13 19:38:07 ==> Preparing data..
Files already downloaded and verified
2020-06-13 19:38:09 ==> Making model..
2020-06-13 19:38:11 The number of parameters of model is 24253410
2020-06-13 19:38:12 Epoch: [0/196]| loss: 2.323 | acc: 8.203 | batch time: 0.752s 
2020-06-13 19:38:27 Epoch: [20/196]| loss: 2.009 | acc: 24.609 | batch time: 0.718s 
2020-06-13 19:38:41 Epoch: [40/196]| loss: 1.891 | acc: 28.725 | batch time: 0.756s 
2020-06-13 19:38:57 Epoch: [60/196]| loss: 1.822 | acc: 31.589 | batch time: 0.783s 
2020-06-13 19:39:12 Epoch: [80/196]| loss: 1.750 | acc: 34.385 | batch time: 0.785s 
2020-06-13 19:39:28 Epoch: [100/196]| loss: 1.696 | acc: 36.622 | batch time: 0.770s 
2020-06-13 19:39:43 Epoch: [120/196]| loss: 1.642 | acc: 38.636 | batch time: 0.785s 
2020-06-13 19:39:59 Epoch: [140/196]| loss: 1.587 | acc: 40.930 | batch time: 0.781s 
2020-06-13 19:40:14 Epoch: [160/196]| loss: 1.539 | acc: 42.894 | batch time: 0.774s 
2020-06-13 19:40:30 Epoch: [180/196]| loss: 1.500 | acc: 44.449 | batch time: 0.779s 
2020-06-13 19:40:41 Training time 0:02:29.470228, Acc 45.556

2020-06-13 20:46:57 epoch 9
2020-06-13 20:46:58 Epoch: [0/196]| loss: 0.354 | acc: 88.281 | batch time: 0.783s 
2020-06-13 20:47:13 Epoch: [20/196]| loss: 0.306 | acc: 89.490 | batch time: 0.786s 
2020-06-13 20:47:28 Epoch: [40/196]| loss: 0.308 | acc: 89.339 | batch time: 0.738s 
2020-06-13 20:47:44 Epoch: [60/196]| loss: 0.305 | acc: 89.485 | batch time: 0.761s 
2020-06-13 20:47:59 Epoch: [80/196]| loss: 0.304 | acc: 89.405 | batch time: 0.737s 
2020-06-13 20:48:14 Epoch: [100/196]| loss: 0.308 | acc: 89.217 | batch time: 0.777s 
2020-06-13 20:48:30 Epoch: [120/196]| loss: 0.308 | acc: 89.214 | batch time: 0.732s 
2020-06-13 20:48:45 Epoch: [140/196]| loss: 0.307 | acc: 89.328 | batch time: 0.737s 
2020-06-13 20:49:00 Epoch: [160/196]| loss: 0.308 | acc: 89.288 | batch time: 0.786s 
2020-06-13 20:49:16 Epoch: [180/196]| loss: 0.307 | acc: 89.339 | batch time: 0.752s 
2020-06-13 20:49:27 Training time 0:02:30.010956, Acc 89.336
"""

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
