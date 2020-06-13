import os
import time
import torch
import argparse
import datetime
import torch.nn as nn
import torch.optim as optim
from model import pyramidnet
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=256 * 4, help='')
parser.add_argument('--num_worker', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=str, default="0, 1, 2, 3", help="")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices


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

    #############################################################
    net = nn.DataParallel(net)
    cudnn.benchmark = True
    #############################################################

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
2020-06-13 19:41:14 ==> Preparing data..
Files already downloaded and verified
2020-06-13 19:41:15 ==> Making model..
2020-06-13 19:41:18 The number of parameters of model is 24253410
2020-06-13 19:41:27 Epoch: [0/49]| loss: 2.350 | acc: 9.570 | batch time: 8.269s 
2020-06-13 19:41:43 Epoch: [20/49]| loss: 1.950 | acc: 27.693 | batch time: 0.812s 
2020-06-13 19:42:00 Epoch: [40/49]| loss: 1.773 | acc: 34.039 | batch time: 0.819s 
2020-06-13 19:42:06 Training time 0:00:48.027100, Acc 36.336

2020-06-13 20:14:20 epoch 9
2020-06-13 20:14:21 Epoch: [0/49]| loss: 0.355 | acc: 86.523 | batch time: 0.894s 
2020-06-13 20:14:39 Epoch: [20/49]| loss: 0.354 | acc: 87.588 | batch time: 0.880s 
2020-06-13 20:14:56 Epoch: [40/49]| loss: 0.364 | acc: 87.333 | batch time: 0.826s 
2020-06-13 20:15:03 Training time 0:00:42.622036, Acc 87.286
"""

if __name__ == '__main__':
    main()
