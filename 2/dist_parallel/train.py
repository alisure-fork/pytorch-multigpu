import os
import time
import torch
import random
import argparse
import datetime
import torch.nn as nn
import torch.optim as optim
from model import pyramidnet
import torch.distributed as dist
import torch.multiprocessing as mp
from alisuretool.Tools import Tools
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def main(root="/mnt/4T/Data/data/CIFAR"):
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, root, args))
    pass


def main_worker(gpu_id, ngpus, root, args):
    batch_size = int(args.batch_size / ngpus)
    num_workers = int(args.num_workers / ngpus)

    Tools.print("Use GPU: {} for training".format(gpu_id))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus, rank=gpu_id)

    Tools.print('==> Making model..')
    net = pyramidnet()
    torch.cuda.set_device(gpu_id)
    net.cuda(gpu_id)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_id])

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    Tools.print('The number of parameters of model is {}'.format(num_params))

    Tools.print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root=root, train=True, download=True, transform=transforms_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,  shuffle=(train_sampler is None),
                              num_workers=num_workers, sampler=train_sampler)
    
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,  momentum=0.9, weight_decay=1e-4)
    cudnn.benchmark = True

    for _ in range(10):
        Tools.print("epoch {}".format(_))
        train(net, criterion, optimizer, train_loader, gpu_id)
    pass


def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
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
2020-06-13 19:53:58 Use GPU: 2 for training
2020-06-13 19:53:58 Use GPU: 0 for training
2020-06-13 19:53:58 Use GPU: 1 for training
2020-06-13 19:53:58 ==> Making model..
2020-06-13 19:53:58 Use GPU: 3 for training
2020-06-13 19:53:58 ==> Making model..
2020-06-13 19:53:59 ==> Making model..
2020-06-13 19:53:59 ==> Making model..
2020-06-13 19:54:03 The number of parameters of model is 24253410
2020-06-13 19:54:03 ==> Preparing data..
2020-06-13 19:54:03 The number of parameters of model is 24253410
2020-06-13 19:54:03 ==> Preparing data..
2020-06-13 19:54:03 The number of parameters of model is 24253410
2020-06-13 19:54:03 ==> Preparing data..
2020-06-13 19:54:03 The number of parameters of model is 24253410
2020-06-13 19:54:03 ==> Preparing data..
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
2020-06-13 19:54:07 Epoch: [0/66]| loss: 2.405 | acc: 5.729 | batch time: 1.033s 
2020-06-13 19:54:07 Epoch: [0/66]| loss: 2.345 | acc: 10.938 | batch time: 0.957s 
2020-06-13 19:54:07 Epoch: [0/66]| loss: 2.346 | acc: 10.938 | batch time: 0.965s 
2020-06-13 19:54:07 Epoch: [0/66]| loss: 2.331 | acc: 10.938 | batch time: 0.951s 
2020-06-13 19:54:18 Epoch: [20/66]| loss: 1.950 | acc: 27.083 | batch time: 0.548s 
2020-06-13 19:54:18 Epoch: [20/66]| loss: 1.953 | acc: 27.183 | batch time: 0.548s 
2020-06-13 19:54:18 Epoch: [20/66]| loss: 1.933 | acc: 25.744 | batch time: 0.548s 
2020-06-13 19:54:18 Epoch: [20/66]| loss: 1.953 | acc: 26.190 | batch time: 0.551s 
2020-06-13 19:54:29 Epoch: [40/66]| loss: 1.766 | acc: 33.587 | batch time: 0.554s 
2020-06-13 19:54:29 Epoch: [40/66]| loss: 1.739 | acc: 34.096 | batch time: 0.556s 
2020-06-13 19:54:29 Epoch: [40/66]| loss: 1.771 | acc: 33.168 | batch time: 0.554s 
2020-06-13 19:54:29 Epoch: [40/66]| loss: 1.777 | acc: 34.019 | batch time: 0.552s 
2020-06-13 19:54:41 Epoch: [60/66]| loss: 1.613 | acc: 39.609 | batch time: 0.559s 
2020-06-13 19:54:41 Epoch: [60/66]| loss: 1.646 | acc: 38.644 | batch time: 0.559s 
2020-06-13 19:54:41 Epoch: [60/66]| loss: 1.646 | acc: 39.344 | batch time: 0.557s 
2020-06-13 19:54:41 Epoch: [60/66]| loss: 1.633 | acc: 38.806 | batch time: 0.555s 
2020-06-13 19:54:43 Training time 0:00:39.303257, Acc 40.12
2020-06-13 19:54:43 Training time 0:00:39.342439, Acc 40.4
2020-06-13 19:54:43 Training time 0:00:39.351259, Acc 39.304
2020-06-13 19:54:43 Training time 0:00:39.363285, Acc 39.608

2020-06-13 20:22:17 epoch 9
2020-06-13 20:22:20 Epoch: [0/66]| loss: 0.382 | acc: 84.896 | batch time: 0.805s 
2020-06-13 20:22:20 Epoch: [0/66]| loss: 0.306 | acc: 88.542 | batch time: 0.651s 
2020-06-13 20:22:20 Epoch: [0/66]| loss: 0.295 | acc: 90.625 | batch time: 0.603s 
2020-06-13 20:22:20 Epoch: [0/66]| loss: 0.359 | acc: 90.104 | batch time: 0.802s 
2020-06-13 20:22:32 Epoch: [20/66]| loss: 0.362 | acc: 87.550 | batch time: 0.616s 
2020-06-13 20:22:32 Epoch: [20/66]| loss: 0.360 | acc: 87.674 | batch time: 0.616s 
2020-06-13 20:22:32 Epoch: [20/66]| loss: 0.358 | acc: 87.971 | batch time: 0.615s 
2020-06-13 20:22:32 Epoch: [20/66]| loss: 0.365 | acc: 87.475 | batch time: 0.614s 
2020-06-13 20:22:44 Epoch: [40/66]| loss: 0.360 | acc: 87.208 | batch time: 0.614s 
2020-06-13 20:22:44 Epoch: [40/66]| loss: 0.349 | acc: 87.932 | batch time: 0.613s 
2020-06-13 20:22:44 Epoch: [40/66]| loss: 0.360 | acc: 87.716 | batch time: 0.614s 
2020-06-13 20:22:44 Epoch: [40/66]| loss: 0.344 | acc: 88.046 | batch time: 0.614s 
2020-06-13 20:22:57 Epoch: [60/66]| loss: 0.343 | acc: 87.807 | batch time: 0.613s 
2020-06-13 20:22:57 Epoch: [60/66]| loss: 0.348 | acc: 88.175 | batch time: 0.613s 
2020-06-13 20:22:57 Epoch: [60/66]| loss: 0.339 | acc: 88.397 | batch time: 0.613s 
2020-06-13 20:22:57 Epoch: [60/66]| loss: 0.337 | acc: 88.183 | batch time: 0.613s 
2020-06-13 20:22:59 Training time 0:00:42.638022, Acc 87.896
2020-06-13 20:22:59 Training time 0:00:42.656286, Acc 88.336
2020-06-13 20:22:59 Training time 0:00:42.742679, Acc 88.232
2020-06-13 20:22:59 Training time 0:00:42.682666, Acc 88.2
"""


if __name__ == '__main__':
    main()
