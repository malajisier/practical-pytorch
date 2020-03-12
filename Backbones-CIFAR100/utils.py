import sys
import numpy as np
import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

def get_acc(outputs, labels):
    total = outputs.shape[0]
    _, pred_label = outputs.max(1)
    num_correct = pred_label.eq(labels).sum().item()
    return num_correct/total


def get_model(args, use_gpu=True):
    if args.net == 'vgg16':
        from models.VGG import VGG16
        net = VGG16()
    elif args.net == 'vgg11':
        from models.VGG import VGG11
        net = VGG11()
    elif args.net == 'vgg13':
        from models.VGG import VGG13
        net = VGG13()
    elif args.net == 'vgg19':
        from models.VGG import VGG19
        net = VGG19()


    else:
        print('the net name you\'ve entered isn\'t supported yet ')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net


def get_train_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    CIFAR100_train = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transforms_train, download=True)
    CIFAR100_trainloader = DataLoader(CIFAR100_train, num_workers=num_workers, shuffle=True, batch_size=batch_size)
    return CIFAR100_trainloader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    CIFAR100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    CIFAR100_testloader = DataLoader(
        CIFAR100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return CIFAR100_testloader


def compute_mean_std(dataset):

    data_r = np.dstack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


# class WarmUpLR(_LRScheduler):
#     # total_iters: totoal_iters of warmup phase
#     def __int__(self, optimizer, total_iters, last_epoch=-1):
#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]