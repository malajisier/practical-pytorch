import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_acc, get_model, get_train_dataloader, get_test_dataloader, WarmUpLR


def train(epoch):
    net.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch_index, (imgs, labels) in enumerate(cifar100_train_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        #imgs, labels = imgs.to(device), labels.to(device)
        imgs, labels = imgs.cuda(), labels.cuda()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += get_acc(outputs, labels)

    print("Epoch %d/%d, train loss: %.4f, train acc: %.4f, LR: %.6f" %(
        epoch, settings.EPOCH, train_loss/len(cifar100_train_loader),
        train_acc/len(cifar100_train_loader), optimizer.param_groups[0]['lr']
    ))


def eval(epoch):
    net.eval()
    test_loss = 0.0
    test_acc = 0.0

    for batch_index, (imgs, labels) in enumerate(cifar100_test_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        outputs = net(imgs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        test_acc += get_acc(outputs, labels)

    print("Test set:    Loss: %.4f,  Accuracy: %.4f" % (
        test_loss / len(cifar100_test_loader), test_acc / len(cifar100_test_loader)
    ))
    print("===============================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = get_model(args, use_gpu=args.gpu)
    net = torch.nn.DataParallel(net)
    # 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率
    # 仅适用于网络的输入数据维度或类型上变化不大
    # 若输入变化较大，cnDNN 每次都会去寻找一遍最优配置，反而降低运行效率
    cudnn.benchmark = True

    # data preprocessing:
    cifar100_train_loader = get_train_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.MILESTONES, gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar100_train_loader)
    total_iters =  iter_per_epoch*args.warm
    warmup_scheduler = WarmUpLR(optimizer, total_iters)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
