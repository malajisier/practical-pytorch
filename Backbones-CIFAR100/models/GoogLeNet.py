# GoogLeNet/Inception v1

import torch
import torch.nn as nn


class Inception():
    def __int__(self, inputs_channel, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool):
        super().__init__()

        # 1x1 conv 
        self.b1 = nn.Sequential(
            nn.Conv2d(inputs_channel, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv --> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(inputs_channel, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv --> 5x5 conv
        self.b3 = nn.Sequential(
            nn.Conv2d(inputs_channel, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 pooling --> 1x1 conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(inputs_channel, pool, kernel_size=1),
            nn.BatchNorm2d(pool),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):
            return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __int__(self, num_class=100):
        super().__init__()

        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

        def forward(self, x):
            output = self.prelayer(x)
            output = self.a3(output)
            output = self.b3(output)

            output = self.maxpool(output)

            output = self.a4(output)
            output = self.b4(output)
            output = self.c4(output)
            output = self.d4(output)
            output = self.e4(output)

            output = self.maxpool(output)

            output = self.a5(output)
            output = self.b5(output)

            output = self.avgpool(output)
            output = self.dropout(output)
            output = output.view(output.size()[0], -1)
            output = self.linear(output)

            return output


def googlenet():
    return GoogLeNet()
