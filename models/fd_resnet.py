import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=3, sigma=1, dim=2):
        super(GaussianSmoothing, self).__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        #sigma = torch.cat([sigma, sigma])
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        #self.sigma = nn.Parameter(torch.tensor([1.0, 1.0]).cuda(), requires_grad=True)
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32).cuda()
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2).cuda()

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.cuda())
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, inputs):
        return self.conv(inputs, weight=self.weight, groups=self.groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut_low = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_low = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.shortcut_high = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_high = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        smoothing_l = GaussianSmoothing(x.shape[1], kernel_size=3, sigma=1)
        smoothing_h = GaussianSmoothing(x.shape[1], kernel_size=3, sigma=1)

        img_pl = F.pad(x, (3//2,)*4, mode='reflect')
        img_l = smoothing_l(img_pl)
        img_ph = F.pad(x, (3//2,)*4, mode='reflect')
        img_h = smoothing_h(img_ph)
        img_h = x-img_h

        out += self.shortcut_low(img_l)
        out += self.shortcut_high(img_h)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut_low = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_low = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.shortcut_high = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_high = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        smoothing_l = GaussianSmoothing(x.shape[1], kernel_size=3, sigma=1)
        smoothing_h = GaussianSmoothing(x.shape[1], kernel_size=5, sigma=1)

        img_pl = F.pad(x, (3//2,)*4, mode='reflect')
        img_l = smoothing_l(img_pl)
        img_l = img_l
        img_ph = F.pad(x, (5//2,)*4, mode='reflect')
        img_h = smoothing_h(img_ph)
        img_h = x-img_h

        out += self.shortcut_low(img_l)
        out += self.shortcut_high(img_h)
        out = F.relu(out, inplace=True)
        print("bottleneck input", out.size())
        return out


class FD_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(FD_ResNet, self).__init__()
        self.in_planes = 64

        # before kernal_size=3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.pool(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out


def FD_ResNet18():
    return FD_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)


def FD_ResNet34():
    return FD_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=256)


def FD_ResNet50():
    return FD_ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)


def FD_ResNet101():
    return FD_ResNet(Bottleneck, [3, 4, 23, 3], num_classes=200)


def ResNet152():
    return FD_ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
