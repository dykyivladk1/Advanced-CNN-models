import torch.nn as nn
import torch


class ResBlock18(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    def forward(self, X):
        idn = X
        if self.downsample is not None:
            idn = self.downsample(idn)
        res = self.relu(self.bn1(self.conv1(X)))
        res = self.bn2(self.conv2(res))
        out = idn + res
        out = self.relu(out)
        return out
class ResNet18(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            in_channels=64, out_channels=64, stride=1)
        self.layer2 = self._make_layer(
            in_channels=64, out_channels=128, stride=2)
        self.layer3 = self._make_layer(
            in_channels=128, out_channels=256, stride=2)
        self.layer4 = self._make_layer(
            in_channels=256, out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock18(in_channels=in_channels,
                       out_channels=out_channels, stride=stride),
            ResBlock18(in_channels=out_channels, out_channels=out_channels)
        )
    def forward(self, X):
        out = self.maxpool(self.relu(self.bn1(self.conv1(X))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
# Dykyi Vladyslav