import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign


class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding = 3//2),
            nn.ReLU(),)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding = 3//2),
            nn.ReLU(),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding = 3//2),
            nn.ReLU(),)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding = 3//2)


    def forward(self, input):
        x = self.conv(input)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        return x


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv01 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding = 1, padding = 3//2),
            nn.ReLU(),)
        self.conv02 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding = 1, padding = 3//2),
            nn.ReLU(),)
        self.conv03 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, output_padding = 1, padding = 3//2),
            nn.ReLU(),)
        self.conv04 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding = 1, padding = 3//2),
            nn.ReLU(),)

        self.conv2 = nn.Conv2d(
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input):
        x = self.conv1(input)

        x = self.conv01(x)

        x = self.conv02(x)

        x = self.conv03(x)

        x = self.conv04(x)

        x = F.tanh(self.conv2(x)) / 2
        
        return x
