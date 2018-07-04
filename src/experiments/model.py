import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels,output_channels,3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownConv,self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(input_channels, output_channels)
        )

    def forward(self,x):
        x = self.down_conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, input_channels,output_channels):
        super(UpConv,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self,x, x_prev):
        x = self.up_conv(x)
        x = torch.cat((x, x_prev), 1)
        x = self.conv(x)
        return x





class Unet(nn.Module):
    def __init__(self,initial_channels):
        super(Unet, self).__init__()
        self.initial_conv = DoubleConv(initial_channels,64)
        self.down1 = DownConv(64,128)
        self.down2 = DownConv(128,256)
        self.down3 = DownConv(256,512)
        self.down4 = DownConv(512,1024)
        self.up1= UpConv(1024,512)
        self.up2= UpConv(512,256)
        self.up3= UpConv(256,128)
        self.up4= UpConv(128,64)
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.final_conv(x)
        return x

    def eval_forward(self,x):
        x = self.forward(x)
        return self.sigmoid(x)












class SimpleUNet(nn.Module):
    def __init__(self, initial_channels):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(initial_channels, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)

        self.deconv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)

        self.conv6 = nn.Conv2d(16, 8, 3, padding=1)

        self.conv7 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv8 = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x3 = F.relu(x3)

        x4 = self.deconv1(x3)
        x4 = torch.cat((x4, x2), 1)
        x4 = self.conv4(x4)
        x4 = F.relu(x4)

        x5 = self.deconv2(x4)
        x5 = torch.cat((x5, x1), 1)
        x5 = self.conv5(x5)
        x5 = F.relu(x5)

        x6 = self.conv6(x5)
        x6 = F.relu(x6)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        return x8


