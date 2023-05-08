import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from scipy.ndimage import zoom
import math

__all__ = ['UNet']

# U-Net convolutional neural network for image segmentation.
class UNet(nn.Module):

  def __init__(self, transfer_learning=False):
    super().__init__()

    self.initial_conv = down_block(3, 64, False) 
    self.down1 = down_block(64, 128, True) 
    self.down2 = down_block(128, 256, True) 
    self.down3 = down_block(256, 512, True) 
    self.down4 = down_block(512, 1024, True) 

    self.up1 = up_block(512)
    self.up2 = up_block(256)
    self.up3 = up_block(128)
    self.up4 = up_block(64)
    self.out = nn.Conv2d(64, 2, 1)

  def forward(self, x):
    self.x1 = self.initial_conv(x)
    self.x2 = self.down1(self.x1)
    self.x3 = self.down2(self.x2)
    self.x4 = self.down3(self.x3)
    self.x5 = self.down4(self.x4)

    x = self.up1(self.x5, self.x4)
    x = self.up2(x, self.x3)
    x = self.up3(x, self.x2)
    x = self.up4(x, self.x1)

    return F.softmax(self.out(x), dim=1)

class down_block(nn.Module):

  def __init__(self, in_channels, out_channels, downsample):
    super().__init__()
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
    self.downsample = downsample
    self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)


  def enable(self):
    for name,param in self.named_parameters():
      param.requires_grad = True
      param.requires_grad = True

  def forward(self, x):
    if self.downsample == True:
      x = self.max_pool(x)
    x = self.double_conv(x)
    return x

class up_block(nn.Module):

  def __init__(self, out_channels):
    super().__init__()

    self.up_sampling = nn.ConvTranspose2d(out_channels*2, out_channels, 2, stride=2)

    self.double_conv = nn.Sequential(
      nn.Conv2d(out_channels*2, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
  def enable(self):
    for name,param in self.named_parameters():
      param.requires_grad = True
      param.requires_grad = True

  def forward(self, x, prev_x):
    x = self.up_sampling(x)
    x = torch.cat((x, prev_x), dim=1)
    x = self.double_conv(x)
    return x

