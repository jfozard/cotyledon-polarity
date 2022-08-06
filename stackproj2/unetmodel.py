import torch
from torch import nn
import torch.nn.functional as F


def down_block(in_ch, out_ch):

    padding = 1
    stride = 1
    kernel_size = 3

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2)
    )


def down_block2(in_ch, out_ch, use_bn):

    padding = 1
    stride = 1
    kernel_size = 3
#    use_bn = True
    if use_bn:
        return nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(inplace=True),
        )

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if stride!=1 or inplanes!=planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            downsample = None

        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        return out

class InBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(InBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if stride!=1 or inplanes!=planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            downsample = None

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

    
class ResUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResUNet, self).__init__()
        self.down0 = InBlock(in_ch, 64)
        self.down1 = BasicBlock(64, 128, 2)
        self.down2 = BasicBlock(128, 256, 2)
        self.down3 = BasicBlock(256, 256, 2)
        self.ups0 = UNetUpsample(scale_factor=2, mode='bilinear')
        self.up0 = BasicBlock(256+256, 128, 1)
        self.ups1 = UNetUpsample(scale_factor=2, mode='bilinear')
        self.up1 = BasicBlock(128+128, 64, 1)
        self.ups2 = UNetUpsample(scale_factor=2, mode='bilinear')
        self.up2 = BasicBlock(64+64, 64, 1)
        self.out = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        xb = x
        x0 = self.down0(xb)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)


        xb = self.ups0(x3)
        xb = torch.cat([xb, x2], dim=1)

        xb = self.up0(xb)

        xb = self.ups1(xb)
                                
        xb = torch.cat([xb, x1], dim=1)

        xb = self.up1(xb)

        xb = self.ups2(xb)
                                       
        xb = torch.cat([xb, x0], dim=1)

        xb = self.up2(xb)

        xb = self.out(xb)

        return torch.sigmoid(xb)


class UNetUpsample(nn.Module):
    def __init__(self, scale_factor, mode):
        super(UNetUpsample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x): 
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


def up_block(in_ch, out_ch):

    padding = 1
    stride = 1
    kernel_size = 3
    return nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        # F.interpolate(scale_factor=2, mode='bilinear', align_corners=True),
        UNetUpsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)

        )
          



def inconv(in_ch, out_ch, use_bn):
    padding = 1
    stride = 1
    kernel_size = 3

#    use_bn = True
    if use_bn:
        return nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        # F.interpolate(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)

        )
    else:
        return nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        # F.interpolate(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(inplace=True)
        )

class up_block2(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn):

        super(up_block2, self).__init__()
        
        self.up = nn.functional.interpolate
        padding = 1
        stride = 1
        kernel_size = 3

#        use_bn = True
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
	        nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
	        nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
 	        nn.LeakyReLU(), #inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(),
 #               nn.BatchNorm2d(out_ch),
                
            )#inplace=True))



    def forward(self, x1, x2):
        x1 = self.up(x1, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
                 

        
class UNet(nn.Module):

    def __init__(self, input_channels, use_bn=True):
        super().__init__()
        self.down0 = down_block(input_channels, 64)
        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 1024)
        self.up4 = up_block(1024, 512)
        self.up3 = up_block(512+512, 256)
        self.up2 = up_block(256+256, 128)
        self.up1 = up_block(128+128, 64)
        self.up0 = up_block(64+64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, xb):
        xs = xb
        x0 = self.down0(xb)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        xb = self.up4(x4)
        xb = torch.cat([xb, x3], dim=1)

        xb = self.up3(xb)
        xb = torch.cat([xb, x2], dim=1)

        xb = self.up2(xb)
        xb = torch.cat([xb, x1], dim=1)

        xb = self.up1(xb)
        xb = torch.cat([xb, x0], dim=1)

        xb = self.up0(xb)

        xb = self.out(xb)

        return torch.sigmoid(xb)


class UNet2(nn.Module):

    def __init__(self, input_channels, out_channels, use_bn=True, sc=16):
        super().__init__()
        self.inc = inconv(input_channels, sc, use_bn)
   #     self.down1 = down_block2(sc, sc, use_bn)
        self.down1 = down_block2(sc, 2*sc, use_bn)
        self.down2 = down_block2(2*sc, 4*sc, use_bn)
        self.down3 = down_block2(4*sc, 8*sc, use_bn)
        self.down4 = down_block2(8*sc, 8*sc, use_bn)
        self.up1 = up_block2(16*sc, 4*sc, use_bn )
        self.up2 = up_block2(8*sc, 2*sc, use_bn  )
        self.up3 = up_block2(4*sc, sc, use_bn  )
        self.up4 = up_block2(2*sc, sc, use_bn )
        self.outc = nn.Conv2d(sc, out_channels, kernel_size=1)

    def forward(self, xb):
        x1 = self.inc(xb)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
     #   x = self.up4(x2, x1)

        x = self.outc(x)
        
        

        return torch.sigmoid(x)


class ProjSegNet(nn.Module):
    def __init__(self, z_depth=32, depth_c=8, sc=32):
        super().__init__()
        self.depth = Depth(depth_c=depth_c)
        self.proj = Proj(z_depth=z_depth)
        self.unet = UNet2(1, 1, sc=sc)

    def forward(self, xb):
        d = self.depth(xb)
        h, sigma, p = self.proj(d, xb)
        u = self.unet(p)
        return h, sigma, p, u


class ProjSegNet2(nn.Module):
    def __init__(self, z_depth=32, depth_c=8, sc=32):
        super().__init__()
        self.depth = Depth(depth_c=depth_c)
        self.proj = Proj(z_depth=z_depth)
        self.unet = UNet2(1, 1, sc=sc)
        self.sm = nn.Softmax(dim=2)
        

    def forward(self, xb):
        d = self.depth(xb)
        #d = d*d
        #d = self.sm(d)
        h, sigma, p = self.proj(d, xb)
        u = self.unet(p)
        return self.sm(d), p, u

"""
class ProjSegNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = Depth()
        self.proj = Proj(z_depth=64)
        self.unet = UNet2(1, 1, sc=32)
        self.sm = nn.Softmax(dim=2)
        

    def forward(self, xb):
        d = self.depth(xb)
        #d = d*d
        #       d = self.sm(d)
        h, sigma, p = self.proj(d, xb)
        u = self.unet(p)
        return self.sm(d), p, u
"""

class Proj(nn.Module):
    def __init__(self, z_depth=32, axis=2):
        super().__init__()
        self.z_depth = z_depth
        self.axis=axis
        self.sm = nn.Softmax(dim=axis)

        
    def forward(self, d, xb):
        z = torch.arange(self.z_depth, dtype=torch.float32).cuda()
        z2 = z*z 
        d = self.sm(d)
        xb = torch.mul(d, xb)
        h = torch.sum(torch.mul(d, z[None, None, :, None, None]),2)
        sigma = torch.sum(torch.mul(d, z2[None, None, :, None, None]),2)-h*h
        return h, sigma, torch.sum(xb, self.axis)



        
class Depth(nn.Module):
    def __init__(self, depth_c=8):
        super().__init__()
        self.n_filters = depth_c
        self.kern = (3, 3, 3)
        self.pool = (1, 2, 2)
        self.axis = 0
        stride = 1
        padding = 1
        in_ch = 1
        nf = self.n_filters
        
        self.compress = nn.Sequential(
            nn.Conv3d(in_ch, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=self.pool),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=self.pool),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=self.pool),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=self.pool)
            )
        

        self.middle = nn.Sequential(
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            )

        self.expand = nn.Sequential(
            nn.Upsample(scale_factor=self.pool, mode='trilinear', align_corners=True),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=self.pool, mode='trilinear', align_corners=True),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=self.pool, mode='trilinear', align_corners=True),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=self.pool, mode='trilinear', align_corners=True),
            nn.Conv3d(nf, nf, kernel_size=self.kern, stride=stride, padding=padding),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True),

        )

        self.output  = nn.Conv3d(nf, 1, kernel_size=self.kern, stride=stride, padding=padding)
            
    def forward(self, x):
        x = self.compress(x)
        x = self.middle(x)
        x = self.expand(x)
        return self.output(x)
