

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam

from tqdm import tqdm

from math import sqrt, pi

from einops import rearrange

from tifffile import imread, imwrite

import sys

from siren_pytorch import SirenNet

from unetmodel import ProjSegNet2

from apply_double_style import test_unet



def get_depth(im):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    im = im.astype(np.float32) - np.mean(im)
    im = (im/np.std(im)).astype(np.float32)

    zdim, ydim, xdim = im.shape
    model = ProjSegNet2(z_depth=zdim, depth_c=8, sc=32)

    model.load_state_dict(torch.load('p_model.pt', map_location=device))
    model.to(device)
    model.eval()

    im = torch.from_numpy(im[np.newaxis, np.newaxis, :, :, :])
    
    depth, _, _ = model(im.to(device))

    depth = torch.mean(torch.arange(zdim).cuda()[:,np.newaxis, np.newaxis]*depth[0], axis=0).detach()

    del model
    
    return depth


# Based on LucidRain's siren_pytorch - https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py

class ProjNet(nn.Module):
    def __init__(self, image_width, image_height, image_depth, sigma=0.1):

        super(ProjNet, self).__init__()
        
        self.sigma = sigma
        self.net = SirenNet(2, 16, 1, 2, w0_initial=4)
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        
        tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
     #   mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

        grid_d = torch.linspace(0, 1, steps = image_depth).unsqueeze(1).unsqueeze(2)
        self.register_buffer('grid_d', grid_d)
        
        self.sigma = torch.nn.Parameter(torch.tensor(sigma))

        
    def forward(self, stack):

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords)
#        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        out = F.sigmoid(out).squeeze(2)

#        print(out.shape)
        out = out.unsqueeze(0)

        
        grid_d = self.grid_d.clone().detach()

#        print(out.shape, grid_d.shape)

        weights = torch.exp(-0.5*(grid_d - out)**2/self.sigma)/(torch.sqrt(2*pi*self.sigma)*self.image_depth)

#        print(self.grid.shape, out.shape, stack.shape, weights.shape)

        proj = (stack*weights).sum(axis=0)

        return out, proj
        

plt.ion()

class GCC(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(GCC, self).__init__()
 
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        #average value
        I_ave, J_ave= I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)
 
#        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)#1e-5
        cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)#1e-5
 
        return -cc #-1.0 * cc + 1

def main():
    im = imread(sys.argv[1])[:,1]
    
    print(im.shape)

    im2 = im

    im_target = imread(sys.argv[2])[0]


    im_target = im_target.astype(float)/np.max(im_target)

    print(im_target.shape)

    plt.figure()
    plt.imshow(im_target)
    plt.draw()
    fig, ax = plt.subplots(1,2, figsize=(20,10))

    
    device = torch.device('cuda:0')

    start_depth = get_depth(im)



    
    model = ProjNet(im.shape[1], im.shape[2], im.shape[0])

    model.to(device)
    model.train()

    stack = torch.tensor(im2.astype(float)/np.max(im2), dtype=torch.float32).to(device)

    target = torch.tensor(im_target, dtype=torch.float32).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)

    loss_depth = torch.nn.MSELoss()

    for i in tqdm(range(10000)):
        depth, input  = model(stack)
        optimizer.zero_grad()
        output = loss_depth(depth, start_depth) #+ 2*model.sigma**2
        output.backward()
        optimizer.step()
        if (i%500==0):
            print(i, 'loss', output.detach())
            print('sigma', model.sigma)

            print(np.max(im_target), input.max())
            ax[0].imshow(input.detach().cpu().numpy())
#            ax[0].imshow(np.dstack((im_target, input.detach().cpu().numpy(), 0*im_target)))
            ax[1].imshow(depth.detach().cpu().numpy()[0])
            plt.draw()
            plt.pause(0.1)
        

    
    loss = GCC()  #torch.nn.BCELoss()
    
    for i in tqdm(range(100000)):
        depth, input  = model(stack)
        optimizer.zero_grad()
        output = loss(input, target) + 2*model.sigma**2
        output.backward()
        optimizer.step()
        if (i%1000==0):
            print(i, output.detach())
            print('sigma', model.sigma)
            print(np.max(im_target), input.max())
            ax[0].imshow(input.detach().cpu().numpy())
#            ax[0].imshow(np.dstack((im_target, input.detach().cpu().numpy(), 0*im_target)))
            ax[1].imshow(depth.detach().cpu().numpy()[0])
            plt.draw()
            plt.pause(0.1)
            g = input.detach().cpu().numpy()
            g = (255*g/np.max(g)).astype(np.uint8)
            
            imwrite('current.tif', g)


    
main()
