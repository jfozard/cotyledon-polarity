#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import deformation as df
#from imageio import imread, imwrite
from tifffile import imread
import sys
import scipy.linalg as la
import scipy.ndimage as nd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon


image_res = 0.692
sc_length = int(50/image_res)

SMALL_SIZE = 10*4
MEDIUM_SIZE = 12*4
BIGGER_SIZE = 14*4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('font', **{'sans-serif':'Arial', 'family':'sans-serif'})


def make_grid(shape):
    u = np.tile(np.eye(2), (8,8))
    u = u.repeat((shape[0]+15)//16, axis=0).repeat((shape[1]+15)//16,axis=1).astype(np.uint8)*128
    u = u[:shape[0],:shape[1]]
    return u

def main(args):

    t0 = imread(args[1])

    print('Start (T0) input image ', args[1])
    print('Image size ', t0.shape)
    if(t0.ndim>2):
#        t0 = np.max(t0, axis=0)
        t0 = t0[1]

    
    t1 = imread(args[2])
    print('End (T1) input image ', args[2])
    print('Image size ', t1.shape)
    if(t1.ndim>2):
#        t1 = np.max(t1, axis=0)
        t1 = t1[1]


    x_data, y_data = df.load_elastic_transform(args[3])
    print('Direct transform from T1 to T0 ', args[3])
    print('transform_grid_size ', x_data.shape)

    x_data_inv, y_data_inv = df.load_elastic_transform(args[4])
    print('Inverse transform from T0 to T1 ', args[4])
    print('transform_grid_size ', x_data_inv.shape) 



    
    mask0 = (1-(nd.gaussian_filter(t0, 3)<10))
        
    mask1 = (1-(nd.gaussian_filter(t1, 3)<10))

   
    res, J, valid, transformation_y, transformation_x, transformation_m = df.interpolate_image_mat(t1, x_data, y_data, out_shape=t0.shape)

    res1, J1, valid1, transformation_y1, transformation_x1, transformation_m1 = df.interpolate_image_mat(t0, x_data_inv, y_data_inv, out_shape=t1.shape)

    overlay_t1 = np.dstack((t1, res1, np.zeros_like(res1)))

    dpi = 100
    fig, ax = plt.subplots(figsize=(2048/dpi, 2048/dpi), dpi=dpi)

    ax.imshow(overlay_t1)    
    plt.axis('off')
#    plt.colorbar(k, ax=ax, format=mtick.PercentFormatter(xmax=1))
    fig.tight_layout() 
    ax.set_position([0,0,1,1])

    plt.savefig(args[7], dpi=dpi)

    plt.close()
    
    roots2, ev2 = df.ev2(transformation_m1)
    # Want largest, not smallest eigenvector so rotate by pi/2
    ev2 = ev2[:,:,[1,0]]
    ev2[:,:,0] = -ev2[:,:,0]

    d = np.sqrt(ev2[:,:,0]**2 + ev2[:,:,1]**2)
    vec = ev2/d[:,:,np.newaxis]
    x, y = np.meshgrid(range(roots2.shape[0]), range(roots2.shape[1]))
    v2 = vec*valid1[:,:,np.newaxis]*mask1[:,:,np.newaxis] #*mask[:,:,np.newaxis]
    N = 32
    qs = 12.0

    start_pts = np.stack([x[::N,::N].flatten()-qs*v2[::N,::N,0].flatten(),y[::N,::N].flatten()-qs*v2[::N,::N,1].flatten()], axis=-1)
    end_pts = np.stack([x[::N,::N].flatten()+qs*v2[::N,::N,0].flatten(),y[::N,::N].flatten()+qs*v2[::N,::N,1].flatten()], axis=-1)


    stretch = (np.sqrt(1.0/roots2[:,:,1])-1.0)*valid1*mask1

    sc = np.max(np.abs(stretch))
    print('stretch scale', sc)


    def fmt(x):
        return '%d'%int(100*x)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(2048/dpi, 2048/dpi), dpi=dpi)
    
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')
    
    ratio = np.sqrt(roots2[:,:,0]/roots2[:,:,1])-1.0

    
    ratio[valid1==0] = np.nan
    ratio[mask1==0] = np.nan
    m = np.nanmean(ratio)

    

    k=ax.imshow(ratio, vmin=0*np.nanmin(ratio), vmax=np.nanmax(ratio))
    #ax.imshow(np.dstack((g,g,g, 255-g)))

    lc = np.stack((start_pts, end_pts), axis=1)

    line_segments = LineCollection(lc, colors='k')
    ax.add_collection(line_segments)
    contours = ax.contour(ratio, [0.0, 0.1, 0.2], colors='r')
    plt.clabel(contours, fmt=fmt, fontsize=70)
#    ax.set_title('Stretch ratio on T1 mean {0:.1f}'.format(100*m))

    w = h = 1024

    sc_box = [ (20, h-20), (20+sc_length, h-20), (20+sc_length, h-10), (20, h-10) ]
    p = Polygon(sc_box, facecolor='w')
    ax.add_patch(p)


    plt.axis('off')
#
    ax.set_position([0,0,1,1])


#    fig.tight_layout() 

    plt.savefig(args[5], dpi=dpi)

    fig, ax = plt.subplots(figsize=(2048/dpi, 2048/dpi), dpi=dpi)
    cbar = plt.colorbar(k, ax=ax, format=mtick.PercentFormatter(xmax=1, decimals=0), ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25])
    cbar.ax.tick_params(labelsize=60) 
    cbar.update_ticks()
    
    #    plt.colorbar(contours, ax=ax)
    ax.remove()
    plt.savefig(args[6], dpi=dpi, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(2048/dpi, 2048/dpi), dpi=dpi)

    
    k=ax.imshow(ratio, vmin=0*np.nanmin(ratio), vmax=np.nanmax(ratio))
    #ax.imshow(np.dstack((g,g,g, 255-g)))

    lc = np.stack((start_pts, end_pts), axis=1)

    line_segments = LineCollection(lc, colors='k')
    ax.add_collection(line_segments)
    
    
    contours = ax.contour(ratio, [0.0, 0.1, 0.2], colors='r')
    
    plt.clabel(contours, fmt=fmt, fontsize=70)
    plt.colorbar(k, ax=ax, format=mtick.PercentFormatter(xmax=1, decimals=0))
    ax.axis('off')

    
    plt.savefig(args[8], dpi=dpi, bbox_inches='tight')
    plt.close()
    
from pathlib import Path

out_path = 'output/stretching_analysis/'
Path(out_path).mkdir(exist_ok=True)

main(['', 'tensor_data/sc_5-brx t0.tif',  'tensor_data/sc_5-brx t1.tif',  'tensor_data/5-brxl-inverse-t1-t0.txt', 'tensor_data/5-brxl-direct-t0-t1.txt', out_path+'5-brx-tensor.svg', out_path+'5-brx-cbar.svg', out_path+'5-brx-overlay.png', out_path+'5-brx-all-overlay.png'])
main(['', 'tensor_data/sc_1-brxl t0 R.tif',  'tensor_data/sc_1-brxl t1 R.tif',  'tensor_data/1-brxl-inverse-t1-t0.txt', 'tensor_data/1-brxl-direct-t0-t1.txt', out_path+'1-brx-tensor.svg', out_path+'1-brx-cbar.svg', out_path+'1-brx-overlay.png',  out_path+'1-brx-all-overlay.png'])
main(['', 'tensor_data/sc_2-brxl-t0.tif',  'tensor_data/sc_2-brxl-t1.tif',  'tensor_data/2-brxl-inverse-t1-t0.txt', 'tensor_data/2-brxl-direct-t0-t1.txt', out_path+'2-brx-tensor.svg', out_path+'2-brx-cbar.svg', out_path+'2-brx-overlay.png', out_path+'2-brx-all-overlay.png'])
main(['', 'tensor_data/sc_1-basl-t0.tif',  'tensor_data/sc_1-basl-t1.tif',  'tensor_data/1-basl-inverse-t1-t0.txt', 'tensor_data/1-basl-direct-t0-t1.txt', out_path+'1-basl-tensor.svg', out_path+'1-basl-cbar.svg', out_path+'1-basl-overlay.png', out_path+'1-basl-all-overlay.png'])




