

# Process all leaves in dataset.
# Outputs to output/leaf_data/
# - pickled LeafView as filename_0 : this contains the leaf image,
#                                    a transformed image (aligned to a template),
#                                    the processed arrows from the ROISet file,
#                                    and the measured midline angle
# - aligned image as filename-aligned.png : leaf with polarity arrows.
#                                           Original arrows in red,
#                                           adjusted arrows in colour if retained after criterion, grey if discared
# - histogram as filename-hist-rose.png : Histogram of angles |\alpha| for that leaf

from data_path import DATA_PATH

from re import L
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import json
from PIL import Image
from math import atan2, pi, sqrt, ceil, sin, cos
import scipy.ndimage as nd

from tifffile import imread
import csv
import matplotlib.cm as cm
import SimpleITK as sitk
from roi_decode import roi_zip
from skimage.segmentation import find_boundaries
from skimage.morphology import closing, disk
from skimage.measure import regionprops

import scipy.ndimage as nd

from process_utils import *

from skimage.filters import threshold_otsu, rank
from apply_double_style import test_unet

from pathlib import Path

from utils import to_ij, map_to_range, regionprop_dict, leaf_boundary

import matplotlib as mpl

import pandas as pd

from rose_plot import rose_plot

from math import pi

from dataclasses import dataclass

from stacked_histogram import stacked_histogram

from leaf import process_leaf_simple, AffineTransform, LeafView, get_rotation_transform, map_image_transform, map_points_transform, get_leaf_bdd_transform

mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
    'font.size': 20, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    'font.family': "sans-serif",
    'font.sans-serif': 'Arial',    
})



p_alpha = 0.05

ANGLE_LOW = 80
ANGLE_HIGH = 100
D_ANGLE = ANGLE_HIGH-ANGLE_LOW

RANGE_LOW = 500
RANGE_HIGH = 750
"""
ds = pd.read_csv('New_data.csv')
ds.columns = ['Identifier', 'Path', 'Filename','Marker', 'SizeCat','Age','CotWidth','TP','StretchSucc','ControlTP','Angle','RoICells','Author','Flipped_t0', 'Flipped_channels', 'Replicate', 'Scale', 'N1','N2']

ds = ds.loc[(~ds.Filename.isnull()) & (~ds.Angle.isnull())] 

ds.CotWidth[ds.CotWidth.isnull()] = 600

idx = ds['CotWidth'].argsort()

ds = ds.iloc[idx]

"""
from get_leaf_dataframe import get_leaf_dataframe_revised


ds = get_leaf_dataframe_revised()

print(ds)



data_path = DATA_PATH



def get_transform(json_file, sc=1):
    with open(json_file, 'r') as f:
        data = json.load(f)


    fn, m, pts = data[0]
    m0 = np.array(m).reshape((3,3)).T
    m0[:2,:2]=m0[:2,:2]/sc
    m = la.inv(m0)
    affine_m = to_ij(m)
    return fn, m0, affine_m, pts

        
def process_leaf(name, image, transform, roi, direction, cot_width, max_length=100, reverse_arrows=False, extra_roi_fn=[]):


    leaf, seg_bdd, sc = process_leaf_simple(name, image, None, roi, direction, return_seg_bdd=True, return_sc=True, extra_roi_fn=extra_roi_fn)

    _, m0_orig, _, _ = get_transform(transform, sc)
    
    h = 800
    off_x = 250
    s = 1024/h
    # scale and shift along x-axis, keep y axis centre in place
    m1 = np.array([[s,0,-off_x],[0,s,512*(1-s)],[0,0,1]])

    # Rotate by 90 degrees, add 1024 to y to fix placement
    m2 = np.array([[0,1,0],[-1,0,1024],[0,0,1]])

    # Composed transformations
    m0 = m2.dot(m1.dot(m0_orig))    
    
    # Inverse tranformation
    m0_inverse = la.inv(m0)

    transform_0 = AffineTransform(m0_inverse, (1024, 1024))
    
    leaf_0 = LeafView(leaf=leaf, transform = transform_0)
    leaf.width = cot_width
    leaf_0.width = cot_width
    
    return leaf_0

def rescale_arrow(a, l_new = 10):
    l = max(1e-6, np.sqrt(a[2]**2 + a[3]**2))
    b = (a[0], a[1], a[2]*l_new/l, a[3]*l_new/l)
    return b
    
def arrow_leaf_plot(leaf, px_scale=1.0):
    
    # Outlines of all cells with annotated arrows
    # Plot these in the coordinate frame 
    
    angle = leaf.angle

    m0 = get_rotation_transform(90-angle)
    
    # Inverse tranformation
    m0_inverse = la.inv(m0)

    transform_0 = AffineTransform(m0_inverse, (1024, 1024))
    
    leaf_0 = LeafView(leaf=leaf, transform = transform_0)

    # Choose colormap
    colormap = cm.hsv
    
    # Plot Wall data, Signal data, Annotated cell outlines (RGB) + while outline for whole leaf
    fig1, ax1 = plt.subplots(figsize=(20,20))
    ax1.set_position([0,0,1,1]) # Use whole axes space

    arrows_0 = leaf_0.get_arrows_dict()
    
    # Image in aligned frame

    bdd = (0.7*leaf_0.get_bdd(valid_only=True)).astype(np.uint8)
    signal = leaf_0.get_signal(valid_only=True)
    cells = leaf_0.get_cells(valid_only=True)

    cell_bdd = find_boundaries(cells) 
    B = np.dstack((bdd, signal, cell_bdd*(0)))

    ax1.imshow(B)

    ## Draw scalebar.
    ## Pixel scaling == px_scale/0.8

    new_px_scale = px_scale/0.8
    bar_length = 100/new_px_scale

    rect = Rectangle((1000-bar_length,950),bar_length,30,linewidth=0,edgecolor='none',facecolor='w')

    ax1.add_patch(rect)
    
    # Process arrows -> if not v long (likely mistake / arrow pointing into unsegmented cell )
    # Add arrow to image, subtract manually estimated direction (Man/Jie estimate, not mine)

    for i in arrows_0:
        a = arrows_0[i]
        if a.below_max_length:
            d = a.adjusted_centroid_angle
            ax1.arrow(*rescale_arrow(a.to_mpl_arrow()), overhang=0.4, color='w', width=1, head_width=12, head_length=10)
    
    return fig1

def data_from_leaf(leaf_0, plot_orig=True, max_length=100):

    leaf = leaf_0.leaf
    
    direction = leaf.angle

    arrow_idx = leaf_0.get_arrows_dict()

    print('name', leaf.name)
    print('N Cells', len(arrow_idx))
    
    # Outlines of all cells with annotated arrows

    # Choose colormap
    colormap = cm.hsv
    
    centroid_angles = []

    # Plot Wall data, Signal data, Annotated cell outlines (RGB) + while outline for whole leaf
    fig1, ax1 = plt.subplots(figsize=(20,20))
    ax1.set_position([0,0,1,1]) # Use whole axes space

    arrows_0 = leaf_0.get_arrows_dict()
    
    # Image in aligned frame

    bdd = leaf_0.get_bdd(valid_only=True)
    signal = leaf_0.get_signal(valid_only=True)
    cells = leaf_0.get_cells(valid_only=True)

    cell_bdd = find_boundaries(cells) 
    B = np.dstack((bdd*(1-cell_bdd), signal, cell_bdd*(64+191*np.isin(cells, list(arrow_idx)))))
    s =  nd.binary_dilation(get_leaf_bdd_transform(bdd))
    B[:,:,2] = np.maximum(B[:,:,2], 255*s)

    ax1.imshow(B)

    # Process arrows -> if not v long (likely mistake / arrow pointing into unsegmented cell )
    # Add arrow to image, subtract manually estimated direction (Man/Jie estimate, not mine)

    for i in arrows_0:
        a = arrows_0[i]
        ax1.arrow(*a.to_mpl_arrow(centroid_end=False), color=(1.0,1.0,1.0), width=1)


    for i in arrows_0:

        a = arrows_0[i]

        if a.below_max_length:
            d = a.adjusted_centroid_angle
            ax1.arrow(*a.to_mpl_arrow(), color=colormap((d+180)/360.0), width=1, alpha=0.8)
            centroid_angles.append(d)
        else:
            d = a.adjusted_centroid_angle
            ax1.arrow(*a.to_mpl_arrow(), color=(0.5, 0.5, 0.5), width=1, alpha=0.8)            
            centroid_angles.append(d)

    ax1.axis('off')        
    ax1.plot([511, 511], [0, 1023],'w-')
    ax1.set_xlim([0, 1023])
    ax1.set_ylim([1023, 0])
    
    print('centroid angle range', np.min(centroid_angles), np.max(centroid_angles))
    
    fig2, ax2 = plt.subplots()
    ax2.hist(np.abs(centroid_angles), bins=18, range=(0,180))
    
    return (fig1,fig2)



leaf_output = 'output/leaf_data/'

Path(leaf_output).mkdir(exist_ok=True, parents=True)

leaf_img_output = leaf_output

import pickle

all_leaves = []
all_hist_data = []

for i in ds.index:
    d = ds.loc[i]

    base_image_path = data_path
    base_arrow_path = data_path

    n = d.Identifier
    image = d.Path + '/' + d.Filename
    arrows = d.Path+'/'+d.Filename+'_RoiSet.zip'
    angle = d.Angle
    cw = d.CotWidth

    leaf_data_0 = process_leaf(n,
                                     base_image_path+image+'_proj.tif',
                                     base_image_path+image+'.json',
                                     base_arrow_path+arrows,
                                     angle,
                                     cw,
                                     max_length=100,
                                     reverse_arrows=False,
                                     extra_roi_fn = [],
    )
            
    print('>>>', n, image, arrows, angle)

    figs = data_from_leaf(leaf_data_0)

    figs[0].savefig(leaf_img_output+f'{n}-aligned.png')
    plt.close(figs[0])
    figs[1].savefig(leaf_img_output+f'{n}-hist-rose.png')
    plt.close(figs[1])

    scale = d.Scale
    if not pd.isna(scale):
        leaf_data = leaf_data_0.leaf
        fig = arrow_leaf_plot(leaf_data, px_scale=scale)
        fig.savefig(leaf_img_output+f'{n}-arrows.png')


    with open(leaf_output+n+'_0','wb') as f:
        pickle.dump(leaf_data_0, f)
            
