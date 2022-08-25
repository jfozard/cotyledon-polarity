
from data_path import DATA_PATH

import matplotlib
matplotlib.use('agg')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
#from PIL import Image
from math import atan2, pi, sqrt, ceil, sin, cos
from tifffile import imread
from transform import Transform
import scipy.ndimage as nd
from matplotlib.collections import LineCollection
from roi_decode import roi_zip
from skimage.segmentation import find_boundaries
from scipy.sparse import coo_matrix
from collections import defaultdict, Counter

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

from base import textext, textext_svg

import random

import ray

from imageio import imwrite

import matplotlib as mpl

import pickle

from collections import Counter


mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
    'font.size': 20, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    
})



from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle

from PIL import Image, ImageDraw

from pathlib import Path


import svgutils.transform as sg
import sys
import base64
import io
from lxml import etree

FS = 2
FS2 = 2.1
IM_S = 4

from matplotlib.patches import Polygon
from skimage.draw import polygon_perimeter

from imageio import imwrite
from PIL import Image, ImageDraw


def draw_poly(A, poly, col, width=4):
    im = Image.fromarray(A)
    draw = ImageDraw.Draw(im)
    poly = [tuple(np.array(p, dtype=int)) for p in poly]
    draw.line(poly+poly[0:1], fill=col, width=width)
    return np.asarray(im)

def overlay_arrows(A, arrows, col=(255,255,255), width=2):

    im = Image.fromarray(A)
    draw = ImageDraw.Draw(im)
    for arrow, _ in arrows:
        draw.line([tuple(x) for x in arrow], fill=col, width=width)
    return np.array(im)

image_res = 0.692
sc_length_microns = 100
sc_length_cell_microns=25
inset_scale = 4



            


def affine(m, p): # apply an affine transformation (2x3 or 3x3 matrix) to a 2d point
    return m[:2,:2].dot(p) + m[:2,2]


def to_ij(m): # xy to ij for matrix (and vice-versa)
    m_ij = np.array(m)
    m_ij[[0,1],:] = m_ij[[1,0],:]
    m_ij[:,[0,1]] = m_ij[:,[1,0]]
    return m_ij

def get_transform(json_file): # Load affine transformation from file -> presumably in XY format
    with open(json_file, 'r') as f:
        data = json.load(f)

    fn, m, pts = data[0]
    m0 = np.array(m).reshape((3,3)).T

    return m0

# Apply an affine transformation to an image
def map_image_affine(im, affine_m, output_shape=None, order=0):
    if output_shape is None:
        return nd.affine_transform(im, affine_m, order=order)
    else:
        return nd.affine_transform(im, affine_m, order=order, output_shape=output_shape)

def centroids_area(seg):
    # Find the centroids and areas of each non-zero labelled region in the segmentation seg
    # returns centroids, areas
    #          centroids - list of cell centroids (I,J)
    return [ f(np.ones_like(seg), labels=seg, index=np.arange(0, np.max(seg)+1)) for f in ( lambda *x, **y: list(map(swap_ij, nd.center_of_mass(*x, **y))), nd.sum ) ]


# Swap two values - I,J <-> X,Y
def swap_ij(x):
    return x[1], x[0]



def get_bbox_diameter(mask):
    obj = nd.find_objects(mask)[0]
    d = sqrt((obj[0].stop - obj[0].start)**2 + (obj[1].stop - obj[1].start)**2 )
    return d


def bbox_to_poly(bbox):
    x1, x2 = bbox[1].start, bbox[1].stop
    y1, y2 = bbox[0].start, bbox[0].stop
    return [(x1,y1), (x2, y1), (x2, y2), (x1,y2)]


def apply_affine_channels(im, aff, output_shape=None):
    if output_shape is None:
        output_shape = im.shape
    return np.stack([map_image_affine(im[i,:,:], aff, output_shape=output_shape) for i in range(im.shape[0])], axis=0)


def extend_to_rgb(im):
    if im.shape[0]<3:
        u = np.zeros((3-im.shape[0],) + im.shape[1:], dtype=im.dtype)
        im = np.concatenate([im[[1,0],:,:], u ], axis=0)
    return im


def min_bbox(pts):
    x = [int(p[0]) for p in pts]
    y = [int(p[1]) for p in pts]
    return slice(min(y), max(y)+1), slice(min(x), max(x)+1)

def add_box_scalebar(im, box, sc_length, draw_orig=True, offset_x=20):
    im_new = np.array(im.transpose((1,2,0)))
    if draw_orig:
        im_new = draw_poly(im_new, box, (0,0,255))

    bbox = min_bbox(box)
    im_new = draw_poly(im_new, bbox_to_poly(bbox), (255,255,255))

    im_new = np.array(im_new)
    im_new[im_new.shape[0]-20:im_new.shape[0]-10, offset_x:offset_x+sc_length,:] = 255
                    
    return im_new, bbox


def draw_circle(A, p, r, col):
    im = Image.fromarray(A)
    draw = ImageDraw.Draw(im)
    draw.arc([tuple(np.array(p)-r), tuple(np.array(p)+r)], 0, 360, fill=col, width=4)
    return np.asarray(im)
    

def add_box_scalebar_pts(im, box, sc_length, pts, radii, draw_orig=True, offset_x=20, offset_y=10):
    im_new = np.array(im.transpose((1,2,0)))
    if draw_orig:
        im_new = draw_poly(im_new, box, (0,0,255))

    if box is not None:
        bbox = min_bbox(box)
        im_new = draw_poly(im_new, bbox_to_poly(bbox), (255,255,255))
    else:
        bbox = None
    for p, r in zip(pts, radii):
        im_new = draw_circle(im_new, p, r, (255, 255, 255))

    
    im_new = np.array(im_new)
    im_new[im_new.shape[0]-offset_y-10:im_new.shape[0]-offset_y, offset_x:offset_x+sc_length,:] = 255
                    
    return im_new, bbox




def get_rotation_transform(theta, sc=0.8):

    a = (pi/180)*(theta)
    
    R0 = sc*np.array(((cos(a), sin(a)), (-sin(a), cos(a))))

    c0 = np.array([[512,512]]).T
    #m2 = np.array([[0,1,0],[-1,0,1024],[0,0,1]])
    m0 = np.vstack((np.hstack((R0, c0-R0.dot(c0))), (0,0,1)))
#    m0 = m2.dot(m0)
    return m0

from leaf import seg_leaf_bdd

def map_box_multiple(im_fn, seg0_fn, seg_pos, transform_fn, transform_pts_fn, angles):

    images = [ imread(f) for f in im_fn ]
    
    print([im.shape for im in images])

    image_sc = 1.0

    sc_length = int(sc_length_microns*image_sc/image_res)

    orig_size = (1024, 1024)
    
    transforms = [ Transform.from_file(f, orig_size) for f in transform_fn ] 
    transforms_pts = [ Transform.from_file(f, orig_size) for f in transform_pts_fn ] 
    
    affine_transforms = [ get_rotation_transform(90-theta, sc=image_sc) for theta in angles ]
    print('affine_transforms', affine_transforms)

    
    inset_scale = 2
    image_affine_transforms = [ to_ij(la.inv(m0)).dot(np.diag([1.0/inset_scale,1.0/inset_scale,1])) for m0 in affine_transforms ]

    image_x2 = [ extend_to_rgb(apply_affine_channels(im,im_aff_trans, output_shape=(1024*inset_scale,1024*inset_scale))) for im, im_aff_trans in zip(images, image_affine_transforms) ]
                                

    # Find cell centroids (in seg0)

#    seg0 = imread(seg0_fn)

    seg0 = seg_leaf_bdd(images[0][1])[1]


    print('seg_pos', seg_pos, seg0.shape)
    
    seg_idx = [seg0[p[1], p[0]] for p in seg_pos]
    
    c0, _ = centroids_area(seg0)

    radii = [ inset_scale*image_sc*0.7*get_bbox_diameter(seg0==idx) for idx in seg_idx]
    
    pts_0_orig = [ c0[idx] for idx in seg_idx ]
    print(pts_0_orig)

    print([seg0[int(p[1]), int(p[0])] for p in pts_0_orig])
    print(seg_idx)
    pts_0 = [ affine(la.inv(to_ij(image_affine_transforms[0])), p) for p in pts_0_orig ]

    
    s = image_x2[0].shape
    ch, cw = s[1]/2, s[2]/2
    w = 200
    h = 160

    box_0 = [(cw - w/2, ch - h/2), (cw + w/2, ch - h/2), (cw + w/2, ch + h/2), (cw - w/2, ch + h/2) ]

    bbox0 = min_bbox(box_0)

    im0_new, bbox0_cp = add_box_scalebar_pts(image_x2[0], box_0, sc_length, pts_0, radii, offset_x = 200)

    print(f'{bbox0} {bbox0_cp}')
    
    imwrite(out_path+'im_t0.png', im0_new)

    im0_inset_new, _ = add_box_scalebar_pts(image_x2[0][:,bbox0[0], bbox0[1]], None, sc_length, [], radii, draw_orig=False)
    
    imwrite(out_path+'im_t0_inset.png', im0_inset_new)

    
    # Map bbox to next image
    
    box_0_orig = [ affine(to_ij(image_affine_transforms[0]), p) for p in box_0 ]

    box_1_orig = [ transforms[0].interpolate(*p) for p in box_0_orig ]

    box_1 = [ affine(la.inv(to_ij(image_affine_transforms[1])), p) for p in box_1_orig ]

    pts_1_orig = [ transforms[0].interpolate(*p) for p in pts_0_orig ]
    pts_1 = [ affine(la.inv(to_ij(image_affine_transforms[1])), p) for p in pts_1_orig ]

    box_2_orig = [ transforms[1].interpolate(*p) for p in box_1_orig ]

    box_2 = [ affine(la.inv(to_ij(image_affine_transforms[2])), p) for p in box_2_orig ]


    pts_2_orig = [ transforms[1].interpolate(*p) for p in pts_1_orig ]
    pts_2 = [ affine(la.inv(to_ij(image_affine_transforms[2])), p) for p in pts_2_orig ]

    
    im1_new, bbox1 = add_box_scalebar_pts(image_x2[1], box_1, sc_length, pts_1, radii, draw_orig=False, offset_x =200)

    im2_new, bbox2 = add_box_scalebar_pts(image_x2[2], box_2, sc_length, pts_2, radii, draw_orig=False, offset_x=200)

    im1_inset_new, _ = add_box_scalebar_pts(image_x2[1][:,bbox1[0], bbox1[1]], None, sc_length, [], radii, draw_orig=False)
    im2_inset_new, _ = add_box_scalebar_pts(image_x2[2][:,bbox2[0], bbox2[1]], None, sc_length, [], radii, draw_orig=False)


    
    fig, ax = plt.subplots()
    ax.imshow(im1_new)

    fig, ax = plt.subplots()
    ax.imshow(im2_new)

    imwrite(out_path+'im_t1.png', im1_new)    
    imwrite(out_path+'im_t1_inset.png', im1_inset_new)


    imwrite(out_path+'im_t2.png', im2_new)    
    imwrite(out_path+'im_t2_inset.png', im2_inset_new)


out_path='output/'
Path(out_path).mkdir(exist_ok=True)
    
data_path = DATA_PATH+'/Brx/stretched/ds1/'

#ray.init()
map_box_multiple([data_path+'5-brx t0_proj.tif',
                  data_path+'5-brx t1_proj.tif',
                  data_path+'5-brx t3_proj.tif' ],
                  data_path+'5-brxl-seg-for-figure.tif',
                   #[250, 316, 852],
#                 [969, 326, 307],
#                  [167,326,969],
                 [(364,234), (501,351), (291, 716)],
                 [ data_path+'5-brxl-inverse-t1-t0.txt',
                   data_path+'5-brxl-inverse-t3-t1.txt' 
                   ],
                 [ data_path+'5-brxl-direct-t0-t1.txt',
                   data_path+'5-brxl-direct-t1-t3.txt' ],
                 [87, 90, 79])
                 



