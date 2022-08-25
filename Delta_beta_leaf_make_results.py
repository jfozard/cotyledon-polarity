
"""
Preprocess imaging data for leaves imaged at two timepoints,
measuring the changes in the angle beta between those timepoints

output to output/delta_beta_results -> a .pkl.gz file containing the two timepoints 
"""

from data_path import DATA_PATH

from delta_beta_tracked_comments import comments_red, comments_blue_purple

#### CROP larger
from re import M
import matplotlib
#matplotlib.use('agg')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#from PIL import Image
from math import atan2, pi, sqrt, sin, cos

import scipy.ndimage as nd
from matplotlib.collections import LineCollection
from skimage.segmentation import find_boundaries
from collections import defaultdict, Counter

import rpy2
from rpy2.robjects.packages import importr

from subprocess import run


import pandas as pd

import matplotlib as mpl

import pickle

import gzip

from collections import Counter

from process_utils import *

from skimage.measure import regionprops

mpl.rcParams.update({
    'figure.facecolor': 'none',
    'figure.edgecolor': 'none',
    'font.size': 20,
    'figure.dpi': 72,
    'font.family': "sans-serif",
    'font.sans-serif': 'Arial',
    'figure.subplot.bottom' : .15,
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',

})


from utils import *

from pathlib import Path


FS = 2
FS2 = 2.1
IM_S = 4

from imageio import imwrite
from PIL import Image, ImageDraw

import sys

image_res = 0.692
sc_length_microns = 100
sc_length_cell_microns=25
inset_scale = 4


a_bins = np.linspace(0,90,10)

def find_closest(array, value, eps=0.05):
    idx = (np.abs(array - value)).argmin()
    if np.abs(array[idx]-value)<=eps:
        return idx
    else:
        return None


def zoom(im, s):
    return np.dstack([nd.zoom(im[:,:,i], s) for i in range(im.shape[2])])





def calc_classify_lineage(results):

    classify_lineage = []

    components = results['components']
    c1_0 = results['leaf1_0'].get_centroids()

    arrows0 = results['leaf0'].get_arrows_keys(only_below_max_length=True)
    arrows1 = results['leaf1'].get_arrows_keys(only_below_max_length=True)

    arrows0_0 = results['leaf0_0'].get_arrows_keys(only_below_max_length=True)
    arrows1_0 = results['leaf1_0'].get_arrows_keys(only_below_max_length=True)


    # Look at the new arrows (from lineage2, so no pruning for size)
    for l in components:
        idx0 = list(l[0])
        idx1 = list(l[1])
        if len(idx0)==1 and len(idx1)<=2: # Sucessfully tracked
            d = [tuple(sorted([1*(i in arrows0) for i in idx0])), tuple([1*(i in arrows1) for i in idx1])]
            # check to see if arrow base has changed

            if len(idx0)==1 and sum(d[0])==1 and len(idx1)>1:
                # Identify which cell is closest to the arrow base in idx0

                arrow_base = arrows0_0[idx0[0]]
                div_cell_centroids = [ c1_0[i] for i in idx1 ]
        #        parent_cell_centroid = c0_0[idx0[0]]

                div_cell_centroid_dists = [la.norm(np.array(c) - np.array(arrow_base)) for c in div_cell_centroids ]
                sort_idx = np.argsort(div_cell_centroid_dists)
                l2 = (idx0, [idx1[i] for i in sort_idx])
        # Need the lineage objects to be sorted lists, not sets

                new_d1 = tuple(d[1][i] for i in sort_idx)
                d[1] = new_d1

            else:
                sort_idx = np.argsort(d[1])
                d[1] = tuple([d[1][i] for i in sort_idx]) # Pick only one order when no polarity of parent cell
                l2 = (idx0, [idx1[i] for i in sort_idx])

            d = tuple(d)
            classify_lineage.append((d, l2))

    classify_lineage.sort(key=lambda u: u[0])

    return classify_lineage

def measure_beta(results):   ### Needs rehash

    arrows0 = results['leaf0'].get_arrows()
    arrows1 = results['leaf1'].get_arrows()

    theta0 = results['leaf0'].get_extra_props()['theta']
    theta1 = results['leaf1'].get_extra_props()['theta']

    beta1 = {}
    beta1_noadjust = {}
    for a in arrows0:
        a.get_extra_props()['beta'] = calc_angle_beta((a.start, a.centroid_end), 90-theta0)
        a.get_extra_props()['beta_end'] = a.centroid_end


    # For beta1, more complex as need to find parent cell and centroid; so it's not just
    # calculated using the arrow angles.

    components = results['components']

    seg1 = results['leaf1'].get_cells(valid_only=True)

    arrows_beta1 = {}
    arrows_beta1_noadjust = {}

    for a in arrows1:
        # Find component in T1 containing cell
        c = get_component(components, a.cell_idx)
        if c is None or len(c[1])==1:
            beta1= calc_angle_beta((a.start, a.centroid_end), 90-theta1)
            beta1_noadjust = beta1
            a.get_extra_props()['beta_end'] = a.centroid_end
            a.get_extra_props()['beta'] = beta1
            a.get_extra_props()['beta_noadjust'] = beta1_noadjust
        else:
            beta1_noadjust = calc_angle_beta((a.start, a.centroid_end), 90-theta1)
            # Centroid of combined daughter cells
            cc = swap_ij(nd.center_of_mass(np.isin(seg1, list(c[1]))))
            a.get_extra_props()['beta_end'] = cc
            beta1 = calc_angle_beta((a.start, cc), 90-theta1)
            a.get_extra_props()['beta'] = beta1
            a.get_extra_props()['beta_noadjust'] = beta1_noadjust
        print('arrow1', a.centroid_end, a.get_extra_props()['beta_end'])



def tracking_summary(results, cl2, of=sys.stdout):

    # cl2 = dictionary k -> x
    # k = ( tuple of marker at T0, tuple of marker at T1) e.g. ( (0,), (0,1) )
    # x = list of

    print('Track summary', file=of)
    print('###################', file=of)
    print([(k, len(x)) for k, x in cl2.items()], file=of)



    seg0 = results['leaf0'].get_cells(valid_only=True)
    arrows0 = results['leaf0'].get_arrows_keys()


    n1 = len(np.unique(seg0)) - 1
    print(np.unique(seg0), file=of)
    print('n1 = all cells in segmentation', n1, file=of)
    results['n1'] = n1

    n2 = len(arrows0)
    print('n2 = all cells with marker at t0', n2, file=of)
    results['n2'] = n2

    n_tracks = sum([len(x) for k,x in cl2.items() if len(k)==2])

    print('n0 = number of tracked cells (at T0)', n_tracks)
    results['n0'] = n_tracks

    print('all good tracks ', n_tracks, file=of)

    n_t0 = sum([sum([len(u[0]) for u in x])  for k,x in cl2.items() if len(k)==2])

    print('tracked cells (t0)', n_t0, file=of)

    n4 = sum([len(x) for k,x in cl2.items() if len(k)==2 and k[0]==(1,)])
    n4_check = len(cl2[((1,),(0,))]) + len(cl2[((1,),(1,))]) + len(cl2[((1,),(0,0))]) + len(cl2[((1,),(1,0))]) + len(cl2[((1,),(0,1))]) + len(cl2[((1,),(1,1))])

    assert n4 == n4_check

    print('n4 = tracked cells (at t0) with marker initially', n4, file=of)
    results['n4'] = n4


    print('tracked cells which retain marker no division', len(cl2[((1,),(1,)  )]), file=of)
    print('tracked cells which retain marker through division', len(cl2[((1,),(1,0)  )])+ len( cl2[((1,),(1,1)  )] ), file=of)
    print('tracked cells which retain marker through division in *both* children', len( cl2[((1,),(1,1)  )] ), file=of)


    n3 = sum([len(x) for k, x in cl2.items() if len(k)==2 and sum(k[0]) and k[1][0] ])
    n3_check = len(cl2[((1,),(1,))]) + len(cl2[((1,),(1,0))]) + len(cl2[((1,),(1,1))])

    assert n3 == n3_check

    print('n3 = tracked cells (at t0) which retain marker through division', n3, file=of)
    results['n3'] = n3

    print('tracked cells which lose marker no division', len(cl2[((1,),(0,)  )]), file=of)
    print('tracked cells which lose marker through division', len(cl2[((1,),(0,0)  )] ), file=of)
    print('tracked cells which lose marker through division but gain in other child', len( cl2[((1,),(0,1)  )] ), file=of )

    n5 = sum([len(x) for k, x in cl2.items() if len(k)==2 and k[0]==(1,) and k[1][0]==0])
    n5_check = len(cl2[((1,),(0,))]) + len(cl2[((1,),(0,0))]) + len(cl2[((1,),(0,1))])

    assert n5 == n5_check

    print('n5 = tracked cells with marker at T0 losing marker', n5, file=of)
    results['n5'] = n5

    assert n3 + n5 == n4


    n6 = len(cl2[((0,),(0,1))]) + len(cl2[((0,),(1,))]) + 2*len(cl2[((0,),(1,1))]) + len(cl2[((1,),(0,1))]) + len(cl2[((1,),(1,1))])

    print('n6 = Number of sucessfully tracked cells at T1 in which the marker appeared at a new location', n6, file=of)
    results['n6'] = n6

    n7 = len(cl2[(0,),(0,1)]) + 2*len(cl2[(0,),(1,1)]) + len(cl2[(1,),(0,1)]) + len(cl2[(1,),(1,1)])

    print('n7 = Number of sucessfully tracked cells at T1 in which the marker appeared at a new location through division', n7, file=of)
    results['n7'] = n7


    n_div = sum(len(x) for k, x in cl2.items() if len(k)==2 and len(k[0])==1 and len(k[1])==2)
    print('number of tracked divisions', n_div, file=of)

    n8 =  sum([ sum([len(u[1]) for u in x]) for k, x in cl2.items() if len(k)==2])
    n8_check = sum([ len(k[1])*len(x) for k, x in cl2.items() if len(k)==2])
    n_t1 = n8

    assert n8 == n8_check

    print('n8 = Number of cells at t1 tracked', n8, file=of)
    results['n8'] = n8


    print('dividing cells', n_div, file=of)

    assert n_t0 + n_div == n_t1


    n9 =  sum([sum(k[1])*len(x) for k, x in cl2.items() if len(k)==2 ])
    n9_check = len(cl2[(1,),(1,)]) + len(cl2[(0,),(1,)]) + len(cl2[(0,),(1,0)]) + len(cl2[(1,),(1,0)]) +  len(cl2[(1,),(0,1)]) + 2* len(cl2[(0,),(1,1)]) +  2* len(cl2[(1,),(1,1)])

    print('n9 = Number of tracked cells with marker at t1', n9, file=of)
    results['n9'] = n9

    seg1 = results['leaf1'].get_cells(valid_only=True)
    arrows1 = results['leaf1'].get_arrows_keys()


    n10 = len(arrows1)
    n11 = len(np.unique(seg1)) - 1

    print('n10 = Number of cells with marker at t1', n10, file=of)
    print('n11 = Number of all cells at t1', n11, file=of)


    n_marker_t0 = n4
    n_marker_t1 = n9
    n_lost = n5
    n_gain = n6
    assert n_marker_t1 == n_marker_t0 - n5 + n6

        
from leaf import *

def make_summary_affine_both(name, im0_fn, seg0_fn, im1_fn, seg1_fn, arrows0_fn, arrows1_fn, transform_fn, transform_pts_fn,  theta0, theta1, of=sys.stdout, flip_t0=False, auto_reverse=False):


    if of:
        print(f'flip_t0={flip_t0}', file=of)

    results = {}

    leaf0 = process_leaf_simple(name+'_t0', im0_fn, seg0_fn, arrows0_fn, theta0, flip_y=flip_t0, return_sc=False, return_seg_bdd=False, auto_reverse=auto_reverse)
 #   leaf1 = process_leaf_simple(name+'_t5', im1_fn, seg1_fn, arrows1_fn, theta1, return_sc=False, return_seg_bdd=False)

    seg0 = leaf0.get_cells(valid_only=True)

    
    print('---->', name, len(leaf0.arrows['default']), flip_t0)

    if flip_t0:
        theta0 = - theta0

    ## results['theta0'] = theta0 # Angle of frame0 midline measured by experimentalist. Anticlockwise from x-axis
    ## results['theta1'] = theta1 # Angle of frame1 midline measured by experimentalist. Anticlockwise from x-axis

    transform = BTransform2.from_file(transform_fn,  (1024,1024)) # BunwarpJ transform between frame 0 and frame 1 (check direction)
    transform_pts = BTransform2.from_file(transform_pts_fn, (1024,1024)) # BunwarpJ transform between frame 1 and frame 0 (check direction)

    results['transform'] = transform
    results['transform_pts'] = transform_pts

    d_transform = DeformableTransform(transform, transform_pts)

    d_transform_1 = DeformableTransform(transform_pts, transform)

    if (seg0_fn is not None):
        if seg1_fn is None:
            seg1 = d_transform_1.map_image((seg0, np.ones_like(seg0)))

            leaf1 = process_leaf_simple(name+'_t5', im1_fn, seg1_fn, arrows1_fn, theta1, return_sc=False, return_seg_bdd=False, seg=seg1, auto_reverse=auto_reverse)
        else:
            leaf1 = process_leaf_simple(name+'_t5', im1_fn, seg1_fn, arrows1_fn, theta1, return_sc=False, return_seg_bdd=False, auto_reverse=auto_reverse)
    else:
        leaf1 = process_leaf_simple(name+'_t5', im1_fn, seg1_fn, arrows1_fn, theta1, return_sc=False, return_seg_bdd=False, auto_reverse=auto_reverse)


    

    # Now transform to aligned coordinates

    leaf1_orig0 = LeafView(leaf=leaf1, transform=d_transform)
    

    # Transform t0 into rotated /scaled version

    IM_S = 2

    # Rotation by (90-theta0) - mapping points in orig frame0 to those in aligned frame 0
    m0_0 = get_rotation_transform(90-theta0)

    m_0 = la.inv(m0_0)
    # Inverse+coordinate order swap of rotation by (90-theta0) - used to rotate orig image to frame 0
    affine_m_0 = m_0

    affine_m_0_x2 = affine_m_0 @ np.diag([1/IM_S,1/IM_S,1])

    transform0_0 = AffineTransform(affine_m_0_x2, (1024*IM_S,1024*IM_S))

    leaf0_0 = LeafView(leaf=leaf0, transform = transform0_0)

    IM_S4 = 4

    affine_m_0_x4 = affine_m_0 @ np.diag([1/IM_S4,1/IM_S4,1])

    transform0_0_x4 = AffineTransform(affine_m_0_x4, (1024*IM_S4,1024*IM_S4))

    leaf0_0_x4 = LeafView(leaf=leaf0, transform = transform0_0_x4)

    #plot_leaf_arrows(leaf0_0)

    transform1_0 = CompositeTransform([transform0_0, d_transform])
    leaf1_0 = LeafView(leaf=leaf1, transform = transform1_0)



    #### T1 segmentation / arrows

    m0_1 = get_rotation_transform(90-theta1)
    m_1 = la.inv(m0_1)
    # Inverse+coordinate order swap of rotation by (90-theta0) - used to rotate orig image to frame 0
    affine_m_1 = m_1

    affine_m_1_x2 = affine_m_1.dot(np.diag([1/IM_S,1/IM_S,1]))

    transform1_1 = AffineTransform(affine_m_1_x2, (1024*IM_S,1024*IM_S))

    leaf1_1 = LeafView(leaf=leaf1, transform = transform1_1)

    affine_m_1_x4 = affine_m_1.dot(np.diag([1/IM_S4,1/IM_S4,1]))

    transform1_1_x4 = AffineTransform(affine_m_1_x4, (1024*IM_S4,1024*IM_S4))

    leaf1_1_x4 = LeafView(leaf=leaf1, transform = transform1_1_x4)

    
    transform0_1 = CompositeTransform([transform1_1, d_transform_1])

    leaf0_1 = LeafView(leaf=leaf0, transform = transform0_1)

    results = {'leaf0': leaf0, 'leaf1': leaf1,
               'leaf0_0': leaf0_0, 'leaf1_0': leaf1_0,
               'leaf0_1': leaf0_1, 'leaf1_1': leaf1_1,
               'theta0': theta0, 'theta1': theta1,
               'leaf0_0_x4': leaf0_0_x4, 'leaf1_1_x4': leaf1_1_x4,
               }


    #### Look at cell lineage from segmentation

    lineage2 = get_lineage(leaf0.get_cells()[0], leaf1_orig0.get_cells()[0], leaf1_orig0.get_cells()[1])
    # Lineage map - list of pairs of cell indices, first in segmentation of frame0, second in segmentation of frame1

    components = calc_bipartite_components(lineage2)
    # Connected components of this lineage map [ set(cells in frame0), set(cells in frame1), list(lineage links)]


    results['components'] = components


    classify_lineage = calc_classify_lineage(results)
    results['classify_lineage'] = classify_lineage
    # Lineage map classified according to the presence of arrows in each of the cells at


    cl2 = defaultdict(list)
    for d, l in classify_lineage:
        cl2[d].append(l)

    results['classify_lineage'] = classify_lineage
    results['cl2'] = cl2



    if of:
        of.write('\nAnalysis of image : ' +name +'\n\n')
        tracking_summary(results, cl2, of=of)

    measure_beta(results)


    # Calculate delta beta


    retained_cells0 = []
    retained_cells1 = []

    delta_beta = {}
    delta_beta_no_div = []
    delta_beta_div = []

    # Non-dividing cells ([1], [1])
    tracked_cells_no_div = cl2[((1,),(1,))]


    arrows0 = leaf0.get_arrows_dict(only_below_max_length=True)
    arrows1 = leaf1.get_arrows_dict(only_below_max_length=True)


    if of:
        print('tracked_cells_no_div', tracked_cells_no_div, file=of)

    for t in tracked_cells_no_div:
        idx0 = t[0][0]
        idx1 = t[1][0]

        beta0 = arrows0[idx0].get_extra_props()['beta']
        beta1 = arrows1[idx1].get_extra_props()['beta']


        d_beta = beta1 - beta0
        delta_beta_no_div.append(d_beta)
        delta_beta[idx0] = d_beta
        retained_cells0.append(idx0)
        retained_cells1.append(idx1)


    # Dividing cells ([1], [1,0]) and ([1], [1,1])
    tracked_cells_div = cl2[((1,),(1,0))] + cl2[((1,), (1,1))]

    if of:
        print('tracked_cells_div', tracked_cells_div, file=of)

    for t in tracked_cells_div:
        idx0 = t[0][0]
        idx1 = t[1][0]

        beta0 = arrows0[idx0].get_extra_props()['beta']
        beta1 = arrows1[idx1].get_extra_props()['beta']


        d_beta = beta1 - beta0

        delta_beta_div.append(d_beta)
        delta_beta[idx0] = d_beta

        retained_cells0.append(idx0)
        retained_cells1.append(idx1)



    results['delta_beta'] = delta_beta
    results['delta_beta_div'] = delta_beta_div
    results['delta_beta_no_div'] = delta_beta_no_div
    results['retained_cells0'] = retained_cells0
    results['retained_cells1'] = retained_cells1




    beta0_loss = {}

    for i, a in arrows0.items():
        if a.cell_idx not in retained_cells0:
            beta0 = a.get_extra_props()['beta']
            beta0_loss[i] = beta0

    beta1_gain = {}

    for i,a in arrows1.items():
        if a.cell_idx not in retained_cells1:
            beta1 = a.get_extra_props()['beta']
            beta1_gain[i] = beta1



    class0 = {}
    for d, l in classify_lineage:
        if d[0]==(1,):
            if d[1][0] == 1:
                class0[l[0][0]] = 'retained'
            else:
                class0[l[0][0]] = 'lost'

    for i, a in arrows0.items():
        if i not in class0:
            class0[i] = 'untracked'


    class1 = {}

    for d, l in classify_lineage:
        if d[0]==(0,):
            if d[1][0] == 1:
                class1[l[1][0]] = 'gained'
            if len(d[1])>1 and d[1][1]==1:
                class1[l[1][1]] = 'gained'
        elif d[0] == (1,):
            if d[1] == (0,1):
                class1[l[1][1]] = 'gained'
            if d[1] == (1,1):
                class1[l[1][0]] = 'retained'
                class1[l[1][1]] = 'gained'
            if d[1] == (1,0):
                class1[l[1][0]] = 'retained'
            if d[1] == (1,):
                class1[l[1][0]] = 'retained'
        else:
            print(d)

    for i, a in arrows1.items():
        if i not in class1:
            class1[i] = 'untracked'

    assert(len(class0)==len(arrows0))

    assert(len(class1)==len(arrows1))

    if of:
            print(Counter(class1.values()), file=of)


    # beta0 / beta1

    beta0 = dict([(k,a.get_extra_props()['beta']) for k,a in arrows0.items() if a.below_max_length])
    beta1 = dict([(k,a.get_extra_props()['beta']) for k,a in arrows1.items() if a.below_max_length])


    return results, delta_beta_no_div, delta_beta_div, beta0, beta1, class0, class1





def process_data(base, size_ds, use_seg=lambda d: False, auto_reverse= lambda d: False, output_results=None):

        stacked_panels = []

       # of = open(base+'_summary.txt', 'w')

        # Drive this from the spreadsheet

        ids = size_ds[0]['Identifier']
        N_cot = len(ids)

        def remove_cot(s):
            if 'cot' in s:
                n = s.index('cot')
                return s[:n]+s[n+4:]
            else:
                return s
        
        N_plants = len(set([remove_cot(s) for s in ids]))

        #print(f'N_cots {N_cot} N_plants {N_plants}', file=of)


        
        out_path =base+'_analysis/'

        results_db = {}
        results_all = []

        do_plots=True

        #for i in list(range(0,2)) + list(range(3, len(size_ds[0]))):
        for i in list(range(len(size_ds[0]))):
            d0 = size_ds[0].iloc[i]
            d5 = size_ds[1].iloc[i]

            id0 = d0['Filename']
            id5 = d5['Filename']

            print('run', i, id0, id5)
	    #'Man' in d0.Path
            #if '5-brx' not in id0:   ### TEMP
            #    continue
            
            results  = make_summary_affine_both(
                d0['Identifier'],
                data_path + d0['Path'] + '/' + id0+'_proj.tif',
                data_path + d0['Path'] + '/' + id0+'_seg.tif' if use_seg(d0) else None,
                data_path + d5['Path'] + '/' + id5+'_proj.tif',
                data_path + d0['Path'] + '/' + id5+'_seg.tif' if use_seg(d5) else None,
                data_path + d0['Path'] + '/' + id0+'_RoiSet.zip',
                data_path + d5['Path'] + '/' + id5+'_RoiSet.zip',
                data_path + d0['Path'] + '/' + id0[:-3]+'-inverse-'+id5[-2:] + '-' + id0[-2:] + '.txt',
                data_path + d0['Path'] + '/' + id0[:-3]+'-direct-'+id0[-2:] + '-' + id5[-2:] + '.txt',
                d0['Angle'],
                d5['Angle'],
                flip_t0=d0['Flipped_t0'],
                auto_reverse = auto_reverse(d0),
            )

            if output_results:
                with gzip.open(output_results+d0['Identifier']+'.pkl.gz', 'wb') as f:
                    pickle.dump(results, f)

from pathlib import Path

from get_leaf_dataframe import get_paired_dataframe

base_all = 'output/'

if __name__=='__main__':

    data_path = DATA_PATH

    results_output_path = base_all+'delta_beta_results/'

    Path(results_output_path).mkdir(exist_ok=True, parents=True)

    base = base_all+'delta_beta'
    size_ds = get_paired_dataframe(marker='brxl', category='stretched')

    
    print(size_ds)
    print('Number of stretched leaves', len(size_ds[0]))


    process_data(base, size_ds, output_results=results_output_path)

    print(' \n\n\n ===== \n\n\n DONE STRETCH \n\n\n =====')
    

    base = base_all+'delta_beta_control'
    size_ds = get_paired_dataframe(marker='brxl', category='control')

    print(size_ds)
    print('Number of control leaves', len(size_ds))


    process_data(base, size_ds, output_results=results_output_path)


    base = base_all+'delta_beta_basl'
    size_ds = get_paired_dataframe(marker='35S_basl', category='stretched')

    print(size_ds)
    print('Number of stretched BASL leaves', len(size_ds[0]))

    
    process_data(base, size_ds, use_seg=lambda d: 'ds1' in d.Path, auto_reverse=lambda d: 'ds1' in d.Path, output_results=results_output_path)

