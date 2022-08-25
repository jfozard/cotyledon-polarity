
# Make histograms of angle alpha and draw Rose plots
# Routines used for both brxl and basl plots


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
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
import sys

import pickle

import scipy.ndimage as nd

from process_utils import *

from utils import binom_pvalue, binom_sig, regionprop_dict, leaf_boundary, MathTextSciFormatter

from apply_double_style import test_unet

from pathlib import Path


import matplotlib as mpl

import pandas as pd

from rose_plot import rose_plot

from math import pi

from dataclasses import dataclass

from stacked_histogram import stacked_histogram

from leaf import process_leaf_simple, AffineTransform, LeafView

from statsmodels.stats.proportion import binom_test
from scipy.stats import chi2_contingency, ks_2samp, ttest_ind

import matplotlib as mpl

mpl.rcParams.update({
    'figure.facecolor': 'none',
    'figure.edgecolor': 'none',
    'font.size': 28,
    'figure.dpi': 72,
    'font.family': "sans-serif",
    'font.sans-serif': 'Arial',
    'figure.subplot.bottom' : .15,
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    'mathtext.default':  'regular',
})


YLABEL_SIZE=28
XLABEL_SIZE=28
TITLE_SIZE=24
XTICK_SIZE=24
YTICK_SIZE=24

leaf_output_path = 'output/leaf_data/'


ANGLE_LOW = 80
ANGLE_HIGH= 100
D_ANGLE = ANGLE_HIGH-ANGLE_LOW

r = (ANGLE_LOW, ANGLE_HIGH)


F_STRING = '$n={}$ $r={:.1f}$% $p={}$ '
F_STRING2 = '$n_L={}$ $n_R={}$ $p={}$ '
bh = 0.05
FS_X = 25
FS_Y = 20


def get_leaf_bdd_transform(signal):
    im = nd.gaussian_filter(signal, 20)
    interior = im > 0.8*threshold_otsu(im)
    interior = nd.binary_fill_holes(interior)
    bdd = find_boundaries(interior)
    return bdd

def arrow_leaf_plot(leaf_0, plot_orig=True, max_length=100):
    
    # Outlines of all cells with annotated arrows

    # Choose colormap
    colormap = cm.hsv
    
    # Plot Wall data, Signal data, Annotated cell outlines (RGB) + while outline for whole leaf
    fig1, ax1 = plt.subplots(figsize=(20,20))
    ax1.set_position([0,0,1,1]) # Use whole axes space

    arrows_0 = leaf_0.get_arrows_dict()
    
    # Image in aligned frame

    bdd = leaf_0.get_bdd(valid_only=True)
    signal = leaf_0.get_signal(valid_only=True)
    cells = leaf_0.get_cells(valid_only=True)

    cell_bdd = find_boundaries(cells) #*np.isin(leaf.cells, arrow_idx)
    B = np.dstack((bdd*(1-cell_bdd), signal, cell_bdd*(64+191*np.isin(cells, list(arrows_0)))))
    s =  nd.binary_dilation(get_leaf_bdd_transform(bdd))
    B[:,:,2] = np.maximum(B[:,:,2], 255*s)

    ax1.imshow(B)

    # Process arrows -> if not v long (likely mistake / arrow pointing into unsegmented cell )
    # Add arrow to image, subtract manually estimated direction (Man/Jie estimate, not mine)
    # (NOT USED) Add angle to centroid_angles array, add (mapped arrow position, colour and angle) to centroid_arrows.

    for i in arrows_0:
        a = arrows_0[i]
        if a.below_max_length:
            d = a.adjusted_centroid_angle
            ax1.arrow(*a.to_mpl_arrow(), color=colormap((d+180)/360.0), width=1, alpha=0.8)
    
    return fig1



def process_data(prefix, ds, hl=[], only_hist=False, marker='brxl', split_sizes=[], font_scale=1.0, leaf_data_path=leaf_output_path):


    of_leaves = open(prefix+'leaf-summary.txt', 'w')
    ids = ds['Identifier']
    N_cot = len(ids)

    def remove_cot(s):
        if 'cot' in s:
            n = s.index('cot')
            return s[:n]+s[n+4:]
        else:
            return s
        
    N_plants = len(set([remove_cot(s) for s in ids]))

    print(f'N_cots {N_cot} N_plants {N_plants}', file=of_leaves)

    of_leaves.close()

    
    of = open(prefix + '_chi2_data.txt', 'w')
    print(prefix + '_chi2_data.txt', 'w')

    
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
    'xtick.labelsize': XTICK_SIZE,
        'ytick.labelsize': YTICK_SIZE,
        
    })

    
    cat_leaves = []
    cat_leaves_0 = []

    idx = 0

    bins = np.linspace(0, 180, 19)
    bin_c = 0.5*(bins[1:] + bins[:-1])


    print('?', ds.Identifier)


    data = []
    for i in ds.index:
        d = ds.loc[i]
        data.append(d.Identifier)

    print('###', data)

    for n in data:
        print('name >>>', n)

#        with open(leaf_output+n,'rb') as f:
#            leaf_data = pickle.load(f)

        with open(leaf_data_path+n+'_0','rb') as f:
            print('loaded', leaf_data_path+n+'_0')
            leaf_data_0 = pickle.load(f)

        leaf_data = leaf_data_0.leaf
        
        cat_leaves.append(leaf_data)
        cat_leaves_0.append(leaf_data_0)



    n = len(cat_leaves)
    nr = (n+4)//5
    fig, ax = plt.subplots(nr, 5, figsize=(5*5, nr*5))
    ax = ax.flatten()
    for i in range(n):
        a = ax[i]
        leaf = cat_leaves[i]
        w = leaf.width
        centroid_angles = leaf.get_adjusted_centroid_angles()
        h = np.histogram(np.abs(centroid_angles), bins=bins)[0]

        
        a.hist(bin_c, bins = bins, weights = h)
        nh = np.sum(h)

        n_cells = leaf.get_total_cell_number() 
        
        a.title.set_text(f'{w} {nh} {nh/n_cells:0.2f}')
        a.title.set_fontsize(TITLE_SIZE)
        
        a.set_xlabel('$\\beta$', fontsize=XLABEL_SIZE)
        a.set_ylabel('Number of cells', fontsize=YLABEL_SIZE)
        a.set_xticks([0,90,180])
        a.set_xlim(0,180)


    for a in ax[n:]:
        a.set_axis_off()
        
    plt.tight_layout()

    plt.savefig(prefix+'histogram_grid.svg')
    plt.savefig(prefix+'histogram_grid.png')
    plt.close()


    # Split the individual histograms according to cell area

    rp = [ regionprop_dict(l.get_cells(valid_only=True)) for l in cat_leaves]

    all_cell_areas = [ ]

    for l, r in zip(cat_leaves, rp):
        for a in l.get_arrows():
            if a.below_max_length:
                all_cell_areas.append(r[a.cell_idx].area)

    # Cell area median for this category only
    a_median = np.median(all_cell_areas)


    n = len(cat_leaves)
    nr = (n+4)//5
    fig, ax = plt.subplots(nr, 5, figsize=(5*5, nr*5))
    ax = ax.flatten()
    for i in range(n):
        a = ax[i]
        leaf = cat_leaves[i]
        w = leaf.width
        centroid_angles = leaf.get_adjusted_centroid_angles()

        cell_areas = np.array([ rp[i][a.cell_idx].area for a in leaf.get_arrows() if a.below_max_length ])

        angles_small = centroid_angles[cell_areas<a_median]

        angles_large = centroid_angles[cell_areas>=a_median]
        
        h_small = np.histogram(np.abs(angles_small), bins=bins)[0]
        h_large = np.histogram(np.abs(angles_large), bins=bins)[0]


        a.hist([bin_c, bin_c], bins = bins, weights = [h_small, h_large], stacked=True)
        nh_small = np.sum(h_small)
        nh_large = np.sum(h_large)
        
        n_cells = leaf.get_total_cell_number() 
        
        a.title.set_text(f'{w} {nh_small} {nh_large} {(nh_small+nh_large)/n_cells:0.2f}')
        
        a.set_xlabel('$\\beta$', fontsize=XLABEL_SIZE)
        a.set_ylabel('Number of cells', fontsize=YLABEL_SIZE)
        a.set_xticks([0,90,180])
        a.set_xlim(0,180)

    for a in ax[n:]:
        a.set_axis_off()
        
    plt.tight_layout()

    plt.savefig(prefix+'histogram_sizes_grid.svg')
    plt.savefig(prefix+'histogram_sizes_grid.png')
    plt.close()


    all_centroid_angles2 = [a for l in cat_leaves for a in l.get_adjusted_centroid_angles() ]


    # Histogram of all angles alpha - with chi2 test for transverse excess

    plt.figure()
    plt.hist(np.abs(all_centroid_angles2), bins=18, range=(0, 180))
    plt.xlim([0, 180])
    plt.ylabel("Number of cells", fontsize=YLABEL_SIZE*font_scale)
    plt.xlabel("$\|\\alpha\|$", fontsize=XLABEL_SIZE*font_scale)
    plt.gca().tick_params(axis="x", labelsize=XTICK_SIZE*font_scale)
    plt.gca().tick_params(axis="y", labelsize=YTICK_SIZE*font_scale)
    
    data = np.abs(all_centroid_angles2)
    p, d = binom_pvalue(data)

    plt.plot((ANGLE_LOW, ANGLE_HIGH), (-bh, -bh), 'r-', clip_on=False, zorder=100, lw=5)

    formatter = MathTextSciFormatter()
    
    plt.title(F_STRING.format(len(data),  d[0]*100, formatter(p)), color='r' if p<0.05 else 'k', fontsize=TITLE_SIZE*font_scale)
    plt.ylim(bottom=0)
    plt.savefig(prefix+'rose_angle_all.svg')
    plt.close()

    # Histograms for different size ranges

    range_angles = {}
    if split_sizes:
        size_ranges = [float('-inf')] + list(split_sizes) + [float('+inf')]
        size_ncot = []
        for j in range(len(size_ranges)-1):
            low_size = size_ranges[j]
            high_size = size_ranges[j+1]
            range_angles[j] = [a for l in cat_leaves for a in l.get_adjusted_centroid_angles() if low_size<=l.width<high_size]
            size_ncot.append(len([l for l in cat_leaves if low_size<=l.width<high_size]))
            plt.figure()

            data = np.abs(range_angles[j])
            plt.hist(data, bins=18, range=(0,180))

            plt.ylabel("Number of cells", fontsize=YLABEL_SIZE)
            plt.xlabel("$\|\\alpha\|$", fontsize=XLABEL_SIZE)
    
            p, d = binom_pvalue(data)

            plt.plot((ANGLE_LOW, ANGLE_HIGH), (-bh, -bh), 'r-', clip_on=False, zorder=100, lw=5)

    
            plt.title(F_STRING.format(len(data),  d[0]*100, formatter(p)), fontsize=TITLE_SIZE)
            plt.xlim(0, 180)
            plt.ylim(bottom=0)
            plt.savefig(prefix+f'split_angle_{j}.svg')
            plt.close()
            with open(prefix+f'range-{j}-alpha.txt', 'w') as a_of:
                for a in range_angles[j]:
                    print(a, file=a_of)

            
        of_compare = open(prefix+'compare_split_angle.txt', 'w')

        print('N Cots ', str(size_ncot), file=of_compare)
        # Compare these histograms
        Nj = len(size_ranges)-1
        for j in range(0, Nj-1):
            for jj in range(j+1, Nj):
                print(f'Compare ranges {j} {jj}', file=of_compare)
                compare_hists(range_angles[j], range_angles[jj], f'range {j}', f'range {jj}', of=of_compare)
        of_compare.close()
    plt.figure()
    plt.hist(np.abs(all_centroid_angles2), bins=18, range=(0, 180))
    plt.xlim([0, 180])
    plt.ylabel("Number of cells", fontsize=YLABEL_SIZE)
    plt.xlabel("$\|\\alpha\|$", fontsize=XLABEL_SIZE)
    
    data = np.abs(all_centroid_angles2)
    p, d = binom_pvalue(data)

    plt.plot((ANGLE_LOW, ANGLE_HIGH), (-bh, -bh), 'r-', clip_on=False, zorder=100, lw=5)

    formatter = MathTextSciFormatter()
    
    plt.title(F_STRING.format(len(data),  d[0]*100, formatter(p)), fontsize=TITLE_SIZE)
    plt.ylim(bottom=0)
    plt.savefig(prefix+'rose_angle_all_nored.svg')
    plt.close()


    
    if only_hist:
        return np.abs(all_centroid_angles2)

    # Write centroid angles out

    with open(prefix+'-alpha.txt', 'w') as a_of:
        for a in all_centroid_angles2:
            print(a, file=a_of)
    
    # Plot of all control leaf outlines and mapped arrows

    plt.figure(figsize=(20,20))

    all_outlines = np.array([leaf_boundary(l.get_bdd(valid_only=True)) for l in cat_leaves_0])

    bg = np.max(all_outlines, axis=0)
    plt.imshow(0.25*bg, cmap=plt.cm.gray, clim=(0,1))



    all_centroid_arrows = []
    for l in cat_leaves_0:
        for a in l.get_arrows():
            if a.below_max_length:
                all_centroid_arrows.append(a)

    # Draw all remapped, centroid-end arrows onto figure

    for a in all_centroid_arrows:
        plt.arrow(*a.to_mpl_arrow(), color=cm.hsv((a.adjusted_centroid_angle+180)/360.0), width=2)
    

    plt.xlim(0,1024)
    plt.ylim(1024,0)

    plt.savefig(prefix+'aligned-rainbow-arrows.png')
    plt.close()

    ### Lots of histograms

    # histograms split by whether the x coordinate of the arrow base is to the left (blue) or right (yellow) of the midline
    plt.figure()
    data_L = np.array([p for a, p in zip(all_centroid_arrows, all_centroid_angles2) if a.start[0]<512])
    data_R = np.array([p for a, p in zip(all_centroid_arrows, all_centroid_angles2) if a.start[0]>=512])

    plt.hist(data_L, bins=18, alpha=0.7, histtype='step', color='b', lw=3)
    plt.hist(data_R, bins=18, alpha=0.7, histtype='step', color='y', lw=3)
    plt.xlabel('$\\alpha$', fontsize=XLABEL_SIZE)
    plt.ylabel('Number of cells', fontsize=YLABEL_SIZE)
    plt.xticks([-180,-90,0,90,180])
    plt.gca().set_xlim(-180,180)


    table = [ [np.sum(data_L<0), np.sum(data_L>=0)], [np.sum(data_R<0), np.sum(data_R>=0)] ]

    print(prefix + ' chi2 contingency LR ', table, ' : ', chi2_contingency(table), file=of)
    
    p = chi2_contingency([ [np.sum(data_L<0), np.sum(data_L>=0)], [np.sum(data_R<0), np.sum(data_R>=0)] ])[1]


    print('aligned FIG SIZE', plt.gcf().get_size_inches(), plt.gcf().dpi)
    plt.title(F_STRING2.format(len(data_L), len(data_R), formatter(p)), fontsize=TITLE_SIZE)
    plt.savefig(prefix+'aligned-histogram-lr.svg')
    plt.close()


    def new_cmap(a):
        if -ANGLE_HIGH<a<-ANGLE_LOW:
            return (1,1,0)
        elif ANGLE_LOW<a<ANGLE_HIGH:
            return (1,0,1)
        else:
            return (0.1, 0.1, 0.1)


        
    plt.figure(figsize=(20,20))
    print(np.array(all_outlines).shape)
    bg = np.max(np.array(all_outlines), axis=0)
    print(np.max(bg))
    plt.imshow(0.25*bg, cmap=plt.cm.gray, clim=(0,1))
    for a in all_centroid_arrows:
        plt.arrow(*a.to_mpl_arrow(), color=cm.hsv((a.adjusted_centroid_angle+180)/360.0), width=2)
        # plt.arrow(*(a[0]), color=new_cmap(a[2]), width=2)

    ax = plt.gca()
    plt.xlim(0,1024)
    plt.ylim(1024,0)

    plt.savefig(prefix+'horiz-arrows.png')
    plt.close()

    of2 = open(prefix+'rose-limits.txt','w')

    if marker=='native_basl':
        fig_rose,  fig_colbar, fig_cols, fig_rows, fig_grid = rose_plot([[np.concatenate((a.start, a.centroid_end-a.start))] for a in all_centroid_arrows], all_centroid_angles2, all_outlines, tests = ['basal', 'medial'],   p_alpha = 0.05, min_length_arrow=1000, of=of2)
    elif marker=='35S_basl':
        fig_rose,  fig_colbar, fig_cols, fig_rows, fig_grid = rose_plot([[np.concatenate((a.start, a.centroid_end-a.start))] for a in all_centroid_arrows], all_centroid_angles2, all_outlines, mean_arrow_type='circular', tests = ['apical', 'medial'], apical_basal_color='cyan', min_length_arrow=50, of=of2)
    else:
        fig_rose,  fig_colbar, fig_cols, fig_rows, fig_grid = rose_plot([[np.concatenate((a.start, a.centroid_end-a.start))] for a in all_centroid_arrows], all_centroid_angles2, all_outlines,  min_length=50, of=of2)

    of2.close()

    
    fig_rose.savefig(prefix+'rose.svg')               
    fig_colbar.savefig(prefix+'rose_colbar.svg', facecolor='none', edgecolor='none', transparent=True)        
    fig_cols.savefig(prefix+'col_hists.svg', facecolor='none', edgecolor='none', transparent=True)
    fig_rows.savefig(prefix+'row_hists.svg', facecolor='none', edgecolor='none', transparent=True)
    fig_grid.savefig(prefix+'grid_hists.svg', facecolor='none', edgecolor='none', transparent=True)
    plt.close('all')


    angle_data = [l.get_adjusted_centroid_angles() for l in cat_leaves]


    leaf_transverse = [ binom_sig(np.abs(x)) for x in angle_data]
    
    angle_num = [len(a) for a in angle_data]
    leaf_cell_num = [l.get_total_cell_number() for l in cat_leaves]



    formatter = MathTextSciFormatter()
    data = np.abs(all_centroid_angles2)
    p, d = binom_pvalue(data)

    
    hist_title = F_STRING.format(len(data),  d[0]*100, formatter(p))

    figs = stacked_histogram(np.linspace(0, 180, 18+1),
                            angle_data,
                            [str(int(l.width)) for l in cat_leaves],
                            #['*' if t else '' for t in leaf_transverse],
                            #['{:.2f}% {}'.format(100*x[0]/x[1], x[0]) for x in zip(angle_num, leaf_cell_num)],
                            normalize_rows=False, highlight=[], combine_axes=False, hist_title= hist_title )

    for i in range(3):
        figs[i].savefig(prefix+f'stacked-{i}.svg')

    of.close()
    print("CLOSE")

    return np.abs(all_centroid_angles2)
    
def get_counts(a, r):
    a = np.array(a)
    s_a =  np.sum((r[0]<=a) & (a < r[1]))
    return [s_a, len(a) - s_a]


def compare_hists(a, b, name_a, name_b, of=sys.stdout):

    counts_a = get_counts(a, r)
    counts_b = get_counts(b, r)


    print(f'contigency table {name_a} vs {name_b} angle range {r}\n', file=of)
    print('{} {} {}\n'.format(name_a, *counts_a), file=of)
    print('{} {} {}\n'.format(name_b, *counts_b), file=of)

    chi2 = chi2_contingency([ counts_a, counts_b ])

    print('chi2 (chi2, p, dof, table) ' +repr(chi2)+'\n', file=of)
    print('chi2 (chi2, p, dof, table) ' +str(chi2)+'\n', file=of)


    ks = ks_2samp(a, b)
    print(f'KS test {name_a} vs {name_b}', ks, file=of)

