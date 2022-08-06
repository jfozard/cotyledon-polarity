

# Make histograms of angle alpha and draw Rose plots

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import scipy.ndimage as nd

import matplotlib as mpl

import pandas as pd

from matplotlib.patches import Rectangle

"""
mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
    'font.size': 28, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    
})
"""

YLABEL_SIZE=28
XLABEL_SIZE=28
TITLE_SIZE=24
XTICK_SIZE=24
YTICK_SIZE=24


def stacked_histogram(bins, angle_data, annot_left=None, annot_right=None, normalize_rows=True, use_orig_for_hist_all=True, highlight=[], tick_label_size=22, combine_axes=False, draw_range=(80,100), hist_title=''):


    mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
#    'font.size': 20, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    'font.family': "sans-serif",
    'font.sans-serif': 'Arial',
    
    })

    cm = plt.cm.get_cmap()

    raw_histograms = [np.histogram(np.abs(a), bins=bins)[0] for a in angle_data]

    hist_raw = np.vstack(raw_histograms)

    N_leaves = len(angle_data)

    
    if normalize_rows:
        row_sums = np.sum(hist_raw, axis=1)
        hist = hist_raw / row_sums[:, np.newaxis]
        hist_all = np.sum(hist_raw, axis=0)

        hist_all_normalized = hist_all/np.sum(hist_all)
        
        h_max = max(np.max(hist), np.max(hist_all_normalized))

    else:
        hist = hist_raw
        hist_all = np.sum(hist_raw, axis=0)

        h_max = np.max(hist) #, np.max(hist_all))

        
        
    if combine_axes:
        fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={'height_ratios': [5*N_leaves/10, 5]},  figsize=(10,5*N_leaves/10 + 5))
        figs = fig
        axes = ax.flatten()
    else:
        fig_stacked, ax_stacked = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 3*N_leaves/10))
        plt.subplots_adjust(bottom=0.05, right=0.9, top=0.99)        
        fig_hist, ax_hist = plt.subplots(nrows=1, ncols=1)#, figsize=(5,2.5))
#        plt.subplots_adjust(bottom=0.05, right=0.95, top=0.99)        
        fig_cbar, ax_cbar = plt.subplots(nrows=1, ncols=1, figsize=(2,6))
        plt.subplots_adjust(bottom=0.05, right=0.9, top=0.99)        

        figs = (fig_stacked, fig_cbar, fig_hist)
        axes = [ax_stacked, ax_cbar, ax_hist]
        print('STACKED', mpl.rcParams)
        
    ax = axes[0]
    im = ax.imshow(hist, extent=(bins[0], bins[-1], len(raw_histograms)-0.5, -0.5), aspect='auto', vmin=0, vmax=h_max)

    # Highlight modal bins

    modal_pos = []
    for h in hist_raw:
        idx = np.argmax(h)
        pos = 0.5*(bins[idx]+bins[idx+1])
        modal_pos.append(pos)

    ax.plot(modal_pos, np.arange(len(modal_pos)), 'rx', ms=10, markeredgewidth=4)


#    for idx in highlight:
#        ax[0].add_patch(Rectangle((bins[0],idx-0.5),bins[-1]-bins[0],1,linewidth=3,edgecolor='r',facecolor='none', clip_on=False))
    
#    ax[0].plot(all_cot_medians, np.arange(len(all_cot_medians)), 'rx', ms=10, markeredgewidth=4)
    ax.set_yticks(np.arange(len(modal_pos))+0.1)
    ax.set_yticklabels(annot_left)

    ax.tick_params(axis='y', labelsize=tick_label_size)
    ax.tick_params(axis='x', labelsize=XTICK_SIZE)


    if annot_right is not None:
        secax = ax.secondary_yaxis('right')
        secax.yaxis.set_ticks(np.arange(len(raw_histograms))+0.1)
        secax.yaxis.set_ticklabels(annot_right)
        secax.tick_params(axis='y', labelsize=tick_label_size)
        for idx in highlight:
            secax.get_yticklabels()[idx].set_color('red')

    for idx in highlight:
        ax.get_yticklabels()[idx].set_color('red')
    ax.set_xlabel('$\|\\alpha\|$', fontsize=XLABEL_SIZE)

    if draw_range is not None:
        ANGLE_LOW, ANGLE_HIGH = draw_range
        bh = len(raw_histograms)-0.5
        ax.plot((ANGLE_LOW, ANGLE_HIGH), (bh, bh), 'r-', clip_on=False, zorder=100, lw=5)
        ax.set_ylim(bottom=len(raw_histograms)-0.5)
    #plt.text(0.8, 0.9, '$p={}$'.format(f(p)), transform=plt.gca().transAxes, fontsize=20)                                                                                       
#    plt.title('$n={}$  $p={}$ $r={:.1f}$%'.format(len(data), f(p), d[0]/d[2]*100), fontsize=20)                                                                                 

#
    
    cbar = plt.colorbar(im, ax=axes[1])

    cbar.ax.tick_params(labelsize=24) 


    bin_c = 0.5*(bins[1:] + bins[:-1])

    ax = axes[2]
    n, bins, patches = ax.hist(bin_c, bins=bins, weights=hist_all, density=False, range=(0,180))

    if normalize_rows:
        for c, p in zip(hist_all, patches):
            plt.setp(p, 'facecolor', cm(c/np.sum(hist_all)/h_max))


    ax.set_xlim((0, 180))
    ax.set_xlabel('$\|\\alpha\|$', fontsize=XLABEL_SIZE)
    ax.set_ylabel('Number of cells', fontsize=YLABEL_SIZE)

    if draw_range is not None:
        ANGLE_LOW, ANGLE_HIGH = draw_range
        bh = -0.05
        ax.plot((ANGLE_LOW, ANGLE_HIGH), (bh, bh), 'r-', clip_on=False, zorder=100, lw=5)
        ax.set_ylim(bottom=0)
    
    if hist_title:
        ax.set_title(hist_title, fontsize=TITLE_SIZE)

    print(combine_axes)
    print('STACKED fig size', figs[2].get_size_inches(), figs[2].dpi)
    
            
    if len(axes)==4:
        axes[-1].set_axis_off()
    axes[1].set_axis_off()
    
#    plt.tight_layout()

    
    return figs
