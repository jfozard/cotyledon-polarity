


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.cm as cm

from scipy.stats import binom_test, ttest_1samp

from scipy.stats import chi2_contingency, chisquare

from utils import binom_pvalue, MathTextSciFormatter, circ_mean, sgn, round_up

from math import pi, atan2, sin, cos

import sys

import matplotlib as mpl

from copy import copy

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
    #    'xtick.labelsize': 28,    
})

YLABEL_SIZE=28
XLABEL_SIZE=28
TITLE_SIZE=24
XTICK_SIZE=24
YTICK_SIZE=24
INSET_SIZE=24

def test(data, prop=0.5):
    a, b = data
    return binom_test(a, a+b, prop, alternative='greater')

def test_two_sided(data, prop=0.5):
    a, b = data
    return binom_test(a, a+b, prop)

@dataclass
class Arrow:
    start: np.ndarray
    end: np.ndarray
    angle: float

def modal_bin_centre(x, bins):
    # Centre of the modal bin for a histogram
    # x - data
    # bins - histogram bin edges
    h, _ = np.histogram(x, bins)
    idx = np.argmax(h)
    return (bins[idx]+bins[idx+1])/2


def rose_plot(
        all_centroid_arrows,
        all_centroid_angles2,
        all_outlines = None,
        p_alpha = 0.05,
        ANGLE_LOW = 80,           # Start of medial angle range (absolute value of angle)
        ANGLE_HIGH= 100,          # End of medial angle range
        Nx = 5,                   # Number of boxes in x-direction
        Ny = 3,                   # Number of boxes in y-direction
        Na = 18,                   # Number of angular bins
        s_y = 100,                # Start position (y) for boxes
        e_y = 900,                # End position (y) for boxes
        s_x = 100,                # Start position (x) for boxes
        e_x = 1024-100,           # End position (x) for boxes
        max_x = 1024,
        max_y = 1024,
        min_length=50,             # Number of arrows in box to show anything
        min_length_arrow=100,     # Min number of arrows to add large directional arrow
        min_test_length=1,        # Min number of arrows for medial test
        figsize=(20,20),
        max_r=None,
        x_bin_classes=['marginal', 'lateral', 'medial', 'lateral',  'marginal'],
      #  x_bin_classes=['marginal', '', 'medial', '',  'marginal'],
        y_bin_classes=['distal', 'central', 'proximal'],
        mean_arrow_type='directional', # or 'circular'
        tests = ['medial'],
        Larrow=60,
        return_box_arrow_data = False,
        apical_basal_color='red',
        of = sys.stdout,
    ):

    
    print('ROSE PLOT PARAMS', mpl.rcParams)
    
    D_ANGLE = ANGLE_HIGH-ANGLE_LOW
    
    bins_x = np.linspace(s_x, e_x, Nx+1) # Positons of  bin edges (x)
    bins_y = np.linspace(s_y, e_y, Ny+1) # Positions of bin edges (y)


    fig = plt.figure(figsize=figsize)
    ax0 = plt.axes([0,0,1,1])
    ax1 = plt.axes([0,0,1,1])
    #ax1.set_zorder(0.1)
    if all_outlines is not None:
        bg = np.max(np.array(all_outlines), axis=0)

    # Overlay (dark grey) transformed leaf outlines
        ax0.imshow(0.25*bg, cmap=plt.cm.gray, clim=(0,1))

    for x in bins_x:
        plt.plot([x,x],[s_y,e_y], color=(0.3,0.3,0.3), lw=5, zorder=1.5)

    for y in bins_y:
        plt.plot([s_x,e_x],[y,y], color=(0.3,0.3,0.3), lw=5, zorder=1.5)

    plt.xlim(0,max_x)
    plt.ylim(max_y,0)     


    f = lambda x : x # Mapping between bin number and bar length - sqrt would give bar areas proportional to counts


    for i, t in enumerate(x_bin_classes):
        plt.text(0.5*(bins_x[i]+bins_x[i+1]), bins_y[0]-20, t, ha='center', fontsize=45, c='w')

    for i, t in enumerate(y_bin_classes):
        plt.text(bins_x[0]-40, 0.5*(bins_y[i]+bins_y[i+1]), t, va='center', fontsize=50, c='w', rotation=90)

    bins = np.linspace(-180, 180, Na+1)
    bin_mid = 0.5*(bins[:-1]+bins[1:])

    bins_rotated = bins+90
    bin_mid_rotated = bin_mid+90


    box_arrow_data = [ [ [] for j in range(Nx) ]  for i in range(Ny) ]
    
    box_histograms = []
    for i in range(Ny):
        row_histograms = []
        for j in range(Nx):
            start_x = bins_x[j]
            end_x = bins_x[j+1]
            start_y = bins_y[i]
            end_y = bins_y[i+1]
            # Include all arrows which *start* within the box
            idx_box = [ k for k, a in enumerate(all_centroid_arrows) if start_x<=a[0][0]<end_x and start_y<=a[0][1]<end_y]
            
            angles = [ all_centroid_angles2[k] for k in idx_box]
            
            hist, _ = np.histogram(angles, bins)

            row_histograms.append(hist)
        box_histograms.append(row_histograms)

    if max_r is None: # Find appropriate maximum radius for rose plots
        max_r = max(max(max(u) for u in row) for row in box_histograms)
        # round up to nearest 5
        max_r = round_up(max_r, 5)

    print(f' max_r={max_r} min_length={min_length} min_length_arrow={min_length_arrow}', file=of)


    grid_axes = []
    
    for i in range(Ny):
        row_histograms = []
        for j in range(Nx):
            start_x = bins_x[j]
            end_x = bins_x[j+1]
            start_y = bins_y[i]
            end_y = bins_y[i+1]
            idx_box = [ k for k, a in enumerate(all_centroid_arrows) if start_x<=a[0][0]<end_x and start_y<=a[0][1]<end_y]

            angles = [ all_centroid_angles2[k] for k in idx_box] # Absolute values of alpha
            angles_side = [sgn(all_centroid_arrows[k][0][0]-max_x/2)*all_centroid_angles2[k] for k in idx_box ] # Angles, sign inverted for arrows which start on the left hand side of the midline

            for k in idx_box: # Draw gray arrows onto the composite images
                a = all_centroid_arrows[k]
                angle = all_centroid_angles2[k]                
                ax0.arrow(*a[0], color=(0.2,0.2,0.2), width=1)
                box_arrow_data[i][j].append(angle)

            print(i, j, len(angles))
            if len(angles)>min_length: # Only draw a circular histogram if sufficiently many arrows starting in that box
                with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':(.5,.5,.5), 'figure.facecolor':'white'}):

                    ax_box = [ start_x/1024, 1.0-end_y/1024, (end_x-start_x)/1024, (end_y-start_y)/1024]

                    cx = (start_x+end_x)/2.0
                    cy = 0.25*start_y + 0.75*end_y

                    ax = fig.add_axes(ax_box, facecolor=None, projection='polar')
                    grid_axes.append(ax)
                    bins = np.linspace(-180, 180, Na+1)
                    bin_mid = 0.5*(bins[:-1]+bins[1:])

                    bins_rotated = bins+90
                    bin_mid_rotated = bin_mid+90

                    hist, _ = np.histogram(angles, bins)

                    
                    N, bins, patches  = ax.hist(bin_mid_rotated*pi/180, bins=bins_rotated*pi/180, weights=f(hist), density=False)

                    for k in range(len(patches)):
                        patches[k].set_facecolor(plt.cm.hsv((bin_mid[k]+180)/360))

    #                ax.text(10/180*pi,0.25,str(len(angles)), color='white', size=15)
                    ax.set_xticks([])
                    ax.set_ylim(0,f(max_r))
                    ax.set_yticks(f([max_r//2, max_r]))
                    if True: #i!=0 or j!=3:
                        ax.set_yticklabels(["",""])
                    else:
                        ax.set_yticklabels(["", f(max_r)], fontdict={'fontsize':40, 'color': 'orange'} )

                    angles = np.array(angles)

                    angles_p = angles[angles>=0]
                    angles_n = angles[angles<0]

                    a_mean_abs = np.mean(np.abs(angles))

                    a_mean = circ_mean(angles)
                    
                    a_mean_p = np.mean(angles_p)
                    a_mean_n = np.mean(angles_n)
                    #
                    a_mode_p = modal_bin_centre(angles_p, np.linspace(0,180, 10))
                    a_mode_n = modal_bin_centre(angles_n, np.linspace(-180, 0, 10))


                    ax.patch.set_alpha(0.0)

                    ax0.text(start_x+25, start_y+35, str(len(angles)), color='white', size=50)

                    
                    if True:
                        stars = []

                        # Test for apical-basal imbalance
                        
                        n_base = np.sum([abs(a)>90 for a in angles])
                        n_tip = np.sum([abs(a)<=90 for a in angles])
                        r_axis_pvalue = test_two_sided([n_base, n_tip]) #   chisquare([n_base, n_tip])
   
                        if r_axis_pvalue<p_alpha:
                            if n_base > n_tip and 'basal' in tests:
                                stars += ['*']
                            elif n_base < n_tip and 'apical' in tests:
                                stars += ['*']
                            else:
                                stars += ['']
                        else:
                            stars += ['']


                        n_med = np.sum([ANGLE_LOW<=abs(a)<ANGLE_HIGH for a in angles])
                        n_other = len(angles) - n_med
                        r_med_pvalue = test([n_med, n_other], (D_ANGLE/180))
                        
                        print(' med/other ', n_med, n_other, (D_ANGLE/180)*len(angles), r_med_pvalue, test([n_med, n_other], (D_ANGLE/180)))
                        if r_med_pvalue<p_alpha:
                            if n_med > (D_ANGLE/180)*len(angles) and 'medial' in tests:
                                stars += ['*']
                            else:
                                stars += ['']
                        else:
                            stars += ['']


                        
                        if len(angles)>min_test_length:
                            for k, (s, col, fs, off_y) in enumerate(zip(stars, [apical_basal_color, 'yellow'], [50, 50], [40, 40])):
                                ax0.text(start_x+100+30*k, start_y+off_y, s, color=col, size=fs)

                    if mean_arrow_type=='directional' and len(angles)>min_length_arrow:
                        n_left = np.sum([a>0 for a in angles])
                        n_right = np.sum([a<=0 for a in angles])
                        r_lr_pvalue = test_two_sided([n_left, n_right])
                        if r_lr_pvalue<p_alpha:
                            if n_left < n_right:
                            #    ax.axvline((a_mean_n+90)*pi/180, lw=2, c='white')
                                arr = ax1.arrow(cx, cy, Larrow*cos(-(a_mode_n+90)*pi/180), Larrow*sin(-(a_mode_n+90)*pi/180), color='w', width=4, head_width=5*4, head_length=4*4, zorder=0.1)
                            else:
                            #    ax.axvline((a_mean_p+90)*pi/180, lw=2, c='white')
                                arr = ax1.arrow(cx, cy, Larrow*cos(-(a_mode_p+90)*pi/180), Larrow*sin(-(a_mode_p+90)*pi/180), color='w', width=4, head_width=5*4, head_length=4*4, zorder=0.1)
                                
                                
                        else:
                           # ax.axvline((a_mean_abs+90)*pi/180, lw=2, c='r')
                           # ax.axvline((-a_mean_abs+90)*pi/180, lw=2, c='g')
                            # ax0.plot([cx, cx+Larrow*cos(-(a_mean_abs+90)*pi/180)], [cy,], color='r', lw=4)
                            # ax0.plot([cx, cx+Larrow*cos(-(-a_mean_abs+90)*pi/180)], [cy, cy+Larrow*sin(-(-a_mean_abs+90)*pi/180)], color='g', lw=4)
                            # ax0.plot([cx, ], [cy, cy+Larrow*sin(-(a_mean_abs+90)*pi/180)], color='r', lw=4)
                            #ax1.plot([cx+Larrow*cos(-(a_mean_abs+90)*pi/180), cx, cx+Larrow*cos(-(-a_mean_abs+90)*pi/180)], [ cy+Larrow*sin(-(a_mean_abs+90)*pi/180), cy, cy+Larrow*sin(-(-a_mean_abs+90)*pi/180)], color='w', lw=4)
                            pass    

                    if mean_arrow_type=='circular' and len(angles)>min_length_arrow:
                        print('amean', a_mean)
                        arr = ax1.arrow(cx, cy, Larrow*cos(-(a_mean+90)*pi/180), Larrow*sin(-(a_mean+90)*pi/180), color='w', width=4, head_width=5*4, head_length=4*4, zorder=0.1)
                        


    ax0.axis('off')
    ax0.set_zorder(1)
#    ax0.zorder = 3
    
    for a in grid_axes:
        a.set_zorder(2)
#        a.zorder = 2
        
    ax1.axis('off')
    ax1.set_zorder(3)
#    ax1.zorder = 0
#    plt.show()
    print(ax0.zorder, a.zorder, ax1.zorder)

#    fig.axes = [fig.axes[0]] + fig.axes[2:] + [fig.axes[1]]


    fig_rose = fig

    
    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=-180, vmax=180)

    old_rc = copy(mpl.rcParams)

    
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)


    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=24, color='w')
    cbar.set_ticks((-180,-90,0,90,180))

    fig_cbar = fig

##    plt.show()



    ff = MathTextSciFormatter(fmt="%1.1e")
    bh = 0.05

    fig, ax = plt.subplots(1,3, figsize=(20,6))
    plt.subplots_adjust(wspace=0.4)
    for idx, (j_e, name) in enumerate([([0,4],  'marginal'),
                                     ([1,2], 'lateral'),
                                     ([3], 'medial')]):

        idx_box = [ k for k, a in enumerate(all_centroid_arrows) if any(bins_x[j]<=a[0][0]<bins_x[j+1] for j in j_e) ]

        angles = [ all_centroid_angles2[k] for k in idx_box]

        bins = np.linspace(0, 180, 18+1)

        ax[idx].hist(np.abs(angles), bins=bins)
        ax[idx].set_ylabel("Number of cells", fontsize=YLABEL_SIZE)
        ax[idx].set_xlabel("$\\alpha$", fontsize=XLABEL_SIZE)
        ax[idx].set_xlim(0, 180)
        ax[idx].set_xticklabels([0,90,180], fontsize=XTICK_SIZE)
        ax[idx].set_xticks([0,90,180])
        ax[idx].tick_params(axis='y', labelsize=YTICK_SIZE)
        p, d = binom_pvalue(np.abs(angles))
        ax[idx].set_title(name, fontdict={'fontsize':TITLE_SIZE})
        ax[idx].plot((ANGLE_LOW, ANGLE_HIGH), (-bh, -bh), 'r-', clip_on=False, zorder=100, lw=5)
        ax[idx].set_ylim(bottom=0)
        prop = d[0]*100

        txt =  f'$n={d[3]}$ ' + f'$r={prop:.1f}$% ' + '$p=' + ff(p) + '$' 


        ax[idx].text(0.5, -0.3, txt, transform=ax[idx].transAxes, fontsize=24, ha='center')#,  fontsize=16)


    fig_cols = fig
###    plt.show()
    
    fig, ax = plt.subplots(1,3, figsize=(20,6))
    plt.subplots_adjust(wspace=0.4)
    for idx, (j_e, name) in enumerate([([0],  'distal'),
                                     ([1], 'central'),
                                     ([2], 'proximal')]):
        idx_box = [ k for k, a in enumerate(all_centroid_arrows) if any(bins_y[j]<=a[0][1]<bins_y[j+1] for j in j_e) ]

        angles = [ all_centroid_angles2[k] for k in idx_box]

        bins = np.linspace(0, 180, 18+1)

        ax[idx].hist(np.abs(angles), bins=bins)
        ax[idx].set_ylabel("Number of cells", fontsize=YLABEL_SIZE)
        ax[idx].set_xlabel("$\\alpha$", fontsize=XLABEL_SIZE)
        ax[idx].set_xlim(0, 180)
        ax[idx].set_xticklabels([0,90,180], fontsize=XTICK_SIZE)
        ax[idx].set_xticks([0,90,180])
        ax[idx].tick_params(axis='y', labelsize=YTICK_SIZE)
        ax[idx].plot((ANGLE_LOW, ANGLE_HIGH), (-bh, -bh), 'r-', clip_on=False, zorder=100, lw=5)
        ax[idx].set_ylim(bottom=0)

        p, d = binom_pvalue(np.abs(angles))
        ax[idx].set_title(name, fontdict={'fontsize':TITLE_SIZE})


        prop = d[0]*100
        txt =  f'n={d[3]} ' + f'r={prop:.1f}% ' + '$p=' + ff(p) + '$' 


        ax[idx].text(0.5, -0.3, txt, transform=ax[idx].transAxes, fontsize=INSET_SIZE, ha='center')#,  fontsize=16)


####    plt.show()
    
    fig_rows = fig

    # Per box histograms for rico question
    for i in range(Ny):
        for j in range(Nx):
            print(f'{i}, {j} : ' + ' '.join(str(a) for a in box_arrow_data[i][j]))

    fig, ax = plt.subplots(Ny, Nx, figsize=(8*Nx, 6*Ny))
    for i in range(Ny):
        for j in range(Nx):
            angles = np.array(box_arrow_data[i][j])
            bins = np.linspace(-180, 180, 18+1)

            angles_p = angles[angles>=0]
            angles_n = angles[angles<0]

            ax[i, j].text(0.1, 0.9, '{:.1f}'.format(np.mean(angles_n)), transform=ax[i,j].transAxes)
            ax[i, j].text(0.8, 0.9, '{:.1f}'.format(np.mean(angles_n)), transform=ax[i,j].transAxes)
            
            
            ax[i, j].hist(angles, bins=bins)
            ax[i, j].set_ylabel("Number of cells", fontsize=YLABEL_SIZE)
            ax[i, j].set_xlabel("$\\alpha$", fontsize=XLABEL_SIZE)
            ax[i, j].set_xlim(-180, 180)
            ax[i, j].set_xticklabels([-180, -90, 0,90,180], fontsize=XTICK_SIZE)
            ax[i, j].set_xticks([-180, -90, 0,90,180])
            ax[i, j].tick_params(axis='y', labelsize=YTICK_SIZE)
            ax[i, j].set_ylim(bottom=0)

    fig_grid = fig

#####    plt.show()

    mpl.rcParams.update(old_rc)

#    fig_rose.savefig('test.png')
#    fig_rose.savefig('test.svg')

    
    if return_box_arrow_data:
        return fig_rose, fig_cbar, fig_cols, fig_rows, fig_grid, box_arrow_data
    else:
        return fig_rose, fig_cbar, fig_cols, fig_rows, fig_grid

