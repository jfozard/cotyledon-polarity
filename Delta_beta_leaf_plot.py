
"""
Make plots for tracked pairs of images, measuring change in angle beta

outputs:
output/delta_beta${X}.txt - measured changes in angle beta
output/delta_beta${X}_summary.txt - summary for each leaf and overall summary statistics used in manuscript
output/delta_beta${X}_analysis/ - intermediate output for each leaf, showing arrows and segmentations used
output/plot_out/delta_beta${X}_delta_beta.svg - histogram of change in beta for tracked cells
where X =  "", "_control", "_basl"

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

#import ray

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

out_path='output/'


FONT = 'Arial, Helvetica'

from utils import *

from pathlib import Path

import svgutils.transform as sg
import sys
from lxml import etree

FS = 2
FS2 = 2.1
IM_S = 4

from imageio import imwrite
from PIL import Image, ImageDraw



image_res = 0.692
sc_length_microns = 100
sc_length_cell_microns=25
inset_scale = 4


svg_width = 180
svg_height = 260

from svgutils.compose import Unit


a_bins = np.linspace(0,90,10)

def find_closest(array, value, eps=0.05):
    idx = (np.abs(array - value)).argmin()
    if np.abs(array[idx]-value)<=eps:
        return idx
    else:
        return None
    
def make_svg_fig(filename, g, height):
    ovWdth = Unit('210mm')
    ovHght = Unit(f'{height}mm')
    fig = sg.SVGFigure(ovWdth,ovHght) #str(svg_height)+"px", str(svg_width)+"px")
    defs = etree.fromstring('<defs> <marker id="arrowhead" style="overflow:visible;" refX="10" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"> <path style="stroke:#ffffff;stroke-width:1pt;stroke-opacity:1;fill:none;" d="M 0.0,0.0 L 10.0,5.0 L 0.0,10 " /> </marker> </defs>')
    fig.root.append(defs)
    fig.append(g)
    fig.save(filename) #.format(page_no))
    run(['inkscape', '-C', '-d', '300',  filename, '-o', filename[:-3]+'png'])
    run(['inkscape', '-C',  filename, '-o', filename[:-3]+'pdf'])



def round_float(v):
    return round(v, 1)

def make_stacked_svg_single(filename, panel_list, page_height=None, header=''):
    
    print('panel_list', panel_list)

    fig_panels = []

    pos = 0

    gap_y = 5
    im_sc2 = 1.0

    off_y = 0
    
    page_no = 0

    if header:
        fig_panels.append(sg.TextElement(5, 10, header, size=8, font=FONT))

        pos += 20
        
    for p1 in panel_list:

        h_max = 0
        
        if p1:
            g1, h1, _ = p1
            print('g1', len(g1.root))
            g1.scale(im_sc2, im_sc2)
            g1.moveto(0, pos)
            fig_panels += [g1]

            h_max = max(h_max, h1)
            
        pos += h_max*im_sc2 + gap_y

        if page_height is not None and pos>page_height:
    
            g = sg.GroupElement(fig_panels)
            g.moveto(5,off_y)
            off_y = 0
            #print(list(g))
            #fig.append(g)
            #print('fig', list(list(fig.root)[1]))
            #print(fig.to_str())
            #fig.save(filename.format(page_no))

            make_svg_fig(filename.format(page_no), g, max(page_height, pos))

            page_no += 1
            fig_panels = []
            pos = 0

    
    print('fig_panels_length', len(fig_panels))

    if len(fig_panels):
        g = sg.GroupElement(fig_panels)
        g.moveto(5,off_y)

        if page_height is not None:
            make_svg_fig(filename.format(page_no), g, max(page_height, pos))
        else:
            make_svg_fig(filename, g, max(page_height, pos))


def draw_measured_cells_svg_panel(measured_cells, results, header_text = "", rw=6, pad=20, red_idx=[], purple_t0_idx=[], blue_t1_idx=[]):

    defs = etree.fromstring('<defs> <marker id="arrowhead" style="overflow:visible;" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"> <path style="stroke:#ffffff;stroke-width:1pt;stroke-opacity:1;fill:none;" d="M 0.0,0.0 L 10.0,5.0 L 0.0,10 z" /> </marker> </defs>')

    panel_groups = []

    if len(measured_cells)>0:

        w = 6

        sp_r = 0.1

        gx = svg_width/(2*w+(w-1)*sp_r)

        w = rw

        h = (2*len(measured_cells)+2*w-1)//max(2*w,1)

        gx2 = (2+sp_r)*gx

        nx = 12

        header_y = 10
        hy = 15

        bdd_int_sc = 0.5
        seg_int = 120

        classify_lineage = results['classify_lineage']

        leaf0 = results['leaf0_0']
        leaf1 = results['leaf1_1']

        seg0 = leaf0.get_cells(valid_only=True)
        seg1 = leaf1.get_cells(valid_only=True)

        im0_bdd = leaf0.get_bdd(valid_only=True)
        im0_signal = leaf0.get_signal(valid_only=True)

        im1_bdd = leaf1.get_bdd(valid_only=True)
        im1_signal = leaf1.get_signal(valid_only=True)

        arrows0 = leaf0.get_arrows_dict()
        arrows1 = leaf1.get_arrows_dict()

        delta_beta = results.get('delta_beta', {})

        # get start and end positions from an arrow
        def to_array(a):
            return np.array((a.start, a.centroid_end))

        # Get start and end positions used for beta calculation from an arrow
        # This is important when a cell divides, as the centroid used is that of the combined daughters.
        def to_beta_array(a):
            return np.array((a.start, a.get_extra_props()['beta_end']))

        page_n_cells = len(measured_cells)

        gy = gx + hy

        ny = ((page_n_cells+(w-1))//w)

        col_height = (gx+hy)*((page_n_cells+(w-1))//w)



        # Background rectangles to indicate t0 / t1

        for k in range(min(w, page_n_cells)):

            r = sg.FigureElement(etree.Element(sg.SVG+f'rect', width=f'{gx}', height=f'{col_height}', style='fill:rgb(200,200,255)'))
            r.moveto(gx2*k, 0)
            r = sg.FigureElement(etree.Element(sg.SVG+f'rect', width=f'{gx}', height=f'{col_height}', style='fill:rgb(240,240,255)'))
            r.moveto(gx2*k+gx, 0)
            panel_groups.append(r)

        # Loop over cells

        cell_idx = 0

        for j in range(ny):
            for k in range(w):
                if cell_idx >= len(measured_cells):
                    break

                c = measured_cells[cell_idx]
                print(c)

                d = list(get_children(classify_lineage, c))

                if c in arrows0:
                    beta0 = arrows0[c].get_extra_props().get('beta', None)
                else:
                    beta0 = None

                if d[0] in arrows1 and 'beta' in arrows1[d[0]].get_extra_props():
                    e = d[0]
                elif len(d)>1: # Should this happen ever?
                    print('beta1 from second cell in the lineage?')
                    e = d[1]
                else:
                    e = d[0]

                if e in arrows1:
                    beta1 = arrows1[e].get_extra_props().get('beta', None)
                else:
                    beta1 = None

                def get_bbox(im, i):
                    return nd.find_objects(im==i)[0]

                c_bbox = get_bbox(seg0, c) #rp0[c-1])

                bbox0 = pad_bbox(c_bbox, im0_bdd.shape, [pad,pad])


                d_bbox = [get_bbox(seg1, i) for i in d]

                bbox1 = d_bbox[0]
                for i in range(1, len(d_bbox)):
                    bbox1 = comb_bbox(bbox1, d_bbox[i])

                bbox1 = pad_bbox( bbox1 , im1_bdd.shape, [pad,pad])


                seg_bdd0 = seg_int*find_boundaries(seg0)*(seg0==c)
                seg_bdd1 = seg_int*find_boundaries(seg1)*np.isin(seg1, d)

                cell_im0 = np.dstack(((bdd_int_sc*im0_bdd[bbox0]).astype(np.uint8), im0_signal[bbox0], seg_bdd0[bbox0]))
                cell_im1 = np.dstack(((bdd_int_sc*im1_bdd[bbox1]).astype(np.uint8), im1_signal[bbox1], seg_bdd1[bbox1]))

                u = max(max(cell_im0.shape), max(cell_im1.shape))

                im_sc = 0.9*gx/u

                g_c0 = [make_image_svg(cell_im0)]

                # Draw arrows

                if c in arrows0:
                    u = to_array(arrows0[c]) - np.array((bbox0[1].start,bbox0[0].start))
                    g_c0.append(make_arrow(u, True))

                c0 = leaf0.get_centroids()
                c1 = leaf1.get_centroids()

                x = c0[c] - np.array((bbox0[1].start,bbox0[0].start))
                g_c0.append(sg.TextElement(x[0], x[1], str(c), color='(100,100,200)', size=4))

                
                g_c0 = sg.GroupElement(g_c0)
                g_c0.scale(im_sc, im_sc)

                im_w = im_sc*cell_im0.shape[1]
                im_h = im_sc*cell_im0.shape[0]

                o_x = 0.5*(gx - im_w)
                o_y = 0.5*(gy-hy - im_h)

                
                g_c0.moveto(k*gx2+o_x, j*gy+o_y)

                panel_groups.append(g_c0)

                g_c1 = [make_image_svg(cell_im1)]


                
                for i in d:
                    if i in arrows1:

                        print('>>>', to_array(arrows1[i]), to_beta_array(arrows1[i]), bbox1)

                        u = to_array(arrows1[i]) - np.array((bbox1[1].start,bbox1[0].start))
                        g_c1.append(make_arrow(u, False))

                    if i in arrows1:
                        u = to_beta_array(arrows1[i]) - np.array((bbox1[1].start,bbox1[0].start))
                        g_c1.append(make_arrow(u, True))

                for i in d:
                    x = c1[i] - np.array((bbox1[1].start,bbox1[0].start))
                    g_c1.append(sg.TextElement(x[0], x[1], str(i), color='(100,200,200)', size=4))


                im_w = im_sc*cell_im1.shape[1]
                im_h = im_sc*cell_im1.shape[0]

                o_x = 0.5*(gx - im_w)
                o_y = 0.5*(gy-hy - im_h)

                if cell_idx in blue_t1_idx:
                    e = sg.FigureElement(etree.Element(sg.SVG+'rect', width=f'{cell_im1.shape[1]}', height=f'{cell_im1.shape[0]}', style='stroke:rgb(0,0,255); stroke-width:6; fill:none;'))
                    g_c1.append(e)

                if cell_idx in purple_t0_idx:
                    e = sg.FigureElement(etree.Element(sg.SVG+'rect', width=f'{cell_im1.shape[1]}', height=f'{cell_im1.shape[0]}', style='stroke:rgb(255,0,255); stroke-width:6; fill:none;'))
                    g_c1.append(e)
                
                g_c1 = sg.GroupElement(g_c1)
                g_c1.scale(im_sc, im_sc)
                g_c1.moveto(k*gx2+gx+o_x, j*gy+o_y)

                # Add text


                
                panel_groups.append(g_c1)

                bh = max(cell_im0.shape[0], cell_im1.shape[0])*im_sc
                bw = gx+cell_im1.shape[1]*im_sc

                
                # red box?

                if cell_idx in red_idx:
                    e = sg.FigureElement(etree.Element(sg.SVG+'rect', width=f'{bw}', height=f'{bh}', style='stroke:rgb(255,0,0); stroke-width:1; fill:none;'))
                    e.moveto(k*gx2+o_x, j*gy+o_y)
                    panel_groups.append(e)
                
                if beta0 is not None:
                    t = sg.TextElement(k*gx2+0.5*gx, (j)*gy + gx + 0.3*hy, f'{beta0:.1f}', size=6, anchor="middle", font=FONT)
                    panel_groups.append(t)

                if beta1 is not None:
                    t = sg.TextElement(k*gx2+1.5*gx, (j)*gy + gx + 0.3*hy, f'{beta1:.1f}', size=6, anchor="middle", font=FONT)
                    panel_groups.append(t)


                db = delta_beta[c]

                g_txt3 = sg.TextElement(k*gx2+gx, (j)*gy + gx + 0.7*hy, f'{db:.1f}', size=6, anchor="middle", font=FONT)


                panel_groups.append(g_txt3)

                cell_idx += 1


        gg = sg.GroupElement(panel_groups)
        gg.moveto(0, header_y)

        txt = sg.TextElement(5, 6, header_text, size=6)#, anchor="middle")

        g = sg.GroupElement([gg, txt])

        panel_height = col_height +header_y

        return g, panel_height, defs



import cv2
        
def draw_arrow(A, u, arrow_head=True):
    print(A.shape, A.dtype, u.shape, u.dtype)
#    A = np.ascontiguousarray(A)

    du = la.norm(u[1] - u[0])

    u = u.astype(np.int32)

    M = 10.0
    
    if arrow_head:
        A = cv2.arrowedLine(A, tuple(u[0]), tuple(u[1]), (255, 255, 255), tipLength=max(0.25, min(0.75, M/du)), thickness=2, line_type=cv2.LINE_AA)
    else:
        A = cv2.line(A, tuple(u[0]), tuple(u[1]), (128, 128, 128), thickness=2, lineType=cv2.LINE_AA)
    return A

from make_xlsx import *

def zoom(im, s):
    return np.dstack([nd.zoom(im[:,:,i], s) for i in range(im.shape[2])])

def draw_measured_cells_xlsx(measured_cells, results, header_text = "", rw=6, pad=20, red_idx=[], purple_t0_idx=[], blue_t1_idx=[]):

    print('draw measured xlsx')
    all_panel_data = []

    all_panel_data.append(TextCell(header_text))

    panel_data = []
    
    if len(measured_cells)>0:

        bdd_int_sc = 0.5
        seg_int = 120

        classify_lineage = results['classify_lineage']

        leaf0 = results['leaf0_0']
        leaf1 = results['leaf1_1']

        seg0 = leaf0.get_cells(valid_only=True)
        seg1 = leaf1.get_cells(valid_only=True)

        im0_bdd = leaf0.get_bdd(valid_only=True)
        im0_signal = leaf0.get_signal(valid_only=True)

        im1_bdd = leaf1.get_bdd(valid_only=True)
        im1_signal = leaf1.get_signal(valid_only=True)

        arrows0 = leaf0.get_arrows_dict()
        arrows1 = leaf1.get_arrows_dict()

        delta_beta = results.get('delta_beta', {})

        # get start and end positions from an arrow
        def to_array(a):
            return np.array((a.start, a.centroid_end))

        # Get start and end positions used for beta calculation from an arrow
        # This is important when a cell divides, as the centroid used is that of the combined daughters.
        def to_beta_array(a):
            return np.array((a.start, a.get_extra_props()['beta_end']))
        
        # Loop over cells

        cell_idx = 0

        w = 6
        ny = ((len(measured_cells)+(w-1))//w)

        
        for j in range(ny):
            panel_data.append(CellBlock(3, 1, [BlankCell(), TextCell('angle beta'), TextCell('change in beta')]))
            for k in range(w):
                if cell_idx >= len(measured_cells):
                    break

                group_data = []
                
                c = measured_cells[cell_idx]
                print(c)

                d = list(get_children(classify_lineage, c))

                if c in arrows0:
                    beta0 = arrows0[c].get_extra_props().get('beta', None)
                else:
                    beta0 = None

                if d[0] in arrows1 and 'beta' in arrows1[d[0]].get_extra_props():
                    e = d[0]
                elif len(d)>1: # Should this happen ever?
                    print('beta1 from second cell in the lineage?')
                    e = d[1]
                else:
                    e = d[0]

                if e in arrows1:
                    beta1 = arrows1[e].get_extra_props().get('beta', None)
                else:
                    beta1 = None

                def get_bbox(im, i):
                    return nd.find_objects(im==i)[0]

                c_bbox = get_bbox(seg0, c) #rp0[c-1])

                bbox0 = pad_bbox(c_bbox, im0_bdd.shape, [pad,pad])

                d_bbox = [get_bbox(seg1, i) for i in d]

                bbox1 = d_bbox[0]
                for i in range(1, len(d_bbox)):
                    bbox1 = comb_bbox(bbox1, d_bbox[i])

                bbox1 = pad_bbox( bbox1 , im1_bdd.shape, [pad,pad])


                seg_bdd0 = seg_int*find_boundaries(seg0)*(seg0==c)
                seg_bdd1 = seg_int*find_boundaries(seg1)*np.isin(seg1, d)

                cell_im0 = np.dstack(((bdd_int_sc*im0_bdd[bbox0]).astype(np.uint8), im0_signal[bbox0], seg_bdd0[bbox0])).astype(np.uint8)
                cell_im1 = np.dstack(((bdd_int_sc*im1_bdd[bbox1]).astype(np.uint8), im1_signal[bbox1], seg_bdd1[bbox1])).astype(np.uint8)

                u = max(max(cell_im0.shape), max(cell_im1.shape))


                g_c0 = zoom(cell_im0, 2)

                # Draw arrows

                if c in arrows0:
                    u = to_array(arrows0[c]) - np.array((bbox0[1].start,bbox0[0].start))
                    draw_arrow(g_c0, u*2, True)

                c0 = leaf0.get_centroids()
                c1 = leaf1.get_centroids()

                if cell_idx in red_idx:
                    group_data.append(ImageCell(g_c0, 'red_t0'))
                else:
                    group_data.append(ImageCell(g_c0))

                g_c1 = zoom(cell_im1, 2)

                
                for i in d:
                    if i in arrows1:

                        print('>>>', to_array(arrows1[i]), to_beta_array(arrows1[i]), bbox1)

                        u = to_array(arrows1[i]) - np.array((bbox1[1].start,bbox1[0].start))
                        draw_arrow(g_c1, u*2, False)

                    if i in arrows1:
                        u = to_beta_array(arrows1[i]) - np.array((bbox1[1].start,bbox1[0].start))
                        draw_arrow(g_c1, u*2, True)


                if cell_idx in blue_t1_idx:
                    group_data.append(ImageCell(g_c1, 'blue'))
                elif cell_idx in purple_t0_idx:
                    group_data.append(ImageCell(g_c1, 'purple'))
                elif cell_idx in red_idx:
                    group_data.append(ImageCell(g_c1, 'red_t1'))
                else:
                    group_data.append(ImageCell(g_c1))
                
                group_data.append(BlankCell())
                
                if beta0 is not None:

                    group_data.append(ValueCell(round_float(beta0)))

                if beta1 is not None:
                    group_data.append(ValueCell(round_float(beta1)))

                group_data.append(BlankCell())


                db = delta_beta[c]

                group_data.append(ValueCell(round_float(db)))

                group_data.append(BlankCell())

                group_data.append(BlankCell())
               
                panel_data.append(CellBlock(3, 3, group_data))

                cell_idx += 1

        all_panel_data.append(CellBlock(ny, w+1, panel_data))
        return all_panel_data









def plot_frame0_frame1(results, sel_arrows0=[], sel_arrows1=[], show_divisions=False, annotate_t0=[], annotate_t1=[], bdd=120, show_tracked=False, view_ch=0, component_filter=lambda x: False):

    def get_ch(l):
        if view_ch ==0:
            return l.get_bdd(valid_only=True)
        else:
            return l.get_signal(valid_only=True)

    def get_seg(l):
        return l.get_cells(valid_only=True)



    seg0_0 = get_seg(results['leaf0_0'])
    seg1_1 = get_seg(results['leaf1_1'])

    im0_0 = get_ch(results['leaf0_0'])
    im1_1 = get_ch(results['leaf1_1'])

    map_im1_0 = get_ch(results['leaf1_0'])
    map_im0_1 = get_ch(results['leaf0_1'])
    map_seg1_0 = get_seg(results['leaf1_0'])

    c0_0 = results['leaf0_0'].get_centroids()
    c1_1 = results['leaf1_1'].get_centroids()
    c1_0 = results['leaf1_0'].get_centroids()

    arrows0_0 = results['leaf0_0'].get_arrows_keys()
    arrows1_0 = results['leaf1_0'].get_arrows_keys()

    arrows0_1 = results['leaf0_1'].get_arrows_keys()
    arrows1_1 = results['leaf1_1'].get_arrows_keys()

    tracked_cells_0 = []
    tracked_cells_1 = []

    if show_tracked:
        cl2 = results['cl2']

        for k in cl2:
            if component_filter(k):
                for c in cl2[k]:
                    tracked_cells_0 += c[0]
                    tracked_cells_1 += c[1]

        seg0_mask = np.isin(seg0_0, tracked_cells_0)*(1+find_boundaries(seg0_0))
    else:
        seg0_mask = find_boundaries(seg0_0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(48,24))
    ax0.imshow(np.dstack((im0_0, map_im1_0, bdd*( seg0_mask ))))
    arrow0_lines = [arrows0_0[i] for i in sel_arrows0] # Confirmation that arrows are XY coordinates at this point
    ax0.add_collection(LineCollection(arrow0_lines, colors="r", lw=1))

    arrow1_lines = [arrows1_0[i] for i in sel_arrows1]
    ax0.add_collection(LineCollection(arrow1_lines, colors="g", lw=1))

    for cells, color in annotate_t0:
        for p in cells:
            if p < len(c0_0):
                c = c0_0[p]
                ax0.text(c[0], c[1], str(p), color=color, fontsize=10)

    if show_divisions:
        components = results['components']

        divisions2 = [ tuple(sorted(u[1])) for u in components if len(u[0])==1 and len(u[1])==2]

        div_lines = [ (c1_0[i], c1_0[j]) for (i,j) in divisions2]
        ax0.add_collection(LineCollection(div_lines, colors="y"))


    if show_tracked:
        seg1_mask = np.isin(seg1_1, tracked_cells_1)
    else:
        seg1_mask = find_boundaries(seg1_1)
    ax1.imshow(np.dstack((map_im0_1, im1_1, bdd*( seg1_mask ))))

    arrow0_lines = [arrows0_1[i] for i in sel_arrows0] # Confirmation that arrows are XY coordinates at this point
    ax1.add_collection(LineCollection(arrow0_lines, colors="r", lw=2))

    arrow1_lines = [arrows1_1[i] for i in sel_arrows1]
    ax1.add_collection(LineCollection(arrow1_lines, colors="g"))

    if show_divisions:
        components = results['components']

        divisions2 = [ tuple(sorted(u[1])) for u in components if len(u[0])==1 and len(u[1])==2]

        div_lines = [ (c1_1[i], c1_1[j]) for (i,j) in divisions2]
        ax1.add_collection(LineCollection(div_lines, colors="y"))

    for cells, color in annotate_t1:
        for p in cells:
            if p < len(c1_1):
                c = c1_1[p]
                ax1.text(c[0], c[1], str(p), color=color, fontsize=10)


    ax1.plot([1024*IM_S/2,1024*IM_S/2],[0, IM_S],'w-')

#    plt.show()

    return fig




def plot_orig_seg_frame0_frame1(results, sel_arrows0=[], sel_arrows1=[], show_divisions=False, annotate_t0=[], annotate_t1=[]):


    def get_seg(l):
        return l.get_cells(valid_only=True)

    seg0 = get_seg(results['leaf0'])
    seg1 = get_seg(results['leaf1'])

    im0_signal = results['leaf0'].get_signal(valid_only=True)
    im0_bdd = results['leaf0'].get_bdd(valid_only=True)

    im1_bdd = results['leaf1'].get_bdd(valid_only=True)
    im1_signal = results['leaf1'].get_signal(valid_only=True)

    c0 = results['leaf0'].get_centroids()
    c1 = results['leaf1'].get_centroids()

    arrows0 = results['leaf0'].get_arrows_keys()
    arrows1 = results['leaf1'].get_arrows_keys()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(48,24))
    ax0.imshow(np.dstack((im0_bdd, im0_signal, 255*( find_boundaries(seg0)))))
    arrow0_lines = [arrows0[i] for i in sel_arrows0] # Confirmation that arrows are XY coordinates at this point
    ax0.add_collection(LineCollection(arrow0_lines, colors="r", lw=1))


    for cells, color in annotate_t0:
        for p in cells:
            c = c0[p]
            ax0.text(c[0], c[1], str(p), color=color, fontsize=10)

    ax1.imshow(np.dstack((im1_bdd, im1_signal, 255*( find_boundaries(seg1)))))


    arrow1_lines = [arrows1[i] for i in sel_arrows1]
    ax1.add_collection(LineCollection(arrow1_lines, colors="g"))

    if show_divisions:
        components = results['components']

        divisions2 = [ tuple(sorted(u[1])) for u in components if len(u[0])==1 and len(u[1])==2]
        merges2 = [ tuple(sorted(u[1])) for u in components if len(u[1])==1 and len(u[0])>=2]

        div_lines = [ (c1[i], c1[j]) for (i,j) in divisions2]
        ax1.add_collection(LineCollection(div_lines, colors="y"))

    for cells, color in annotate_t1:
        for p in cells:
            c = c1[p]
            ax1.text(c[0], c[1], str(p), color=color, fontsize=10)

    return fig



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




def plot_tracked_component_svg(prefix, results, component, use_orig=False, pad=5):

    fig, ax_grid = plt.subplots(1, 2, figsize=(12,6))

    if not use_orig:

        leaf0 = results['leaf0_0_x4']
        leaf1 = results['leaf1_1_x4']

    else:

        leaf0 = results['leaf0']
        leaf1 = results['leaf1']

    seg0 = leaf0.get_cells(valid_only=True)
    seg1 = leaf1.get_cells(valid_only=True)

    print('seg0 shape', seg0.shape)

    seg0_bdd = find_boundaries(seg0)
    seg1_bdd = find_boundaries(seg1)

    im0_signal = leaf0.get_signal(valid_only=True)
    im0_bdd = leaf0.get_bdd(valid_only=True)

    im1_signal = leaf1.get_signal(valid_only=True)
    im1_bdd = leaf1.get_bdd(valid_only=True)

    arrows0 = leaf0.get_arrows_dict()
    arrows1 = leaf1.get_arrows_dict()

    theta0 = leaf0.get_extra_props()['theta']
    theta1 = leaf1.get_extra_props()['theta']


    classify_lineage = results['classify_lineage']

    c = list(component[0])[0]
    d = list(component[1])

    if c in arrows0:
        beta0 = arrows0[c].get_extra_props().get('beta', None)
    else:
        beta0 = None

    if d[0] in arrows1 and 'beta' in arrows1[d[0]].get_extra_props():
        e = d[0]
    elif len(d)>1:
        e = d[1]
    else:
        e = d[0]

    if e in arrows1:
        beta1 = arrows1[e].get_extra_props().get('beta', None)
    else:
        beta1 = None

    # Find the annotation for the tracked cell

    k = None
    for u in classify_lineage:
        if u[1]==component:
            k = u[0]


    ax0, ax1 = ax_grid

    if beta0 is not None:
        ax0.annotate(f'{k} {beta0:.1f}', (0.05, 0.05), xycoords="axes fraction", color="w")
    else:
        ax0.annotate(f'{k}', (0.05, 0.05), xycoords="axes fraction", color="w")

    if beta1 is not None:
        ax1.annotate(f'{beta1:.1f}', (0.05, 0.05), xycoords="axes fraction", color="w")


    # Plot cells before and after
    # Cells at t0 (in frame of figure 1)


    def get_bbox(im, i):
        return nd.find_objects(im==i)[0]

    c_bbox = get_bbox(seg0, c) #rp0[c-1])

    bbox0 = pad_bbox(c_bbox, im0_bdd.shape, [pad,pad])

    d_bbox = [get_bbox(seg1, i) for i in d]

    bbox1 = d_bbox[0]
    for i in range(1, len(d_bbox)):
        bbox1 = comb_bbox(bbox1, d_bbox[i])

    bbox1 = pad_bbox( bbox1 , im1_bdd.shape, [pad,pad])



    seg_bdd0 = 200*seg0_bdd[bbox0]*(seg0[bbox0]==c)
    seg_bdd1 = 200*seg1_bdd[bbox1]*np.isin(seg1[bbox1], d)


    cell_im0 = np.dstack((im0_bdd[bbox0], im0_signal[bbox0], seg_bdd0))

    cell_im1 = np.dstack((im1_bdd[bbox1], im1_signal[bbox1], seg_bdd1))


    def to_array(a):
        return np.array((a.start, a.centroid_end))

    def to_beta_array(a):
        return np.array((a.start, a.get_extra_props()['beta_end']))

    # Draw arrows

    b0 = []
    if c in arrows0:
        u = to_array(arrows0[c]) - np.array((bbox0[1].start,bbox0[0].start))
        b0.append((u, True))
        if u[1][0] - u[0][0] > 0:
            u2 = (u[0], u[0] + np.array(([30,0])))
        else:
            u2 = (u[0], u[0] - np.array(([30,0])))
        b0.append((u2, False))

    b1 = []
    for i in d:

        if i in arrows1:
            u = to_beta_array(arrows1[i]) - np.array((bbox1[1].start,bbox1[0].start))


            b1.append((u, True))
            if u[1][0] - u[0][0] > 0:
                u2 = (u[0], u[0] + np.array(([30,0])))
            else:
                u2 = (u[0], u[0] - np.array(([30,0])))
            b1.append((u2, False))


    cell_arrows0 = overlay_arrows(cell_im0.astype(np.uint8), b0)
    cell_arrows1 = overlay_arrows(cell_im1.astype(np.uint8), b1)

    print(prefix+' cell arrows0 shape', cell_arrows0.shape)

    imwrite(prefix+"arrows_t0.png", cell_arrows0)
    imwrite(prefix+"arrows_t1.png", cell_arrows1)


    sc_length_cell = int(sc_length_cell_microns/image_res)

    print('svg_image_arrow', beta0, beta1)

    svg_image_arrow(prefix+"arrows_t0.svg", cell_im0.astype(np.uint8), b0, beta0, sc_length=sc_length_cell, text='$\\beta_1={:.1f}^\\circ$', o_text=-10)
    svg_image_arrow(prefix+"arrows_t1.svg", cell_im1.astype(np.uint8), b1, beta1, sc_length=sc_length_cell, text='$\\beta_2={:.1f}^\\circ$')

    
    
    print('DONE SVG', prefix)

    print('delta beta', beta1-beta0)
    
    svg_delta_beta(prefix+"delta_beta_text.svg", beta1-beta0)


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

def plot_results(name, results, prefix, cell_list=None, plot_all_tracked=False, plot_measured_cells=True, of=None, stacked_panels=None, highlight_cells=False, svg_title="", l_idx=0):

    Path(prefix).mkdir(exist_ok=True)

    leaf0 = results['leaf0']
    leaf1 = results['leaf1']
    seg0 = leaf0.get_cells(valid_only=True)
    seg1 = leaf1.get_cells(valid_only=True)

    imwrite(prefix+'/seg0.tif', seg0)
    imwrite(prefix+'/seg1.tif', seg1)
    
    
    theta0 = results['theta0']
    theta1 = results['theta1']

    leaf0_0 = results['leaf0_0']

    leaf0_1 = results['leaf0_1']

    leaf1_0 = results['leaf1_0']

    leaf1_1 = results['leaf1_1']


    fig = plot_frame0_frame1(results, sel_arrows0=leaf0.get_arrows_keys(only_below_max_length=True), sel_arrows1=leaf1.get_arrows_keys(only_below_max_length=True), annotate_t0=[(leaf0.get_arrows_keys(), 'w')] , annotate_t1=[(leaf1.get_arrows_keys(), 'w')])

    fig.savefig(prefix+'/arrows.png')
    plt.close(fig)

    retained_cells0 = results['retained_cells0']
    retained_cells1 = results['retained_cells1']

    fig = plot_frame0_frame1(results, sel_arrows0=retained_cells0, sel_arrows1=retained_cells1, annotate_t0=[(retained_cells0, 'w')] , annotate_t1=[(retained_cells1, 'w')])

    fig.savefig(prefix+'/tracked-arrows.png')
    plt.close(fig)

    components = results['components']

    classify_lineage = results['classify_lineage']

    cl2 = results['cl2']


    of.write('\nAnalysis of image : ' + prefix[:-1]+'\n\n')
    print('\nAnalysis of image : ' + prefix[:-1]+'\n\n')

    tracking_summary(results, cl2, of=of)



    if cell_list:

        Path(prefix+'/components_svg').mkdir(exist_ok=True)
#
        for j, idx in enumerate(cell_list):
            c = get_component(components, idx, idx=0)
            plot_tracked_component_svg(prefix+'/components_svg'+f'/{j}_', results, c)
            print('TRACKED')



    if plot_measured_cells:
        delta_beta = results['delta_beta']
        delta_beta_div = results['delta_beta_div']
        delta_beta_no_div = results['delta_beta_no_div']

        tracked_cells_no_div = cl2[((1,),(1,))]

        tracked_cells_div = cl2[((1,),(1,0))] + cl2[((1,), (1,1))]




        if stacked_panels is not None:
            # p1 = draw_measured_cells_svg_panel(prefix+'no_div_{}.svg', [tracked_cells_no_div[idx[i]][0][0] for i in range(len(tracked_cells_no_div))], results, "Change in beta for tracked cells without division", rw=6)

            
            if highlight_cells:
                idx = np.argsort(delta_beta_no_div)
                red_angles = comments_red[name]

                pb_angles = comments_blue_purple[name]

                idx_red_nodiv = []
                idx_purple_nodiv = []

                print('?', idx, type(idx), type(delta_beta_no_div), red_angles[0], delta_beta_no_div)
                print(idx.dtype, idx.shape, len(delta_beta_no_div))
                
                for a in red_angles[0]:
                    print(a)
                    i = find_closest(np.array(delta_beta_no_div)[idx], a)
                    if i is not None:
                        idx_red_nodiv.append(i)



                for a in pb_angles[0]:
                    print(a)
                    i = find_closest(np.array(delta_beta_no_div)[idx], a)
                    if i is not None:
                        idx_purple_nodiv.append(i)

                        
                
                        
#                p1 = draw_measured_cells_svg_panel(prefix+'no_div_{}.svg', [tracked_cells_no_div[idx[i]][0][0] for i in range(len(tracked_cells_no_div))], results, name+" without division", rw=6, red_idx = idx_red_nodiv, purple_t0_idx = idx_purple_nodiv)
#                p1 = draw_measured_cells_svg_panel([tracked_cells_no_div[idx[i]][0][0] for i in range(len(tracked_cells_no_div))], results, f"Cotyledon {l_idx+1}", rw=6, red_idx = idx_red_nodiv, purple_t0_idx = idx_purple_nodiv)
                p1 = draw_measured_cells_xlsx([tracked_cells_no_div[idx[i]][0][0] for i in range(len(tracked_cells_no_div))], results, f"Cotyledon {l_idx+1}", rw=6, red_idx = idx_red_nodiv, purple_t0_idx = idx_purple_nodiv)
                idx = np.argsort(delta_beta_div)
                # p2 = draw_measured_cells_svg_panel(prefix+'div_{}.svg', [tracked_cells_div[idx[i]][0][0] for i in range(len(tracked_cells_div))], results, "Change in beta for tracked cells with division", rw=6)

                idx_blue_div = []
                idx_red_div = []

                print('?', idx, type(idx), type(delta_beta_div), red_angles[1], delta_beta_div)
                print(idx.dtype, idx.shape, len(delta_beta_div))
                for a in red_angles[1]:
                    print(a)
                    i = find_closest(np.array(delta_beta_div)[idx], a)
                    if i is not None:
                        idx_red_div.append(i)
                        
                for a in pb_angles[1]:
                    i = find_closest(np.array(delta_beta_div)[idx], a)
                    if i is not None:
                        idx_blue_div.append(i)


#                p2 = draw_measured_cells_svg_panel(prefix+'div_{}.svg', [tracked_cells_div[idx[i]][0][0] for i in range(len(tracked_cells_div))], results, name+" with division", rw=6, red_idx = idx_red_div, blue_t1_idx=idx_blue_div)
                p2 = draw_measured_cells_xlsx([tracked_cells_div[idx[i]][0][0] for i in range(len(tracked_cells_div))], results,  f"Cotyledon {l_idx+1}", rw=6, red_idx = idx_red_div, blue_t1_idx=idx_blue_div)
                stacked_panels.append([p1,p2])

            else:
                idx = np.argsort(delta_beta_no_div)
                p1 = draw_measured_cells_xlsx([tracked_cells_no_div[idx[i]][0][0] for i in range(len(tracked_cells_no_div))], results, name+" without division", rw=6)
                idx = np.argsort(delta_beta_div)
                # p2 = draw_measured_cells_svg_panel(prefix+'div_{}.svg', [tracked_cells_div[idx[i]][0][0] for i in range(len(tracked_cells_div))], results, "Change in beta for tracked cells with division", rw=6)
                p2 = draw_measured_cells_xlsx([tracked_cells_div[idx[i]][0][0] for i in range(len(tracked_cells_div))], results, name+" with division", rw=6)
                stacked_panels.append([p1,p2])




#result_output_path = 'stretched_data/'
#Path(result_output_path).mkdir(exist_ok=True)

import gzip

result_path = out_path+'delta_beta_results/'

def process_data(base, size_ds, use_seg=lambda d: False, auto_reverse= lambda d: False, two_sided=False, highlight_cells=False, stacked_base=None, output_plots_base=None):

        stacked_panels = []

        of = open(base+'_summary.txt', 'w')

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

        print(f'N_cots {N_cot} N_plants {N_plants}', file=of)


        
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

            with gzip.open(result_path+d0['Identifier']+'.pkl.gz', 'rb') as in_file:
                results  = pickle.load(in_file)

            if do_plots:
                if '5-brx' in id0: #### TEMP

                    # Find the appropriate cells!
                    seg0 = results[0]['leaf0'].get_cells(valid_only=True)

                    cell_list = []
                    for p in [(364, 234), (501, 351), (291, 716)]:
                        cell_list +=[seg0[p[1], p[0]]]
                    """
                    old_seg0 = imread('segmented/5-brxl.tif')
                    links = auto_lineage(old_seg0, seg0)
                    cell_list = []
                    for i in [167, 326, 969]: #[969, 326, 307]:
                        cell_list += [j for (k,j) in links if k==i]
                    """
                    print('cell_list', cell_list)
                    
                    plot_results(
                        d0['Identifier'],
                        results[0],
                        out_path+id0[:-3]+'/',
                        of = of,
                        stacked_panels = stacked_panels,
                        cell_list = cell_list,
                        highlight_cells = highlight_cells,
                        l_idx=i,
                    )
                    print('plot_results')
                    #quit()   #### TEMP

                else:
                    plot_results(
                        d0['Identifier'],
                        results[0],
                        out_path+id0[:-3]+'/',
                        of = of,
                        stacked_panels = stacked_panels,
                        highlight_cells = highlight_cells,
                        l_idx=i,
                    )

            results_all.append(results)

            #with gzip.open(result_output_path+ d0['Identifier'] + '.gz', 'wb') as pf:
            #    pickle.dump(results, pf)


        if do_plots:
#            make_stacked_svg_single(stacked_base+'_tracked_cells_nodiv_{}.svg', [p[0] for p in stacked_panels], page_height=250, header='Tracked cells without cell division')
#            make_stacked_svg_single(stacked_base+'_tracked_cells_div_{}.svg', [p[1] for p in stacked_panels], page_height=250, header='Tracked cells with cell division')
            print('make_xlsx')
            make_xlsx(stacked_base+'_tracked_cells.xlsx',
                      [[p[0] for p in stacked_panels], [p[1] for p in stacked_panels]],
                      headers=['Tracked cells without cell division', 'Tracked cells with cell division'])

        
        results, db_no_div, db_div, beta0, beta1, class0, class1 = zip(*results_all)


        #with gzip.open(base+'_output.gz', 'wb') as pf:
        #    pickle.dump((results_all, size_ds), pf)




        plt.figure()
        beta0_all = sum([list(b.values()) for b in beta0], [])
        plt.hist(beta0_all, bins=a_bins)
        plt.savefig(output_plots_base+'_beta0.png')

        plt.figure()
        beta1_all = sum([list(b.values()) for b in beta1], [])
        plt.hist(beta1_all, bins=a_bins)
        plt.savefig(output_plots_base+'_beta1.png')
#        plt.show()



        results1 = results[0]

        counts = {}
        for k in sorted(results1):
            if k[0]=='n':
                d = [r[k] for r in results]
                print(k, d,  sum(d), file=of)
                counts[k] = sum(d)

                
            

        c0 = Counter(sum([list(c.values()) for c in class0], []))# in list(class0_1.values()) + list(class0_2.values()) + list(class0_3.values()))
        c1 = Counter(sum([list(c.values()) for c in class1], []))# list(class1_1.values()) + list(class1_2.values()) + list(class1_3.values()))

        print('counter 0', c0, file=of)
        print('counter 1', c1, file=of)



        db = np.array(sum(db_no_div, []) + sum(db_div, []))

        with open(base+'.txt', 'w') as f:
            for d in db:
                f.write(str(d)+'\n')


#        with open(base+'_counts.pkl', 'rb') as cf:
#            counts = pickle.load(cf)
        db = np.loadtxt(base+".txt")


        from scipy.stats import chisquare, binom_test, ttest_1samp
        from statsmodels.stats.weightstats import ztest

        test = binom_test([np.sum(db<0), np.sum(db>0)], alternative='greater')


        print('delta beta', file=of)
        print('num -ve', np.sum(db<0), file=of)
        print('num +ve', np.sum(db>=0), file=of)

        print(' chi2 ', chisquare([np.sum(db<0), np.sum(db>=0)]), file=of)
        print('binom 1-sided ', binom_test([np.sum(db<0), np.sum(db>=0)], alternative='greater'), file=of)
        print('binom 2-sided ', binom_test([np.sum(db<0), np.sum(db>=0)]), file=of)

        ### Paired t-test
        print('z test less', ztest(db, value=0.0, alternative='smaller'), file=of)
        print('t test 1samp', ttest_1samp(db, popmean=0.0, alternative='less'), file=of)
        print('t test 1samp greater', ttest_1samp(db, popmean=0.0, alternative='greater'), file=of)
        print('t test 1samp 2 sided', ttest_1samp(db, popmean=0.0), file=of)
        print('mean db', np.mean(db), file=of)
        print('estimated stddev', np.std(db, ddof=1), file=of)
        print('n', len(db), file=of)

        stats = importr('stats')
        data = rpy2.robjects.FloatVector(db)
        r_tt_gt = stats.t_test(data, mu=0, alternative="greater")
        print(' R one-sided t-test (greater)', r_tt_gt, file=of)

        r_tt = stats.t_test(data, mu=0, alternative="less")
        print(' R one-sided t-test (less)', r_tt, file=of)
        r_tt2 = stats.t_test(data, mu=0, alternative="two.sided")
        print(' R two-sided t-test two sided', r_tt2, file=of)

        if two_sided:
            test = r_tt2.rx2('p.value')[0]
        else:
            test = r_tt.rx2('p.value')[0]
        conf_int = r_tt2.rx2("conf.int")

        n = len(db)

        plt.style.use('dark_background')
        plt.figure()
        plt.hist(db, bins=np.linspace(-90,90, 19))
        plt.xlabel('$\\Delta \\beta = \\beta_2 - \\beta_1$', size=24)
        plt.ylabel('Number of cells', size=24)
        plt.text(0.05, 0.9, f'p={test:.2f}', fontsize=24, transform=plt.gca().transAxes)
        plt.text(0.05, 0.75, f'n={n}', fontsize=24, transform=plt.gca().transAxes)
        plt.xticks([-90,-45,0,45,90], fontsize=18)
        plt.xlim(-90,90)
        plt.yticks(fontsize=18)
        plt.axvline(0, lw=2, color='r')
        plt.axvline(np.mean(db), lw=2, color='blue')
        plt.axvline(conf_int[0], ls='--', lw=2, color='blue')
        plt.axvline(conf_int[1], ls='--', lw=2, color='blue')
        plt.savefig(output_plots_base+'_delta_beta.svg', facecolor='none', edgecolor='none', transparent=True)



        print(file=of)
        print('#### Numbers for manuscript ####', file=of)

        print('Before stretching, polarized BRXL2 signal could be seen in about ' + '{:.1f}'.format(100*counts['n2']/counts['n1']) + ' ( {} / {} ; n2 / n1 )'.format(counts['n2'], counts['n1']), file=of)

        #s. In the first two classes, BRXL2 signal was detected in the cell before stretching (377/3503? tracked cells John is this correct?)

        print('In the first two classes, BRXL2 signal was detected in the cell before stretching in ' + '{:.1f}'.format(100*counts['n4']/counts['n0']) +' {} / {}; n4 / n0'.format(counts['n4'], counts['n0']), file=of)

        print('In ' + '{:.1f}'.format(100*counts['n3']/counts['n4']) + ' ( {} / {} ; n3 / n4 ) of these cells, signal was still detected in the cell after the 7h stretch'.format(counts['n3'], counts['n4']), file=of)

        print('In the remaining ' + '{:.1f}'.format(100*counts['n5']/counts['n4']) + ' ( {} / {} ; n5 / n4 ), signal disappeared from the cell after the stretch period'.format(counts['n5'], counts['n4']), file=of)

        print('This behaviour was observed for ' + '{:.1f}'.format(100*counts['n6']/counts['n8']) + ' ( {} / {} ; n6 / n8 ), of tracked cells (at T1) '.format(counts['n6'], counts['n8']), file=of)

        print('and was often ' + '{:.1f}'.format(100*counts['n7']/counts['n6']) + ' ( {} / {} ; n7 / n6 ) associated with cell division '.format(counts['n7'], counts['n6']), file=of)


        print('#### Delta beta statistics ####', file=of)

        print(r_tt.names)

        print(' Mean delta beta was {:.1f} (standard deviation {:.1f}, n={} ) and not significantly less than zero (p = {:.2f} one-sample one-sided t-test)'.format(np.mean(db), np.std(db, ddof=1), len(db), r_tt.rx2("p.value")[0]), file=of)







        of.close()
        print('FINAL')


from pathlib import Path

from get_leaf_dataframe import get_paired_dataframe


Path(out_path+'plot_out').mkdir(exist_ok=True, parents=True)


if __name__=='__main__':

    data_path = DATA_PATH


    
    plt.style.use("dark_background")

    base = out_path+'delta_beta'

    Path(base+'_analysis').mkdir(exist_ok=True, parents=True)

    stacked_base = out_path+'tracked_cells/'

    Path(stacked_base).mkdir(exist_ok=True, parents=True)
    
    stacked_base += 'delta_beta'
    
    marker = 'brxl'

    size_ds = get_paired_dataframe(marker='brxl', category='stretched')
    
    print(size_ds)

    print('Number of stretched leaves', len(size_ds))


    process_data(base, size_ds, highlight_cells=True, stacked_base=stacked_base, output_plots_base=out_path+'plot_out/delta_beta')


    print(' \n\n\n ===== \n\n\n DONE STRETCH \n\n\n =====')

    base = out_path+'delta_beta_control'

    Path(base+'_analysis').mkdir(exist_ok=True, parents=True)


    size_ds = get_paired_dataframe(marker='brxl', category='control')
    
    print(size_ds)

    print('Number of control leaves', len(size_ds))


    pf = 'delta_beta_control'
    stacked_base = out_path+'tracked_cells/'+pf
    process_data(base, size_ds, stacked_base=stacked_base, output_plots_base=out_path+'plot_out/'+pf)

    base = out_path+'delta_beta_basl'

    Path(base+'_analysis').mkdir(exist_ok=True, parents=True)


    marker = '35S_basl'

    size_ds = get_paired_dataframe(marker='35S_basl', category='stretched')
    
    print(size_ds)

    print('Number of stretched leaves', len(size_ds))

    pf = 'delta_beta_basl'
    stacked_base = out_path+'tracked_cells/'+pf

    process_data(base, size_ds, use_seg=lambda d: 'ds1' in d.Path, auto_reverse=lambda d: 'ds1' in d.Path, two_sided=True, stacked_base=stacked_base, output_plots_base=out_path+'plot_out/'+pf)

