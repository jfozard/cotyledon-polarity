
from imageio import imwrite
from PIL import Image, ImageDraw

from apply_double_style import test_unet

from base import textext, textext_svg

import numpy as np
import scipy.ndimage as nd

import json

from math import atan2, pi, sqrt, sin, cos

import svgutils.transform as sg
import io
from lxml import etree

from scipy.sparse import coo_matrix
from roi_decode import roi_zip


from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

from statsmodels.stats.proportion import binom_test

from process_utils import *

from skimage.filters import threshold_otsu, rank


def circ_mean(x): # Circular mean of a list of angles (in degrees)
    a = pi*np.array(x)/180
    vx = np.mean(np.cos(a))
    vy = np.mean(np.sin(a))
    a_mean = atan2(vy, vx)/pi*180
    print('circ_mean', x, a_mean)
    return a_mean

def sgn(x):
    return 1 if x>0 else (-1 if x<0 else 0)


def round_up(x, n):
    return ((x+n-1)//n)*n


def leaf_boundary(signal):
    im = nd.gaussian_filter(signal, 20)
    interior = im > 0.8*threshold_otsu(im)
    interior = nd.binary_fill_holes(interior)
    bdd = find_boundaries(interior)
    return bdd


ANGLE_LOW = 80
ANGLE_HIGH= 100
D_ANGLE = ANGLE_HIGH-ANGLE_LOW



def binom_pvalue(data, r=(ANGLE_LOW,ANGLE_HIGH)):
    assert(np.min(data)>=0)
    assert(np.max(data)<=180)
    prop = (r[1]-r[0])/180
    observed = np.sum((r[0] <= data) & (data< r[1]))
    expected = len(data)*prop
    total = len(data)


    return binom_test(np.sum((r[0] <= data) & (data< r[1])), len(data), prop, alternative='larger'), (observed/total, observed, expected, total)


def binom_sig(data, r=(ANGLE_LOW,ANGLE_HIGH), alpha=0.05):
    assert(np.min(data)>=0)
    assert(np.max(data)<=180)
    prop = (r[1]-r[0])/180
    observed = np.sum((r[0] <= data) & (data< r[1]))
    expected = len(data)*prop
    total = len(data)
       
    return binom_test(np.sum((r[0] <= data) & (data< r[1])), len(data), prop, alternative='larger')<alpha
    



def segment(im):
    seg_bdd = test_unet('G-249.pt', im)
    cells = watershed_no_labels(seg_bdd, h=10)[0,:,:]
    return cells


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


def svg_delta_beta(svg_fn, db):
    svg = textext_svg('$\\Delta \\beta = {:.1f}^\\circ$'.format(db), fill="#ffffff")
    svg.getroot().set('style', 'fill:#ffffff;')
    svg.write(svg_fn)


def svg_image_arrow(svg_fn, image, arrows, angle, sc_length=0, text='$\\beta_1={:.1f}^\\circ$', o_text=0):

    h, w = image.shape[:2]
    fig = sg.SVGFigure(str(w)+"px", str(h)+"px")

    defs = etree.fromstring('<defs> <marker id="arrowhead" style="overflow:visible;" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"> <path style="fill-rule:evenodd;stroke:#ffffff;stroke-width:1pt;stroke-opacity:1;fill:#ffffff;fill-opacity:1" d="M 0.0,0.0 L 10.0,5.0 L 0.0,10 z" /> </marker> </defs>')
    fig.root.append(defs)
    
    with io.BytesIO() as output:
        imwrite(output, image, format="PNG")
#        contents = output.getvalue()
        output.seek(0)
        image_panel = sg.ImageElement(output, image.shape[1], image.shape[0])

    a = []
    for arrow, arrow_head in arrows:
        arrow = sg.LineElement(arrow, color="white")
        if arrow_head:
            arrow.root.set("marker-end", "url(#arrowhead)" )
        a.append(arrow)

    if sc_length:
        line = sg.LineElement([(5, image.shape[0]-5), (5+sc_length, image.shape[0]-5)], color="white")
        a.append(line)
#        line.set('stroke-width',2)

    min_y = min(min(a[0][0][1], a[0][1][1]) for a in arrows)
    if min_y < h/4:
        txt_y = 3*h/4
    else:
        txt_y = h/4

        
    angle_txt = f'{angle:.1f}'

    txt = sg.GroupElement([sg.FigureElement(textext(text.format(angle), fill="#ffffff"))])
    txt.root.set('style', 'fill:#ffffff;')
    txt.scale(1.9,1.9)
    txt.moveto(o_text, image.shape[0]+10)

    
    #txt = sg.TextElement(w/2, txt_y, angle_txt, size="12", color="white", anchor="middle")

    
    
    fig.append([image_panel]+a+[txt])
    fig.save(svg_fn)
                    


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

# Utility functions for bounding boxes
def comb_slice(s1, s2):
    return slice(min(s1.start, s2.start), max(s1.stop, s2.stop))

def comb_bbox(bbox1, bbox2):
    return tuple([comb_slice(bbox1[i], bbox2[i]) for i in range(2)])

def pad_slice(s, d, p):
    return slice(max(0, s.start-p), min(d, s.stop+p))

def pad_bbox(bbox, d, p):
    return tuple([pad_slice(bbox[i], d[i], p[i]) for i in range(2)])


# Find all indices in segmentation touching the edges of the image (likely incomplete cells).
def edge_idx(seg):
    a, b = seg.shape
    return list(set(np.unique(seg[0,:])) | set(np.unique(seg[:,0])) | set(np.unique(seg[a-1,:])) | set(np.unique(seg[:,b-1])))



# Write tests for this
def calc_bipartite_components(links): # Split the bipartite map into connected components 
    #f_map, r_map = calc_forward_map(links), calc_reverse_map(links)
    l_all = list(links)
    visited_x = set()
    visited_y = set()
    components = []
    while l_all:
        l = l_all.pop()
        x, y = l
        if x not in visited_x:
            if y not in visited_y:
                visited_x.add(x)
                visited_y.add(y)
                components.append([set([x]), set([y]), [l]])
            else:
                # find component containing y
                c = [u for u in components if y in u[1]][0] # slow step; union-find / visited as dict
                c[0].add(x)
                c[2].append(l)
                visited_x.add(x)
        else:
            if y not in visited_y:
                # Find component containing x
                c = [u for u in components if x in u[0]][0] # slow step; union-find?
                c[1].add(y)
                c[2].append(l)
                visited_y.add(y)
            else:
                # Join pair of components
                c_x = [u for u in components if x in u[0]][0]
                c_y = [u for u in components if y in u[1]][0]
                c_x[0].update(c_y[0])
                c_x[1].update(c_y[1])
                c_x[2] = c_x[2] + c_y[2] + [l]
                components.remove(c_y)
    return components # [[labels in t0], [labels in t1], [links]]




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


def regionprop_dict(im):
    rp = regionprops(im)
    return dict((i, r) for i,r in zip(np.unique(im), rp))


# Extrack directed line segments ("arrows") from Fiji / ImageJ ROI manager file
def get_arrows(seg, filename, use_centroids=True, reverse_arrows=False, flip_y=False):
    roi_data = roi_zip(filename)
    new_data = {}
#    stack += 1                                                                                                                                                          
    centroids = nd.center_of_mass(np.ones_like(seg), labels=seg, index=np.arange(0, np.max(seg)+1))

#    print(centroids, stack)                                                                                                                                             
    for r in roi_data.values():
        if reverse_arrows:
            r.X1, r.Y1, r.X2, r.Y2 = r.X2, r.Y2, r.X1, r.Y1

        if flip_y:
            r.Y1 = 1024-r.Y1 # or 1023? Check this.
            r.Y2 = 1024-r.Y2

        idx = seg[int(r.Y2), int(r.X2)]
        if idx>0:
            if use_centroids:
                c = centroids[idx]
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((c[1], c[0]))]
            else:
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((r.X2, r.Y2))]

    print('ARROW FILE', filename, len(new_data))
    return new_data # Arrows in XY format

def get_arrows_auto(seg, filename, use_centroids=True):
    roi_data = roi_zip(filename)
    new_data = {}
#    stack += 1                                                                                                                                                          
    centroids = nd.center_of_mass(np.ones_like(seg), labels=seg, index=np.arange(0, np.max(seg)+1))


    bdd = find_boundaries(seg)
    bdd_dist = nd.distance_transform_edt(1-bdd)
    
#    print(centroids, stack)
    for r in roi_data.values():

        d2 = bdd_dist[int(r.Y2), int(r.X2)]
        d1 = bdd_dist[int(r.Y1), int(r.X1)]

        if d1>d2:
            r.X1, r.Y1, r.X2, r.Y2 = r.X2, r.Y2, r.X1, r.Y1

        idx = seg[int(r.Y2), int(r.X2)]
        if idx>0:
            if use_centroids:
                c = centroids[idx]
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((c[1], c[0]))]
            else:
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((r.X2, r.Y2))]
    return new_data # Arrows in XY format


def make_image_svg(image):
    with io.BytesIO() as output:
        imwrite(output, image, format="PNG")
        #        contents = output.getvalue()
        output.seek(0)
        image_panel = sg.ImageElement(output, image.shape[1], image.shape[0])
    return image_panel

def make_arrow(arrow, arrow_head):
    if arrow_head:
        arrow = sg.LineElement(arrow, color="white")
        arrow.root.set("marker-end", "url(#arrowhead)" )
    else:
        arrow = sg.LineElement(arrow, color="red")
    return arrow



def get_component(components, i, idx=1):
    return next((c for c in components if i in c[idx] ), None)

def get_children(classify_lineage, i):
    return next((c[1] for _,c in classify_lineage if i==c[0]), None)

# Calculate matrix entries used for overlap calculations
def calc_cM(seg, exact, weights=None):

    X = seg
    Y = exact

    # Calculation of IoU                                                                                                                                                 
    if weights is None:
        w = np.ones(X.shape)
        n = np.product(X.shape)
    else:
        w = weights
        n = np.sum(w)
        
    lX, iX = np.unique(X, return_inverse=True)
    lY, iY = np.unique(Y, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]                                                                                                                           
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))
    
    cX = nd.sum(w, labels=X, index=lX)
    cY = nd.sum(w, labels=Y, index=lY)

    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]
    
    M = coo_matrix((w.flatten(), (I, J))).tocsr()

    # loop over all objects in seg
    
    return lX, lY, cX, cY, n, M


class MathTextSciFormatter(object):
    
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return s #"${}$".format(s)


def calc_overlap(X, Y, w=None, threshold=0.5, return_criterion=False):
    # Max_g ((Ra ^ Rg) /Rg) for each a in X                                                                                                                              
    lX, lY, aX, aY, n, M = calc_cM(X, Y, w)
    overlap_map = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]                 
            c = intersection_area/i_area
            if len(c>0):
                k = np.argmax(c)
                if c[k]>=threshold:
                    overlap_map[li] = lj[k]
                else:
                    overlap_map[li] = None
            else:
                overlap_map[li] = None
    return overlap_map


def find_outside(X, valid):
    # return list of all cells which overlap a pixel for which valid is False
    # X - labelled image
    # valid - boolean mask image
    lX = np.unique(X)
    u = nd.sum(1-valid, labels=X, index=lX)
    return [i for i,v in zip(lX, u) if v>0]

def auto_lineage(l0, l1):

    # Map taking cells in l1 to their parent (largest overlap) in l0 ; 
    parent_map = calc_overlap(l1, l0, threshold=0.5)
    # Map taking cells in l0 to their child (largest overlap) in l1
    child_map = calc_overlap(l0, l1, threshold=0.5)

    lineage_map = list(child_map.items()) + list((v,k) for k,v in parent_map.items())
    lineage_map = list(set(lineage_map))
    
    return lineage_map
    

def get_lineage(seg0, map_seg1, valid):
    lineage2 = auto_lineage(seg0, map_seg1)
    lineage2 = [l for l in lineage2 if l[0] and l[1]]

    # Remove all of the links from cells which overlap points in T0 which are outside
    # the region with a corresponding point in T1
    outside = find_outside(seg0, valid)
    for o in outside:
        to_remove = [ i for i, p in enumerate(lineage2) if p[0]==o]
        for i in to_remove[::-1]:
            del lineage2[i]
    lineage2.sort()
    return lineage2

# Convert 
def arrow_beta(p):
    d = p[1] - p[0]
    angle = (180/pi)*atan2(abs(d[1]), abs(d[0]))
    return angle

def map_to_range(d): # [-180, 180 )
    while d<-180:
        d+=360
    while d>=180:
        d-=360
    return d


def calc_angle_beta(p, theta=0.0):
    d = p[1] - p[0]
    angle = (180/pi)*atan2(d[1], d[0])
    angle = map_to_range(angle - theta)
    angle = abs(angle)
    if angle > 90:
        angle = 180-angle
    return angle

# Swap two values - I,J <-> X,Y
def swap_ij(x):
    return x[1], x[0]

    
# Extract an (the first) item in a set
def SetFirst(s):
    for e in s:
        break
    return e



def draw_bbox(ax, bbox):
    ax.add_patch(Rectangle((bbox[1].start,bbox[0].start), bbox[1].stop - bbox[1].start, bbox[0].stop - bbox[0].start, fc='None', ec='b'))


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
