

# Make histograms of angle alpha and draw Rose plots

#from numba.core.typing.templates import make_intrinsic_template
import numpy as np
import scipy.linalg as la
import json
from math import atan2, pi, sqrt
import scipy.ndimage as nd

from copy import deepcopy


from tifffile import imread

from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

import scipy.ndimage as nd

from process_utils import *

from utils import map_to_range

from skimage.filters import threshold_otsu
from apply_double_style import test_unet

from math import pi

from dataclasses import dataclass

from transform_reduced import Transform as BTransform

def centroids_area(seg):
    def swap_ij(x):
        return (x[1],x[0])
    # Find the centroids and areas of each non-zero labelled region in the segmentation seg
    # returns centroids, areas
    #          centroids - list of cell centroids (I,J)
    return [ f(np.ones_like(seg), labels=seg, index=np.arange(0, np.max(seg)+1)) for f in ( lambda *x, **y: list(map(swap_ij, nd.center_of_mass(*x, **y))),
 nd.sum ) ]


"""
def affine_inverse(m):
    u = np.zeros_like(m)
    mat_inv = la.inv(m[:2,:2])
    u[:2,:2] = mat_inv
    u[:2,2] = -mat_inv @ m[:2,2]
"""

def affine_inverse(m):
    return la.inv(m)


def get_transform(json_file, sc=1):
    with open(json_file, 'r') as f:
        data = json.load(f)

    fn, m, pts = data[0]
    m0 = np.array(m).reshape((3,3)).T
    m0[:2,:2]=m0[:2,:2]/sc
    m = la.inv(m0)
    affine_m = to_ij(m)
    return fn, m0, affine_m, pts

def get_leaf_bdd_transform(signal):
    im = nd.gaussian_filter(signal, 20)
    interior = im > 0.8*threshold_otsu(im)
    interior = nd.binary_fill_holes(interior)
    bdd = find_boundaries(interior)
    return bdd


def map_image_transform(im, affine_m):
    return nd.affine_transform(im, affine_m, order=0)

def map_points_transform(p0, m):
    return np.array([affine(m, x) for x in p0])


@dataclass
class Arrow:
    start: np.ndarray
    manual_end: np.ndarray
    centroid_end: np.ndarray
    reverse: bool
    
    orig_angle: float
    orig_centroid_angle: float

    adjusted_angle: float
    adjusted_centroid_angle: float

    cell_idx: int
    below_max_length: bool
    extra_props: dict

    def to_mpl_arrow(self, centroid_end=True):
        if centroid_end:
            return (self.start[0], self.start[1], self.centroid_end[0]-self.start[0], self.centroid_end[1]-self.start[1])            
        else:
            return (self.start[0], self.start[1], self.manual_end[0]-self.start[0], self.manual_end[1]-self.start[1])

    def to_line_segment(self, centroid_end=True):
        if centroid_end:
#            return ((self.start[0], self.centroid_end[0]), (self.start[1], self.centroid_end[1]))   
            return ((self.start[0], self.start[1]), (self.centroid_end[0], self.centroid_end[1]))        
        else:
            return ((self.start[0], self.start[1]), (self.manual_end[0], self.manual_end[1]))        

    def get_extra_props(self):
        return self.extra_props
        
@dataclass
class Leaf:
    arrows: dict       # List of arrows  
    angle: float       # Angle of leaf (anti-clockwise from right)
    name: str          # filename

    signal: np.ndarray   # marker channel (original coordinates, but scaled up to 1024^2)
    bdd: np.ndarray      # boundary channel (original coordinates, but scaled up to 1024^2)
    cells: np.ndarray    # segmentation of bdd

    sc: float            # Scale factor used to scale image to 1024^2 

    extra_props: dict


    def get_arrows(self, key='default', only_below_max_length=False):
        return [a for a in self.arrows[key] if (not only_below_max_length) or a.below_max_length]

    def get_arrows_keys(self, key='default', only_below_max_length=False):
        arrow_map = {}
        for a in self.arrows[key]:
            if not only_below_max_length or a.below_max_length:
                arrow_map[a.cell_idx] = a.to_line_segment()
        return arrow_map

    def get_arrow_cell_indices(self, key='default'):
        return [ a.cell_idx for a in self.arrows[key] ]
        
    def get_adjusted_centroid_angles(self, key='default'):
        return np.array([ a.adjusted_centroid_angle for a in self.arrows[key] if a.below_max_length ])
        
    def get_total_cell_number(self):
        return(len(np.unique(self.cells)))

    def get_obj(self):
        return nd.find_objects(self.cells+1)

    def get_centroids(self):
        c, _ = centroids_area(self.cells)
        return c

    def get_cells(self, valid_only=False):
        if valid_only:
            return self.cells
        else:
            return self.cells, np.ones(self.cells.shape, dtype=np.int32)

    def get_signal(self, valid_only=False):
        if valid_only:
            return self.signal
        else:
            return self.signal, np.ones(self.signal.shape, dtype=np.int32)

    def get_bdd(self, valid_only=False):
        if valid_only:
            return self.bdd
        else:
            return self.bdd, np.ones(self.bdd.shape, dtype=np.int32)

    def get_extra_props(self):
        return self.extra_props

    def get_arrows_dict(self, key='default', only_below_max_length=False):
        arrow_map = {}
        for a in self.arrows[key]:
            if not only_below_max_length or a.below_max_length:
                arrow_map[a.cell_idx] = a
        return arrow_map

def interpolate_transformation(im_valid, transformation_x, transformation_y, out_shape=None):
    """
    Interpolate image (using ndimage.map_coordinates for interpolation)

    arguments:
        im
        transformation_y, transformation_x
    keyword arguments:
        out_shape

    returns: 
    out, 
    valid, 
  
    """

    im, im_valid = im_valid

    if out_shape is None:
        out_shape = im.shape
    m, n = out_shape

    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))

    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)

 #   if len(im.shape)==2:
    out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
    out_valid =  nd.map_coordinates(im_valid, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
 
 #   else:
 #       out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])

    return out, out_valid*valid


class BaseTransform(object):
    def __init__(self, domain_shape=None, range_shape=None):

        self.domain_shape = domain_shape
        self.range_shape = range_shape if range_shape is not None else domain_shape
        self.transformation_x = None
        self.transformation_y = None
        self.transformation_valid = None
        # Convert to LRU cache
        self.hashed_results = {}
    
    def map_image(self, source, order=0, return_valid=False):
        if self.transformation_x is None:
            g_y, g_x = np.mgrid[:self.domain_shape[0], :self.domain_shape[1]].astype(np.float64)
            g_coords = np.c_[(g_x.flatten(), g_y.flatten())]
            print('new_map', g_coords.shape)
            coords, t_valid = self.map_image_coords(g_coords, return_valid=True)
            im_m, im_n = self.range_shape
#            valid = ((coords[:,0]>=0) & (coords[:,0]<=(im_m-1)) &
#             (coords[:,1]>=0) & (coords[:,1]<=(im_n-1))).reshape(self.domain_shape)
            self.transformation_x = coords[:,0].reshape(self.domain_shape)
            self.transformation_y = coords[:,1].reshape(self.domain_shape)
            self.transformation_valid = t_valid.reshape(self.domain_shape)

        sh = hash(tuple([s.data.tobytes() for s in source]))
        if sh in self.hashed_results:
            print('hashed')
            out, valid = self.hashed_results[sh]
        else:
            print('unhashed')
            out, valid = interpolate_transformation(source, self.transformation_x, self.transformation_y, out_shape=self.domain_shape)
            valid = valid * self.transformation_valid
            self.hashed_results[sh] = (out, valid)

        if return_valid:
            return out, valid
        else:
            return out

    def reset_cache(self):
        self.hashed_results = {}


    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        del state["transformation_x"]
        del state["transformation_y"]
        del state["transformation_valid"]
        del state["hashed_results"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self.transformation_x = None
        self.transformation_y = None
        self.transformation_valid = None
        self.hashed_results = {}


import deformation_reduced as df


# Deformed mapping of multiple timepoints onto a single reference timepoint

        
class BTransform2(BaseTransform):
    def __init__(self, x_data, y_data, im_shape, out_shape=None):
        super().__init__(im_shape, im_shape) # Fixme for im / out shape different

        self.x_data = x_data
        self.y_data = y_data
        self.im_shape = im_shape
        if out_shape is None:
            self.out_shape = self.im_shape
        else:
            self.out_shape = out_shape
            
        self.transformation_y = None
        self.transformation_x = None
        self.hashed_results = {}
        self.stored_pre_affine = False

    def map_image_coords(self, pts, return_valid=False):
        # Map points (in x-y coordinates) from the domain to the range of the transform    
        if return_valid == False:
            return df.get_transform_points(self.x_data, self.y_data, self.out_shape, pts)
        else:
            out_coords = df.get_transform_points(self.x_data, self.y_data, self.out_shape, pts)

            valid_in = (0<=pts[...,0]) & (pts[...,0]<=self.out_shape[0]-1) & (0<=pts[...,1]) & (pts[...,1]<=self.out_shape[1]-1)

            valid_out = (0<=out_coords[...,0]) & (out_coords[...,0]<=self.im_shape[0]-1) & (0<=out_coords[...,1]) & (out_coords[...,1]<=self.im_shape[1]-1)
            return out_coords, valid_in & valid_out


    @classmethod
    def from_file(cls, fn, im_shape):
        x_data, y_data = df.load_elastic_transform(fn)
        return BTransform2(x_data, y_data, im_shape)


class CompositeTransform(BaseTransform):
    def __init__(self, transform_list):
        super().__init__(domain_shape=transform_list[0].domain_shape)
        self.transform_list = transform_list

    def map_image_coords(self, pts, return_valid=False):
        u = pts
        if return_valid == False:
            for t in self.transform_list:
                print('map rv False ', t.domain_shape)
                u = t.map_image_coords(u, return_valid=False)
            return u
        else:
            u = pts
            v = np.ones(u.shape[:-1], dtype=np.int32)
            for t in self.transform_list:
                u, valid = t.map_image_coords(u, return_valid=True)
                v = v & valid
            return u, v

    def map_source_coords(self, pts):
        for t in self.transform_list[::-1]:
            pts = t.map_source_coords(pts)
        return pts

@dataclass
class DeformableTransform:

    def __init__(self, forwards_map: BTransform2, reverse_map: BTransform2):
        self.forwards_map = forwards_map
        self.reverse_map = reverse_map
        self.domain_shape = forwards_map.domain_shape

    def map_image(self, *args, **kwargs):
        return self.forwards_map.map_image(*args, **kwargs)

    def map_image_coords(self, *args, **kwargs):
        return self.forwards_map.map_image_coords(*args, **kwargs)

    def map_source_coords(self, pts):
        return self.reverse_map.map_image_coords(pts)

@dataclass
class ATransform(BaseTransform):

    def __init__(self, matrix: np.ndarray, domain_shape: tuple):
        super().__init__(domain_shape, domain_shape)
        self.matrix = matrix
        self.domain_shape = domain_shape


    #def map_image(self, image, return_valid=False):
    #    return map_image_affine(image, self.matrix, output_shape=self.domain_shape, return_valid=return_valid)

    def map_image_coords(self, coord_grid, return_valid=False):
        if return_valid==False:
            return (self.matrix[:2,:2] @ coord_grid.T).T + self.matrix[:2,2][np.newaxis, :]
        else:
            out_coords = (self.matrix[:2,:2] @ coord_grid.T).T + self.matrix[:2,2][np.newaxis, :]
            return out_coords, np.ones(coord_grid.shape[:-1], dtype=np.int32)

        


class AffineTransform:
    forwards_map: ATransform
    reverse_map: ATransform

    def __init__(self, matrix, target_domain_shape, source_domain_shape=None):
        self.forwards_map = ATransform(matrix=matrix, domain_shape=target_domain_shape)
        self.reverse_map = ATransform(matrix=affine_inverse(matrix), domain_shape=source_domain_shape if source_domain_shape is not None else target_domain_shape)
        self.domain_shape = target_domain_shape

    def map_image(self, *args, **kwargs):
        return self.forwards_map.map_image(*args, **kwargs)

    def map_image_coords(self, *args, **kwargs):
        return self.forwards_map.map_image_coords(*args, **kwargs)

    def map_source_coords(self, pts):
        return self.reverse_map.map_image_coords(pts)

@dataclass
class LeafView:
    leaf: Leaf          # Base leaf object
    transform: np.ndarray # Transform mapping view coordinates into leaf coordinates

    def get_centroids(self):
        c, _ = centroids_area(self.get_cells(valid_only=True))
        return c

    def get_cells(self, valid_only=False):
        im, v = self.transform.map_image(self.leaf.get_cells(), return_valid=True)
        if valid_only:
            return im*v
        else:
            return im, v

    def get_signal(self, valid_only=False):
        im, v = self.transform.map_image(self.leaf.get_signal(), return_valid=True)
        if valid_only:
            return im*v
        else:
            return im, v

    def get_bdd(self, valid_only=False):
        im, v = self.transform.map_image(self.leaf.get_bdd(), return_valid=True)
        print(im.shape, v.shape)
        if valid_only:
            return im*v
        else:
            return im, v

    def get_arrows(self, key='default', only_below_max_length=False):

        # This should map the arrows
        new_arrows = list( self.get_arrows_dict(key, only_below_max_length).values() )
        return new_arrows

    
    def get_arrows_keys(self, key='default', only_below_max_length=False):
        # This should map the arrows

        arrow_map = {}
        d = self.get_arrows_dict(key, only_below_max_length)
        arrow_map = { k : b.to_line_segment() for k,b in d.items() }
        return arrow_map

    def get_arrows_dict(self, key='default', only_below_max_length=False):
        # This should map the arrows

        arrow_map = {}
        for k, a in self.leaf.get_arrows_dict(key, only_below_max_length).items():
            b = deepcopy(a)
            if 'beta_end' in a.extra_props:
                v = np.vstack((a.start, a.manual_end, a.centroid_end, a.extra_props['beta_end']))
                v = self.transform.map_source_coords(v)
                b.start, b.manual_end, b.centroid_end, b.extra_props['beta_end'] = v
            else:
                v = np.vstack((a.start, a.manual_end, a.centroid_end))
                v = self.transform.map_source_coords(v)
                b.start, b.manual_end, b.centroid_end = v
            arrow_map[k] = b
            
        return arrow_map
    
    def get_extra_props(self):
        return self.leaf.get_extra_props()

"""
def polarity_from_fiji(stack, filename, use_centroids=True, reverse_arrows=False, scale_roi=1):
    roi_data = roi_zip(filename)
    new_data = {}
#    stack += 1
    centroids = nd.center_of_mass(np.ones_like(stack), labels=stack, index=np.arange(np.max(stack)))

    
#    print(centroids, stack)
    for r in roi_data.values():
        if reverse_arrows:
            r.X1, r.Y1, r.X2, r.Y2 = r.X2, r.Y2, r.X1, r.Y1

        r.X1 *= scale_roi
        r.X2 *= scale_roi
        r.Y1 *= scale_roi
        r.Y2 *= scale_roi
        
        idx = stack[int(r.Y2), int(r.X2)]
        if idx>0:
            if use_centroids:
                c = centroids[idx]
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((c[1], c[0]))]
            else:
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((r.X2, r.Y2))]
    return new_data
"""
 
# Extrack directed line segments ("arrows") from Fiji / ImageJ ROI manager file
def get_arrows(seg, filename, use_centroids=True, reverse_arrows=False, scale_roi=1, flip_y=False, auto_reverse=False):
    roi_data = roi_zip(filename)
    new_data = {}
#    stack += 1                                                                                                                                                          
    centroids = nd.center_of_mass(np.ones_like(seg), labels=seg, index=np.arange(0, np.max(seg)+1))
    

    bdd = find_boundaries(seg)
    bdd_dist = nd.distance_transform_edt(1-bdd)

    reversed_data = {}
    start_bdd_dist = {}
    
#    print(centroids, stack)                                                                                                                                             
    for r in roi_data.values():
        reversed = reverse_arrows
        if reverse_arrows:
            r.X1, r.Y1, r.X2, r.Y2 = r.X2, r.Y2, r.X1, r.Y1
            
        r.X1 *= scale_roi
        r.X2 *= scale_roi
        r.Y1 *= scale_roi
        r.Y2 *= scale_roi

        if flip_y:
            r.Y1 = 1023-r.Y1 # or 1023? Check this.
            r.Y2 = 1023-r.Y2

        if auto_reverse:
            d2 = bdd_dist[int(r.Y2), int(r.X2)]
            d1 = bdd_dist[int(r.Y1), int(r.X1)]

            if d1>d2:
                r.X1, r.Y1, r.X2, r.Y2 = r.X2, r.Y2, r.X1, r.Y1
                reversed = not reversed
                

        mid_x = 0.5*(r.X1 + r.X2)
        mid_y = 0.5*(r.Y1 + r.Y2)
        idx = seg[int(mid_y), int(mid_x)]

        start_bdd_dist[idx] = bdd_dist[int(r.Y1), int(r.X1)]
        
        if idx>0:
            if use_centroids:
                c = centroids[idx]
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((c[1], c[0]))]
            else:
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((r.X2, r.Y2))]
        reversed_data[idx] = reversed

    print('ARROW FILE', filename, len(new_data))
    return new_data, reversed_data, start_bdd_dist  # Arrows in XY format

def seg_leaf_bdd(bdd_im):
    print('segmenting unet')
    seg_bdd = test_unet('G-249.pt', bdd_im)
    cells = watershed_no_labels(seg_bdd, h=10)[0,:,:]
    return seg_bdd, cells

def process_leaf_simple(name, image_fn, seg_fn, roi_fn, direction, max_length=100, reverse_arrows=False, auto_reverse=False, flip_y=False, return_seg_bdd=False, return_sc=False, extra_roi_fn=[], seg=None):

    
    t0_im = imread(image_fn)
    
    if flip_y:
        print('flip', t0_im.shape, direction)
        t0_im = t0_im[:,::-1,:]
        direction = -direction


    t0_signal = (255.0*t0_im[0,:,:]/np.max(t0_im[0,:,:])).astype(np.uint8)
    t0_bdd = (255.0*t0_im[1,:,:]/np.max(t0_im[1,:,:])).astype(np.uint8)

    if t0_signal.shape[0]==512:
        t0_signal = nd.zoom(t0_signal, 2.0)
        t0_bdd = nd.zoom(t0_bdd, 2.0)
        sc = 2
    else:
        sc = 1
    
    print('scale', sc)

    if seg is not None:
        cells = seg
    else:
        if seg_fn is None:
            seg_bdd, cells = seg_leaf_bdd(t0_bdd)
        else:
            cells = imread(seg_fn)
            if sc!=1:
                cells = nd.zoom(cells, sc, order=0)
            if flip_y:
                cells = cells[::-1,:]
            print('cells shape', cells.shape)
        
    print('shapes', t0_signal.shape, t0_bdd.shape, cells.shape)


    roi_files = [roi_fn] + extra_roi_fn

    all_arrows = {}

    for r, n in zip(roi_files, (['default'] + [f'replicate{i}' for i in range(1, len(extra_roi_fn)+1)  ])):
    #    assert np.min(cells)>0

        # Arrows with cell centroid corrected

        polarity_t0, reversed_t0, bdd_dist_t0 = get_arrows(cells, r, use_centroids=True, reverse_arrows=reverse_arrows, auto_reverse=auto_reverse, scale_roi=sc, flip_y=flip_y)

        print('name', n, 'orig roi length ', len(polarity_t0))

        # Raw annotated arrows

        polarity_t0_orig, reversed_t0_orig, bdd_dist_t0_orig = get_arrows(cells, r, use_centroids=False, reverse_arrows=reverse_arrows, auto_reverse=auto_reverse, scale_roi=sc, flip_y=flip_y)

        arrow_data = []

        for i in polarity_t0:
            l_centroid = polarity_t0[i]
            l_orig = polarity_t0_orig[i]


            # Check max length criterion for centroid adjusted angles

            l0 = (l_centroid[1][0], l_centroid[1][1]) # end
            l1 = (l_centroid[0][0], l_centroid[0][1]) # start
            dx = l0[0] - l1[0]
            dy = l0[1] - l1[1]
            length_centroid = sqrt(dx*dx+dy*dy)

            arrow_angle_centroid = atan2( -(l0[1] - l1[1]), (l0[0] - l1[0]))*180.0/pi 
            adjusted_angle_centroid = map_to_range(arrow_angle_centroid-direction)


            l0 = (l_orig[1][0], l_orig[1][1]) 
            l1 = (l_orig[0][0], l_orig[0][1]) 
            dx = l0[0] - l1[0]
            dy = l0[1] - l1[1]

            assert la.norm(l_orig[0] - l_centroid[0])<0.01

            length_orig = sqrt(dx*dx + dy*dy)

            arrow_angle_orig = atan2( -(l0[1] - l1[1]), (l0[0] - l1[0]))*180.0/pi 
            adjusted_angle_orig = map_to_range(arrow_angle_orig-direction)

            # Check whether distance between adjusted end + original end < original length

            good_arrow = (la.norm(l_orig[1] - l_centroid[1]) < 1.5*length_orig) and (bdd_dist_t0[i]<5)

            a = Arrow(l_orig[0], l_orig[1], l_centroid[1], reversed_t0[i], arrow_angle_orig, arrow_angle_centroid, adjusted_angle_orig, adjusted_angle_centroid, i, good_arrow, {})

            arrow_data.append(a)

        all_arrows[n] = arrow_data

        print('name', n, 'good arrows', len([a for a in all_arrows[n] if a.below_max_length]))

        
    l= Leaf(
        arrows = all_arrows,
        angle = direction,
        name = name,        
        signal = t0_signal,
        bdd = t0_bdd,      # boundary channel (original coordinates, but scaled up to 1024^2)
        cells = cells,     # segmentation of bdd
        sc = sc,
        extra_props = { 'theta': direction },
    )

    r = [l]
    if return_seg_bdd:
        r.append(seg_bdd)
    if return_sc:
        r.append(sc)

    if len(r)>1:
        return tuple(r)
    else:
        return r[0]

def plot_leaf(leaf):
    plt.figure()
    im = np.dstack((leaf.get_bdd(valid_only=True), leaf.get_signal(valid_only=True), 255*find_boundaries(leaf.get_cells(valid_only=True))))
    plt.imshow(im)


def plot_leaf_arrows(leaf):
    plt.figure()
    im = np.dstack((leaf.get_bdd(valid_only=True), leaf.get_signal(valid_only=True), 255*find_boundaries(leaf.get_cells(valid_only=True))))
    plt.imshow(im)
    arrows = leaf.get_arrows()
    for a in arrows:
        plt.arrow(*a.to_mpl_arrow(centroid_end=False), width=2, color='w')
        plt.arrow(*a.to_mpl_arrow(), width=2, color='g')


def get_rotation_transform(theta, sc=0.8):

    a = (pi/180)*(theta)
    
    R0 = sc*np.array(((cos(a), sin(a)), (-sin(a), cos(a))))

    c0 = np.array([[512,512]]).T
    #m2 = np.array([[0,1,0],[-1,0,1024],[0,0,1]])
    m0 = np.vstack((np.hstack((R0, c0-R0.dot(c0))), (0,0,1)))
#    m0 = m2.dot(m0)
    return m0

"""
def process(name, im0_fn, seg0_fn, im1_fn, seg1_fn, arrows0_fn, arrows1_fn, transform_fn, transform_pts_fn, theta0, theta1, flip_t0=False):
    leaf0 = process_leaf_simple(name+'_t0', im0_fn, seg0_fn, arrows0_fn, theta0, flip_y=flip_t0)
    leaf1 = process_leaf_simple(name+'_t5', im1_fn, seg1_fn, arrows1_fn, theta1)

    #plot_leaf(leaf0) # Make a version that also shows the arrows
    #plot_leaf(leaf1)

    transform = BTransform2.from_file(transform_fn,  (1024,1024)) # BunwarpJ transform between frame 0 and frame 1 (check direction)
    transform_pts = BTransform2.from_file(transform_pts_fn, (1024,1024)) # BunwarpJ transform between frame 1 and frame 0 (check direction)

    d_transform = DeformableTransform(transform, transform_pts)

    leaf1_orig0 = LeafView(leaf=leaf1, transform=d_transform)

    # Now transform to aligned coordinates

    plot_leaf(leaf1_orig0) 
    plt.show()


    plot_leaf_arrows(leaf1_orig0) 



    # Transform t0 into rotated /scaled version

    IM_S = 2
 
    # Rotation by (90-theta0) - mapping points in orig frame0 to those in aligned frame 0
    m0_0 = get_rotation_transform(90-theta0)

    print('m0_0', m0_0)

    m_0 = la.inv(m0_0)
    # Inverse+coordinate order swap of rotation by (90-theta0) - used to rotate orig image to frame 0
    affine_m_0 = m_0
    
    print('affine_m_0', affine_m_0) #,
    print('map centre', affine_m_0 @ np.array([512,512,1]))
    print('map origin', affine_m_0 @ np.array([0,0,1]))
    print('map f', affine_m_0 @ np.array([1024,1024,1]))

    affine_m_0_x2 = affine_m_0 @ np.diag([1/IM_S,1/IM_S,1])
   # affine_m_0_x2 = np.diag([1/IM_S,1/IM_S,1])
    affine_m_0_x2_d2 = np.diag([IM_S,IM_S,1]) @ affine_m_0 @ np.diag([1/IM_S, 1/IM_S, 1.0])

    transform0_0 = AffineTransform(affine_m_0_x2, (1024*IM_S,1024*IM_S))

    leaf0_0 = LeafView(leaf=leaf0, transform = transform0_0)

    plot_leaf_arrows(leaf0_0)

    transform1_0 = CompositeTransform([transform0_0, d_transform])
    leaf1_0 = LeafView(leaf=leaf1, transform = transform1_0)

    plot_leaf_arrows(leaf1_0)

    im1 = leaf1_0.get_bdd()
    im2 = leaf0_0.get_bdd()

    plt.figure()
    plt.imshow(np.dstack((im1[0]*im1[1], im2[0]*im2[1], im1[0])))
   # plt.show()

    plt.show()
"""
