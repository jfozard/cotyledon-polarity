
import numpy as np
import numpy.random as npr
import scipy.ndimage as nd
from skimage.segmentation import find_boundaries

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt
except ImportError:
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
    from PyQt4.QtCore import Qt

from PIL import Image, ImageDraw

import deformation as df

def to_qimg_rgb(m):
    m = np.ascontiguousarray(m)
    height, width, channels = m.shape
    bytesPerLine = width*3

    return QtGui.QImage(QtCore.QByteArray(m.tostring()), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)    

binary_lut =  np.array([[50, 120, 50], [120, 50, 120], [180, 50, 50],[50, 50, 180]])

# Deformed mapping of multiple timepoints onto a single reference timepoint

        
class Transform(object):
    def __init__(self, x_data, y_data, im_shape, out_shape=None):
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

        
    def interpolate(self, x, y):
        t = (x/(self.im_shape[1]-1.0))*(self.x_data.shape[1]-3)+1
        s = (y/(self.im_shape[0]-1.0))*(self.x_data.shape[0]-3)+1
        return np.array([ df.Interpolate(t, s, self.x_data),  df.Interpolate(t, s, self.y_data) ])

    
    def map_image(self, source, order=0, return_valid=False, pre_affine=None):
        if self.transformation_y is None or pre_affine is not self.stored_pre_affine:
            print('precalc', pre_affine)
            out, valid, self.transformation_y, self.transformation_x = df.interpolate_image(source, self.x_data, self.y_data, out_shape=self.out_shape, pre_affine=pre_affine)
            self.stored_pre_affine = pre_affine
            self.hashed_results = {}
        else:
            print('use calc', pre_affine)
            sh = hash(source.data.tobytes())
            if sh in self.hashed_results:
                out, valid = self.hashed_results[sh]
            else:
                out, valid = df.interpolate_transformation(source, self.transformation_x, self.transformation_y, out_shape=self.out_shape)
                self.hashed_results[sh] = (out, valid)
        if return_valid:
            return out, valid
        else:
            return out

    def edit_transform(self, u0, u1):
        # Invalidate transformation cache
        self.transformation_x = None
        self.transformation_y = None

        self.hashed_results = {}
        
        v0 = self.interpolate(*u0)
        v1 = self.interpolate(*u1)

        # Change x_data and y_data


    def apply_map(self, f):
        self.transformation_x = None
        self.transformation_y = None

        self.hashed_results = {}

        for i in range(self.x_data.shape[0]):
            for j in range(self.x_data.shape[1]):
                p = (self.x_data[i,j], self.y_data[i,j])
                p = f(p)
                self.x_data[i,j], self.y_data[i,j] = p
        
        
    @classmethod
    def from_file(cls, fn, im_shape):
        x_data, y_data = df.load_elastic_transform(fn)
        return Transform(x_data, y_data, im_shape)

