#1;4205;0c1;4205;0c

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
from PIL import Image
from math import atan2, pi, sqrt, ceil, sin, cos
from tifffile import imread
import csv
import matplotlib.cm as cm
import SimpleITK as sitk
from roi_decode import roi_zip
from skimage.segmentation import find_boundaries
import scipy.ndimage as nd

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


def polarity_from_fiji_auto(stack, filename, use_centroids=True, reverse_arrows=False):
    roi_data = roi_zip(filename)
    new_data = {}
#    stack += 1
    centroids = nd.center_of_mass(np.ones_like(stack), labels=stack, index=np.arange(np.max(stack)))

    bdd = find_boundaries(stack)
    bdd_dist = nd.distance_transform_edt(1-bdd)
    
#    print(centroids, stack)
    for r in roi_data.values():

        d2 = bdd_dist[int(r.Y2), int(r.X2)]
        d1 = bdd_dist[int(r.Y1), int(r.X1)]

        if d1>d2:
            r.X1, r.Y1, r.X2, r.Y2 = r.X2, r.Y2, r.X1, r.Y1

        idx = stack[int(r.Y2), int(r.X2)]
        
        if idx>0:
            if use_centroids:
                c = centroids[idx]
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((c[1], c[0]))]
            else:
                new_data[idx] = [np.array((r.X1, r.Y1)), np.array((r.X2, r.Y2))]
    return new_data


def watershed_no_labels(s, sigma=1.0, h=1.0):
    s = np.ascontiguousarray(255*(s/float(np.max(s))))
    signal = sitk.GetImageFromArray(s)
    if sigma>0:
        signal = sitk.DiscreteGaussian(signal, sigma)
    signal = sitk.Cast(signal, sitk.sitkInt16)
    labels = sitk.MorphologicalWatershed(signal, level=h, markWatershedLine = False, fullyConnected=False)
    labels = sitk.GetArrayFromImage(labels)
    return labels.astype(np.int32)


