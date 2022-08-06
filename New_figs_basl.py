

# Make histograms of angle alpha and draw Rose plots

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
from PIL import Image
from math import atan2, pi, sqrt, ceil, sin, cos
import scipy.ndimage as nd

from tifffile import imread
import csv
import sys

import pickle

import scipy.ndimage as nd

from process_utils import *

from pathlib import Path


import matplotlib as mpl

import pandas as pd

from New_figs_base import process_data, arrow_leaf_plot, get_counts, compare_hists

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





ds = pd.read_csv('New_data.csv')
ds.columns=['Identifier', 'Path', 'Filename','Marker', 'SizeCat','Age','CotWidth','StretchTP','StretchSucc','ControlTP','Angle','RoICells','Author','Flipped_t0', 'Flipped_channels', 'Replicate', 'Scale', 'N1','N2']

ds = ds.loc[ (~ds.Filename.isnull()) & (~ds.Angle.isnull())] #& (ds.Marker=='basl') ]


print(ds.loc[ds.CotWidth.isnull()])

ds.CotWidth[ds.CotWidth.isnull()]=600



print('ds, ', ds)

ds2 = ds.loc[ds.Path=='Native BASL']

idx = ds2['CotWidth'].argsort()

ds_range = ds2.iloc[idx]

prefix = 'output/plot_out/native-basl-'

print('>>>', prefix, ds_range)

process_data(prefix, ds_range, marker='native_basl')



ds2 = ds.loc[(ds.Marker=='35S_basl') & (ds.StretchTP!='t5')]


print('ds, ', ds2)

idx = ds2['CotWidth'].argsort()

ds_range = ds2.iloc[idx]


prefix = 'output/plot_out/35S-basl-'

print('>>>', prefix)
print( ds_range)

process_data(prefix, ds_range, marker='35S_basl')

