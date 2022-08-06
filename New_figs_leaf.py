
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

import pickle

import scipy.ndimage as nd

from process_utils import *

from pathlib import Path


import matplotlib as mpl

import pandas as pd

plot_output = 'output/plot_out/'


mpl.rcParams.update({ 
    'figure.facecolor': 'none', 
    'figure.edgecolor': 'none',       
    'font.size': 28, 
    'figure.dpi': 72, 
    'figure.subplot.bottom' : .15, 
    'savefig.edgecolor': 'none',
    'savefig.facecolor': 'none',
    'font.family': "sans-serif",
    'font.sans-serif': 'Arial',
})

from New_figs_base import process_data, arrow_leaf_plot, get_counts, compare_hists

ds = pd.read_csv('New_data.csv')
ds.columns=['Identifier', 'Path', 'Filename','Marker', 'SizeCat','Age','CotWidth','StretchTP','StretchSucc','ControlTP','Angle','RoICells','Author','Flipped_t0', 'Flipped_channels', 'Replicate', 'Scale', 'N1','N2']

ds = ds.loc[ (~ds.Filename.isnull()) & (~ds.Angle.isnull()) & (ds.Marker=='brxl') ]

print(ds.loc[ds.CotWidth.isnull()])

ds.CotWidth[ds.CotWidth.isnull()]=600


Path(plot_output).mkdir(exist_ok=True, parents=True)

mask = (ds.StretchTP=='t5') & (~ds.StretchSucc.isnull())
idx = ds.loc[mask]['CotWidth'].argsort()
ds_range=ds.loc[mask].iloc[idx]
prefix = plot_output + 'stretch-t5-'

print('>>>', prefix, ds_range)

stretch_t5_data = process_data(prefix, ds_range)

mask = ((ds.StretchTP=='no stretch') & (ds.ControlTP!='t5')) | (ds.StretchTP=='t0')
idx = ds.loc[mask]['CotWidth'].argsort()
ds_range=ds.loc[mask].iloc[idx]

highlight = np.where((ds_range.StretchTP=='t0') &  (~ds_range.StretchSucc.isnull()))[0]

prefix = plot_output + 'aggregate-t0-'

print('>>>', prefix, ds_range)

print('hl', highlight)

print(len(ds_range[ds_range.CotWidth<550]), len(ds_range[(550<=ds_range.CotWidth)*(ds_range.CotWidth<650)]), len(ds_range[ds_range.CotWidth>=650]))

aggregate_t0_data = process_data(prefix, ds_range, hl= highlight, split_sizes=[550, 650])

sample_data = {}

mask = ((ds.StretchTP=='no stretch') & (ds.ControlTP!='t5')) | (ds.StretchTP=='t0')
idx = ds.loc[mask]['CotWidth'].argsort()
ds_range=ds.loc[mask].iloc[idx]


import numpy.random as npr

npr.seed(1234)

for i in range(20):

    # Sample 13
    idx = np.sort(npr.choice(np.arange(len(ds_range)), 13, replace=False))

    print('idx', idx, type(idx), idx.dtype, len(ds_range))
    
    ds_red = ds_range.iloc[idx]

    print('ds_red', ds_red.Identifier)

    prefix = plot_output + 'aggregate-t0-sample-'+str(i)

    print('prefix', prefix)
    
    sample_data[i] = process_data(prefix, ds_red, only_hist=True, font_scale=1.3)

r = (80, 100)

compare_output = 'output/compare_output/'
Path(compare_output).mkdir(exist_ok=True, parents=True)

compare_hists(aggregate_t0_data, stretch_t5_data, 'aggregate_t0', 'stretch_t5', of=open(compare_output+'compare-agg_t0_stretch_t5.txt','w'))
#compare_hists(sample_data[2], stretch_t5_data, 'aggregate_t0_split_2', 'stretch_t5', of=open(compare_output+'compare-agg_t0_split_2_stretch_t5.txt','w'))
