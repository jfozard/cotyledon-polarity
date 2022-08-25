

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

from get_leaf_dataframe import get_leaf_dataframe_revised

output_path = 'output/'
plot_output = output_path+'plot_out/'


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






ds_range = get_leaf_dataframe_revised(marker='basl_basl', category='aggregate-t0')

prefix = plot_output+'native-basl-'

print('>>>', prefix, ds_range)

process_data(prefix, ds_range, marker='native_basl', leaf_data_path=output_path+'leaf_data/')


ds_range = get_leaf_dataframe_revised(marker='35S_basl', category='aggregate-t0')

prefix = plot_output+'35S-basl-'

print('>>>', prefix)
print( ds_range)

process_data(prefix, ds_range, marker='35S_basl', leaf_data_path=output_path+'leaf_data/')

