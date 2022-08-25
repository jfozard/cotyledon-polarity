

"""
Code to import dataframe from CSV files, renaming columns
Also routines to filter for specific subsets (unstretched, stretched, different markers etc.)
"""

import pandas as pd
import numpy as np

def get_leaf_dataframe_orig(marker=None, category=None):
    ds = pd.read_csv('New_data.csv')
    ds.columns = ['Identifier', 'Path', 'Filename','Marker', 'SizeCat','Age','CotWidth','TP','StretchSucc','ControlTP','Angle','RoICells','Author','Flipped_t0', 'Flipped_channels', 'Replicate', 'Scale', 'N1','N2']

    if marker:
        ds = ds.loc[(~ds.Filename.isnull()) & (~ds.Angle.isnull()) & (ds.Marker==marker)]
    else:
        ds = ds.loc[(~ds.Filename.isnull()) & (~ds.Angle.isnull())]

    print('Cotyledons without width measurement', ds.loc[ds.CotWidth.isnull()])


    return ds

def get_leaf_dataframe_revised(marker=None, category=None, sort=True):
    ds = pd.read_csv('New_data_revised.csv')
    ds.columns = ['Identifier', 'Path', 'Filename','Marker', 'SizeCat','Age','CotWidth','TP','StretchSucc','ControlTP','Angle','RoICells','Author','Flipped_t0', 'Flipped_channels', 'Replicate', 'Scale', 'N1','N2']

    if marker:
        ds = ds.loc[(~ds.Filename.isnull()) & (~ds.Angle.isnull()) & (ds.Marker==marker)]
    else:
        ds = ds.loc[(~ds.Filename.isnull()) & (~ds.Angle.isnull())]

    print('Cotyledons without width measurement', ds.loc[ds.CotWidth.isnull()])

    
    if category=='stretch-t5':
        mask = (ds.TP=='t5') & (~ds.StretchSucc.isnull())

        ds = ds.loc[mask]
        
        ds.CotWidth[ds.CotWidth.isnull()] = 600

        if sort:
            idx = ds['CotWidth'].argsort()
            ds = ds.iloc[idx]
            
    elif category =='aggregate-t0':
        mask = ((ds.TP=='no stretch') & (ds.ControlTP!='t5')) | (ds.TP=='t0')

        ds = ds.loc[mask]

        ds.CotWidth[ds.CotWidth.isnull()] = 600

        if sort:
            idx = ds['CotWidth'].argsort()
            ds=ds.iloc[idx]

    else:
        ds.CotWidth[ds.CotWidth.isnull()] = 600

        if sort:
            idx = ds['CotWidth'].argsort()
            ds = ds.iloc[idx]

    return ds

def get_paired_dataframe(category, marker=None):
    ds = get_leaf_dataframe_revised(marker=marker, sort=False)
    if category == 'stretched':
        idx = ds.loc[(ds.TP=='t0') & (ds.StretchSucc=='Y')]['CotWidth'].argsort()

        
        size_ds = []
        size_ds.append(ds.loc[(ds.TP=='t0') & (ds.StretchSucc=='Y')].iloc[idx])
        size_ds.append(ds.loc[(ds.TP=='t5') & (ds.StretchSucc=='Y')].iloc[idx])

        return size_ds
    elif category == 'control':
        idx = ds.loc[(ds.ControlTP=='t0')]['CotWidth'].argsort()

        size_ds = []
        size_ds.append(ds.loc[(ds.ControlTP=='t0') & (ds.Marker==marker)].iloc[idx])
        size_ds.append(ds.loc[(ds.ControlTP=='t5') & (ds.Marker==marker)].iloc[idx])
        return size_ds
    else:
        raise ValueError

def test_orig_revised():

    ds1 = get_leaf_dataframe_orig()
    ds2 = get_leaf_dataframe_revised()

    assert(np.all(ds1['Identifier'].array == ds2['Identifier'].array))



def test_all_orig_files_used():
    from data_path import DATA_PATH
    ds2 = get_leaf_dataframe_revised()
    used_files = []
    for d in ds2.itertuples():
        fn = DATA_PATH + d.Path + '/' + d.Filename+ '.tif'
        used_files.append(fn)
    # Search subdirectories of DATA_PATH
    from pathlib import Path

    ds_files = []
    p = Path(DATA_PATH)
    for f in p.rglob('*.tif'):
        if 'proj' not in f.name:
            ds_files.append(str(f))
        
    print('orig tif', len(used_files), len(ds_files))

    for f in used_files:
        if f not in ds_files:
            print('Not in DS', f)

    for f in ds_files:
        if f not in used_files:
            print('Unused file' , f)
#            print(f, file=of)



def test_all_proj_files_used():
    from data_path import DATA_PATH
    ds2 = get_leaf_dataframe_revised()
    used_files = []
    for d in ds2.itertuples():
        fn = DATA_PATH + d.Path + '/' + d.Filename+ '_proj.tif'
        used_files.append(fn)
    # Search subdirectories of DATA_PATH
    from pathlib import Path

    ds_files = []
    p = Path(DATA_PATH)
    for f in p.rglob('*.tif'):
        if 'proj' in f.name:
            ds_files.append(str(f))
        
    print('proj', len(used_files), len(ds_files))

    for f in used_files:
        if f not in ds_files:
            print('Not in DS', f)

    for f in ds_files:
        if f not in used_files:
            print('Unused file' , f)
#            print(f, file=of)

def test_all_json_files_used():
    from data_path import DATA_PATH
    ds2 = get_leaf_dataframe_revised()
    used_files = []
    for d in ds2.itertuples():
        fn = DATA_PATH + d.Path + '/' + d.Filename+ '.json'
        used_files.append(fn)
    # Search subdirectories of DATA_PATH
    from pathlib import Path

    ds_files = []
    p = Path(DATA_PATH)
    for f in p.rglob('*.json'):
       # if 'proj' in f.name:
            ds_files.append(str(f))
        
    print('json', len(used_files), len(ds_files))

    for f in used_files:
        if f not in ds_files:
            print('Not in DS', f)

    for f in ds_files:
        if f not in used_files:
            print('Unused file' , f)
#            print(f, file=of)


def test_all_roiset_files_used():
    from data_path import DATA_PATH
    ds2 = get_leaf_dataframe_revised()
    used_files = []
    for d in ds2.itertuples():
        fn = DATA_PATH + d.Path + '/' + d.Filename+ '_RoiSet.zip'
        used_files.append(fn)
    # Search subdirectories of DATA_PATH
    from pathlib import Path

    ds_files = []
    p = Path(DATA_PATH)
    for f in p.rglob('*.zip'):
       # if 'proj' in f.name:
#            print(f)
            ds_files.append(str(f))
        
    print('RoiSet', len(used_files), len(ds_files))

    for f in used_files:
        if f not in ds_files:
            print('Not in DS', f)

    for f in ds_files:
        if f not in used_files:
            print('Unused file' , f)
#            print(f, file=of)

def test_all():
#    of = open('unused_files.txt', 'w')
    test_all_orig_files_used()
    test_all_proj_files_used()

    test_all_json_files_used()

    test_all_roiset_files_used()
#    of.close()

#test_all()
