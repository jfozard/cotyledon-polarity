
import click
import torch
#import dtoolcore

import numpy as np

from torchvision import transforms

from imageio import imread, imsave

from unetmodel import UNet2

from tifffile import TiffWriter, TiffFile, imsave

import scipy.ndimage as nd

from unetmodel import ProjSegNet, ProjSegNet2
#from torchsummary import summary

from torch import nn


def load_tiff(fn):
    with TiffFile(fn) as tiff:
        data = tiff.series[0].asarray()
    data = np.squeeze(data)
    return data

def write_tiff(fn, A):
#    A = np.transpose(A, (2, 0, 1))
    imsave(fn, A[np.newaxis, :, np.newaxis, :, :, np.newaxis])

def write_tiff_col(fn, A):
#    A = np.transpose(A, (2, 0, 1))
    imsave(fn, A[np.newaxis, np.newaxis, :, :, :], imagej=True)

    
TS=96+128+96
OVERLAP=96

def scale_to_uint8(array):

    scaled = array.astype(np.float32)

    if scaled.max() - scaled.min() == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = 255 * (scaled - scaled.min()) / (scaled.max() - scaled.min())

    return scaled.astype(np.uint8)

def scale_limited(array):

    scaled = array.astype(np.float32)
    
    if scaled.max() - scaled.min() == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    print('image scale', scaled.max())
    scaled = 255 * scaled #- scaled.min()) / (scaled.max() - scaled.min())

    return scaled.astype(np.uint8)



"""
TS - O, TS - 2O, TS - 2O, TS - O

i.e. N = 2 + round_up((D - 2*(TS-O))/(TS-2*O))
N*TS - (N-1)*O = D
O = (N*TS - D)/(N-1)

"""

def pad_image_to_tile_size(im, ts=TS):

    zdim, ydim, xdim = im.shape
    ny = ydim//ts
    nx = xdim//ts

    pad_y = ts * (ny + 1) - ydim

    pad_x = ts * (nx + 1) - xdim

    impad = np.pad(im, ((0,0), (0, pad_y), (0, pad_x)), 'edge')

    im = im[:,:,np.newaxis]

    return impad


def round_up(a, b):
    return (a + (b-1))//b

def round(a):
    return int(a+0.5)

def get_overlaps(dim, ts, min_overlap):
    print(dim, ts, min_overlap)
    #N = 2 + round_up(dim - 2*(ts-min_overlap), (ts-2*min_overlap))
    N = round_up(dim-2*min_overlap, ts-2*min_overlap)
    if N==1:
        O = 0
    else:
        O = (N*ts - dim)/(N-1)/2
    print('N,O', N,O)

    start_pos = []
    end_pos = []
    start_use = []
    end_use = []

    
    idx_start = 0
    idx_start_use = 0
    idx_end_use = ts - O
    for i in range(N):
        idx_start = i*(ts - 2*O)
        start_pos += [round(idx_start)]
        end_pos += [round(idx_start)+ts]
        start_use += [round(idx_start_use)]
        end_use += [round(idx_end_use)]
        idx_start_use = idx_end_use
        idx_end_use += (ts - 2*O)
    end_use[-1] = dim

    print(list(zip(start_pos, end_pos)), list(zip(start_use, end_use)))
    for i in range(N):
        print(start_use[i] - start_pos[i], end_pos[i] - end_use[i])

    return N, list(zip(start_pos, end_pos)), list(zip(start_use, end_use))
        


    """
    0 -> TS , 0 -> TS - O/2
    TS-O -> 2*TS-O, TS - O/2 -> 2*TS - O/2
    2*TS - 2*O -> 3*TS - 2*O, 2*TS - 3*O/2 - 3*TS - 3*O/2
    ...
    """


    
def image_to_model_input(im, ts=TS):

    zdim, ydim, xdim = im.shape

    
    ny, y_pos, y_use = get_overlaps(ydim, ts, OVERLAP)
    nx, x_pos, x_use = get_overlaps(xdim, ts, OVERLAP)

    print(y_pos, x_pos)

#    impad = pad_image_to_tile_size(im)

    tensorpad = torch.from_numpy(im[np.newaxis,:,:,:])



    
    tiles = []
    for y in range(ny):
        for x in range(nx):
            tile = tensorpad[:,:,y_pos[y][0]:y_pos[y][1],x_pos[x][0]:x_pos[x][1]]
            tiles.append(tile)

    network_input = torch.stack(tiles)
    print(network_input.shape)
    
    return network_input


def grouper(n, iterable):
    args = [iter(iterable)] * n
    return map(list, zip(*args))


def reassemble_image(stack, cols, ydim, xdim, ts):

    n, _, _ = stack.shape

    ny, y_pos, y_use = get_overlaps(ydim, ts, OVERLAP)
    nx, x_pos, x_use = get_overlaps(xdim, ts, OVERLAP)

    nl = []
    idx = 0
#    for y in range(ny):
    for y in range(ny):
        nly = []
        for x in range(nx):
            s = stack[idx,
                      (y_use[y][0]-y_pos[y][0]):(y_use[y][1]-y_pos[y][0]),
                      (x_use[x][0]-x_pos[x][0]):(x_use[x][1]-x_pos[x][0])]
            nly.append(s)
            idx += 1
        nl.append(nly)

#    print([x.shape for x in nl])

#    print(list(grouper(cols, nl)))
    
    return np.block(nl)


def reassemble_image_3d(stack, cols, ydim, xdim, ts):

    n, _, _, _ = stack.shape
    ny, y_pos, y_use = get_overlaps(ydim, ts, OVERLAP)
    nx, x_pos, x_use = get_overlaps(xdim, ts, OVERLAP)

    nl = []
    idx = 0
    for y in range(ny):
        nly = []
        for x in range(nx):
            s = stack[idx, :,
                      (y_use[y][0]-y_pos[y][0]):(y_use[y][1]-y_pos[y][0]),
                      (x_use[x][0]-x_pos[x][0]):(x_use[x][1]-x_pos[x][0])]
            nly.append(s)
            idx += 1
        nl.append(nly)

    return np.block(nl)



def test_unet(model_fpath, im_all, outputimall_fpath,  ts=TS, channel=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    channels = 1

    im = im_all[:,channel,:,:]
    
    im = im.astype(np.float32) - np.mean(im)
    im = (im/np.std(im)).astype(np.float32)

    zdim, ydim, xdim = im.shape

    print(im.shape)
    
    model = ProjSegNet2(z_depth=zdim, depth_c=8, sc=32)
#    model = ProjSegNet() 
#    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_fpath, map_location=device))
    model.to(device)
    model.eval()
                
#    summary(model, (zdim, TS, TS), device=device)

    print('done summary')
    network_input = image_to_model_input(im)
    
    n = network_input.shape[0]
#    predmasktiles_np = []
#    predimtiles_np = []
    preddepthtiles_np = []

    
    bs = 1
    with torch.no_grad():
        for i in range((n+bs-1)//bs):
            print(i)
            part = network_input[i*bs:min(n, (i+1)*bs),:,:,:,:]
            part = part.to(device)
            print(part.shape)
            
            preddepthtiles, predimtiles, predmasktiles = model(part)
            
#            predmasktiles_np.append(predmasktiles.cpu().numpy())
#            predimtiles_np.append(predimtiles.cpu().numpy())
            preddepthtiles_np.append(preddepthtiles.cpu().numpy())

            

    zdim, ydim, xdim = im.shape
    nx = xdim//ts

    """
    predmasktiles_np = np.concatenate(predmasktiles_np)

    print(predmasktiles_np.shape)
    reassembled = reassemble_image(predmasktiles_np[:,0,:,:], nx, ydim, xdim, TS)
    print('reassembled_shape', reassembled.shape)

    cropped = reassembled#[:ydim,:xdim]
    as_image = scale_to_uint8(cropped)


    print('save unet output', output_fpath, as_image.dtype, as_image.shape, cropped.dtype, np.max(cropped), np.max(as_image))
    imsave(output_fpath, as_image)

    predimtiles_np = np.concatenate(predimtiles_np)
    print(predimtiles_np.shape, predmasktiles_np.shape)
    reassembled = reassemble_image(predimtiles_np[:,0,:,:], nx, ydim, xdim, TS)
    cropped = reassembled#[:ydim,:xdim]
    as_image = scale_to_uint8(cropped)

    print('save image output', outputim_fpath, as_image.dtype, as_image.shape, cropped.dtype, np.max(cropped), np.max(as_image))
    imsave(outputim_fpath, as_image)
    """

    preddepthtiles_np = np.concatenate(preddepthtiles_np)
    reassembled = reassemble_image_3d(preddepthtiles_np[:,0,:,:,:], nx, ydim, xdim, TS)
#    cropped = reassembled#[:zdim, :ydim,: xdim]
#    as_image = scale_limited(cropped)

#    print('save depth output', outputdepth_fpath, as_image.dtype, as_image.shape, cropped.dtype, np.max(cropped), np.max(as_image))
#    write_tiff(outputdepth_fpath, as_image)

    projected_all = np.sum(reassembled[:,np.newaxis,:,:]*im_all, axis=0)
    as_image = (projected_all).astype(np.uint8)
    print(as_image.shape)
    write_tiff_col(outputimall_fpath, as_image)
    print('save all image output', outputimall_fpath, as_image.dtype, as_image.shape, projected_all.dtype, np.max(projected_all))
     
    
    
@click.command()
@click.argument('input_fpath')
@click.argument('outputimall_fpath')
@click.argument('channel', default=1)
@click.argument('order', default=0)
@click.argument('num_ch', default=2)
@click.argument('rev_ch', default=0)
def main(input_fpath, outputimall_fpath, channel, order, num_ch, rev_ch):

    im = load_tiff(input_fpath)
    channel = int(channel)
    
    if len(im.shape)==3:
        im = im[:,np.newaxis,:,:]

    im = im[:,:num_ch]

    if rev_ch:
        im = im[:,::-1]

        
    if order==1:
        print('invert')
        im = im[::-1, :, :,:]
#    im = im[:,:,:256,:256]
    print(im.shape, channel)

 #   quit()
    
    model_input = im
    model_fpath = '../data/model.pt'
    
    test_unet(model_fpath, model_input, outputimall_fpath,  ts=TS, channel=channel)


if __name__ == "__main__":
    main()
