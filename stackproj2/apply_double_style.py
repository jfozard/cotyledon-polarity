#65;5802;1c
import torch

import numpy as np

from torchvision import transforms
import torch.nn.functional as F

from imageio import imread, imsave
from tifffile import imread

from unetmodel import UNet2

import scipy.ndimage as nd

import matplotlib.pyplot as plt

TS_C = 128

TS = 96+TS_C+96
OVERLAP = 96

def round_up(a, b):
    return (a + (b-1))//b

def round(a):
    return int(a+0.5)

def get_overlaps(dim, ts, min_overlap):
    print(dim, ts, min_overlap)
    #N = 2 + round_up(dim - 2*(ts-min_overlap), (ts-2*min_overlap))
    N = round_up(dim-2*min_overlap, ts-2*min_overlap) 
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


def scale_to_uint8(array):

    scaled = array.astype(np.float32)

    if scaled.max() - scaled.min() == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = 255 * (scaled - scaled.min()) / (scaled.max() - scaled.min())

    return scaled.astype(np.uint8)


def pad_image_to_tile_size(im, ts=TS):

    xdim, ydim, _ = im.shape
    nx = xdim//ts
    ny = ydim//ts

    pad_x = ts * (nx + 1) - xdim
    pad_y = ts * (ny + 1) - ydim

    impad = np.pad(im, ((0, pad_x), (0, pad_y), (0, 0)), 'edge')

    return impad


def image_to_model_input(im, ts=TS):
    tt = transforms.ToTensor()

    xdim, ydim, _ = im.shape
    nx = xdim//ts
    ny = ydim//ts

    pad_x = ts * (nx + 1) - xdim
    pad_y = ts * (ny + 1) - ydim

    impad = np.pad(im, ((0, pad_x), (0, pad_y), (0, 0)), 'edge')

    tensorpad = tt(impad)

    tiles = []
    for x in range(nx+1):
        for y in range(ny+1):
            tile = tensorpad[:,x*ts:(x+1)*ts,y*ts:(y+1)*ts]
            tiles.append(tile)

    network_input = torch.stack(tiles)

    return network_input


def grouper(n, iterable):
    args = [iter(iterable)] * n
    return map(list, zip(*args))


def reassemble_image(stack, cols):

    n, _, _ = stack.shape

    nl = [stack[i,:,:] for i in range(n)]

    return np.block(list(grouper(cols, nl)))


def _test_unet(model_fpath, im, output_fpath=None, ts=TS, rotate=0, flip=0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    print('image shape', im.shape, device, np.mean(im), im.dtype)
    channels = 1


    if im.ndim>2:
        im = im[0,:,:]


    im = im.astype(np.float32) - np.mean(im)
    im = (im/np.std(im)).astype(np.float32)

    if flip:
        im = im[:,::-1]
    if rotate!=0:
        im = nd.rotate(im, 90*rotate)
    im = im[:,:,np.newaxis]

    model = UNet2(1, 1, use_bn=True, sc=64)

    d = torch.load(model_fpath, map_location=device)
    if 'model_state_dict' in d:
        model.load_state_dict(d['model_state_dict'])
    else:
        model.load_state_dict(d)
                              
    model.to(device)
 
    model.eval()

    print('new im shape', im.shape, np.mean(im))
    
    network_input = image_to_model_input(im)

    n = network_input.shape[0]
    bs = 8
    
    all_output = []
    with torch.no_grad():
        for i in range((n+bs-1)//bs):
            print('tile {}:{} / {}'.format(i*bs, min(n, (i+1)*bs), network_input.shape[0]))
            part = network_input[i*bs:min(n, (i+1)*bs),:,:,:]
            part = part.to(device)
            print(part.shape)
            
            predmasktiles = model(part)

        
            predmasktiles_np = predmasktiles.cpu().numpy()
            all_output.append(predmasktiles_np)


    predmasktiles_np = np.concatenate(all_output, axis=0)
    xdim, ydim, _ = im.shape
    ny = ydim//ts

    reassembled = reassemble_image(np.squeeze(predmasktiles_np), ny+1)
    cropped = reassembled[:xdim,:ydim]
    as_image = scale_to_uint8(cropped)

    #as_image = nd.rotate(as_image, -90*rotate)

    print('out im', as_image.shape, np.mean(as_image))
    
    if output_fpath:
        imsave(output_fpath, as_image)


    im = as_image[np.newaxis,:,:]/np.max(as_image)
    if rotate!=0:
        im = nd.rotate(im[0], -90*rotate)[np.newaxis,:,:]
    if flip:
        im = im[:,:,::-1]

    print(im.shape)
    return im

def test_unet(model_fpath, im):
    print('test_unet', im.shape)
    res = np.mean(np.stack([_test_unet(model_fpath, im, rotate=r, flip=f) for r in range(4) for f in range(2)], axis=-1), axis=-1)
    return res


def main(model_fpath, input_fpath, output_fpath, rotate):

    input_image = imread(input_fpath)

    print(input_image.shape)
    if len(input_image.shape) == 2:
        pass
    else:
        input_image = input_image[0,:,:]
    input_image = input_image - np.min(input_image)
    input_image = input_image.astype(float)/np.max(input_image)*255
    print(np.std(input_image))
    plt.imshow(input_image)
    plt.colorbar()
    plt.show()

    


if __name__ == "__main__":
    main()


