import numpy as np
import sys
import scipy.ndimage as nd
import numba
import time
import numpy.linalg as la
import functools



@numba.njit
def Interpolate_derivs(x, y, data):
    """
    Interpolate data using cardinal b-splines (spacing 1).
    Also calculate x- and y- derivatives of the interpolated data
    Values at a single x,y point.

    returns ival, ival_dx, ival_dy

    """
    
    widthToUse = data.shape[1]
    heightToUse = data.shape[0]

    ix=numba.int32(x)
    iy=numba.int32(y)

    xIndex = np.zeros((4,), dtype=numba.int32)
    yIndex = np.zeros((4,), dtype=numba.int32)

# Set X indexes
# p is the index of the rightmost influencing spline
    p = (ix + 2) if  (0.0 <= x) else (ix + 1)
    for k in range(4):
        xIndex[k] = -1 if (p<0 or p>=widthToUse) else p
        p -= 1

# Set Y indexes
    p = (iy + 2) if  (0.0 <= y) else (iy + 1)
    for k in range(4):
        yIndex[k] = -1 if (p<0 or p>=heightToUse) else p
        p -= 1


# Compute how much the sample depart from an integer position
# [ conditional because int rounds down for positive numbers and up for negative numbers ]

    ex = x - ((ix) if (0.0 <= x) else (ix - 1))
    ey = y - ((iy) if (0.0 <= y) else (iy - 1))

    xWeight = np.zeros((4,), dtype=numba.float64)
    dxWeight = np.zeros((4,), dtype=numba.float64)
    yWeight = np.zeros((4,), dtype=numba.float64)
    dyWeight = np.zeros((4,), dtype=numba.float64)

# Set X weights for the image and derivative interpolation
    for (weight, e) in [(xWeight, ex), (yWeight, ey)]:
        s = 1.0 - e
        weight[0]  = 0.5 * e * e * e / 3.0 
        weight[1]  = 2.0 / 3.0 - (2.0 - s) * 0.5 * s * s
        weight[2]  = 2.0 / 3.0 - (2.0 - e) * 0.5 * e * e 
        weight[3]  = 0.5 * s * s * s / 3.0 

    for (dweight, e) in [(dxWeight, ex), (dyWeight, ey)]:
        s = 1.0 - e
        dweight[0]  = 0.5 * e * e 
        dweight[1]  = - ( 1.5 * s * s - 2 * s)
        dweight[2]  = 1.5 * e * e - 2 * e
        dweight[3]  = - 0.5 * s * s 



    ival = 0.0
    ival_dx = 0.0
    ival_dy = 0.0
    for j in range(4):
        s = 0.0
        s_dx = 0.0
#        s_dy = 0.0
        iy=yIndex[j]
        if iy != -1:
            for i in range(4):
                ix=xIndex[i]
                if ix!=-1:
                    s += xWeight[i]*data[iy][ix]
                    s_dx += dxWeight[i]*data[iy][ix]
                    #s_dy += xWeight[i]*data[iy][ix]
            s_dy = s
            ival += yWeight[j] * s
            ival_dx += yWeight[j] * s_dx
            ival_dy += dyWeight[j] * s_dy
    return ival, ival_dx, ival_dy


@numba.njit
def Interpolate(x, y, data):
    """
    Interpolate data using cardinal b-splines (spacing 1).
    Values at a single x,y point.

    returns ival

    """

    widthToUse = data.shape[1]
    heightToUse = data.shape[0]

    ix=numba.int32(x)
    iy=numba.int32(y)

    xIndex = np.zeros((4,), dtype=numba.int32)
    yIndex = np.zeros((4,), dtype=numba.int32)

# Set X indexes
# p is the index of the rightmost influencing spline
    p = (ix + 2) if  (0.0 <= x) else (ix + 1)
    for k in range(4):
        xIndex[k] = -1 if (p<0 or p>=widthToUse) else p
        p -= 1

# Set Y indexes
    p = (iy + 2) if  (0.0 <= y) else (iy + 1)
    for k in range(4):
        yIndex[k] = -1 if (p<0 or p>=heightToUse) else p
        p -= 1


# Compute how much the sample depart from an integer position
# [ conditional because int rounds down for positive numbers and up for negative numbers ]

    ex = x - ((ix) if (0.0 <= x) else (ix - 1))
    ey = y - ((iy) if (0.0 <= y) else (iy - 1))

    xWeight = np.zeros((4,), dtype=numba.float64)
    yWeight = np.zeros((4,), dtype=numba.float64)


# Set X weights for the image and derivative interpolation
    for (weight, e) in [(xWeight, ex), (yWeight, ey)]:
        s = 1.0 - e
        weight[0]  = 0.5 * e * e * e / 3.0 
        weight[1]  = 2.0 / 3.0 - (2.0 - s) * 0.5 * s * s
        weight[2]  = 2.0 / 3.0 - (2.0 - e) * 0.5 * e * e 
        weight[3]  = 0.5 * s * s * s / 3.0 



    ival = 0.0
    for j in range(4):
        s = 0.0
        iy=yIndex[j]
        if iy != -1:
            for i in range(4):
                ix=xIndex[i]
                if ix!=-1:
                    s += xWeight[i]*data[iy][ix]
            ival+=yWeight[j] * s
    return ival

@numba.njit
def get_weights(x, n):
    """
    Get weights for interpolation at a particular point.
    Only weights for value interpolation (not derivatives)

    x - position
    n - ?number of intervals?
    
    returns xIndex, xWeight
    """
    
    widthToUse = n+3

    ix=numba.int32(x)

    xIndex = np.zeros((4,), dtype=numba.int32)
# Set X indexes
# p is the index of the rightmost influencing spline
    p = (ix + 2) if  (0.0 <= x) else (ix + 1)
    for k in range(4):
        xIndex[k] = -1 if (p<0 or p>=widthToUse) else p
        p -= 1

# Compute how much the sample depart from an integer position
# [ conditional because int rounds down for positive numbers and up for negative numbers ]

    ex = x - ((ix) if (0.0 <= x) else (ix - 1))

    xWeight = np.zeros((4,), dtype=numba.float64)
#    dxWeight = np.zeros((4,), dtype=numba.float64)

# Set X weights for the image and derivative interpolation
    e = ex
    s = 1.0 - e
    xWeight[0]  = 0.5 * e * e * e / 3.0 
    xWeight[1]  = 2.0 / 3.0 - (2.0 - s) * 0.5 * s * s
    xWeight[2]  = 2.0 / 3.0 - (2.0 - e) * 0.5 * e * e 
    xWeight[3]  = 0.5 * s * s * s / 3.0 

    """
    for (dweight, e) in [(dxWeight, ex), (dyWeight, ey)]:
        s = 1.0 - e
        dweight[0]  = 0.5 * e * e 
        dweight[1]  = - ( 1.5 * s * s - 2 * s)
        dweight[2]  = 1.5 * e * e - 2 * e
        dweight[3]  = - 0.5 * s * s 
    """


    return xIndex, xWeight

def load_elastic_transform(filename):
    """
    Load elastic transform saved by bUnwarpJ

    filename

    returns x_data, y_data

    """
    with open(filename,'r') as f:
        u = f.read()
        u = u.split('\n')
        intervals = int(u[0].split('=')[1])
#        print(intervals)
        N = intervals + 3
        x_data = []
        for i in range(N):
            s = u[i+3].split(' ')
            x_data.append([float(v) for v in s if v!=''])
#        print(x_data)
        y_data = []
        for i in range(N):
            s = u[N+i+5].split(' ')
            y_data.append([float(v) for v in s if v!=''])
#        print y_data
        return np.array(x_data), np.array(y_data)

@numba.njit
def get_pre_affine(out_shape, pre_affine):
    valid_shape = out_shape
    m, n = out_shape
    valid = np.zeros(out_shape, dtype=numba.float64)
    MM = pre_affine[:2,:2].copy()
    c = pre_affine[:2,2].copy()
    for v in numba.prange(m):
        p = np.zeros((2,), dtype=numba.float64)
        for u in range(n):
            p[0] = numba.float64(u)
            p[1] = numba.float64(v)
            p = MM.dot(p) + c
            valid[v,u] = ((p[0]>=0) & (p[0]<=(valid_shape[0]-1)) & (p[1]>=0) & (p[1]<=(valid_shape[1]-1)))
    return valid



@numba.njit(parallel=True)
def get_transform(x_data, y_data, out_shape, pre_affine=None):
    """
    Find source point for each point in image

    x_data, y_data : The mapping
    out_shape : shape of target image

    returns:
    transformation_x, transformation_y : arrays of x- and y- coordinates for each point in the image

    """
    m = out_shape[0]
    n = out_shape[1]
    transformation_x = np.zeros((m,n), dtype=numba.float64)
    transformation_y = np.zeros((m,n), dtype=numba.float64)
    intervals = x_data.shape[0]-3
    if pre_affine is None:
        for v in numba.prange(m):
            tv = numba.float64(v*intervals)/(m-1) + 1.0
            for u in range(n):
                tu = numba.float64(u*intervals)/(n-1) + 1.0
                transformation_x[v,u] = Interpolate(tu, tv, x_data)
                transformation_y[v,u] = Interpolate(tu, tv, y_data)
    else:
#        print(pre_affine[:2, :2].dtype)
#        MM = np.array(pre_affine[:2,:2], dtype=numba.float64)
#        c = np.array(pre_affine[:2,2], dtype=numba.float64)
        MM = pre_affine[:2,:2].copy()
        c = pre_affine[:2,2].copy()
        for v in numba.prange(m):
            p = np.zeros((2,), dtype=numba.float64)
            for u in range(n):
                p[0] = numba.float64(u)
                p[1] = numba.float64(v)
                p = MM.dot(p) + c
                tv = numba.float64(p[1]*intervals)/(m-1) + 1.0
                tu = numba.float64(p[0]*intervals)/(n-1) + 1.0
                transformation_x[v,u] = Interpolate(tu, tv, x_data)
                transformation_y[v,u] = Interpolate(tu, tv, y_data)

        
    return transformation_x, transformation_y
                
@numba.njit(parallel=True)
def get_transform_points(x_data, y_data, out_shape, pts):
    """
    Find source point for each point in image

    x_data, y_data : The mapping
    out_shape : shape of target image

    returns:
    transformation_x, transformation_y : arrays of x- and y- coordinates for each point in the image

    """
    m = out_shape[0]
    n = out_shape[1]
    N = pts.shape[0]
    output = np.zeros((N,2), dtype=numba.float64)
    intervals = x_data.shape[0]-3
    for v in numba.prange(N):
        tv = numba.float64(pts[v,1]*intervals)/(m-1) + 1.0
        tu = numba.float64(pts[v,0]*intervals)/(n-1) + 1.0
        output[v,0] = Interpolate(tu, tv, x_data)
        output[v,1] = Interpolate(tu, tv, y_data)

    return output

@numba.njit
def get_all_weights(M, N, m, n):
    """
    Precalculate all interpolation weights (for value interpolation)
    """
    # Precalculate weights
    p_xIndex = np.zeros((n,4), dtype=numba.int32)
    p_xWeights = np.zeros((n,4), dtype=numba.float64)
    p_yIndex = np.zeros((m,4), dtype=numba.int32) 
    p_yWeights = np.zeros((m,4), dtype=numba.float64) 

    for u in numba.prange(n):
        tu = numba.float64(u*N)/(n-1) + 1.0
        p_xIndex[u,:], p_xWeights[u,:] = get_weights(tu, n-1)
        
    for v in numba.prange(m):
        tv = numba.float64(v*M)/(m-1) + 1.0
        p_yIndex[v,:], p_yWeights[v,:] = get_weights(tv, m-1)

    return p_xIndex, p_xWeights, p_yIndex, p_yWeights

    
@numba.njit
def get_transform_grid(x_data, y_data, out_shape):
    """
    Find source point for each point in image

    (Like get_transform, but precalculating the weights for each 
    x and y position )

    x_data, y_data : The mapping
    out_shape : shape of target image

    returns:
    transformation_x, transformation_y : arrays of x- and y- coordinates for 
    each point in the image

    """


    m, n = out_shape
    transformation_x = np.zeros(out_shape, dtype=numba.float64)
    transformation_y = np.zeros(out_shape, dtype=numba.float64)
    intervals = x_data.shape[0]-3

    # Precalculate weights

    p_xIndex = np.zeros((out_shape[1],4), dtype=numba.int32)
    p_xWeights = np.zeros((out_shape[1],4), dtype=numba.float64)
    p_yIndex = np.zeros((out_shape[0],4), dtype=numba.int32) 
    p_yWeights = np.zeros((out_shape[0],4), dtype=numba.float64) 

    for u in numba.prange(n):
        tu = numba.float64(u*intervals)/(n-1) + 1.0
        p_xIndex[u,:], p_xWeights[u,:] = get_weights(tu, n-1)
        
    for v in numba.prange(m):
        tv = numba.float64(v*intervals)/(m-1) + 1.0
        p_yIndex[v,:], p_yWeights[v,:] = get_weights(tv, m-1)
    
    
    for v in numba.prange(m):
        for u in range(n):
            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*x_data[iy,ix]
                ival+=p_yWeights[v,j] * s

            transformation_x[v,u] = ival

            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*y_data[iy,ix]
                ival+=p_yWeights[v,j] * s

            
            transformation_y[v,u] = ival

    return transformation_x, transformation_y



@numba.njit
def get_transform_grid_weights(x_data, y_data, out_shape):
    """
    Find source point for each point in image

    (Like get_transform_grid, but returning the weights at each row and column)

    x_data, y_data : The mapping
    out_shape : shape of target image

    returns:
    transformation_x, transformation_y : arrays of x- and y- coordinates for 
    each point in the image
    p_xIndex, p_xWeights, p_yIndex, p_yWeights

    """


    m, n = out_shape
    transformation_x = np.zeros(out_shape, dtype=numba.float64)
    transformation_y = np.zeros(out_shape, dtype=numba.float64)
    M = x_data.shape[0]-3
    N = x_data.shape[1]-3

    # Precalculate weights

    p_xIndex = np.zeros((out_shape[1],4), dtype=numba.int32)
    p_xWeights = np.zeros((out_shape[1],4), dtype=numba.float64)
    p_yIndex = np.zeros((out_shape[0],4), dtype=numba.int32) 
    p_yWeights = np.zeros((out_shape[0],4), dtype=numba.float64) 

    for u in numba.prange(n):
        tu = numba.float64(u*N)/(n-1) + 1.0
        p_xIndex[u,:], p_xWeights[u,:] = get_weights(tu, N)
        
    for v in numba.prange(m):
        tv = numba.float64(v*M)/(m-1) + 1.0
        p_yIndex[v,:], p_yWeights[v,:] = get_weights(tv, M)
    
    for v in numba.prange(m):
        for u in range(n):
            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*x_data[iy][ix]
                ival+=p_yWeights[v,j] * s

            transformation_x[v,u] = ival

            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*y_data[iy][ix]
                ival+=p_yWeights[v,j] * s

            
            transformation_y[v,u] = ival

    return transformation_x, transformation_y, p_xIndex, p_xWeights, p_yIndex, p_yWeights


@numba.njit
def get_transform_grid_weights_subsample(x_data, y_data, ss_x, ss_y, out_shape):
    """
    UNSURE
    """
    m, n = out_shape
    M = x_data.shape[0]-3
    N = x_data.shape[1]-3

    # Precalculate weights

    m_ss = (out_shape[0]+(ss_y-1))//ss_y
    n_ss = (out_shape[1]+(ss_x-1))//ss_x
    
    transformation_x = np.zeros((m_ss, n_ss), dtype=numba.float64)
    transformation_y = np.zeros((m_ss, n_ss), dtype=numba.float64)
    
    p_xIndex = np.zeros((n_ss,4), dtype=numba.int32)
    p_xWeights = np.zeros((n_ss,4), dtype=numba.float64)
    p_yIndex = np.zeros((m_ss,4), dtype=numba.int32) 
    p_yWeights = np.zeros((m_ss,4), dtype=numba.float64) 

    
    
    for u in numba.prange(0,n,ss_x):
        tu = numba.float64((u*N)/(n-1) + 1.0)
        p_xIndex[u//ss_x,:], p_xWeights[u//ss_x,:] = get_weights(tu, N)

    for v in numba.prange(0,m,ss_y):
        tv = numba.float64((v*M)/(m-1) + 1.0)
        p_yIndex[v//ss_y,:], p_yWeights[v//ss_y,:] = get_weights(tv, M)
    
    for v in range(m_ss):
        for u in range(n_ss):
            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*x_data[iy][ix]
                ival+=p_yWeights[v,j] * s

            transformation_x[v,u] = ival

            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*y_data[iy][ix]
                ival+=p_yWeights[v,j] * s

            
            transformation_y[v,u] = ival

    return transformation_x, transformation_y, p_xIndex, p_xWeights, p_yIndex, p_yWeights

@numba.njit
def get_spline_grid(data, out_shape):
    """
    Interpolate spline-data at a grid of points.
    """
    m, n = out_shape
    output = np.zeros(out_shape, dtype=numba.float64)
    M = data.shape[0]-3
    N = data.shape[1]-3

    # Precalculate weights

    p_xIndex = np.zeros((out_shape[1],4), dtype=numba.int32)
    p_xWeights = np.zeros((out_shape[1],4), dtype=numba.float64)
    p_yIndex = np.zeros((out_shape[0],4), dtype=numba.int32) 
    p_yWeights = np.zeros((out_shape[0],4), dtype=numba.float64) 

    for u in numba.prange(n):
        tu = numba.float64(u*N)/(n-1) + 1.0
        p_xIndex[u,:], p_xWeights[u,:] = get_weights(tu, n-1)
        
    for v in numba.prange(m):
        tv = numba.float64(v*M)/(m-1) + 1.0
        p_yIndex[v,:], p_yWeights[v,:] = get_weights(tv, m-1)
    
    
    for v in numba.prange(m):
        for u in range(n):
            ival = 0.0
            for j in range(4):
                s = 0.0
                iy=p_yIndex[v,j]
                if iy != -1:
                    for i in range(4):
                        ix=p_xIndex[u,i]
                        if ix!=-1:
                            s += p_xWeights[u,i]*data[iy][ix]
                ival+=p_yWeights[v,j] * s

            output[v,u] = ival

    return output




@numba.njit
def get_transform_deriv(x_data, y_data, out_shape):
    """
    Given a transformation x_data, y_data, evaluate (simple loop over grid points)

    returns:
    transformation_x, transformation_y, 
    transformation_x_dx, transformation_y_dx - derivatives with respect to the x-position
    transformation_x_dy, transformation_y_dy - derivatives with respect to the y-position
    """
    m, n = out_shape
    transformation_x = np.zeros(out_shape, dtype=numba.float64)
    transformation_y = np.zeros(out_shape, dtype=numba.float64)
    transformation_x_dx = np.zeros(out_shape, dtype=numba.float64)
    transformation_y_dx = np.zeros(out_shape, dtype=numba.float64)
    transformation_x_dy = np.zeros(out_shape, dtype=numba.float64)
    transformation_y_dy = np.zeros(out_shape, dtype=numba.float64)
    intervals = x_data.shape[0]-3
    for v in numba.prange(m):
        tv = numba.float64(v*intervals)/(m-1) + 1.0
        for u in range(n):
            tu = numba.float64(u*intervals)/(n-1) + 1.0
            transformation_x[v,u], transformation_x_dx[v,u], transformation_x_dy[v,u] = Interpolate_derivs(tu, tv, x_data)
            transformation_y[v,u], transformation_y_dx[v,u], transformation_y_dy[v,u] = Interpolate_derivs(tu, tv, y_data)
    return transformation_x, transformation_y, transformation_x_dx, transformation_y_dx, transformation_x_dy, transformation_y_dy 



@numba.njit
def interpolate_image_grid(im, x_data, y_data, out_shape=None):
    """
    Interpolate an image using the transformation x_data, y_data

    returns:
    out - interpolated image 
    valid - region in interpolated image mapping to point in im
    transformation_y, transformation_x - weights from grid transform (not sure where these are later used)
    """

    if out_shape is None:
        out_shape = im.shape
    m, n = out_shape

    transformation_x, transformation_y, p_xIndex, p_xWeights, p_yIndex, p_yWeights = get_transform_grid_weights(x_data, y_data, out_shape) 

    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))

    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im.shape[0]-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im.shape[1]-1))).reshape(out_shape)
    
    
    if len(im.shape)==2:
        out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=3).reshape(out_shape)
    else:
        out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])
    return out, valid, transformation_y, transformation_x


@numba.njit
def interpolate_image_grad(im, im_spl, x_data, y_data, out_shape=None):
    """
    Interpolate an image using the transformation x_data, y_data
        im - image to interpolate 
        im_spl - cardinal b-spline representation of image
        x_data, y_data - transform from new_image coordinates to points in im

    returns:
    out - interpolated image
    out_dx, out_dy - x- and y-derivatives of the interpolated image (at the target points)
    valid - region for which points in the interpolated image map into points in im
    p_xIndex, p_xWeights, p_yIndex, p_yWeights - various interpolation weights (?)

    """

    if out_shape is None:
        out_shape = im.shape
    m, n = out_shape
    transformation_x, transformation_y, p_xIndex, p_xWeights, p_yIndex, p_yWeights = get_transform_grid_weights(x_data, y_data, out_shape) 

    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))
    
    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)
    
    out = np.zeros(out_shape, dtype=numba.float64)
    out_dx = np.zeros(out_shape, dtype=numba.float64)
    out_dy = np.zeros(out_shape, dtype=numba.float64)
    for v in numba.prange(out_shape[0]):
        for u in range(out_shape[1]):
            out[v,u], out_dx[v,u], out_dy[v,u] = Interpolate_derivs(transformation_x[v,u]+1.0, transformation_y[v,u]+1.0, im_spl)
    return out, out_dx, out_dy, valid, p_xIndex, p_xWeights, p_yIndex, p_yWeights



@numba.njit
def interpolate_image_grad_subsample(im, im_spl, x_data, y_data, ss_x, ss_y, out_shape=None):
    if out_shape is None:
        out_shape = im.shape

    m_ss = (out_shape[0]+(ss_y-1))//ss_y
    n_ss = (out_shape[1]+(ss_x-1))//ss_x

    
    transformation_x, transformation_y, p_xIndex, p_xWeights, p_yIndex, p_yWeights = get_transform_grid_weights_subsample(x_data, y_data, ss_x, ss_y, out_shape) 

    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))

    out_shape = (m_ss, n_ss)

    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)
    
    out = np.zeros(out_shape, dtype=numba.float64)
    out_dx = np.zeros(out_shape, dtype=numba.float64)
    out_dy = np.zeros(out_shape, dtype=numba.float64)
    for v in numba.prange(out_shape[0]):
        for u in range(out_shape[1]):
            out[v,u], out_dx[v,u], out_dy[v,u] = Interpolate_derivs(transformation_x[v,u]+1.0, transformation_y[v,u]+1.0, im_spl)
    return out, out_dx, out_dy, valid, p_xIndex, p_xWeights, p_yIndex, p_yWeights



#@numba.jit
def interpolate_transformation(im, transformation_x, transformation_y, out_shape=None):
    """
    Interpolate image (using ndimage.map_coordinates for interpolation)

    returns: 
    out, 
    valid, 
    transformation_y, transformation_x
    """

    if out_shape is None:
        out_shape = im.shape
    m, n = out_shape

    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))

    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)

    if len(im.shape)==2:
        out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
    else:
        out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])

    return out, valid




#@numba.jit
def interpolate_image(im, x_data, y_data, out_shape=None, pre_affine=None):
    """
    Interpolate image (using ndimage.map_coordinates for interpolation)

    returns: 
    out, 
    valid, 
    transformation_y, transformation_x
    """

    if out_shape is None:
        out_shape = im.shape
    m, n = out_shape

    transformation_x, transformation_y = get_transform(x_data, y_data, out_shape, pre_affine=pre_affine) 
    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))

    
    
    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)

    if pre_affine is not None:
        valid = valid * get_pre_affine(out_shape, pre_affine)
    
    if len(im.shape)==2:
        out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
    else:
        out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])

    return out, valid, transformation_y, transformation_x


@numba.jit
def interpolate_image_part(im, x_data, y_data, source_range, target_range, out_shape=None):
    """
    Interpolate image (using ndimage.map_coordinates for interpolation)

    im is source_image[source_range]. output is target_image[target_range] provided source for this point is in
    source_range

    returns: 
    out, 
    valid, 
    transformation_y, transformation_x
    """

    if out_shape is None:
        out_shape = im.shape
    m, n = out_shape

    transformation_x, transformation_y = get_transform(x_data, y_data, out_shape) 
    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))

    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)
    
    if len(im.shape)==2:
        out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
    else:
        out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])
    return out, valid, transformation_y, transformation_x




@numba.jit
def interpolate_image_J(im, x_data, y_data, out_shape=None):
    """
    Interpolate image and calculate Jacobian of mapping (using ndimage.map_coordinates for interpolation)
    returns: 
    out, 
    valid, 
    transformation_y, transformation_x
    """
    if out_shape is None:
        out_shape = im.shape

    m, n = out_shape
    
    transformation_x, transformation_y, transformation_x_dx, transformation_y_dx, transformation_x_dy, transformation_y_dy = get_transform_deriv(x_data, y_data, out_shape)
        
    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))


    jacobian = (transformation_x_dx*transformation_y_dy -
               transformation_x_dy*transformation_y_dx)/float(m-1)/float(n-1)*(x_data.shape[0]-3)*(x_data.shape[1]-3)

    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)
    
    if len(im.shape)==2:
        out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
    else:
        out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])
    return out, jacobian, valid, transformation_y, transformation_x


#@numba.jit
def interpolate_image_mat(im, x_data, y_data, out_shape=None):
    """
    Interpolate image and calculate Jacobian of mapping (using ndimage.map_coordinates for interpolation)
    returns: 
    out, 
    valid, 
    transformation_y, transformation_x
    """
    if out_shape is None:
        out_shape = im.shape

    m, n = out_shape
    
    transformation_x, transformation_y, transformation_x_dx, transformation_y_dx, transformation_x_dy, transformation_y_dy = get_transform_deriv(x_data, y_data, out_shape)
        
    coords = np.dstack((transformation_y.flatten(), transformation_x.flatten()))


    jacobian = (transformation_x_dx*transformation_y_dy -
               transformation_x_dy*transformation_y_dx)/float(m-1)/float(n-1)*(x_data.shape[0]-3)*(x_data.shape[1]-3)


    sc_y = (x_data.shape[0]-3)/float(m-1)
    sc_x = (x_data.shape[1]-3)/float(n-1)
    

    transformation_mat = np.stack([np.stack([transformation_x_dx*sc_x, transformation_y_dx*sc_x], axis=-1),
                                   np.stack([transformation_x_dy*sc_y, transformation_y_dy*sc_y], axis=-1)],axis=-1)
                                   
    
    im_m, im_n = im.shape
    valid = ((coords[:,:,0]>=0) & (coords[:,:,0]<=(im_m-1)) &
             (coords[:,:,1]>=0) & (coords[:,:,1]<=(im_n-1))).reshape(out_shape)
    
    
    if len(im.shape)==2:
        out = nd.map_coordinates(im, coords.reshape((-1,2)).T, order=0).reshape(out_shape)
    else:
        out = np.dstack([nd.map_coordinates(im[:,:,i], coords.reshape((-1,2)).T, order=0).reshape(out_shape) for i in range(im.shape[2])])
    return out, jacobian, valid, transformation_y, transformation_x,  transformation_mat



def fit_bsplines2d(data, M, N):
    """
    Fit a 2d cardinal product b-spline to data for later evaluation
    of derivatives of data at interpolated (off-grid) points
    
    returns:
       data - array of interpolated values

    """
    m, n = data.shape

    data_ext = np.zeros((m+2,n+2), dtype=data.dtype)
    data_ext[:m,:n] = data
    
    
    p_xIndex, p_xWeights, p_yIndex, p_yWeights = get_all_weights(M, N, m, n)

    Bx = np.zeros((n+2, N+3))
    for i in range(n):
        for j in range(4):
            idx = p_xIndex[i,j]
            if idx!=-1:
                Bx[i, idx] = p_xWeights[i, j]
    Bx[n,0] = 1
    Bx[n,1] = -1
    Bx[n+1,-2] = -1
    Bx[n+1, -1] = 1
                
    data_ext = la.solve(Bx, data_ext.T).T
    By = np.zeros((m+2, M+3))
    for i in range(m):
        for j in range(4):
            idx = p_yIndex[i,j]
            if idx!=-1:
                By[i, idx] = p_yWeights[i, j]
    By[m,0] = 1
    By[m,1] = -1
    By[m+1,-2] = -1
    By[m+1, -1] = 1

    data = la.solve(By, data_ext)
    return data
    
                
def ev2(m):
    m = np.einsum('ijxl,ijxm->ijlm', m, m)#np.dot(m,m)#0.5*(m + np.transpose(m, [0,1,3,2]))
    T = m[:,:,0,0] + m[:,:,1,1]
    D = m[:,:,0,0]*m[:,:,1,1] - m[:,:,0,1]*m[:,:,1,0]
    roots = np.zeros(m.shape[:3])
    g = np.sqrt(T*T-4*D)
    roots[:,:,0] = (T+g)/2
    roots[:,:,1] = (T-g)/2
    print(m.shape, roots.shape)
    u = m - roots[:,:,1,np.newaxis,np.newaxis]*np.eye(2)[np.newaxis,np.newaxis,:,:]
    # one of these must be nonzero
    s = np.abs(u[:,:,0,0])>np.abs(u[:,:,1,1])
    ev1 = s[:,:,np.newaxis]*u[:,:,:,0]+(~s)[:,:,np.newaxis]*u[:,:,:,1]
    #ev1[:,:,1] *= -1
    #ev1=ev1[:,:,[1,0]]
    return roots, ev1


    


