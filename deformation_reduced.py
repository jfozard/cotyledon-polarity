
"""
Routines for loading and analysing bUnwarpJ output
"""


from typing import Tuple
import numpy as np
import sys
import scipy.ndimage as nd
import numba
import time
import numpy.linalg as la
import functools


@numba.njit
def Interpolate(x:float, y:float, data:np.ndarray) -> float: 
    """

    Interpolate data using cardinal (cubic) b-splines (spacing 1).
    Splines of support [-2, 2] 
                ----
              /      \
             /        \
         ---/          \---
    ----/                  \----
    -2     -1     0     1      2

    Calculates values at a single x,y point.

    Here y=i=row of data matrix, x=j=column of data matrix.

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
# [ conditional because int rounds down for positive numbers and up for negative numbers ]

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
# [ again conditional because int rounds down for positive numbers and up for negative numbers ]

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

# Loop over the subset of the contributing splines, and sum their values
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


"""
bUnwarpJ output is of the form

Intervals=4

X Coeffs -----------------------------------
  -24.520870413411462    -25.70161426483417    -27.42054034732776   -28.947576550002317    -27.39762496358135   -25.276173080435665   -24.353947474517792 
 -0.08333098317685227  -0.42262344668961305    0.4839875446991823    0.9160749716118679   0.16254967992678193   0.16611757998619547   -0.8581201416735658 
   23.803534707504813    24.861216096220264    24.638623404898087    24.483787882854855    24.591908839993504    24.972623757273542    22.904230826710254 
    47.85831403760983     50.07803420635153    49.397029331719395    49.776923289658825    49.410525560992056     49.83360162103759    47.325516183100184 
    72.79959438611448     73.60539657455125      74.2536524009545     74.07219761694299     74.14449431593447     74.45480607237675     73.52891519821335 
    97.93697903029248    100.61058102846668     99.21699570067219     99.30066537474308     99.15862225173764     98.68970956848935     99.92098449974823 
   122.58007124741764    123.98142316195154    122.90968686024169    121.89856612181391    122.94948462837287     124.5608002113927     124.5937511528738 

Y Coeffs -----------------------------------
     267.039512726613    208.02615431992314      151.449042704305    101.49603372756202    50.920397192853926     2.655288829625424    -46.75667657861386 
   261.82592008039995     196.2567178685406     149.1142450575289     98.96798002811536     49.65062102977404   -0.5064521347494045    -50.06765415531324 
   256.10958987787046    198.82077817566795    149.80422691702768     99.43268177001431      49.7990991899858    0.4279294780142648    -52.42857253445828 
    253.5889002792627    197.86569696361192    149.18567616707296     99.50464770737915     49.58757982661836    0.2545240458372513   -52.724546617478836 
    256.3343053744982    199.04423050620588    149.53226955353918     99.51556507611417     49.73362224835832    0.2737850771507432    -49.90540984229237 
   259.31870111553104    195.04768933546546    150.18670071106226     98.78439571472764     50.34505539939112   -1.1094048354237471     -47.0363596353464 
   267.61866490994635    206.16138498170787    151.54158646057297     99.56987801006443     49.79379118863506    1.3606560878652747    -46.47961186801768 

This is the inverse transform (reg1_inverse.txt) between reg2 (source, 200x100) and reg1 (target, 100x200), where reg2 is a 90 degrees clockwise rotation of reg1
(I assume that) this maps pixel coordinates from reg2 to reg1. The first and last rows and columns are pading for the transformation grid.
Everything on the top/ bottom row of reg2 maps to the left/right size of reg1.
The left hand size of reg2 maps to the bottom of reg1.

"""

def load_elastic_transform(filename:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load elastic transform saved by bUnwarpJ

    input: filename

    returns: x_data[N,N], y_data[N,N] where N is the size of the transform
    """
    with open(filename,'r') as f:

        u = f.read()
        u = u.split('\n')
        intervals = int(u[0].split('=')[1])
        N = intervals + 3

        x_data = []
        for i in range(N):
            s = u[i+3].split(' ')
            x_data.append([float(v) for v in s if v!=''])
        
        y_data = []
        for i in range(N):
            s = u[N+i+5].split(' ')
            y_data.append([float(v) for v in s if v!=''])

        return np.array(x_data), np.array(y_data)

    
@numba.njit(parallel=True)
def get_transform_points(x_data:np.ndarray, y_data:np.ndarray, out_shape: tuple, pts:np.ndarray) -> np.ndarray:
    """
    Find source point for each point in points

    arguments:
        x_data, y_data : The mapping (of dimension m rows by n columns) 
        pts: input point array of dimension [:, 2]
        These are in x-y format, scaled such that the domain of the mapping is the unit square
        i.e. 0<=x<=1, 0<=y<=1. The interpolation will still work outside that range, but
        no extrapolation of the spline is performed.

    returns:
        array [:, 2] of mapped x- and y- coordinates for each point in pts

    """

    N = pts.shape[0]
    m, n = out_shape
    output = np.zeros((N,2), dtype=numba.float64)
    intervals = x_data.shape[0]-3
    for v in numba.prange(N):
        tu = numba.float64(pts[v,0]*intervals)/(n-1) + 1.0 # Map x and y 
        tv = numba.float64(pts[v,1]*intervals)/(m-1) + 1.0
        output[v,0] = Interpolate(tu, tv, x_data)
        output[v,1] = Interpolate(tu, tv, y_data)

    return output





    


