
"""
   This class decodes an ImageJ .roi file. 
	<p>
	Format of the original 64 byte ImageJ/NIH Image
	.roi file header. Two byte numbers are big-endian
	signed shorts. The JavaScript example at
	http://wsr.imagej.net/macros/js/DecodeRoiFile.js
	demonstrates how to use this information to 
	decode a .roi file.
	<pre>
	0-3		"Iout"
	4-5		version (>=217)
	6-7		roi type (encoded as one byte)
	8-9		top
	10-11	left
	12-13	bottom
	14-15	right
	16-17	NCoordinates
	18-33	x1,y1,x2,y2 (straight line) | x,y,width,height (double rect) | size (npoints)
	34-35	stroke width (v1.43i or later)
	36-39   ShapeRoi size (type must be 1 if this value>0)
	40-43   stroke color (v1.43i or later)
	44-47   fill color (v1.43i or later)
	48-49   subtype (v1.43k or later)
	50-51   options (v1.43k or later)
	52-52   arrow style or aspect ratio (v1.43p or later)
	53-53   arrow head size (v1.43p or later)
	54-55   rounded rect arc size (v1.43p or later)
	56-59   position
	60-63   header2 offset
	64-       x-coordinates (short), followed by y-coordinates
	<pre>
	@see <a href="http://wsr.imagej.net/macros/js/DecodeRoiFile.js">DecodeRoiFile.js</a>
"""

import struct
import zipfile

class ROI(object):
    def __init__(self, f):
            head = f.read(64)
            self.roi_type = struct.unpack('>h', head[6:8])[0]/256
            self.top = struct.unpack('>h', head[8:10])[0]
            self.left = struct.unpack('>h', head[10:12])[0]
            self.bottom = struct.unpack('>h', head[12:14])[0]
            self.right = struct.unpack('>h', head[14:16])[0]
            N = struct.unpack('>h', head[16:18])[0]
            self.X1 = struct.unpack('>f', head[18:22])[0]
            self.Y1 = struct.unpack('>f', head[22:26])[0]
            self.X2 = struct.unpack('>f', head[26:30])[0]
            self.Y2 = struct.unpack('>f', head[30:34])[0]

            poly = []
            if N:
                data = f.read(8*N)
                for i in range(N):
                    xp = self.left + struct.unpack('>h',  data[2*i:2*i+2])[0]
                    yp = self.top + struct.unpack('>h',  data[2*N+2*i:2*N+2*i+2])[0]
                    poly.append((xp, yp))
                self.poly = poly

def roi_zip(fn):
    data = {}
    with open(fn, 'rb') as f:
        z = zipfile.ZipFile(f)
        for n in sorted(z.namelist()):
            if n[-4:]=='.roi':
                with z.open(n) as f:
                    r = ROI(f)
                    data[n[:-4]] = r

    return data

def main():
    import sys
    import matplotlib.pyplot as plt
    from tifffile import imread
    import numpy as np

#    im = imread(sys.argv[2])

#    im = np.dstack([im[1,:,:], im[0,:,:], np.zeros_like(im[0,:,:])])

    data = roi_zip(sys.argv[1])

#    print(data)

    for r, v in data.items():
        print(r, '--', v.X1, v.X2, v.Y1, v.Y2, v.roi_type)
    
#    plt.figure()
#    plt.imshow(im)
#    for r in data.values():
#        plt.plot([r.X1, r.X2], [r.Y1, r.Y2], 'w-')
#    plt.show()


if __name__=='__main__':
    main()
