#65;5802;1c
import svgutils.transform as sg
import sys
from lxml import etree
from subprocess import run

from svgutils.compose import Unit
from pathlib import Path

ovWdth = Unit('174mm')
ovHght = Unit('60mm')

from os.path import getmtime
import time

FONT = 'Arial, Helvetica'

def get_m_time(fn):
    return time.asctime(time.gmtime(getmtime(fn)))


from imageio import imread

def get_image(filename):
    im = imread(filename)
    return sg.ImageElement(open(filename,'rb'), im.shape[1], im.shape[0])[0][0]

def generate_figure():

    all_fig0 = sg.SVGFigure(ovWdth,ovHght)

    all_fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='white')))
    
    MPL_SCALE = 0.15*2
    JL_SCALE = 0.08

    def get_file(fn, scale=MPL_SCALE, pos=None):
        print('get file', fn, get_m_time(fn))
        g = sg.fromfile(fn).getroot()
        g.scale(scale, scale)
        if pos is not None:
            g.moveto(*pos)
        return g


    path = 'output/plot_out/'

    fig0 = []
    
    W = 140
    s = 5
    panels = [ get_file(path+fn, pos=pp, scale=sc) for fn, pp, sc in
               [ ('aggregate-t0-split_angle_0.svg', (23, 0), MPL_SCALE*0.85),
                 ('aggregate-t0-split_angle_1.svg', (23+W, 0), MPL_SCALE*0.85),
                 ('aggregate-t0-split_angle_2.svg', (23+2*W, 0), MPL_SCALE*0.85) ] ]
                                                    
    labels =  [ sg.TextElement(*p, size='10', color='black', font=FONT) for p in
                [(5, 15, 'A'), (5+W, 15, 'B'), (5+2*W, 15, 'C') ] ]
    
    g_all = sg.GroupElement(panels + labels)

    g_all.scale(0.4,0.4)
    g_all.moveto(0, 20)

    all_fig0.append(g_all)

    overlay = sg.fromfile(overlay_path+"ranges_overlay.svg").getroot()
    overlay.scale(0.66, 0.66)
    
    all_fig0.append(overlay)
    
    all_fig0.save(fig_out+"fig_ranges.svg")
    run(['inkscape', '-C', '-d', '300',  fig_out+'fig_ranges.svg', '-o', fig_out+'fig_ranges.png'])

overlay_path = 'overlay/'
fig_out = 'output/figures/'

Path(fig_out).mkdir(exist_ok=True, parents=True)

generate_figure() 
