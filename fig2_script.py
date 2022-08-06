#65;5802;1c
import svgutils.transform as sg
import sys
from lxml import etree
from subprocess import run

from svgutils.compose import Unit

ovWdth = Unit('174mm')
ovHght = Unit('210mm')

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
    panels = [ get_file(path+fn, pos=pp, scale=sc) for fn, pp, sc in [ ('aggregate-t0-stacked-2.svg', (23, 0), MPL_SCALE*0.85),
                                                                       ('aggregate-t0-stacked-1.svg', (110, 100+s), MPL_SCALE*0.85), #*0.75),
                                                                       ('aggregate-t0-stacked-0.svg', (23, 105+s), MPL_SCALE*0.85),
                                                                       ('stretch-t5-stacked-2.svg', (W + 23, 0), MPL_SCALE*0.85),
                                                                       ('stretch-t5-stacked-1.svg', (W + 110, 100-5+s), MPL_SCALE*0.85), #*0.75),
                                                                       ('stretch-t5-stacked-0.svg', (W + 23, 105+s), MPL_SCALE*0.85),
                                                                       ('aggregate-t0-sample-14rose_angle_all_nored.svg',  (W + 23, 210+s), MPL_SCALE*0.85),
                                                                       ('aggregate-t0-sample-6rose_angle_all_nored.svg', (W + 23, 300+s), MPL_SCALE*0.85), 
                                                                       ('aggregate-t0-aligned-histogram-lr.svg',  (W + 23, 390+s), MPL_SCALE*0.85),
                                                                       ('native-basl-stacked-1.svg', (2*W + 110, 110-10-5+s), MPL_SCALE*0.85), #*0.75),
                                                                       ('native-basl-stacked-2.svg', (2*W+23,0), MPL_SCALE*0.85),
                                                                       ('native-basl-stacked-0.svg', (2*W+23,105+s), MPL_SCALE*0.85),
                                                                       ('native-basl-aligned-histogram-lr.svg', (2*W+23,300+s), MPL_SCALE*0.85), 
                                                                       ('35S-basl-stacked-2.svg', (2*W+23,210+s), MPL_SCALE*0.85),
                                                                       ('35S-basl-aligned-histogram-lr.svg', (2*W+23,390+s), MPL_SCALE*0.85),  ] ]
                                                    
    labels =  [ sg.TextElement(*p, size=12, color='black', font=FONT) for p in  [(5, 15, 'A'), (5, 105+s, 'B'), 
                (W+10, 15, 'C'), (W+10, 105+s, 'D'), (W+10, 215+s, 'E'), (W+10, 305+s, 'F'), (W+10, 395+s, 'G'),
                (2*W+10, 15, 'H'), (2*W+10, 105+s, 'I'), (2*W+10, 215+s, 'J'), (2*W+10, 305+s, 'K'), (2*W+10, 395+s, 'L')

                ] ]
    
    g_all = sg.GroupElement(panels + labels)

    g_all.scale(0.6,0.6)

    fig0.append(g_all)

    overlay_a = sg.fromfile(overlay+"fig2_overlay_a2.svg").getroot()
    overlay_b = sg.fromfile(overlay+"fig2_overlay_b.svg").getroot()
    overlay_b.moveto(0.6*(W + 23), 0.6*390+s)

    fig0.append(overlay_a)
    fig0.append(overlay_b)

    fig0 = sg.GroupElement(fig0)
    fig0.scale(0.65, 0.65)
    fig0.moveto(0, 15)
    
    all_fig0.append(fig0)

    all_fig0.save(fig_out+"fig2.svg")
    run(['inkscape', '-C', '-d', '300',  fig_out+'fig2.svg', '-o', fig_out+'fig2.png'])

overlay = 'overlay/'
fig_out = 'output/figures/'

#Path(fig_out).mkdir(exist_ok=True, parents=True)

generate_figure() 
