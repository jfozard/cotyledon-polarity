#65;5802;1c
import svgutils.transform as sg
import sys
from lxml import etree
from subprocess import run

from imageio import imread

from svgutils.compose import Unit

ovWdth = Unit('174mm')
ovHght = Unit('180mm')


def get_image(filename):
    im = imread(filename)
    return sg.ImageElement(open(filename,'rb'), im.shape[1], im.shape[0])

def generate_figure():
    fig0 = sg.SVGFigure(ovWdth,ovHght)

    fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='white')))

    
    MPL_SCALE = 0.15
    JL_SCALE = 0.08

    def get_file(fn, scale=MPL_SCALE, pos=None):
        g = sg.fromfile(fn).getroot()
        g.scale(scale, scale)
        if pos is not None:
            g.moveto(*pos)
        return g

    path = './'

    N = 4
    SX = 75
    SY = 65
    
    panels = []
    for i in range(20):
        fn = f'output/plot_out/aggregate-t0-sample-{i}rose_angle_all.svg'
        panels.append( get_file(path+fn, pos=((i%N)*SX, (i//N)*SY) ) )

    g_all = sg.GroupElement(panels)

    g_all.scale(0.55,0.55)
    g_all.moveto(7,5)
    
    
    fig0.append(g_all)

    fig0.save(fig_out+"samples.svg")
    run(['inkscape', '-C', '-d', '300',  fig_out+'samples.svg', '-o', fig_out+'samples.png'])

fig_out = 'output/figures/'
generate_figure()
