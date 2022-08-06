#65;5802;1c
import svgutils.transform as sg
import sys
from lxml import etree
from subprocess import run
from imageio import imread

from svgutils.compose import Unit

ovWdth = Unit('174mm')
ovHght = Unit('98mm')

from os.path import getmtime
import time

def get_m_time(fn):
    return time.asctime(time.gmtime(getmtime(fn)))


def get_image(filename):
    im = imread(filename)
    return sg.ImageElement(open(filename,'rb'), im.shape[1], im.shape[0])


def generate_figure():
    fig0 = sg.SVGFigure( ovWdth, ovHght)

    fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='black')))

    MPL_SCALE = 0.15
    JL_SCALE = 0.08

    def get_file(fn, scale=MPL_SCALE, pos=None):
        print('get file', fn, get_m_time(fn))
        g = sg.fromfile(fn).getroot()
        g.scale(scale, scale)
        if pos is not None:
            g.moveto(*pos)
        return g


    def get_png(fn, scale=1.0, pos=None):
        print('get file', fn, get_m_time(fn))
        image = imread(fn)
        image_panel = sg.ImageElement(open(fn, 'rb'), image.shape[1], image.shape[0])
        image_panel.scale(scale, scale)
        if pos is not None:
            image_panel.moveto(*pos)
        return image_panel

    def get_file2(fn, scale=MPL_SCALE, pos=None):
        print('get file', fn, get_m_time(fn))
        g = sg.fromfile(fn).getroot()[-1][0]
        g.scale(scale, scale)
        if pos is not None:
            g.moveto(*pos)
        return g

    def get_file3(fn, scale=MPL_SCALE, pos=None):
        print('get file', fn, get_m_time(fn))
        g = sg.fromfile(fn).getroot()[-2]
        g.scale(scale, scale)
        if pos is not None:
            g.moveto(*pos)
        return g


    output = 'output/plot_out/'
    output_leaf = 'output/leaf_data/'

    panel_brxl = [ get_file(output+'aggregate-t0-rose.svg'),
                   get_file3(output+'aggregate-t0-rose_colbar.svg', pos=(40,195), scale=0.26*1.2),
                   get_file2(overlay+'rose_annotate.svg', pos=(52,215), scale=0.37*1.2),
                   get_png(output_leaf+'brx_24Aug_seed2cot2_t0-arrows.png', pos=(200,0), scale=0.16),
                   get_file(overlay+'alpha_fig2.svg', pos=(195, 195), scale=2.0),

                    ]
                
                   
    panel_brxl = sg.GroupElement(panel_brxl)

    panel_brxl.scale(0.5, 0.5)

    labels = [ sg.TextElement(*p, size=6, color='white') for p in  [(2, 10, 'A'), (112, 10, 'B') ] ]

    labels = sg.GroupElement(labels)

    g_all = sg.GroupElement([panel_brxl, labels])
    g_all.scale(0.8, 0.8)
    
    fig0.append(g_all)

    fig0.save(fig_out+"fig3.svg")
    run(['inkscape', '-C', '-d', '300',  fig_out+'fig3.svg', '-o', fig_out+'fig3.png'])

overlay = 'overlay/'
fig_out = 'output/figures/'

#Path(fig_out).mkdir(exist_ok=True, parents=True)

generate_figure()
