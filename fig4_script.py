#65;5802;1c
import svgutils.transform as sg
import sys
from lxml import etree
from subprocess import run

from imageio import imread

from svgutils.compose import Unit
from os.path import getmtime
import time

def get_m_time(fn):
    return time.asctime(time.gmtime(getmtime(fn)))


ovWdth = Unit('174mm')
ovHght = Unit('244mm')

def get_image(filename):
    im = imread(filename)
    return sg.ImageElement(open(filename,'rb'), im.shape[1], im.shape[0])

def generate_figure():
    
    fig0 = sg.SVGFigure( ovWdth,ovHght )

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

    path = 'output/plot_out/'

    sp_y = 230

    panel_basl = [ get_file(path+'native-basl-rose.svg'),
                   get_file3(path+'native-basl-rose_colbar.svg', pos=(40,195), scale=0.26*1.2),
                   get_file2(overlay+'rose_annotate.svg', pos=(52,215), scale=0.37*1.2),
                   get_png('output/leaf_data/basl_basl_pm_25_sept_21.lif - basl_basl_pm_05-arrows.png', pos=(200,0), scale=0.14),
                   get_file(overlay+'alpha_fig2.svg', pos=(195, 195), scale=2.0),

                   get_file(path+'35S-basl-rose.svg', pos=(0,sp_y)),
                   get_png('output/leaf_data/1-basl-R-arrows.png', pos=(200,0+sp_y), scale=0.14),
                    ]
                
    
                   
    panel_basl = sg.GroupElement(panel_basl)
    panel_basl.scale(0.5, 0.5)


    panel_basl2 = [ get_file('output/plot_out/delta_beta_basl_delta_beta.svg', pos=(75, 225)),
                    ]

    panel_basl2 = sg.GroupElement(panel_basl2)

    labels = [ sg.TextElement(*p, size=6, color='white') for p in  [(2, 10, 'A'), (112, 10, 'B'),
                                                                    (2, 10+sp_y/2, 'C'), (112, 10+sp_y/2, 'D'),
                                                                    (62, 235, 'E') ] ]

    labels = sg.GroupElement(labels)

    g_all = sg.GroupElement([panel_basl, panel_basl2, labels])

    g_all.scale(0.87, 0.87)
    fig0.append(g_all)

    fig0.save(fig_out+"fig4.svg")
    run(['inkscape', '-C', '-d', '300',  fig_out+'fig4.svg', '-o', fig_out+'fig4.png'])

overlay = 'overlay/'
fig_out = 'output/figures/'

#Path(fig_out).mkdir(exist_ok=True, parents=True)

generate_figure()
