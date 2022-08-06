#65;5802;1c
import svgutils.transform as sg
import sys
from lxml import etree

from imageio import imread
from svgutils.compose import Unit
from subprocess import run

from pathlib import Path

ovWdth = Unit('174mm')
ovHght = Unit('200mm')

from os.path import getmtime
import time

FONT = 'Arial, Helvetica'

def get_m_time(fn):
    return time.asctime(time.gmtime(getmtime(fn)))



def get_image(filename):
    im = imread(filename)
    return sg.ImageElement(open(filename,'rb'), im.shape[1], im.shape[0])

input_path='output/'


def generate_figure():
    fig0 = sg.SVGFigure(ovWdth,ovHght)


    fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='black')))

    MPL_SCALE = 0.4
    JL_SCALE = 0.08

    def get_file(fn, scale=MPL_SCALE, pos=None):
        print('get file', fn, get_m_time(fn))
        g = sg.fromfile(fn).getroot()
        g.scale(scale, scale)
        if pos is not None:
            g.moveto(*pos)
        return g

    panel_stretcher = []
    panel_stretcher.append(sg.fromfile('overlay/stretcher.svg').getroot())

    panel_stretcher = sg.GroupElement(panel_stretcher)
    panel_stretcher.moveto(20, -20)
    panel_stretcher.scale(3.3, 3.3)


    panel_stretch = []

    stretch = get_file(input_path+'stretching_analysis/5-brx-tensor.svg', scale=1.0)
    stretch_sb = get_file(input_path+'stretching_analysis/5-brx-cbar.svg', scale=1.0)
    
    stretch.scale(0.208, 0.208)
    panel_stretch.append(stretch)
    stretch_sb.scale(0.25,  0.25)
    stretch_sb.moveto(300, 0)
    panel_stretch.append(stretch_sb)

    panel_stretch = sg.GroupElement(panel_stretch)
    panel_stretch.moveto(782, 34)
    
    panel_T0 = []
    T0 = get_image(input_path+'im_t0.png')
    T0.scale(0.15, 0.15)

    panel_T0.append(T0)

    T0_inset = get_image(input_path+'im_t0_inset.png')
    T0_inset.scale(0.6, 0.6)
    T0_inset.moveto(-30, -40)

    panel_T0.append(T0_inset)

    panel_T0 += [sg.TextElement(*p, size=26, color='white', font=FONT) for p in [(81, 105, '(i)'), (160, 115, '(ii)'),
                                                                      (95, 208, '(iii)') ]]


    panel_T0 = sg.GroupElement(panel_T0)
    panel_T0.moveto(90, 400)
    
    fig0.append([panel_T0])

    panel_T1 = []
    
    T1 = get_image(input_path+'im_t1.png')
    T1.scale(0.15, 0.15)

    panel_T1.append(T1)

    T1_inset = get_image(input_path+'im_t1_inset.png')
    T1_inset.scale(0.6, 0.6)
    T1_inset.moveto(-20, -40)

    panel_T1.append(T1_inset)

    panel_T1 = sg.GroupElement(panel_T1)
    panel_T1.moveto(440, 400)


    panel_T2= []
    T2 = get_image(input_path+'im_t2.png')
    T2.scale(0.15, 0.15)
    panel_T2.append(T2)

    T2_inset = get_image(input_path+'im_t2_inset.png')
    T2_inset.scale(0.6, 0.6)
    T2_inset.moveto(-20, -50)

    panel_T2.append(T2_inset)

    panel_T2 += [sg.TextElement(*p, size=26, color='white', font=FONT) for p in [(68, 115, '(i)'), (158, 135, '(ii)'),
                                                                      (88, 255, '(iii)') ]]

    
    panel_T2 = sg.GroupElement(panel_T2)
    panel_T2.moveto(792, 400)

    labels = [ sg.TextElement(*p, size=32, color='white', font=FONT) for p in  [(50, 40, 'A'), (738, 38, 'B'), (25, 390, 'C'), (388, 390, 'D'), (738, 390, 'E'), (70, 750, 'F'),
                                                                     (130, 970, 'G'),  (600, 970, 'H')] ]


    g_labels = sg.GroupElement(labels)



    panel_base = 'output/delta_beta_analysis/5-brx/components_svg/'


    
    def make_panel(p, cells_t0_fn, cells_t0_p, cells_t1_fn, cells_t1_p, db_fn, db_p, beta_pos):
        b = get_file('overlay/white_beta.svg')[-1][0]
        b.scale(4,4)
        b.moveto(*beta_pos)
        g = [ sg.TextElement(*p, size=32, color='white', font=FONT),
              get_file(panel_base+cells_t0_fn, scale=1.0, pos=cells_t0_p), 
              get_file(panel_base+cells_t1_fn, scale=1.0, pos=cells_t1_p), 
              get_file(panel_base+db_fn, scale=3.0, pos=db_p),
              b ]
        return sg.GroupElement(g)
    
    panel_i =  make_panel( (0, 0, '(i)') , 
                            '0_arrows_t0.svg', (105-58,769-20-780),
                            '0_arrows_t1.svg', (216-58,766-20-780),
                            '0_delta_beta_text.svg', (105-38,869-750), (105-58, 808-0-780) )
    panel_i.moveto(130, 760)

    panel_ii = make_panel( (0, 0, '(ii)'),
                             '1_arrows_t0.svg', (100-42, 937+100-1060),
                             '1_arrows_t1.svg', (218-42, 937+100-1060),
                             '1_delta_beta_text.svg', (125-42, 140), (112+32-42, 806+180+100-1060))
    panel_ii.moveto(470, 760)

    panel_iii =  make_panel( (0, 0, '(iii)'),
                             '2_arrows_t0.svg', (100-43, 885-900),
                             '2_arrows_t1.svg', (213-43, 872-900),
                             '2_delta_beta_text.svg', (100-30, 140), (107+0-43, 806+95+30-900)) 
    panel_iii.moveto(800, 760)


    g_cells = sg.GroupElement([panel_i, panel_ii, panel_iii])




    delta_beta = get_file(input_path+'plot_out/delta_beta_delta_beta.svg', scale=1.0)
    delta_beta.moveto(130,940)

    delta_beta_control = get_file(input_path+'plot_out/delta_beta_control_delta_beta.svg', scale=1.0)
    delta_beta_control.moveto(620,940)


    g = sg.GroupElement([panel_stretcher, panel_stretch, panel_T0, panel_T1, panel_T2, g_labels, g_cells, delta_beta, delta_beta_control])
    g.scale(0.15,0.15)
    
    fig0.append(g)

    fig0.root.set('style', 'background-color: black;')
    
    fig0.save(fig_out+"fig1.svg")
    run(['inkscape', '-C', '-d', '300',  fig_out+'fig1.svg', '-o', fig_out+'fig1.png'])

fig_out = 'output/figures/'

Path(fig_out).mkdir(exist_ok=True, parents=True)
    
generate_figure() 
