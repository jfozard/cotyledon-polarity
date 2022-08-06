"""
=======
textext
=======

:Author: Pauli Virtanen <pav@iki.fi>
:Date: 2008-04-26
:Author: Pit Garbe <piiit@gmx.de>
:Date: 2014-02-03
:Author: TexText developers
:Date: 2019-04-05
:License: BSD

Textext is an extension for Inkscape_ that allows adding
LaTeX-generated text objects to your SVG drawing. What's more, you can
also *edit* these text objects after creating them.

This brings some of the power of TeX typesetting to Inkscape.

Textext was initially based on InkLaTeX_ written by Toru Araki,
but is now rewritten.

Thanks to Sergei Izmailov, Robert Szalai, Rafal Kolanski, Brian Clarke,
Florent Becker and Vladislav Gavryusev for contributions.

.. note::
   Unfortunately, the TeX input dialog is modal. That is, you cannot
   do anything else with Inkscape while you are composing the LaTeX
   text snippet.

   This is because I have not yet worked out whether it is possible to
   write asynchronous extensions for Inkscape.

.. note::
   Textext requires Pdflatex and Pstoedit_ compiled with the ``plot-svg`` back-end

.. _Pstoedit: http://www.pstoedit.net/pstoedit
.. _Inkscape: http://www.inkscape.org/
.. _InkLaTeX: http://www.kono.cis.iwate-u.ac.jp/~arakit/inkscape/inklatex.html
"""

from __future__ import print_function
import hashlib
import logging
import logging.handlers
import math
import re
import os
import platform
import sys
import uuid
from io import open # ToDo: For open utf8, remove when Python 2 support is skipped
from subprocess import run


from lxml import etree

TEXTEXT_NS = u"http://www.iki.fi/pav/software/textext/"
SVG_NS = u"http://www.w3.org/2000/svg"
XLINK_NS = u"http://www.w3.org/1999/xlink"

ID_PREFIX = "textext-"

NSS = {
    u'textext': TEXTEXT_NS,
    u'svg': SVG_NS,
    u'xlink': XLINK_NS,
}


class TexToPdfConverter(object):
    """
    Base class for Latex -> SVG converters
    """
    DOCUMENT_TEMPLATE = r"""
    \documentclass{article}
    %s
    \pagestyle{empty}
    \begin{document}
    %s
    \end{document}
    """

    LATEX_OPTIONS = ['-interaction=nonstopmode',
                     '-halt-on-error']

    def __init__(self):
        self.tmp_base = 'tmp'

    # --- Internal
    def tmp(self, suffix):
        """
        Return a file name corresponding to given file suffix,
        and residing in the temporary directory.
        """
        return self.tmp_base + '.' + suffix

    def tex_to_svg(self, tex_command, latex_text, preamble_file, fill=None):
        """
        Create a PDF file from latex text
        """

        # Read preamble
        #preamble_file = os.path.abspath(preamble_file)
        preamble = preamble_file
        """
        if os.path.isfile(preamble_file):
            with open(preamble_file, 'r') as f:
                preamble += f.read()
        """
        # Options pass to LaTeX-related commands

        texwrapper = self.DOCUMENT_TEMPLATE % (preamble, latex_text)

        # Convert TeX to PDF

        # Write tex
        with open(self.tmp('tex'), mode='w', encoding='utf-8') as f_tex:
            f_tex.write(texwrapper)

        # Exec tex_command: tex -> pdf
        run([tex_command, self.tmp('tex')] + self.LATEX_OPTIONS)
        """Convert the PDF file to a SVG file"""

        run([
            "inkscape", 
            "--pdf-poppler",
            "--pdf-page=1",
            "--export-type=svg",
            "--export-text-to-path",
            "--export-area-drawing",
            "--export-filename", self.tmp('svg'),
            self.tmp('pdf')
        ]
        )


        svg = etree.parse(self.tmp('svg'))
        
        rename_map = {}

        # replace all ids with unique random uuid

        idx = 0
        for el in svg.iterfind('.//*[@id]'):
            old_id = el.attrib["id"]
            new_id = 'id-' + str(uuid.uuid4())
            idx +=1
            el.attrib["id"] = new_id
            rename_map[old_id] = new_id

            # find usages of old ids and replace them
            def replace_old_id(m):
                old_name = m.group(1)
                try:
                    replacement = rename_map[old_name]
                except KeyError:
                    replacement = old_name
                return "#{}".format(replacement)
            regex = re.compile(r"#([^)(]*)")

            
            for el in svg.iter():
                for name, value in el.items():
                    new_value = regex.sub(replace_old_id, value)
                    el.attrib[name] = new_value

        print('rename', rename_map)


        if fill:
            for el in svg.iterfind(".//*[@style]"):
                s = el.attrib['style']
                if 'fill' in s:
                    s = s.replace("fill:#000000", "fill:"+fill)
                    el.attrib['style']=s
        
        return svg

PRE = r""" \usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc} 
\usepackage{
            lmodern,
            babel,
            uarial,
            textcomp,
            amsmath
           }
\renewcommand{\familydefault}{\sfdefault}
\usepackage[italic]{mathastext}
"""

def textext(text, fill=None):
    tt =  TexToPdfConverter()
    print('textext', text)

    svg = tt.tex_to_svg("pdflatex", text, PRE, fill=fill)
    return svg.getroot()

def textext_svg(text, fill=None):
    tt =  TexToPdfConverter()
    print('textext', text)

    svg = tt.tex_to_svg("pdflatex", text, PRE, fill=fill)
    return svg

