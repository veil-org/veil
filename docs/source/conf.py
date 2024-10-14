# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
#Location of Sphinx files
sys.path.insert(0, os.path.abspath('../../veil'))


project = 'veil'
copyright = '2024, Angelo Impedovo, Giuseppe Rizzo, Antonio Di Mauro'
author = 'Angelo Impedovo, Giuseppe Rizzo, Antonio Di Mauro'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]
source_suffix = ['.rst', '.md']
templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'
