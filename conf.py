import os
import sys
sys.path.insert(0, os.path.abspath('./source'))

project = 'Skeytch2Image'
copyright = '2024, Amine EL Hend'
author = 'Amine EL Hend'
release = '6.12.24'
master_doc = 'index'
extensions = []
"""
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]
"""
templates_path = ['_templates']
exclude_patterns = []


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
