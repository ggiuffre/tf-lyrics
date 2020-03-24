import os
import sys
import sphinx_material



# Path setup

sys.path.insert(0, os.path.abspath('../..'))



# Project information

project = 'tflyrics'
copyright = '2020, Giorgio Giuffrè'
author = 'Giorgio Giuffrè'
release = '0.1'



# General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_material',
    'm2r'
]

# templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# Options for HTML output

html_theme = 'sphinx_material'
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()

html_theme_options = {
    'color_primary': 'blue',
    'color_accent': 'light-blue',
    'repo_url': 'https://github.com/ggiuffre/tf-lyrics',
    'repo_name': 'tflyrics',
    'logo_icon': 'subject'
}

tml_static_path = ['_static']
