# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BayesMBAR"
copyright = "2024, Xinqiang Ding"
author = "Xinqiang Ding"
# release = '0.0.1'

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../../"))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
#    "autoapi.extension",
]


templates_path = ["_templates"]
exclude_patterns = []

# autodoc_default_options = {
#     'member-order': 'bysource',
# }
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/DingGroup/BayesMBAR',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"