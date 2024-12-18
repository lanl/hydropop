# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hpdocs"
copyright = "2023, Jon Schwenk"
author = "Jon Schwenk"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
import hydropop
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary"]

autosummary_generate = True
autodoc_member_order = "bysource"
# Remove parentheses from functions cross-referenced
add_function_parentheses = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# Custom css (used for disabling table scrollbar)
def setup(app):
    app.add_css_file("custom.css")
    app.add_css_file("s4defs-roles.css")


# For colored text
rst_prolog = """
.. include:: <s5defs.txt>

"""
