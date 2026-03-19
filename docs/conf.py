
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ACID_code_v2'
copyright = '2025, Benjamin Cadell'
author = 'Benjamin Cadell'
root_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx.ext.doctest",
              "sphinx.ext.viewcode",
              "sphinx.ext.autosummary"
              ]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'undoc-members': True,
}
autoclass_content = "both"
autodoc_member_order = "bysource"

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "FloatLike": "ACID_code_v2.FloatLike",
    "IntLike": "ACID_code_v2.IntLike",
    "Scalar": "ACID_code_v2.Scalar",
    "NumericArray": "ACID_code_v2.NumericArray",
    "Array1D": "ACID_code_v2.Array1D",
    "Array2D": "ACID_code_v2.Array2D",
    "ArrayAnyD": "ACID_code_v2.ArrayAnyD",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
