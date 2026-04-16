
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ACID_code'
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
    "FloatLike": "ACID_code.utils.FloatLike",
    "IntLike": "ACID_code.utils.IntLike",
    "Scalar": "ACID_code.utils.Scalar",
    "NumericArray": "ACID_code.utils.NumericArray",
    "Array1D": "ACID_code.utils.Array1D",
    "Array2D": "ACID_code.utils.Array2D",
    "ArrayAnyD": "ACID_code.utils.ArrayAnyD",
}

napoleon_preprocess_types = True
napoleon_type_aliases = {
    "FloatLike": "ACID_code.FloatLike",
    "IntLike": "ACID_code.IntLike",
    "Scalar": "ACID_code.Scalar",
    "NumericArray": "ACID_code.NumericArray",
    "Array1D": "ACID_code.Array1D",
    "Array2D": "ACID_code.Array2D",
    "ArrayAnyD": "ACID_code.ArrayAnyD",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'


_ALIAS_REPLACEMENTS = {
    "ACID_code.FloatLike": "FloatLike",
    "ACID_code.IntLike": "IntLike",
    "ACID_code.Scalar": "Scalar",
    "ACID_code.NumericArray": "NumericArray",
    "ACID_code.Array1D": "Array1D",
    "ACID_code.Array2D": "Array2D",
    "ACID_code.ArrayAnyD": "ArrayAnyD",

    # Old references
    "ACID_code.utils.FloatLike": "FloatLike",
    "ACID_code.utils.IntLike": "IntLike",
    "ACID_code.utils.Scalar": "Scalar",
    "ACID_code.utils.NumericArray": "NumericArray",
    "ACID_code.utils.Array1D": "Array1D",
    "ACID_code.utils.Array2D": "Array2D",
    "ACID_code.utils.ArrayAnyD": "ArrayAnyD",
}


def _clean_signature_text(text: str | None) -> str | None:
    if text is None:
        return None

    for full_name, short_name in _ALIAS_REPLACEMENTS.items():
        text = text.replace(f"TypeAliasForwardRef('{full_name}')", short_name)
        text = text.replace(f'TypeAliasForwardRef("{full_name}")', short_name)

    return text

def process_signature(app, what, name, obj, options, signature, return_annotation):
    signature = _clean_signature_text(signature)
    return_annotation = _clean_signature_text(return_annotation)
    return signature, return_annotation

def setup(app):
    app.connect("autodoc-process-signature", process_signature)