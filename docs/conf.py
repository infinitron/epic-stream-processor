"""Sphinx configuration."""
project = "Hypermodern Python"
author = "Karthik"
copyright = "2022, LoCo Lab"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
autosummary_generate = True
html_theme = "furo"
