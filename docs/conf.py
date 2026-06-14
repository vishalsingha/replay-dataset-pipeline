"""Sphinx configuration for Replay Dataset Pipeline documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Replay Dataset Pipeline"
copyright = "2026, Vishal Singh(vishalsingha)"
author = "Vishal Singh(vishalsingha)"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstrings = True
napoleon_numpy_docstrings = False
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_baseurl = "https://vishalsingha.github.io/replay-dataset-pipeline/"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

