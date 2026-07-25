"""Sphinx configuration for the Quasar documentation set.

Build the docs locally with::

    pip install -r docs/requirements.txt
    sphinx-build -b html docs build/html

The configuration intentionally uses only widely-available extensions
(autosectionlabel, mathjax) so that contributors do not need to install
anything beyond ``sphinx`` and the chosen theme.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------

project   = "Quasar"
author    = "Quasar contributors"
copyright = f"{datetime.now().year}, {author}"

# Keep the literal in lock-step with CMakeLists.txt and
# python/quasar/__init__.py.
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth        = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_title       = "Quasar Documentation"
html_static_path = ["_static"] if os.path.isdir(
    os.path.join(os.path.dirname(__file__), "_static")) else []

# Prefer sphinx_rtd_theme when installed; fall back to the bundled alabaster
# theme so the build still works without extra dependencies.
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"
