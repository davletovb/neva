"""Sphinx configuration for the Neva documentation."""

from __future__ import annotations

import datetime
import os
import sys
from importlib import metadata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

project = "Neva"
author = "Neva Contributors"
current_year = datetime.datetime.now().year
copyright = f"{current_year}, {author}"

try:
    release = metadata.version("neva")
except metadata.PackageNotFoundError:  # pragma: no cover - metadata unavailable when building locally
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.ipynb"]

html_theme = "alabaster"
html_static_path: list[str] = ["_static"]
