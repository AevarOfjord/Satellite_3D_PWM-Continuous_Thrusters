# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Satellite Thruster Control"
copyright = "2024, Aevar Oefjoerd"
author = "Aevar Oefjoerd"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",  # For markdown support
]

# MyST (Markdown) settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,  # Don't show undocumented members by default
    "exclude-members": "__weakref__,__dict__,__module__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_mock_imports = ["mujoco", "osqp", "gurobipy"]  # Mock external dependencies

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Satellite Thruster Control System"

# Create _static if it doesn't exist
os.makedirs("_static", exist_ok=True)

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
