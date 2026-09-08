# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys
import datetime
import sphinx_rtd_theme

# The repo root, so that ``import SeisCL`` finds the *package* (SeisCL/) rather
# than SeisCL/SeisCL.py shadowing it as a top-level module.
sys.path.insert(0, os.path.abspath('..'))

# nbsphinx shells out to the `pandoc` *binary*, which is not a Python package.
# If it isn't already on PATH, fall back to the copy bundled by the
# `pypandoc-binary` wheel (see requirements.txt) so that `pip install -r
# requirements.txt` is enough to build the docs on a bare machine.
if shutil.which('pandoc') is None:
    try:
        import pypandoc
        os.environ['PATH'] = (os.path.dirname(pypandoc.get_pandoc_path())
                              + os.pathsep + os.environ['PATH'])
    except (ImportError, OSError):
        pass

# -- Project information -----------------------------------------------------

project = 'SeisCL'
copyright = '2021, Gabriel Fabien-Ouellet'
author = 'Gabriel Fabien-Ouellet'

# The full version, including alpha/beta/rc tags
release = '1.0.0'



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.doctest',
              'sphinx.ext.viewcode',
              'sphinx.ext.extlinks',
              "sphinx.ext.intersphinx",
              'matplotlib.sphinxext.plot_directive',
              'm2r2',
              'numpydoc',
              'nbsphinx',
]

autoapi_dirs = ['../SeisCL']
autoclass_content = 'both'

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

## Generate autodoc stubs with summaries from code
autosummary_generate = True

## Include Python objects as they appear in source files
autodoc_member_order = 'bysource'

## Default options used by autodoc directives
## (autodoc_default_flags was removed in Sphinx 4.0)
autodoc_default_options = {'members': True}

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False


# nbsphinx_prolog = """
# Download this notebook: {{ env.doc2path(env.docname, base=None) }}
#
# ----
# """

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ['png']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
# The encoding of source files.
source_encoding = 'utf-8-sig'
master_doc = 'index'

# General information about the project
year = datetime.date.today().year
project = 'SeisCL'
copyright = '{}, Gabriel Fabien-Ouellet'.format(year)

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(year=year)


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',  '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

html_last_updated_fmt = '%b %d, %Y'
html_title = 'SeisCL'
html_short_title = 'SeisCL'
html_logo = '_static/seiscl_logo_small.png'
html_favicon = '_static/seiscl_logo.png'
html_static_path = ['_static']
html_extra_path = []
pygments_style = 'default'
add_function_parentheses = False

html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': True,
}

html_context = {
    'menu_links_name': 'Repository',
    'menu_links': [
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/gfabieno/SeisCL.git'),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    'doc_path': 'docs',
    'github_project': 'gfabieno',
    'github_repo': 'SeisCL',
    'github_version': 'master',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_execute = 'never'

autoapi_python_class_content = 'both'

# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_css_file("style.css")