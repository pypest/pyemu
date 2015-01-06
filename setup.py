import sys
from distutils.core import setup


long_description = \
"""pyemu is a set of python modules for linear-based model-independent uncertainty analyses.
""" 

setup(name="pyemu",
      description=long_description,
      long_description=long_description,      
      author="Jeremy White",
      author_email='jwhite@usgs.gov',
      url='https://code.google.com/p/flopy/',
      license='GNU GPL',
      platforms='Windows, Mac OS-X',
      py_modules = ["pyemu","mat_handler","pst_handler"],
      version="0.1")
