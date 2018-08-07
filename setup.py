import sys
#from distutils.core import setup
from setuptools import setup

long_description = \
"""pyemu is a set of python modules for linear-based model-independent uncertainty analyses.
""" 

setup(name="pyemu",
      description=long_description,
      long_description=long_description,      
      author="Jeremy White",
      author_email='jtwhite1000@gmail.com',
      url='https://github.com/jtwhite79/pyemu',
      download_url = 'https://github.com/jtwhite79/pyemu/tarball/0.6',
      license='New BSD',
      platforms='Windows, Mac OS-X',
      packages = ["pyemu","pyemu.pst","pyemu.plot","pyemu.utils","pyemu.mat"],
      version="0.6")
