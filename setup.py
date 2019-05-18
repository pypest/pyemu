import sys
#from distutils.core import setup
from setuptools import setup

long_description = \
"""pyemu is a set of python modules for interfacing with PEST and PEST++.
""" 

setup(name="pyemu",
      description=long_description,
      long_description=long_description,      
      author="Jeremy White, Mike Fienen",
      author_email='jtwhite1000@gmail.com,mnfienen@usgs.gov',
      url='https://github.com/jtwhite79/pyemu',
      download_url = 'https://github.com/jtwhite79/pyemu/tarball/0.7',
      license='New BSD',
      platforms='Windows, Mac OS-X, linux',
      packages = ["pyemu","pyemu.pst","pyemu.plot","pyemu.utils","pyemu.mat","pyemu.prototypes"],
      version="0.8")
