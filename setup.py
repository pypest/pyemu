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
      author_email='jwhite@usgs.gov',
      url='https://github.com/jtwhite79/pyemu',
      download_url = 'https://github.com/jtwhite79/pyemu/tarball/0.4',
      license='New BSD',
      platforms='Windows, Mac OS-X',
<<<<<<< HEAD
      packages = ["pyemu","pyemu.pst","pyemu.plot","pyemu.utils","pyemu.mat"],
=======
      packages = ["pyemu","pyemu.pst","pyemu.utils","pyemu.mat"],
>>>>>>> abcff190f517ac068298e5bbefea7026046d4830
      version="0.4")
