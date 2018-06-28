.. pyemu documentation master file, created by
   sphinx-quickstart on Thu Sep 14 11:12:03 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#################################
Welcome to pyEMU's documentation!
#################################

pyEMU [WFD16]_ is a set of python modules for performing linear and non-linear
uncertainty analysis including parameter and forecast analyses, data-worth
analysis, and error-variance analysis. These python modules can interact with
the PEST [DOH15]_ and PEST++ [WWHD15]_  suites and use terminology consistent
with them.   pyEMU is written in an object-oriented programming style, and
thus expects that users will write, or adapt, client code in python to
implement desired analysis.  :doc:`source/oop` are provided in this documentation.

pyEMU is available via github_ .

.. _github: https://github.com/jtwhite79/pyemu
.. _Notes: 

********
Contents
********

.. toctree::
   :maxdepth: 2

   source/oop
   source/glossary

***********************
Technical Documentation
***********************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

********** 
References 
********** 

.. [DOH15] Doherty, J., 2015. Calibration and
   Uncertainty Analysis for Complex Environmental Models:  Brisbane, Australia, Watermark Numerical
   Computing, http://www.pesthomepage.org/Home.php .

.. [WFD16] White, J.T., Fienen, M.N., and Doherty, J.E., 2016, A python framework
   for environmental model uncertainty analysis:  Environmental Modeling &
   Software, v. 85, pg. 217-228, https://doi.org/10.1016/j.envsoft.2016.08.017 .
   
.. [WWHD15] Welter, D.E., White, J.T., Hunt, R.J., and Doherty, J.E., 2015,
   Approaches in highly parameterized inversion: PEST++ Version 3, a Parameter
   ESTimation and uncertainty analysis software suite optimized for large
   environmental models: U.S. Geological Survey Techniques and Methods, book 7,
   section C12, 54 p., https://doi.org/10.3133/tm7C12 .