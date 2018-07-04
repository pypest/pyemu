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


Contents
********

.. toctree::
   :maxdepth: 1

   source/Monte_carlo_page
   source/oop
   source/glossary
   
.. _Technical:

Technical Documentation
***********************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References 
********** 

.. [BPL+16] Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T.,
   Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW model development using
   Python and FloPy: Groundwater, v. 54, p. 733-739, https://doi.org/10.1111/gwat.12413.

.. [DOH15] Doherty, John, 2015. Calibration and
   Uncertainty Analysis for Complex Environmental Models:  Brisbane, Australia, Watermark Numerical
   Computing, http://www.pesthomepage.org/Home.php .
      
.. [FB16] Fienen, M.N., and Bakker, Mark, 2016, HESS Opinions: Repeatable research--- what hydrologists can learn 
   from the Duke cancer research scandal: Hydrology and Earth System Sciences, v. 20, no. 9, pg. 3739-3743,
   https://doi.org/10.5194/hess-20-3739-2016 .
   
.. [KB09] Beven, Keith, 2009, Environmental modelling--- an uncertain future?: London, Routledge, 310 p.

.. [KRP+16] Kluyver, Thomas; Ragan-Kelley, Benjamin; Perez, Fernado; Granger, Brian; Bussonnier, Matthias;
   Federic, Jonathan; Kelley, Kyle; Hamrick, Jessica; Grout, Jason; Corlay, Sylvain; Ivanov, Paul; Avila, Damian;
   Aballa, Safia; Willing, Carol; and Jupyter Development Team, 2016, Jupyter Notebooks-- a publishing
   format for reproducible computational workflows:  in Positioning and Power in Academic
   Publishing-- Players, Agents, and Agendas. F. Loizides and B. Schmidt (eds). 
   IOS Press, https://doi.org/10.3233/978-1-61499-649-1-87 and https://jupyter.org .
   
.. [MCK10] McKinney, Wes, 2010, Data structures for statistical computing in python:
   in Proceedings of the 9th Python in Science Conference, Stefan van
   der Walt and Jarrod Millman (eds.), p. 51-56, https://pandas.pydata.org/index.html .

.. [PSF18] The Python Software Foundation, 2018, Documentation, The python tutorial, 
   9. Classes: https://docs.python.org/3.6/tutorial/classes.html .
   
.. [RS16] Reges, Stuart, and Stepp, Marty, 2016, Building Java Programs--- A Back to Basics 
   Approach, Fourth Edition:  Boston, Pearson, 1194 p.

.. [WFD16] White, J.T., Fienen, M.N., and Doherty, J.E., 2016, A python framework
   for environmental model uncertainty analysis:  Environmental Modeling &
   Software, v. 85, pg. 217-228, https://doi.org/10.1016/j.envsoft.2016.08.017 .
   
.. [WWHD15] Welter, D.E., White, J.T., Hunt, R.J., and Doherty, J.E., 2015,
   Approaches in highly parameterized inversion: PEST++ Version 3, a Parameter
   ESTimation and uncertainty analysis software suite optimized for large
   environmental models: U.S. Geological Survey Techniques and Methods, book 7,
   section C12, 54 p., https://doi.org/10.3133/tm7C12 .