pyEMU
=====

python modules for model-independent FOSM (first-order, second-moment) (a.k.a linear-based, a.k.a. Bayes linear) 
uncertainty analyses and data-worth analyses, non-linear uncertainty analyses and interfacing with PEST and PEST++.  
pyEMU also has a pure python (pandas and numpy) implementation of ordinary kriging for geostatistical interpolation and 
support for generating high-dimensional PEST(++) model interfaces, including support for (very) high-dimensional 
ensemble generation and handling   

Main branch:
[![pyemu continuous integration](https://github.com/pypest/pyemu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pypest/pyemu/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/pypest/pyemu/badge.svg?branch=main)](https://coveralls.io/github/pypest/pyemu?branch=main)
[![codecov](https://codecov.io/gh/pypest/pyemu/branch/main/graph/badge.svg?token=bnrI1JKvbk)](https://codecov.io/gh/pypest/pyemu)

Develop branch:
[![pyemu continuous integration](https://github.com/pypest/pyemu/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/pypest/pyemu/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/pypest/pyemu/badge.svg?branch=develop)](https://coveralls.io/github/pypest/pyemu?branch=develop)
[![codecov](https://codecov.io/gh/pypest/pyemu/graph/badge.svg?token=bnrI1JKvbk)](https://codecov.io/gh/pypest/pyemu)

Documentation
=============

Complete user's guide:

[https://pyemu.readthedocs.io/en/latest/](https://pyemu.readthedocs.io/en/latest/)

The pyEMU documentation is being treated as a first-class citizen!  Also see the [example notebooks in the repo](https://github.com/pypest/pyemu/tree/main/examples), these are `.ipynb` files.

What is pyEMU?
================

pyEMU is a set of python modules for model-independent, user-friendly, computer model uncertainty analysis.  pyEMU is tightly coupled to the open-source suite PEST (Doherty 2010a and 2010b, and Doherty and other, 2010) and PEST++ (Welter and others, 2015, Welter and other, 2012), which are tools for model-independent parameter estimation.  However, pyEMU can be used with generic array objects, such as numpy ndarrays.

Several equations are implemented, including Schur's complement for conditional uncertainty propagation (a.k.a. Bayes Linear estimation) (the foundation of the PREDUNC suite from PEST) and error variance analysis (the foundation of the PREDVAR suite of PEST).  pyEMU has easy-to-use routines for parameter and data worth analyses, which estimate how increased parameter knowledge and/or additional data effect forecast uncertainty in linear, Bayesian framework.  Support is also provided for high-dimensional Monte Carlo analyses via `ObservationEnsemble` and `ParameterEnsemble` class, including the null-space monte carlo approach of Tonkin and Doherty (2009); these ensemble classes also play nicely with PESTPP-IES.

pyEMU also includes lots of functionality for dealing with PEST(++) datasets, such as:
* manipulation of PEST control files, including the use of pandas for sophisticated editing of the parameter data and observation data sections
* creation of PEST control files from instruction and template files
* going between site sample files and pandas dataframes - really cool for observation processing
* easy-to-use observation (re)weighting via residuals or user-defined functions
* handling Jacobian and covariance matrices, including functionality to go between binary and ASCII matrices, reading and writing PEST uncertainty files.  Covariance matrices can be instantiated from relevant control file sections, such as parameter bounds or observation weights.  The base Matrix class overloads most common linear algebra operators so that operations are automatically aligned by row and column name.  Builtin SVD is also included in all Matrix instances.
* geostatistics including geostatistical structure support, reading and writing PEST structure files and creating covariance matrices implied by nested geostatistical structures, and ordinary kriging (in the utils.geostats.OrdinaryKrige object), which replicates the functionality of pest utility ``ppk2fac``. 
* composite scaled sensitivity calculations
* calculation of correlation coefficient matrix from a given covariance matrix
* Karhunen-Loeve-based parameterization as an alternative to pilot points for spatially-distributed parameter fields
* a helper functions to start a group of tcp/ip workers on a local machine for parallel PEST++/BeoPEST runs
* full support for prior information equations in control files
* preferred differencing prior information equations where the weights are based on the Pearson correlation coefficient
* verification-based tests based on results from several PEST utilities

Version => 1.1 includes the `PstFrom` setup class to support generating PEST(++) interfaces in the 100,000 to 1,000,000 parameter range with all the bells and whistles.  A publication documenting the `PstFrom` class can be found here:

[https://doi.org/10.1016/j.envsoft.2021.105022](https://doi.org/10.1016/j.envsoft.2021.105022)

A publication documenting pyEMU and an example application can be found here:

[http://dx.doi.org/10.1016/j.envsoft.2016.08.017](http://dx.doi.org/10.1016/j.envsoft.2016.08.017)


Funding
=======

pyEMU was originally developed with support from the U.S. Geological Survey. The New Zealand Strategic Science Investment Fund as part of GNS Science’s (https://www.gns.cri.nz/) Groundwater Research Programme has also funded contributions 2018-present.  Intera, Inc. has also provided funding for pyEMU development and support

Examples
========

Several example ipython notebooks are provided to demonstrate typical workflows for FOSM parameter and forecast uncertainty analysis as well as techniques to investigate parameter contributions to forecast uncertainty and observation data worth. Example models include the Henry saltwater intrusion problem (Henry 1964) and the model of Freyberg (1988)

There is a whole world of detailed learning material for script-based approaches to parameter estimation and uncertainty quantification using PEST(++) at https://github.com/gmdsi/GMDSI_notebooks. These are and excellent resource for people picking up pyEMU for the first time and for those needing to revisit elements.

Related Links
=============

[PEST++ on GitHub](https://github.com/usgs/pestpp)

[PEST](http://www.pesthomepage.org/)

[Groundwater Modelling Decision Support Initiative](https://gmdsi.org)


How to get started with pyEMU
=============================

pyEMU is available through pyPI and conda. To install pyEMU type:

    >>>conda install -c conda-forge pyemu

or

    >>>pip install pyemu

pyEMU needs `numpy` and `pandas`.  For plotting, `matplotloib`, `pyshp`, and `flopy` to take advantage of the auto interface construction

After pyEMU is installed, the PEST++ software suite can be installed for your operating system  using the command:

    get-pestpp :pyemu

See the [documentation](https://pyemu.readthedocs.io/en/latest/get_pestpp) for more information.

Found a bug? Got a smart idea? Contributions welcome.
====================================================
Feel free to raise and issue or submit a pull request.

pyEMU CI testing, using GitHub actions, has recently been switched over to run with [pytest](https://docs.pytest.org/).
We make use of [pytest-xdist](https://pytest-xdist.readthedocs.io/en/latest/) for parallel execution. 
Some notes that might be helpful for building your PR and testing:
* Test files are in [./autotest](https://github.com/pypest/pyemu/tree/develop/autotest)
* Pytest settings are in [./autotest/conftest.py](./autotest/conftest.py) and [./pyproject.toml](autotest/pyproject.toml)
* Currently, files ending `_tests.py` are collected
* Functions starting `test_` or ending `_test` are collected
* ipython notebooks in [./examples](./examples) are also run
* As tests are run in parallel, where tests require read/write access to files it is safest to sandbox runs. 
Pytest has a built-in fixture `tmp_path` that can help with this. 
Setting optional argument `--basetemp` can be helpful for accessing the locally run files. 
## Running tests locally
To be able to make clean use of pytests fixture decorators etc., 
it is recommended to run local tests through `pytest` (rather than use from script execution and commenting in 
__main__ block). For e.g.:
### Run all tests: 
> pytest --basetemp=runner autotest

with pytest-xdist, local runs can be parallelized:
> pytest --basetemp=runner -n auto autotest

### Run all tests in a file: 
> pytest --basetemp=runner -n auto autotest/testfile_tests.py

### Run a specific test [`this_test()`]:
> pytest --basetemp=runner autotest/testfile_tests.py::this_test

### Test dependencies

Python dependencies for the test suite can be installed via `pip install -e .[optional, test]` (they are recorded in [`./pyproject.toml`](./pyproject.toml)). You should also include binaries for various integrations:

* [PEST++](https://github.com/usgs/pestpp/releases), e.g. `pestpp-ies`, `pestpp-mou`, etc. These can be automatically installed via the `get-pestpp` script included with pyEMU (see above).
* [MODFLOW exectuables](https://github.com/MODFLOW-ORG/executables), i.e. `mf6`, `mfnwt`, `mfusg_gsi`. These can be automatically installed using flopy's [`get-modflow`](https://flopy.readthedocs.io/en/latest/md/get_modflow.html) command. 

These should be placed in the relevant `./bin/win`, `./bin/linux`, or `./bin/mac` directories (depending on your OS).

### Using an IDE:
Most modern, feature-rich editors and IDEs support launching pytest within debug or run consoles. 
Some might need "encouraging" to recognise the non-standard test tags used in this library. 
For example, in pycharm, to support click-and-run testing, the
[pytest-imp](https://plugins.jetbrains.com/plugin/14202-pytest-imp) plugin is required to 
pickup test functions that end with `_test` (a nosetest hangover in pyEMU).

