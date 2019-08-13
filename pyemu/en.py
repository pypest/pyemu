from __future__ import print_function, division
import os
from datetime import datetime
import copy
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
from .pyemu_warnings import PyemuWarning
import math
import numpy as np
import pandas as pd

from pyemu.pst.pst_handler import Pst

from pyemu.mat.mat_handler import get_common_elements,Matrix,Cov
from pyemu.pst.pst_utils import write_parfile,read_parfile
from pyemu.plot.plot_utils import ensemble_helper
from .utils.os_utils import run_sweep

SEED = 358183147 #from random.org on 5 Dec 2016
"""`float`: random seed for stochatics"""
np.random.seed(SEED)

class EnLoc(object):
    def __init__(self,mydf):
        self._mydf = mydf

    def __getitem__(self,item):
        return type(self._mydf)(self._mydf.pst,df=self._mydf._df.loc[item])

class EnIloc(object):
    def __init__(self,df):
        self._mydf = df

    def __getitem__(self,item):
        return type(self._mydf)(self._mydf.pst,df=self._mydf._df.iloc[item])

class Ensemble(object):
    """ The base class type for parameter and observation ensembles.
        This class should not be instantiated directly. Instead
        use `ParameterEnsemble` and `ObservationEnsemble`

    Args:
        pst (`pyemu.Pst`): a control file instance
        df (`pandas.DataFrame`): a dataframe with columns of variable
            (parameter or observation) names and index of realization
            names
        istransformed (`bool`, optional): flag for tracking parameter transformation
            status (needed in the parent class for inheritance)


    Example::

        pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=10000)

    """
    def __init__(self,pst,df,istransformed=False):

        if not isinstance(df,pd.DataFrame):
            if isinstance(df,pd.Series):
                df = pd.DataFrame(df)
            else:
                raise Exception("'df' must be a dataframe, not {0}".format(type(df)))
        if isinstance(pst,str):
            pst = pyemu.Pst(pst)
        if not isinstance(pst,Pst):
            raise Exception("'pst' must be a pyemu.Pst, not {0}".format(type(pst)))

        self._df = df.copy() # for safety
        self._df.columns = [c.lower() for c in self._df.columns]
        self.pst = pst
        self.loc = EnLoc(self)
        self.iloc = EnIloc(self)
        self._istransformed = bool(istransformed)
        self.index = self._df.index
        self.columns = self._df.columns

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    def max(self,*args,**kwargs):
        return self._df.max(*args,**kwargs)

    def min(self,*args,**kwargs):
        return self._df.min(*args,**kwargs)

    def mean(self,*args,**kwargs):
        return self._df.mean(*args,**kwargs)

    def apply(self,*args,**kwargs):
        return type(self)(pst=self.pst,df=self._df.apply(*args,**kwargs),istransformed=self.istransformed)

    @property
    def istransformed(self):
        """property decorated method to get the current
        transformation status of the ParameterEnsemble

        Returns:
            `bool`: transformation status

        """
        return copy.copy(self._istransformed)

    @property
    def shape(self):
        return self._df.shape

    def as_pyemu_matrix(self,typ=Matrix):
        """
        Create a pyemu.Matrix from the Ensemble.

        Args:
            typ (`pyemu.Matrix` or derived type): the type of matrix to return

        Returns:
            `pyemu.Matrix`

        """
        x = self._df.values.copy().astype(np.float)
        return typ(x=x,row_names=list(self._df.index),
                      col_names=list(self._df.columns))

    def drop(self,arg):
        """ overload of pandas.DataFrame.drop()

        Args:
            arg ([object]): positional argument(s) to pass to pandas.DataFrame.drop()

        Returns:
            `Ensemble`
        
        """
        df = self._df.drop(arg)
        return type(self)(df=df,pst=self.pst,istransformed=self.istransformed)

    def dropna(self,*args,**kwargs):
        """overload of pandas.DataFrame.dropna()

        Args:
            *args ([object]): positional arguments to pass to pandas.DataFrame.dropna()
            **kwargs (`dict`): keyword args to pass to pandas.DataFrame.dropna()

        Returns
        -------
        Ensemble : Ensemble
        
        """
        df = super(Ensemble,self).dropna(*args,**kwargs)
        return type(self)(df=df,pst=self.pst)

    def plot(self,bins=10,facecolor='0.5',plot_cols=None,
                    filename="ensemble.pdf",func_dict = None,
                    **kwargs):
        """plot ensemble histograms to multipage pdf

        Args:
            bins (`int`): number of bins for the histogram(s)
            facecolor (`str`): matplotlib color ('r','g','m',etc)
            plot_cols ([`str`]): subset of ensemble columns to plot.  If None, all are plotted.
                Default is None
            filename (`str`): pdf filename. Default is "ensemble.pdf"
            func_dict (`dict`): a dict of functions to apply to specific columns
                for example: {"par1":np.log10}
            **kwargs (`dict`): keyword args to pass to `pyemu.plot_utils.ensemble_helper()`

        Example::

            pst = pyemu.Pst("my.pst")
            cov = pyemu.Cov.from_parameter_data(pst,sigma_range=6)
            pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=1000)
            func_dict = {p:np.log10 for p in pst.par_names)
            pe.plot(bins=20,func_dict=func_dict)

        """
        ensemble_helper(self,bins=bins,facecolor=facecolor,plot_cols=plot_cols,
                        filename=filename)

    def __sub__(self,other):
        diff = self._df - other._df
        return type(self)(pst=self.pst,df=diff)

    # jwhite - 13 Aug 2019 - dont need this since the constructor now takes a df
    @classmethod
    def from_dataframe(cls,**kwargs):
        """deprecated class method constructor to create an Ensemble from
        a pandas.DataFrame.  Please use the Ensemble constructor directly
        now

        Args:
            **kwargs (`dict`): optional args to pass to the
                Ensemble Constructor.  Expects 'df' in kwargs.keys()
                that must be a pandas.DataFrame instance

        Returns
        -------
            Ensemble : Ensemble
        """
        pst = kwargs.pop("pst")
        df = kwargs.pop("df")
        assert isinstance(df,pd.DataFrame)
        return cls(pst=pst,df=df)

    @classmethod
    def from_binary(cls, pst, filename):
        """instantiate an observation ensemble from a PEST binary-type file

        Args:
            pst (`pyemu.Pst`): a Pst instance
            filename (`str`): the binary file name

        Returns:
            oe : ObservationEnsemble

        Notes:
            uses `pyemu.Matrix.from_binary()`

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename="obs.jcb")

        """
        m = Matrix.from_binary(filename)
        return cls(pst=pst, df=m.to_dataframe())


    @staticmethod
    def reseed():
        """method to reset the numpy.random seed using the `pyemu.en.SEED`
         global variable

        """
        np.random.seed(SEED)

    def copy(self):
        """make a deep copy of self

        Returns:
            `Ensemble`

        """
        return type(self)(df=self._df.copy(),pst=self.pst.get(),istransformed=self.istransformed)

    def covariance_matrix(self,localizer=None):
        """calculate the empirical covariance matrix implied by the ensemble using
        mean-differencing operation at the core of EnKF

        Args:
            localizer (`pyemu.Matrix`): covariance localizer to apply

        Returns:
            `pyemu.Cov`: empirical covariance matrix

        Example::

            cov = pe.covariance_matrix()
            cov.to_binary("cov.jcb")

        """

        mean = np.array(self._df.mean(axis=0))
        delta = self.as_pyemu_matrix(typ=Cov)
        for i in range(self.shape[0]):
            delta.x[i, :] -= mean
        delta *= (1.0 / np.sqrt(float(self.shape[0] - 1.0)))

        if localizer is not None:
            delta = delta.T * delta
            return delta.hadamard_product(localizer)

        return delta.T * delta



    def get_deviations(self):
        """get the deviations of the ensemble value from the mean vector

        Returns:
            `pyemu.Ensemble`: Ensemble of deviations from the mean
        """
        bt = False
        if not self.istransformed:
            bt = True
            self._transform()

        mean_vec = self._df.mean()

        df = self._df.loc[:, :].copy()
        for col in df.columns:
            df.loc[:, col] -= mean_vec[col]
        if bt:
            self._back_transform()
        return type(self).from_dataframe(pst=self.pst, df=df)



class ObservationEnsemble(Ensemble):
    """ Ensemble derived type for observations.

     Args:
        pst (`pyemu.Pst`): a control file instance
        df (`pandas.DataFrame`): a dataframe with columns of variable
            (parameter or observation) names and index of realization
            names

    Example::

        oe = pyemu.ParameterEnsemble.from_iid_gaussian_draw(pst=pst,num_reals=10000)


    Note:
        This class is primarily used to generate realizations of observation noise.
            These are typically generated from the weights listed in the control file.
            However, a general covariance matrix can be explicitly used. Does not
            generate noise realizations for observations with zero weight
    """

    def __init__(self,pst,df,istransformed=False):
        super(ObservationEnsemble,self).__init__(pst,df,istransformed)


    def _transform(self,inplace=True):
        """


        :param inplace:
        :return:
        """
        self._istransformed = True
        if inplace:
            return
        else:
            return ObservationEnsemble(pst=self.pst.get(),df=self._df.copy())

    def _back_transform(self, inplace=True):
        """


        :param inplace:
        :return:
        """
        self._istransformed = False
        if inplace:
            return
        else:
            return ObservationEnsemble(pst=self.pst.get(), df=self._df.copy())


    @property
    def names(self):
        """property decorated method to get current non-zero weighted
        column names.  Uses ObservationEnsemble.pst.nnz_obs_names

        Returns:
            [`str`]: non-zero weight observation names which are the column
                names in the `ObservationEnsemble`
        """
        return self.pst.nnz_obs_names


    @property
    def mean_values(self):
        """ property decorated method to get mean values of observation noise.
        This is a zero-valued pandas.Series

        Returns:
            `pandas Series`: series with 0.0 for non-zero weighted observations
                (since we assume observation noise has zero mean)

        """
        vals = self.pst.observation_data.obsval.copy()
        vals.loc[self.names] = 0.0
        return vals


    # def draw(self,cov,num_reals):
    #     """ draw realizations of observation noise and add to mean_values
    #     Note: only draws noise realizations for non-zero weighted observations
    #     zero-weighted observations are set to mean value for all realizations
    #
    #     Parameters
    #     ----------
    #     cov : pyemu.Cov
    #         covariance matrix that describes the support volume around the
    #         mean values.
    #     num_reals : int
    #         number of realizations to draw
    #
    #     """
    #     super(ObservationEnsemble,self).draw(cov,num_reals,
    #                                          names=self.pst.nnz_obs_names)
    #     self.loc[:,self.names] += self.pst.observation_data.obsval

    @property
    def nonzero(self):
        """ property decorated method to get a new ObservationEnsemble
        of only non-zero weighted observations

        Returns
        -------
        ObservationEnsemble : ObservationEnsemble

        """
        df = self._df.loc[:,self.pst.nnz_obs_names]
        return ObservationEnsemble(df=df,
                pst=self.pst.get(obs_names=self.pst.nnz_obs_names))

    @classmethod
    def from_id_gaussian_draw(cls,pst,num_reals):
        """ this is the primary method to generate obseration noise
        `ObservationEnsembles`.

        Args:
            pst (`pyemu.Pst`): a control file instance
            num_reals (`int`): number of realizations to draw

        Returns:
            `ObservationEnsemble`: ensemble of observation noise realizations.  The
            standard deviation of each column is the inverse of the weight in the
            control file

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble,from_id_gaussian_draw(pst=pst,num_reals=10000)
            oe.to_binary("obs.jcb")

        """
        # set up some column names
        real_names = np.arange(num_reals,dtype=np.int64)
        #arr = np.empty((num_reals,len(pst.obs_names)))
        obs = pst.observation_data
        stds = {name:1.0/obs.loc[name,"weight"] for name in pst.nnz_obs_names}
        nz_names = set(pst.nnz_obs_names)
        arr = np.random.randn(num_reals,pst.nobs)
        for i,oname in enumerate(pst.obs_names):
            if oname in nz_names:
                arr[:,i] *= stds[oname]
            else:
                arr[:,i] = 0.0
        df = pd.DataFrame(arr,index=real_names,columns=pst.obs_names)
        df.loc[:,pst.obs_names] += pst.observation_data.obsval
        new_oe = cls(pst=pst,df=df)
        return new_oe

    def to_binary(self, filename):
        """write the observation ensemble to an extended jco-style binary file.

        Args:
            filename (`str`):  the filename to write

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst,num_reals=10000)
            oe.to_binary("obs.jcb")

        """
        self.as_pyemu_matrix().to_coo(filename)




    @property
    def phi_vector(self):
        """property decorated method to get a vector of L2 norms (phi)
        for the realizations.

        Return:
            `pandas.DataFrame`: dataframe of realization names and corresponding
            phi value

        Notes:
            The ObservationEnsemble.pst.weights can be updated prior
            to calling this method to evaluate new weighting strategies

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename="my.2.obs.jcb")
            phi1 = oe.phi_vector
            pst.observation_data.loc[pst.nnz_obs_names,"weight] *= 10
            phi2 = oe.phi_vector
            ax = plt.subplot(111)
            phi1.hist(ax=ax)
            phi2.hist(ax=ax)
            plt.show()

        """
        weights = self.pst.observation_data.loc[self.names,"weight"]
        obsval = self.pst.observation_data.loc[self.names,"obsval"]
        phi_vec = []
        for idx in self._df.index.values:
            simval = self._df.loc[idx,self.names]
            phi = (((simval - obsval) * weights)**2).sum()
            phi_vec.append(phi)
        return pd.Series(data=phi_vec,index=self.index)


    def add_base(self):
        """ add "base" control file values as a realization

        """
        if "base" in self._df.index:
            raise Exception("'base' already in index")
        self._df.loc["base",:] = self.pst.observation_data.loc[self.columns,"obsval"]


class ParameterEnsemble(Ensemble):
    """ Ensemble derived type for parameter ensembles

    Args:
        pst (`pyemu.Pst`): a control file instance
        df ('pandas.DataFrame`): dataframe of realized values

        istransformed : bool
            flag indicating the transformation status (log10) of the arguments be passed
        bound_tol (`float`):  fractional amount to reset bounds transgression within the bound.
            This has been shown to be very useful for the subsequent recalibration
            because it moves parameters off of their bounds, so they are not treated as frozen in
            the upgrade calculations. defaults to 0.0

    Notes:
        implements bounds enforcement, log10 transformation,
        fixed parameters and null-space projection
        Users are probably most interested in the `ParameterEnsemble.from_gaussian_draw()`
        classmethod to generate a parameter ensemble.

    Example::

        pst = pyemu.Pst("my.pst")
        cov = pyemu.Cov.from_binary("cov.jcb")
        pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=1000)
        pe.to_binary("par.jcb")




    """

    def __init__(self,pst,df,istransformed=False,bound_tol=0.0):

        super(ParameterEnsemble,self).__init__(pst,df)
        # a flag for current log transform status
        self._istransformed = bool(istransformed)
        if "tied" in list(self.pst.parameter_data.partrans.values):
            warnings.warn("tied parameters are treated as fixed in "+\
                         "ParameterEnsemble",PyemuWarning)
        self.bound_tol = bound_tol


    @property
    def mean_values(self):
        """ the mean value vector while respecting log transform

        Returns:
            `pandas.Series`: effective mean parameter values
                (just the `Pst.parameter_data.parval1` values)
        Notes:
            if `ParameterEnsemble.istransformed` is `True`, then
            `mean_values` are transformed also (with respect to log
            transform only)

        """
        if not self.istransformed:
            return self.pst.parameter_data.parval1.copy()
        else:
            vals = self.pst.parameter_data.parval1.copy()
            vals[self.log_indexer] = np.log10(vals[self.log_indexer])
            return vals

    @property
    def names(self):
        """ Get the names of the parameters in the ParameterEnsemble

        Returns:
            [`str`]: list of parameter names

        """
        return self.pst.par_names

    @property
    def adj_names(self):
        """ Get the names of adjustable parameters in the ParameterEnsemble

        Returns:
            [`str`]: adjustable parameter names

        """
        return self.pst.adj_par_names

    @property
    def ubnd(self):
        """ the upper bound vector while respecting log transform

        Returns:
            `pandas.Series`: (possibly transformed) upper parameter bound (depending on
                `ParameterEnsemble.istransformed`)

        """
        if not self.istransformed:
            return self.pst.parameter_data.parubnd.copy()
        else:
            ub = self.pst.parameter_data.parubnd.copy()
            ub[self.log_indexer] = np.log10(ub[self.log_indexer])
            return ub

    @property
    def lbnd(self):
        """ the lower bound vector while respecting log transform

       Returns:
            `pandas.Series`: (possibly transformed) lower parameter bound (depending on
                `ParameterEnsemble.istransformed`)

        """
        if not self.istransformed:
            return self.pst.parameter_data.parlbnd.copy()
        else:
            lb = self.pst.parameter_data.parlbnd.copy()
            lb[self.log_indexer] = np.log10(lb[self.log_indexer])
            return lb

    @property
    def log_indexer(self):
        """ indexer for log transform

        Returns:
            `pandas.Series`: boolean indexer of log transformed parameters

        """
        istransformed = self.pst.parameter_data.partrans == "log"
        return istransformed.values

    @property
    def fixed_indexer(self):
        """ indexer for fixed status

        Returns:
            `pandas.Series`: boolean indexer of fixed/tied parameters

        """
        #isfixed = self.pst.parameter_data.partrans == "fixed"
        isfixed = self.pst.parameter_data.partrans.\
            apply(lambda x : x in ["fixed","tied"])
        return isfixed.values



    # def draw(self,cov,num_reals=1,how="normal",enforce_bounds=None):
    #     """draw realizations of parameter values
    #
    #     Parameters
    #     ----------
    #     cov : pyemu.Cov
    #         covariance matrix that describes the support around
    #         the mean parameter values
    #     num_reals : int
    #         number of realizations to generate
    #     how : str
    #         distribution to use to generate realizations.  Options are
    #         'normal' or 'uniform'.  Default is 'normal'.  If 'uniform',
    #         cov argument is ignored
    #     enforce_bounds : str
    #         how to enforce parameter bound violations.  Options are
    #         'reset' (reset individual violating values), 'drop' (drop realizations
    #         that have one or more violating values.  Default is None (no bounds enforcement)
    #
    #     """
    #     how = how.lower().strip()
    #     if not self.istransformed:
    #             self._transform()
    #     if how == "uniform":
    #         self._draw_uniform(num_reals=num_reals)
    #     else:
    #         super(ParameterEnsemble,self).draw(cov,num_reals=num_reals)
    #         # replace the realizations for fixed parameters with the original
    #         # parval1 in the control file
    #         self.pst.parameter_data.index = self.pst.parameter_data.parnme
    #         fixed_vals = self.pst.parameter_data.loc[self.fixed_indexer,"parval1"]
    #         for fname,fval in zip(fixed_vals.index,fixed_vals.values):
    #             #if fname not in self.columns:
    #             #    continue
    #             self.loc[:,fname] = fval
    #     istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
    #     self.loc[:,istransformed] = 10.0**self.loc[:,istransformed]
    #     self._istransformed = False
    #
    #     #self._applied_tied()
    #
    #
    #     self.enforce(enforce_bounds)

    def _draw_uniform(self,num_reals=1):
        """ Draw parameter realizations from a (log10) uniform distribution
        described by the parameter bounds.  Respect Log10 transformation

        Parameters
        ----------
        num_reals : int
            number of realizations to generate

        """
        if not self.istransformed:
            self._transform()
        self.loc[:,:] = np.NaN
        self.dropna(inplace=True)
        ub = self.ubnd
        lb = self.lbnd
        for pname in self.names:
            if pname in self.adj_names:
                self.loc[:,pname] = np.random.uniform(lb[pname],
                                                      ub[pname],
                                                      size=num_reals)
            else:
                self.loc[:,pname] = np.zeros((num_reals)) + \
                                    self.pst.parameter_data.\
                                         loc[pname,"parval1"]

    @classmethod
    def from_uniform_draw(cls,pst,num_reals):
        """ instantiate a parameter ensemble from uniform draws

        Parameters
        ----------
        pst : pyemu.Pst
            a control file instance
        num_reals : int
            number of realizations to generate

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble

        """
        #if not pe.istransformed:
        #    pe._transform()
        #ub = pe.ubnd
        #lb = pe.lbnd
        li = pst.parameter_data.partrans == "log"
        ub = pst.parameter_data.parubnd.copy()
        ub.loc[li] = ub.loc[li].apply(np.log10)
        ub = ub.to_dict()
        lb = pst.parameter_data.parlbnd.copy()
        lb.loc[li] = lb.loc[li].apply(np.log10)
        lb = lb.to_dict()

        # set up some column names
        #real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
        real_names = np.arange(num_reals,dtype=np.int64)
        arr = np.empty((num_reals,len(ub)))
        adj_par_names = set(pst.adj_par_names)
        for i,pname in enumerate(pst.parameter_data.parnme):
            #print(pname,lb[pname],ub[pname])
            if pname in adj_par_names:
                arr[:,i] = np.random.uniform(lb[pname],
                                                      ub[pname],
                                                      size=num_reals)
            else:
                arr[:,i] = np.zeros((num_reals)) + \
                                    pst.parameter_data.\
                                         loc[pname,"parval1"]
        #print("back transforming")

        df = pd.DataFrame(arr,index=real_names,columns=pst.par_names)
        df.loc[:,li] = 10.0**df.loc[:,li]

        new_pe = cls(pst=pst,df=pd.DataFrame(data=arr,columns=pst.par_names))
        #new_pe._applied_tied()
        return new_pe

    @classmethod
    def from_triangular_draw(cls, pst, num_reals):
        """instantiate a parameter ensemble from triangular distribution

        Parameters
        ----------
        pst : pyemu.Pst
            a control file instance
        num_reals : int
            number of realizations to generate

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble


        """

        li = pst.parameter_data.partrans == "log"
        ub = pst.parameter_data.parubnd.copy()
        ub.loc[li] = ub.loc[li].apply(np.log10)

        lb = pst.parameter_data.parlbnd.copy()
        lb.loc[li] = lb.loc[li].apply(np.log10)

        pv = pst.parameter_data.parval1.copy()
        pv.loc[li] = pv[li].apply(np.log10)

        ub = ub.to_dict()
        lb = lb.to_dict()
        pv = pv.to_dict()

        # set up some column names
        # real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
        real_names = np.arange(num_reals, dtype=np.int64)
        arr = np.empty((num_reals, len(ub)))
        adj_par_names = set(pst.adj_par_names)
        for i, pname in enumerate(pst.parameter_data.parnme):
            #print(pname, lb[pname], ub[pname])
            if pname in adj_par_names:
                arr[:,i] = np.random.triangular(lb[pname],
                                                pv[pname],
                                                ub[pname],
                                                size=num_reals)
            else:
                arr[:, i] = np.zeros((num_reals)) + \
                            pst.parameter_data. \
                                loc[pname, "parval1"]


        #print("back transforming")

        df = pd.DataFrame(arr, index=real_names, columns=pst.par_names)
        df.loc[:, li] = 10.0 ** df.loc[:, li]

        new_pe = cls(pst=pst, df=pd.DataFrame(data=arr, columns=pst.par_names))
        # new_pe._applied_tied()
        return new_pe



    @classmethod
    def from_gaussian_draw(cls,pst,cov,num_reals=1,use_homegrown=True,group_chunks=False,
                           fill_fixed=True,enforce_bounds=False):
        """ instantiate a parameter ensemble from a covariance matrix

        Parameters
        ----------
        pst : pyemu.Pst
            a control file instance
        cov : (pyemu.Cov)
            covariance matrix to use for drawing
        num_reals : int
            number of realizations to generate
        use_homegrown : bool
            flag to use home-grown full cov draws...much faster
            than numpy...
        group_chunks : bool
            flag to break up draws by par groups.  Only applies
            to homegrown, full cov case. Default is False
        fill_fixed : bool
            flag to fill in fixed parameters from the pst into the
            ensemble using the parval1 from the pst.  Default is True
        enforce_bounds : bool
            flag to enforce parameter bounds from the pst.  realized
            parameter values that violate bounds are simply changed to the
            value of the violated bound.  Default is False

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble


        """

        # set up some column names
        #real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
        real_names = np.arange(num_reals,dtype=np.int64)

        li = pst.parameter_data.partrans == "log"
        vals = pst.parameter_data.parval1.copy()
        vals[li] = vals.loc[li].apply(np.log10)

        # make sure everything is cool WRT ordering
        if list(vals.index.values) != cov.row_names:
            common_names = get_common_elements(vals.index.values,
                                               cov.row_names)
            if len(common_names) == 0:
                raise Exception("ParameterEnsemble::from_gaussian_draw() error: cov and pst share no common names")
            vals = vals.loc[common_names]
            cov = cov.get(common_names)
        else:
            common_names = cov.row_names

        li = pst.parameter_data.partrans.loc[common_names] == "log"
        if cov.isdiagonal:
            #print("making diagonal cov draws")
            #print("building mean and std dicts")
            arr = np.zeros((num_reals,len(vals)))
            stds = {pname:std for pname,std in zip(common_names,np.sqrt(cov.x.flatten()))}
            means = {pname:val for pname,val in zip(common_names,vals)}
            #print("numpy draw")
            arr = np.random.randn(num_reals,len(common_names))
            #print("post-processing")
            adj_pars = set(pst.adj_par_names)
            for i,pname in enumerate(common_names):
                if pname in adj_pars:
                    #s = stds[pname]
                    #v = means[pname]
                    #arr[:,i] = np.random.normal(means[pname],stds[pname],
                    #                            size=num_reals)
                    arr[:,i] = (arr[:,i] * stds[pname]) + means[pname]
                else:
                    arr[:,i] = means[pname]
            #print("build df")
            df = pd.DataFrame(data=arr,columns=common_names,index=real_names)
        else:
            if use_homegrown:
                print("making full cov draws with home-grown goodness")
                # generate standard normal vectors


                # jwhite - 18-dec-17: the cholesky version is giving significantly diff
                # results compared to eigen solve, so turning this off for now - need to
                # learn more about this...
                # use_chol = False
                # if use_chol:
                #     a = np.linalg.cholesky(cov.as_2d)
                #
                # else:
                # decompose...
                if group_chunks:
                    par_cov = pst.parameter_data.loc[cov.names,:]
                    par_cov.loc[:,"idxs"] = np.arange(cov.shape[0])
                    #print("algning cov")
                    #cov.align(list(par_cov.parnme))
                    pargps = par_cov.pargp.unique()
                    #print("reserving reals matrix")
                    reals = np.zeros((num_reals,cov.shape[0]))

                    for ipg,pargp in enumerate(pargps):
                        pnames = list(par_cov.loc[par_cov.pargp==pargp,"parnme"])
                        idxs = par_cov.loc[par_cov.pargp == pargp, "idxs"]
                        #print("{0} of {1} drawing for par group '{2}' with {3} pars "
                        #      .format(ipg+1,len(pargps),pargp, len(idxs)))

                        s,e = idxs[0],idxs[-1]
                        #print("generating snv matrix")
                        snv = np.random.randn(num_reals, len(pnames))

                        cov_pg = cov.get(pnames)
                        if len(pnames) == 1:
                            std = np.sqrt(cov_pg.x)
                            reals[:,idxs] = vals[pnames].values[0] + (snv * std)
                        else:
                            try:
                                cov_pg.inv
                            except:
                                covname = "trouble_{0}.cov".format(pargp)
                                #print('saving toubled cov matrix to {0}'.format(covname))
                                cov_pg.to_ascii(covname)
                                #print(cov_pg.get_diagonal_vector())
                                raise Exception("error inverting cov for par group '{0}',"+\
                                                "saved trouble cov to {1}".
                                                format(pargp,covname))
                            v, w = np.linalg.eigh(cov_pg.as_2d)
                            # check for near zero eig values

                            #vdiag = np.diag(v)
                            for i in range(v.shape[0]):
                                if v[i] > 1.0e-10:
                                    pass
                                else:
                                    print("near zero eigen value found",v[i],\
                                          "at index",i," of ",v.shape[0])
                                    v[i] = 0.0
                            vsqrt = np.sqrt(v)
                            vsqrt[i:] = 0.0
                            v = np.diag(vsqrt)
                            a = np.dot(w, v)
                            pg_vals = vals[pnames]
                            for i in range(num_reals):
                                #v = snv[i,:]
                                #p = np.dot(a,v)
                                reals[i,idxs] =  pg_vals + np.dot(a,snv[i,:])
                else:

                    #print("generating snv matrix")
                    snv = np.random.randn(num_reals, cov.shape[0])

                    #print("eigen solve for full cov")
                    v, w = np.linalg.eigh(cov.as_2d)
                    #w, v, other = np.linalg.svd(cov.as_2d,full_matrices=True,compute_uv=True)
                    # vdiag = np.diag(v)
                    for i in range(v.shape[0]):
                        if v[i] > 1.0e-10:
                            pass
                        else:
                            print("near zero eigen value found", v[i], \
                                  "at index", i, " of ", v.shape[0])
                            v[i] = 0.0
                    # form projection matrix
                    #print("form projection")
                    a = np.dot(w, np.sqrt(np.diag(v)))
                    #print(a)
                    # project...
                    reals = []
                    for vec in snv:
                        real = vals + np.dot(a, vec)
                        reals.append(real)

                df = pd.DataFrame(reals, columns=common_names, index=real_names)

            #vals = pe.mean_values
            else:
                #print("making full cov draws with numpy")
                df = pd.DataFrame(data=np.random.multivariate_normal(vals, cov.as_2d,num_reals),
                                  columns = common_names,index=real_names)
            #print(df.shape,cov.shape)


        df.loc[:,li] = 10.0**df.loc[:,li]

        # replace the realizations for fixed parameters with the original
        # parval1 in the control file
        #print("handling fixed pars")
        #pe.pst.parameter_data.index = pe.pst.parameter_data.parnme
        if fill_fixed:
            par = pst.parameter_data
            fixed_vals = par.loc[par.partrans.apply(lambda x: x in ["fixed","tied"]),"parval1"]
            for fname,fval in zip(fixed_vals.index,fixed_vals.values):
                #print(fname)
                df.loc[:,fname] = fval

            #print("apply tied")
        new_pe = cls(pst=pst,df=df)
        if enforce_bounds:
            new_pe.enforce()
        return new_pe


    @classmethod
    def from_mixed_draws(cls,pst,how_dict,default="gaussian",num_reals=100,cov=None,sigma_range=6,
                         enforce_bounds=True,partial=False):
        """instaniate a parameter ensemble from stochastic draws using a mixture of
        distributions.  Available distributions include (log) "uniform", (log) "triangular",
        and (log) "gaussian". log transformation is respected.

        Parameters
        ----------
        pst : pyemu.Pst
            a Pst instance
        how_dict : dict
            a dictionary of parnme keys and 'how' values, where "how" can be "uniform",
            "triangular", or "gaussian".
        default : str
            the default distribution to use for parameter not listed in how_dict
        num_reals : int
            number of realizations to draw
        cov : pyemu.Cov
            an optional Cov instance to use for drawing from gaussian distribution.  If None,
            and "gaussian" is listed in how_dict (or default), then a diagonal covariance matrix
            is constructed from the parameter bounds in the pst.  Default is None
        sigma_range : float
             the number of standard deviations implied by the bounds in the pst.  Only used if
             "gaussian" is in how_dict (or default) and cov is None.  Default is 6.
        enforce_bounds : boolean
            flag to enforce parameter bounds in resulting ParameterEnsemble.
            Only matters if "gaussian" is in values of how_dict.  Default is True.
        partial : bool
            flag to allow a partial ensemble (not all pars included). Default is False

        """

        # error checking
        accept = {"uniform", "triangular", "gaussian"}
        assert default in accept,"ParameterEnsemble.from_mixed_draw() error: 'default' must be in {0}".format(accept)
        par_org = pst.parameter_data.copy()
        pset = set(pst.adj_par_names)
        hset = set(how_dict.keys())
        missing = pset.difference(hset)
        #assert len(missing) == 0,"ParameterEnsemble.from_mixed_draws() error: the following par names are not in " +\
        #    " in how_dict: {0}".format(','.join(missing))
        if not partial and len(missing) > 0:
            print("{0} par names missing in how_dict, these parameters will be sampled using {1} (the 'default')".\
                  format(len(missing),default))
            for m in missing:
                how_dict[m] = default
        missing = hset.difference(pset)
        assert len(missing) == 0, "ParameterEnsemble.from_mixed_draws() error: the following par names are not in " + \
                                  " in the pst: {0}".format(','.join(missing))

        unknown_draw = []
        for pname,how in how_dict.items():
            if how not in accept:
                unknown_draw.append("{0}:{1}".format(pname,how))
        if len(unknown_draw) > 0:
            raise Exception("ParameterEnsemble.from_mixed_draws() error: the following hows are not recognized:{0}"\
                            .format(','.join(unknown_draw)))


        # work out 'how' grouping
        how_groups = {how:[] for how in accept}
        for pname,how in how_dict.items():
            how_groups[how].append(pname)

        # gaussian
        pes = []
        if len(how_groups["gaussian"]) > 0:
            gset = set(how_groups["gaussian"])
            par_gaussian = par_org.loc[gset, :]
            #par_gaussian.sort_values(by="parnme", inplace=True)
            par_gaussian.sort_index(inplace=True)
            pst.parameter_data = par_gaussian

            if cov is not None:
                cset = set(cov.row_names)
                #gset = set(how_groups["gaussian"])
                diff = gset.difference(cset)
                assert len(diff) == 0,"ParameterEnsemble.from_mixed_draws() error: the 'cov' is not compatible with " +\
                        " the parameters listed as 'gaussian' in how_dict, the following are not in the cov:{0}".\
                        format(','.join(diff))
            else:

                cov = Cov.from_parameter_data(pst,sigma_range=sigma_range)
            pe_gauss = cls.from_gaussian_draw(pst,cov,num_reals=num_reals,
                                                            enforce_bounds=enforce_bounds)
            pes.append(pe_gauss)

        if len(how_groups["uniform"]) > 0:
            par_uniform = par_org.loc[how_groups["uniform"],:]
            #par_uniform.sort_values(by="parnme",inplace=True)
            par_uniform.sort_index(inplace=True)
            pst.parameter_data = par_uniform
            pe_uniform = cls.from_uniform_draw(pst,num_reals=num_reals)
            pes.append(pe_uniform)

        if len(how_groups["triangular"]) > 0:
            par_tri = par_org.loc[how_groups["triangular"],:]
            #par_tri.sort_values(by="parnme", inplace=True)
            par_tri.sort_index(inplace=True)
            pst.parameter_data = par_tri
            pe_tri = cls.from_triangular_draw(pst,num_reals=num_reals)
            pes.append(pe_tri)


        df = pd.DataFrame(index=np.arange(num_reals),columns=par_org.parnme.values)

        df.loc[:,:] = np.NaN
        fixed_tied = par_org.loc[par_org.partrans.apply(lambda x: x in ["fixed","tied"]),"parval1"].to_dict()
        for p,v in fixed_tied.items():
            df.loc[:,p] = v

        for pe in pes:
            df.loc[pe.index,pe.columns] = pe

        if partial:
            df = df.dropna(axis=1)
        elif df.shape != df.dropna().shape:
            raise Exception("ParameterEnsemble.from_mixed_draws() error: NaNs in final parameter ensemble")
        pst.parameter_data = par_org
        return cls(df=df,pst=pst)


    def _back_transform(self,inplace=True):
        """ Private method to remove log10 transformation from ensemble

        Parameters
        ----------
        inplace: bool
            back transform self in place

        Returns
        ------
        ParameterEnsemble : ParameterEnsemble
            if inplace if False

        Note
        ----
        Don't call this method unless you know what you are doing

        """
        if not self.istransformed:
            raise Exception("ParameterEnsemble already back transformed")

        istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
        if inplace:
            self._df.loc[:,istransformed] = 10.0**(self._df.loc[:,istransformed])
            self._df.loc[:,:] = (self._df.loc[:,:] -\
                             self.pst.parameter_data.offset)/\
                             self.pst.parameter_data.scale

            self._istransformed = False
        else:
            vals = (self.pst.parameter_data.parval1 -\
                    self.pst.parameter_data.offset) /\
                    self.pst.parameter_data.scale
            new_en = ParameterEnsemble(pst=self.pst.get(),
                                       df=self._df.loc[:,:].copy(),
                                       istransformed=False)
            new_en._df.loc[:,istransformed] = 10.0**(self._df.loc[:,istransformed])
            new_en._df.loc[:,:] = (new_en._df.loc[:,:] -\
                             new_en.pst.parameter_data.offset)/\
                             new_en.pst.parameter_data.scale
            return new_en


    def _transform(self,inplace=True):
        """ Private method to perform log10 transformation for ensemble

        Parameters
        ----------
        inplace: bool
            transform self in place

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble
            if inplace is False

        Note
        ----
        Don't call this method unless you know what you are doing

        """
        if self.istransformed:
            #raise Exception("ParameterEnsemble already transformed")
            return

        istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
        if inplace:
            #self.loc[:,istransformed] = np.log10(self.loc[:,istransformed])
            self._df.loc[:,:] = (self._df.loc[:,:] * self.pst.parameter_data.scale) +\
                             self.pst.parameter_data.offset
            self._df.loc[:,istransformed] = self._df.loc[:,istransformed].applymap(lambda x: math.log10(x))

            self._istransformed = True
        else:
            vals = self.pst.parameter_data.parval1.copy()
            new_en = ParameterEnsemble(pst=self.pst.get(),
                                       data=self._dfloc[:,:].copy(),
                                       istransformed=True)
            new_en._df.loc[:,:] = (new_en._df.loc[:,:] * self.pst.parameter_data.scale) +\
                             new_en.pst.parameter_data.offset
            new_en._df.loc[:,istransformed] = self._df.loc[:,istransformed].applymap(lambda x: math.log10(x))
            return new_en



    def project(self,projection_matrix,inplace=True,log=None,
                enforce_bounds="reset"):
        """ project the ensemble using the null-space Monte Carlo method

        Parameters
        ----------
        projection_matrix : pyemu.Matrix
            projection operator - must already respect log transform

        inplace : bool
            project self or return a new ParameterEnsemble instance

        log: pyemu.Logger
            for logging progress

        enforce_bounds : str
            parameter bound enforcement flag. 'drop' removes
            offending realizations, 'reset' resets offending values

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble
            if inplace is False

        """

        if self.istransformed:
            self._back_transform()

        li = self.pst.parameter_data.loc[:,"partrans"] == "log"
        self.loc[:,li] = self.loc[:,li].applymap(lambda x: math.log10(x))
        self._istransformed = True

        #make sure everything is cool WRT ordering
        common_names = get_common_elements(self.adj_names,
                                                 projection_matrix.row_names)
        base = self.mean_values.loc[common_names]
        projection_matrix = projection_matrix.get(common_names,common_names)

        if not inplace:
            new_en = ParameterEnsemble(pst=self.pst.get(),df=self._df.loc[:,:].copy(),
                                       istransformed=self.istransformed)

        for real in self.index:
            if log is not None:
                log("projecting realization {0}".format(real))

            # null space projection of difference vector
            pdiff = self._df.loc[real,common_names] - base
            pdiff = np.dot(projection_matrix.x,
                           (self._df.loc[real,common_names] - base)\
                           .values)

            if inplace:
                self._df.loc[real,common_names] = base + pdiff
            else:
                new_en._df.loc[real,common_names] = base + pdiff

            if log is not None:
                log("projecting realization {0}".format(real))
        if not inplace:
            new_en.enforce(enforce_bounds)
            new_en._df.loc[:,istransformed] = 10.0**new_en._df.loc[:,istransformed]
            new_en._istransformed = False

            #new_en._back_transform()
            return new_en

        self.enforce(enforce_bounds)
        self._df.loc[:,istransformed] = 10.0**self._df.loc[:,istransformed]
        self._istransformed = False

    def enforce(self,enforce_bounds="reset"):
        """ entry point for bounds enforcement.  This gets called for the
        draw method(s), so users shouldn't need to call this

        Parameters
        ----------
        enforce_bounds : str
            can be 'reset' to reset offending values or 'drop' to drop
            offending realizations

        """
        if isinstance(enforce_bounds,bool):
            import warnings
            warnings.warn("deprecation warning: enforce_bounds should be "+\
                          "either 'reset', 'drop', 'scale', or None, not bool"+\
                          "...resetting to None.",PyemuWarning)
            enforce_bounds = None
        if enforce_bounds is None:
            return

        if enforce_bounds.lower() == "reset":
            self.enforce_reset()
        elif enforce_bounds.lower() == "drop":
            self.enforce_drop()
        elif enforce_bounds.lower() == "scale":
            self.enfore_scale()
        else:
            raise Exception("unrecognized enforce_bounds arg:"+\
                            "{0}, should be 'reset' or 'drop'".\
                            format(enforce_bounds))

    def enfore_scale(self):
        """
        enforce parameter bounds on the ensemble by finding the
        scaling factor needed to bring the most violated parameter back in bounds

        Note
        ----
        this method not fully implemented.

        Raises
        ------
        NotImplementedError if called.

        """
        raise NotImplementedError()


    def enforce_drop(self):
        """ enforce parameter bounds on the ensemble by dropping
        violating realizations

        """
        ub = self.ubnd
        lb = self.lbnd
        drop = []
        for id in self.index:
            #mx = (ub - self.loc[id,:]).min()
            #mn = (lb - self.loc[id,:]).max()
            if (ub - self._df.loc[id,:]).min() < 0.0 or\
                            (lb - self._df.loc[id,:]).max() > 0.0:
                drop.append(id)
        self._df.loc[drop,:] = np.NaN
        self.dropna(inplace=True)


    def enforce_reset(self):
        """enforce parameter bounds on the ensemble by resetting
        violating vals to bound
        """

        ub = (self.ubnd * (1.0+self.bound_tol)).to_dict()
        lb = (self.lbnd * (1.0 - self.bound_tol)).to_dict()
        #for iname,name in enumerate(self.columns):
            #self.loc[self.loc[:,name] > ub[name],name] = ub[name] * (1.0 + self.bound_tol)
            #self.loc[self.loc[:,name] < lb[name],name] = lb[name].copy() * (1.0 - self.bound_tol)
        #    self.loc[self.loc[:,name] > ub[name],name] = ub[name]
        #    self.loc[self.loc[:,name] < lb[name],name] = lb[name]

        val_arr = self._df.values
        for iname, name in enumerate(self.columns):
            val_arr[val_arr[:,iname] > ub[name],iname] = ub[name]
            val_arr[val_arr[:, iname] < lb[name],iname] = lb[name]


    @classmethod
    def from_parfiles(cls,pst,parfile_names,real_names=None):
        """ create a parameter ensemble from parfiles.  Accepts parfiles with less than the
        parameters in the control (get NaNs in the ensemble) or extra parameters in the
        parfiles (get dropped)

        Parameters:
            pst : pyemu.Pst

            parfile_names : list of str
                par file names

            real_names : str
                optional list of realization names. If None, a single integer counter is used

        Returns:
            pyemu.ParameterEnsemble


        """
        if isinstance(pst,str):
            pst = pyemu.Pst(pst)
        dfs = {}
        if real_names is not None:
            assert len(real_names) == len(parfile_names)
        else:
            real_names = np.arange(len(parfile_names))

        for rname,pfile in zip(real_names,parfile_names):
            assert os.path.exists(pfile), "ParameterEnsemble.read_parfiles() error: " + \
                                          "file: {0} not found".format(pfile)
            df = read_parfile(pfile)
            #check for scale differences - I don't who is dumb enough
            #to change scale between par files and pst...
            diff = df.scale - pst.parameter_data.scale
            if diff.apply(np.abs).sum() > 0.0:
                warnings.warn("differences in scale detected, applying scale in par file",
                              PyemuWarning)
                #df.loc[:,"parval1"] *= df.scale

            dfs[rname] = df.parval1.values

        df_all = pd.DataFrame(data=dfs).T
        df_all.columns = df.index

        if len(pst.par_names) != df_all.shape[1]:
            #if len(pst.par_names) < df_all.shape[1]:
            #    raise Exception("pst is not compatible with par files")
            pset = set(pst.par_names)
            dset = set(df_all.columns)
            diff = pset.difference(dset)
            if len(diff) > 0:
                warnings.warn("the following parameters are not in the par files (getting NaNs) :{0}".
                             format(','.join(diff)),PyemuWarning)
                blank_df = pd.DataFrame(index=df_all.index,columns=diff)

                df_all = pd.concat([df_all,blank_df],axis=1)

            diff = dset.difference(pset)
            if len(diff) > 0:
                warnings.warn("the following par file parameters are not in the control (being dropped):{0}".
                              format(','.join(diff)),PyemuWarning)
                df_all = df_all.loc[:, pst.par_names]

        return cls.from_dataframe(pst=pst,df=df_all)


    def to_csv(self,*args,**kwargs):
        """overload of pandas.DataFrame.to_csv() to account
        for parameter transformation so that the saved
        ParameterEnsemble csv is not in Log10 space

        Parameters
        ----------
        *args : list
            positional arguments to pass to pandas.DataFrame.to_csv()
        **kwrags : dict
            keyword arguments to pass to pandas.DataFrame.to_csv()

        Note
        ----
        this function back-transforms inplace with respect to
        log10 before writing

        """
        retrans = False
        if self.istransformed:
            self._back_transform(inplace=True)
            retrans = True
        if self.isnull().values.any():
            warnings.warn("NaN in par ensemble",PyemuWarning)
        super(ParameterEnsemble,self)._df.to_csv(*args,**kwargs)
        if retrans:
            self._transform(inplace=True)


    def to_binary(self,filename):
        """write the parameter ensemble to a jco-style binary file

        Parameters
        ----------
        filename : str
            the filename to write

        Returns
        -------
        None


        Note
        ----
        this function back-transforms inplace with respect to
        log10 before writing

        """

        retrans = False
        if self.istransformed:
            self._back_transform(inplace=True)
            retrans = True
        if self._df.isnull().values.any():
            warnings.warn("NaN in par ensemble",PyemuWarning)
        self.as_pyemu_matrix().to_coo(filename)
        if retrans:
            self._transform(inplace=True)


    def to_parfiles(self,prefix):
        """
            write the parameter ensemble to PEST-style parameter files

        Parameters
        ----------
        prefix: str
            file prefix for par files

        Note
        ----
        this function back-transforms inplace with respect to
        log10 before writing

        """
        if self.isnull().values.any():
            warnings.warn("NaN in par ensemble",PyemuWarning)
        if self.istransformed:
            self._back_transform(inplace=True)

        par_df = self.pst.parameter_data.loc[:,
                 ["parnme","parval1","scale","offset"]].copy()

        for real in self.index:
            par_file = "{0}{1}.par".format(prefix,real)
            par_df.loc[:,"parval1"] =self._df.loc[real,:]
            write_parfile(par_df,par_file)


    def add_base(self):
        """ add "base" control file values as a realization

        """
        if "base" in self.index:
            raise Exception("'base' already in index")
        self._df.loc["base",:] = self.pst.parameter_data.loc[self.columns,"parval1"]

