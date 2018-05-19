from __future__ import print_function, division
import os
from datetime import datetime
import copy
import warnings
import math
import numpy as np
import pandas as pd

from pyemu.mat.mat_handler import get_common_elements,Matrix,Cov,SparseMatrix
from pyemu.pst.pst_utils import write_parfile,read_parfile
from pyemu.plot.plot_utils import ensemble_helper

#warnings.filterwarnings("ignore",message="Pandas doesn't allow columns to be "+\
#                                         "created via a new attribute name - see"+\
#                                         "https://pandas.pydata.org/pandas-docs/"+\
#                                         "stable/indexing.html#attribute-access")
warnings.filterwarnings("ignore",category=UserWarning,module="pandas")
SEED = 358183147 #from random.org on 5 Dec 2016
#print("setting random seed")
np.random.seed(SEED)


class Ensemble(pd.DataFrame):
    """ The base class type for handling parameter and observation ensembles.
        It is directly derived from pandas.DataFrame.  This class should not be
        instantiated directly.

    Parameters
    ----------
    *args : list
        positional args to pass to pandas.DataFrame()
    **kwargs : dict
        keyword args to pass to pandas.DataFrame().  Must contain
        'columns' and 'mean_values'

    Returns
    -------
    Ensemble : Ensemble

    """
    def __init__(self,*args,**kwargs):
        """constructor for base Ensemble type.  'columns' and 'mean_values'
          must be in the kwargs
        """

        assert "columns" in kwargs.keys(),"ensemble requires 'columns' kwarg"

        mean_values = kwargs.pop("mean_values",None)

        super(Ensemble,self).__init__(*args,**kwargs)

        if mean_values is None:
            raise Exception("Ensemble requires 'mean_values' kwarg")
        self._mean_values = mean_values

    def as_pyemu_matrix(self,typ=Matrix):
        """
        Create a pyemu.Matrix from the Ensemble.

        Parameters
        ----------
            typ : pyemu.Matrix or derived type
                the type of matrix to return

        Returns
        -------
        pyemu.Matrix : pyemu.Matrix

        """
        x = self.values.copy().astype(np.float)
        return typ(x=x,row_names=list(self.index),
                      col_names=list(self.columns))

    def drop(self,arg):
        """ overload of pandas.DataFrame.drop()

        Parameters
        ----------
        arg : iterable
            argument to pass to pandas.DataFrame.drop()

        Returns
        -------
        Ensemble : Ensemble
        
        """
        df = super(Ensemble,self).drop(arg)
        return type(self)(data=df,pst=self.pst)

    def dropna(self,*args,**kwargs):
        """overload of pandas.DataFrame.dropna()

        Parameters
        ----------
        *args : list
            positional args to pass to pandas.DataFrame.dropna()
        **kwargs : dict
            keyword args to pass to pandas.DataFrame.dropna()

        Returns
        -------
        Ensemble : Ensemble
        
        """
        df = super(Ensemble,self).dropna(*args,**kwargs)
        return type(self)(data=df,pst=self.pst)

    def draw(self,cov,num_reals=1,names=None):
        """ draw random realizations from a multivariate
            Gaussian distribution

        Parameters
        ----------    
        cov: pyemu.Cov
            covariance structure to draw from
        num_reals: int
            number of realizations to generate
        names : list
            list of columns names to draw for.  If None, values all names
            are drawn

        """
        real_names = np.arange(num_reals,dtype=np.int64)

        # make sure everything is cool WRT ordering
        if names is not None:
            vals = self.mean_values.loc[names]
            cov = cov.get(names)
        elif self.names != cov.row_names:
            names = get_common_elements(self.names,
                                        cov.row_names)
            vals = self.mean_values.loc[names]
            cov = cov.get(names)
            pass
        else:
            vals = self.mean_values
            names = self.names

        # generate random numbers
        if cov.isdiagonal: #much faster
            val_array = np.array([np.random.normal(mu,std,size=num_reals) for\
                                  mu,std in zip(vals,np.sqrt(cov.x))]).transpose()
        else:
            val_array = np.random.multivariate_normal(vals, cov.as_2d,num_reals)

        self.loc[:,:] = np.NaN
        self.dropna(inplace=True)

        # this sucks - can only set by enlargement one row at a time
        for rname,vals in zip(real_names,val_array):
            self.loc[rname, names] = vals
            # set NaNs to mean_values
            idx = pd.isnull(self.loc[rname,:])
            self.loc[rname,idx] = self.mean_values[idx]

    # def enforce(self):
    #     """ placeholder method for derived ParameterEnsemble type
    #     to enforce parameter bounds
    # 
    # 
    #     Raises
    #     ------
    #         Exception if called
    #     """
    #     raise Exception("Ensemble.enforce() must overloaded by derived types")

    def plot(self,bins=10,facecolor='0.5',plot_cols=None,
                    filename="ensemble.pdf",func_dict = None,
                    **kwargs):
        """plot ensemble histograms to multipage pdf

        Parameters
        ----------
        bins : int
            number of bins
        facecolor : str
            color
        plot_cols : list of str
            subset of ensemble columns to plot.  If None, all are plotted.
            Default is None
        filename : str
            pdf filename. Default is "ensemble.pdf"
        func_dict : dict
            a dict of functions to apply to specific columns (e.g., np.log10)

        **kwargs : dict
            keyword args to pass to plot_utils.ensemble_helper()

        Returns
        -------
        None

        """
        ensemble_helper(self,bins=bins,facecolor=facecolor,plot_cols=plot_cols,
                        filename=filename)


    def __sub__(self,other):
        """overload of pandas.DataFrame.__sub__() operator to difference two
        Ensembles

        Parameters
        ----------
        other : pyemu.Ensemble or pandas.DataFrame
            the instance to difference against

        Returns
        -------
            Ensemble : Ensemble
        
        """
        diff = super(Ensemble,self).__sub__(other)
        return Ensemble.from_dataframe(df=diff)


    @classmethod
    def from_dataframe(cls,**kwargs):
        """class method constructor to create an Ensemble from
        a pandas.DataFrame

        Parameters
        ----------
        **kwargs : dict
            optional args to pass to the
            Ensemble Constructor.  Expects 'df' in kwargs.keys()
            that must be a pandas.DataFrame instance

        Returns
        -------
            Ensemble : Ensemble
        """
        df = kwargs.pop("df")
        assert isinstance(df,pd.DataFrame)
        df.columns = [c.lower() for c in df.columns]
        mean_values = kwargs.pop("mean_values",df.mean(axis=0))
        e = cls(data=df,index=df.index,columns=df.columns,
                mean_values=mean_values,**kwargs)
        return e

    @staticmethod
    def reseed():
        """method to reset the numpy.random seed using the pyemu.en
        SEED global variable

        """
        np.random.seed(SEED)

    def copy(self):
        """make a deep copy of self

        Returns
        -------
        Ensemble : Ensemble

        """
        df = super(Ensemble,self).copy()
        return type(self).from_dataframe(df=df)



    def covariance_matrix(self,localizer=None):
        """calculate the approximate covariance matrix implied by the ensemble using
        mean-differencing operation at the core of EnKF

        Parameters
        ----------
            localizer : pyemu.Matrix
                covariance localizer to apply

        Returns
        -------
            cov : pyemu.Cov
                covariance matrix

        """




        mean = np.array(self.mean(axis=0))
        delta = self.as_pyemu_matrix(typ=Cov)
        for i in range(self.shape[0]):
            delta.x[i, :] -= mean
        delta *= (1.0 / np.sqrt(float(self.shape[0] - 1.0)))

        if localizer is not None:
            delta = delta.T * delta
            return delta.hadamard_product(localizer)

        return delta.T * delta



class ObservationEnsemble(Ensemble):
    """ Ensemble derived type for observations.  This class is primarily used to
    generate realizations of observation noise.  These are typically generated from
    the weights listed in the control file.  However, a general covariance matrix can
    be explicitly used.

    Note:
        Does not generate noise realizations for observations with zero weight
    """

    def __init__(self,pst,**kwargs):
        """ObservationEnsemble constructor.

        Parameters
        ----------
        pst : pyemu.Pst
            required Ensemble constructor kwwargs
            'columns' and 'mean_values' are generated from pst.observation_data.obsnme
            and pst.observation_data.obsval resepctively.

        **kwargs : dict
            keyword args to pass to Ensemble constructor

        Returns
        -------
        ObservationEnsemble : ObservationEnsemble

        """
        kwargs["columns"] = pst.observation_data.obsnme
        kwargs["mean_values"] = pst.observation_data.obsval
        super(ObservationEnsemble,self).__init__(**kwargs)
        self.pst = pst
        self.pst.observation_data.index = self.pst.observation_data.obsnme

    def copy(self):
        """overload of Ensemble.copy()

        Returns
        -------
        ObservationEnsemble : ObservationEnsemble

        """
        df = super(Ensemble,self).copy()
        return type(self).from_dataframe(df=df,pst=self.pst.get())

    @property
    def names(self):
        """property decorated method to get current non-zero weighted
        column names.  Uses ObservationEnsemble.pst.nnz_obs_names

        Returns
        -------
        list : list
            non-zero weight observation names
        """
        return self.pst.nnz_obs_names


    @property
    def mean_values(self):
        """ property decorated method to get mean values of observation noise.
        This is a zero-valued pandas.Series

        Returns
        -------
        mean_values : pandas Series

        """
        vals = self.pst.observation_data.obsval.copy()
        vals.loc[self.names] = 0.0
        return vals


    def draw(self,cov,num_reals):
        """ draw realizations of observation noise and add to mean_values
        Note: only draws noise realizations for non-zero weighted observations
        zero-weighted observations are set to mean value for all realizations

        Parameters
        ----------
        cov : pyemu.Cov
            covariance matrix that describes the support volume around the
            mean values.
        num_reals : int
            number of realizations to draw

        """
        super(ObservationEnsemble,self).draw(cov,num_reals,
                                             names=self.pst.nnz_obs_names)
        self.loc[:,self.names] += self.pst.observation_data.obsval

    @property
    def nonzero(self):
        """ property decorated method to get a new ObservationEnsemble
        of only non-zero weighted observations

        Returns
        -------
        ObservationEnsemble : ObservationEnsemble

        """
        df = self.loc[:,self.pst.nnz_obs_names]
        return ObservationEnsemble.from_dataframe(df=df,
                        pst=self.pst.get(obs_names=self.pst.nnz_obs_names))

    @classmethod
    def from_id_gaussian_draw(cls,pst,num_reals):
        """ this is an experiemental method to help speed up independent draws
        for a really large (>1E6) ensemble sizes.

        Parameters
        ----------
        pst : pyemu.Pst
            a control file instance
        num_reals : int
            number of realizations to draw

        Returns
        -------
            ObservationEnsemble : ObservationEnsemble

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
        new_oe = cls.from_dataframe(pst=pst,df=df)
        return new_oe

    def to_binary(self, filename):
        """write the observation ensemble to a jco-style binary file.  The
        ensemble is transposed in the binary file so that the 20-char obs
        names are carried

        Parameters
        ----------
        filename : str
            the filename to write

        Returns
        -------
        None


        Note
        ----
        The ensemble is transposed in the binary file

        """
        self.as_pyemu_matrix().T.to_binary(filename)


    @classmethod
    def from_binary(cls,pst,filename):
        """instantiate an observation obsemble from a jco-type file

        Parameters
        ----------
        pst : pyemu.Pst
            a Pst instance
        filename : str
            the binary file name

        Returns
        -------
        oe : ObservationEnsemble

        """
        m = Matrix.from_binary(filename)
        return ObservationEnsemble(data=m.T.x,pst=pst)


    @property
    def phi_vector(self):
        """property decorated method to get a vector of L2 norm (phi)
        for the realizations.  The ObservationEnsemble.pst.weights can be
        updated prior to calling this method to evaluate new weighting strategies

        Return
        ------
        pandas.DataFrame : pandas.DataFrame

        """
        weights = self.pst.observation_data.loc[self.names,"weight"]
        obsval = self.pst.observation_data.loc[self.names,"obsval"]
        phi_vec = []
        for idx in self.index.values:
            simval = self.loc[idx,self.names]
            phi = (((simval - obsval) * weights)**2).sum()
            phi_vec.append(phi)
        #return pd.DataFrame({"phi":phi_vec},index=self.index)
        return pd.Series(data=phi_vec,index=self.index)


    def add_base(self):
        """ add "base" control file values as a realization

        """
        if "base" in self.index:
            raise Exception("'base' already in index")
        self.loc["base",:] = self.pst.observation_data.loc[self.columns,"obsval"]


class ParameterEnsemble(Ensemble):
    """ Ensemble derived type for parameters
        implements bounds enforcement, log10 transformation,
        fixed parameters and null-space projection
        Note: uses the parnme attribute of Pst.parameter_data from column names
        and uses the parval1 attribute of Pst.parameter_data as mean values

    Parameters
    ----------
    pst : pyemu.Pst
        The 'columns' and 'mean_values' args need for Ensemble
        are derived from the pst.parameter_data.parnme and pst.parameter_data.parval1
        items, respectively
    istransformed : bool
        flag indicating the transformation status (log10) of the arguments be passed
    **kwargs : dict
        keyword arguments to pass to Ensemble constructor.
    bound_tol : float
        fractional amount to reset bounds transgression within the bound.
        This has been shown to be very useful for the subsequent recalibration
        because it moves parameters off of their bounds, so they are not treated as frozen in
        the upgrade calculations. defaults to 0.0

    Returns
    -------
    ParameterEnsemble : ParameterEnsemble

    """

    def __init__(self,pst,istransformed=False,**kwargs):
        """ ParameterEnsemble constructor.


        """
        kwargs["columns"] = pst.parameter_data.parnme
        kwargs["mean_values"] = pst.parameter_data.parval1

        super(ParameterEnsemble,self).__init__(**kwargs)
        # a flag for current log transform status
        self.__istransformed = bool(istransformed)
        self.pst = pst
        if "tied" in list(self.pst.parameter_data.partrans.values):
            #raise NotImplementedError("ParameterEnsemble does not " +\
            #                          "support tied parameters")
            import warnings
            warnings.warn("tied parameters are treated as fixed in "+\
                         "ParameterEnsemble")
        self.pst.parameter_data.index = self.pst.parameter_data.parnme
        self.bound_tol = kwargs.get("bound_tol",0.0)

    def dropna(self, *args, **kwargs):
        """overload of pandas.DataFrame.dropna()

        Parameters
        ----------
        *args : list
            positional args to pass to pandas.DataFrame.dropna()
        **kwargs : dict
            keyword args to pass to pandas.DataFrame.dropna()

        Returns
        -------
        Ensemble : Ensemble

        """
        df = super(Ensemble, self).dropna(*args, **kwargs)
        if df is not None:
            pe = ParameterEnsemble.from_dataframe(df=df,pst=self.pst)
            pe.__istransformed = self.istransformed
            return pe

    def copy(self):
        """ overload of Ensemble.copy()

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble

        """
        df = super(Ensemble,self).copy()
        pe = ParameterEnsemble.from_dataframe(df=df,pst=self.pst.get())
        pe.__istransformed = self.istransformed
        return pe

    @property
    def istransformed(self):
        """property decorated method to get the current
        transformation status of the ParameterEnsemble

        Returns
        -------
        istransformed : bool

        """
        return copy.copy(self.__istransformed)

    @property
    def mean_values(self):
        """ the mean value vector while respecting log transform

        Returns
        -------
        mean_values : pandas.Series

        """
        if not self.istransformed:
            return self.pst.parameter_data.parval1.copy()
        else:
            # vals = (self.pst.parameter_data.parval1 *
            #         self.pst.parameter_data.scale) +\
            #         self.pst.parameter_data.offset
            vals = self.pst.parameter_data.parval1.copy()
            vals[self.log_indexer] = np.log10(vals[self.log_indexer])
            return vals

    @property
    def names(self):
        """ Get the names of the parameters in the ParameterEnsemble

        Returns
        -------
        list : list
            parameter names

        """
        return list(self.pst.parameter_data.parnme)

    @property
    def adj_names(self):
        """ Get the names of adjustable parameters in the ParameterEnsemble

        Returns
        -------
        list : list
            adjustable parameter names

        """
        return list(self.pst.parameter_data.parnme.loc[~self.fixed_indexer])

    @property
    def ubnd(self):
        """ the upper bound vector while respecting log transform

        Returns
        -------
        ubnd : pandas.Series

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

        Returns
        -------
        lbnd : pandas.Series

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

        Returns
        -------
        log_indexer : pandas.Series

        """
        istransformed = self.pst.parameter_data.partrans == "log"
        return istransformed.values

    @property
    def fixed_indexer(self):
        """ indexer for fixed status

        Returns
        -------
        fixed_indexer : pandas.Series

        """
        #isfixed = self.pst.parameter_data.partrans == "fixed"
        isfixed = self.pst.parameter_data.partrans.\
            apply(lambda x : x in ["fixed","tied"])
        return isfixed.values



    def draw(self,cov,num_reals=1,how="normal",enforce_bounds=None):
        """draw realizations of parameter values

        Parameters
        ----------
        cov : pyemu.Cov
            covariance matrix that describes the support around
            the mean parameter values
        num_reals : int
            number of realizations to generate
        how : str
            distribution to use to generate realizations.  Options are
            'normal' or 'uniform'.  Default is 'normal'.  If 'uniform',
            cov argument is ignored
        enforce_bounds : str
            how to enforce parameter bound violations.  Options are
            'reset' (reset individual violating values), 'drop' (drop realizations
            that have one or more violating values.  Default is None (no bounds enforcement)

        """
        how = how.lower().strip()
        if not self.istransformed:
                self._transform()
        if how == "uniform":
            self._draw_uniform(num_reals=num_reals)
        else:
            super(ParameterEnsemble,self).draw(cov,num_reals=num_reals)
            # replace the realizations for fixed parameters with the original
            # parval1 in the control file
            self.pst.parameter_data.index = self.pst.parameter_data.parnme
            fixed_vals = self.pst.parameter_data.loc[self.fixed_indexer,"parval1"]
            for fname,fval in zip(fixed_vals.index,fixed_vals.values):
                #if fname not in self.columns:
                #    continue
                self.loc[:,fname] = fval
        istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
        self.loc[:,istransformed] = 10.0**self.loc[:,istransformed]
        self.__istransformed = False

        #self._applied_tied()


        self.enforce(enforce_bounds)

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
        """ this is an experiemental method to help speed up uniform draws
        for a really large (>1E6) ensemble sizes.  WARNING: this constructor
        transforms the pe argument

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
        for i,pname in enumerate(pst.parameter_data.parnme):
            print(pname,lb[pname],ub[pname])
            if pname in pst.adj_par_names:
                arr[:,i] = np.random.uniform(lb[pname],
                                                      ub[pname],
                                                      size=num_reals)
            else:
                arr[:,i] = np.zeros((num_reals)) + \
                                    pe.pst.parameter_data.\
                                         loc[pname,"parval1"]
        print("back transforming")

        df = pd.DataFrame(arr,index=real_names,columns=pst.par_names)
        df.loc[:,li] = 10.0**df.loc[:,li]

        new_pe = cls.from_dataframe(pst=pst,df=pd.DataFrame(data=arr,columns=pst.par_names))
        #new_pe._applied_tied()
        return new_pe


    @classmethod
    def from_sparse_gaussian_draw(cls,pst,cov,num_reals):
        """ instantiate a parameter ensemble from a sparse covariance matrix.
        This is an advanced user method that assumes you know what you are doing
        - few guard rails...

        Parameters
        ----------
        pst : pyemu.Pst
            a control file instance
        cov : (pyemu.SparseMatrix)
            sparse covariance matrix to use for drawing
        num_reals : int
            number of realizations to generate

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble

        """


        assert isinstance(cov,SparseMatrix)
        real_names = np.arange(num_reals, dtype=np.int64)

        li = pst.parameter_data.partrans == "log"
        vals = pst.parameter_data.parval1.copy()
        vals.loc[li] = vals.loc[li].apply(np.log10)


        par_cov = pst.parameter_data.loc[cov.row_names, :]
        par_cov.loc[:, "idxs"] = np.arange(cov.shape[0])
        # print("algning cov")
        # cov.align(list(par_cov.parnme))
        pargps = par_cov.pargp.unique()
        print("reserving reals matrix")
        reals = np.zeros((num_reals, cov.shape[0]))

        for ipg, pargp in enumerate(pargps):
            pnames = list(par_cov.loc[par_cov.pargp == pargp, "parnme"])
            idxs = par_cov.loc[par_cov.pargp == pargp, "idxs"]
            print("{0} of {1} drawing for par group '{2}' with {3} pars "
                  .format(ipg + 1, len(pargps), pargp, len(idxs)))

            snv = np.random.randn(num_reals, len(pnames))

            print("...extracting cov from sparse matrix")
            cov_pg = cov.get_matrix(col_names=pnames,row_names=pnames)
            if len(pnames) == 1:
                std = np.sqrt(cov_pg.x)
                reals[:, idxs] = vals[pnames].values[0] + (snv * std)
            else:
                try:
                    cov_pg.inv
                except:
                    covname = "trouble_{0}.cov".format(pargp)
                    print('saving toubled cov matrix to {0}'.format(covname))
                    cov_pg.to_ascii(covname)
                    print(cov_pg.get_diagonal_vector())
                    raise Exception("error inverting cov for par group '{0}'," + \
                                    "saved trouble cov to {1}".
                                    format(pargp, covname))
                v, w = np.linalg.eigh(cov_pg.as_2d)
                # check for near zero eig values

                # vdiag = np.diag(v)
                for i in range(v.shape[0]):
                    if v[i] > 1.0e-10:
                        pass
                    else:
                        print("near zero eigen value found", v[i], \
                              "at index", i, " of ", v.shape[0])
                        v[i] = 0.0
                vsqrt = np.sqrt(v)
                vsqrt[i:] = 0.0
                v = np.diag(vsqrt)
                a = np.dot(w, v)
                pg_vals = vals[pnames]
                for i in range(num_reals):
                    # v = snv[i,:]
                    # p = np.dot(a,v)
                    reals[i, idxs] = pg_vals + np.dot(a, snv[i, :])

        df = pd.DataFrame(reals, columns=cov.row_names, index=real_names)
        df.loc[:, li] = 10.0 ** df.loc[:, li]

        # replace the realizations for fixed parameters with the original
        # parval1 in the control file
        print("handling fixed pars")
        # pe.pst.parameter_data.index = pe.pst.parameter_data.parnme
        par = pst.parameter_data
        fixed_vals = par.loc[par.partrans == "fixed", "parval1"]
        for fname, fval in zip(fixed_vals.index, fixed_vals.values):
            # print(fname)
            df.loc[:, fname] = fval

        # print("apply tied")
        new_pe = cls.from_dataframe(pst=pst, df=df)

        return new_pe

    @classmethod
    def from_gaussian_draw(cls,pst,cov,num_reals=1,use_homegrown=True,group_chunks=False):
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
            pass
        else:
            common_names = cov.row_names

        if cov.isdiagonal:
            print("making diagonal cov draws")
            print("building mean and std dicts")
            arr = np.zeros((num_reals,len(vals)))
            stds = {pname:std for pname,std in zip(common_names,np.sqrt(cov.x.flatten()))}
            means = {pname:val for pname,val in zip(common_names,vals)}
            print("numpy draw")
            arr = np.random.randn(num_reals,len(common_names))
            print("post-processing")
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
            print("build df")
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
                    print("reserving reals matrix")
                    reals = np.zeros((num_reals,cov.shape[0]))

                    for ipg,pargp in enumerate(pargps):
                        pnames = list(par_cov.loc[par_cov.pargp==pargp,"parnme"])
                        idxs = par_cov.loc[par_cov.pargp == pargp, "idxs"]
                        print("{0} of {1} drawing for par group '{2}' with {3} pars "
                              .format(ipg+1,len(pargps),pargp, len(idxs)))

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
                                print('saving toubled cov matrix to {0}'.format(covname))
                                cov_pg.to_ascii(covname)
                                print(cov_pg.get_diagonal_vector())
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

                    print("generating snv matrix")
                    snv = np.random.randn(num_reals, cov.shape[0])

                    print("eigen solve for full cov")
                    v, w = np.linalg.eigh(cov.as_2d)
                    #w, v, other = np.linalg.svd(cov.as_2d,full_matrices=True,compute_uv=True)

                    # form projection matrix
                    print("form projection")
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
                print("making full cov draws with numpy")
                df = pd.DataFrame(data=np.random.multivariate_normal(vals, cov.as_2d,num_reals),
                                  columns = common_names,index=real_names)
            #print(df.shape,cov.shape)


        df.loc[:,li] = 10.0**df.loc[:,li]

        # replace the realizations for fixed parameters with the original
        # parval1 in the control file
        print("handling fixed pars")
        #pe.pst.parameter_data.index = pe.pst.parameter_data.parnme
        par = pst.parameter_data
        fixed_vals = par.loc[par.partrans.apply(lambda x: x in ["fixed","tied"]),"parval1"]
        for fname,fval in zip(fixed_vals.index,fixed_vals.values):
            #print(fname)
            df.loc[:,fname] = fval

        #print("apply tied")
        new_pe = cls.from_dataframe(pst=pst,df=df)

        return new_pe

    @classmethod
    def from_binary(cls, pst, filename):
        """instantiate an parameter obsemble from a jco-type file

        Parameters
        ----------
        pst : pyemu.Pst
            a Pst instance
        filename : str
            the binary file name

        Returns
        -------
        pe : ParameterEnsemble

        """
        m = Matrix.from_binary(filename).to_dataframe()

        return ParameterEnsemble.from_dataframe(df=m, pst=pst)

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
            self.loc[:,istransformed] = 10.0**(self.loc[:,istransformed])
            self.loc[:,:] = (self.loc[:,:] -\
                             self.pst.parameter_data.offset)/\
                             self.pst.parameter_data.scale

            self.__istransformed = False
        else:
            vals = (self.pst.parameter_data.parval1 -\
                    self.pst.parameter_data.offset) /\
                    self.pst.parameter_data.scale
            new_en = ParameterEnsemble(pst=self.pst.get(),data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=vals,istransformed=False)
            new_en.loc[:,istransformed] = 10.0**(self.loc[:,istransformed])
            new_en.loc[:,:] = (new_en.loc[:,:] -\
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
            raise Exception("ParameterEnsemble already transformed")

        istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
        if inplace:
            #self.loc[:,istransformed] = np.log10(self.loc[:,istransformed])
            self.loc[:,:] = (self.loc[:,:] * self.pst.parameter_data.scale) +\
                             self.pst.parameter_data.offset
            self.loc[:,istransformed] = self.loc[:,istransformed].applymap(lambda x: math.log10(x))

            self.__istransformed = True
        else:
            vals = self.pst.parameter_data.parval1.copy()
            new_en = ParameterEnsemble(pst=self.pst.get(),data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=vals,istransformed=True)
            new_en.loc[:,:] = (new_en.loc[:,:] * self.pst.parameter_data.scale) +\
                             new_en.pst.parameter_data.offset
            new_en.loc[:,istransformed] = self.loc[:,istransformed].applymap(lambda x: math.log10(x))
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

        istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
        self.loc[:,istransformed] = self.loc[:,istransformed].applymap(lambda x: math.log10(x))
        self.__istransformed = True

        #make sure everything is cool WRT ordering
        common_names = get_common_elements(self.adj_names,
                                                 projection_matrix.row_names)
        base = self.mean_values.loc[common_names]
        projection_matrix = projection_matrix.get(common_names,common_names)

        if not inplace:
            new_en = ParameterEnsemble(pst=self.pst.get(),data=self.loc[:,:].copy(),
                                       columns=self.columns,
                                       mean_values=self.mean_values.copy(),
                                       istransformed=self.istransformed)

        for real in self.index:
            if log is not None:
                log("projecting realization {0}".format(real))

            # null space projection of difference vector
            pdiff = self.loc[real,common_names] - base
            pdiff = np.dot(projection_matrix.x,
                           (self.loc[real,common_names] - base)\
                           .as_matrix())

            if inplace:
                self.loc[real,common_names] = base + pdiff
            else:
                new_en.loc[real,common_names] = base + pdiff

            if log is not None:
                log("projecting realization {0}".format(real))
        if not inplace:
            new_en.enforce(enforce_bounds)
            new_en.loc[:,istransformed] = 10.0**new_en.loc[:,istransformed]
            new_en.__istransformed = False

            #new_en._back_transform()
            return new_en

        self.enforce(enforce_bounds)
        self.loc[:,istransformed] = 10.0**self.loc[:,istransformed]
        self.__istransformed = False

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
                          "...resetting to None.")
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
        # ub = self.ubnd
        # lb = self.lbnd
        # for id in self.index:
        #     mx_diff = (self.loc[id,:] - ub) / ub
        #     mn_diff = (lb - self.loc[id,:]) / lb
        #
        #     # if this real has a violation
        #     mx = max(mx_diff.max(),mn_diff.max())
        #     if mx > 1.0:
        #         scale_factor = 1.0 / mx
        #         self.loc[id,:] *= scale_factor
        #
        #     mx = ub - self.loc[id,:]
        #     mn = lb - self.loc[id,:]
        #     print(mx.loc[mx<0.0])
        #     print(mn.loc[mn>0.0])
        #     if (ub - self.loc[id,:]).min() < 0.0 or\
        #                     (lb - self.loc[id,:]).max() > 0.0:
        #         raise Exception()

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
            if (ub - self.loc[id,:]).min() < 0.0 or\
                            (lb - self.loc[id,:]).max() > 0.0:
                drop.append(id)
        self.loc[drop,:] = np.NaN
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

        val_arr = self.values
        for iname, name in enumerate(self.columns):
            val_arr[val_arr[:,iname] > ub[name],iname] = ub[name]
            val_arr[val_arr[:, iname] < lb[name],iname] = lb[name]


    def read_parfiles_prefix(self,prefix):
        """ thin wrapper around read_parfiles using the pnulpar prefix concept.  Used to
        fill ParameterEnsemble from PEST-type par files

        Parameters
        ----------
        prefix : str
            the par file prefix

        """
        raise Exception("ParameterEnsemble.read_parfiles_prefix() is deprecated.  Use ParameterEnsemble.from_parfiles()")

        # pfile_count = 1
        # parfile_names = []
        # while True:
        #     pfile_name = prefix +"{0:d}.par".format(pfile_count)
        #     if not os.path.exists(pfile_name):
        #         break
        #     parfile_names.append(pfile_name)
        #     pfile_count += 1
        #
        # if len(parfile_names) == 0:
        #     raise Exception("ParameterEnsemble.read_parfiles_prefix() error: " + \
        #                     "no parfiles found with prefix {0}".format(prefix))
        #
        # return self.read_parfiles(parfile_names)


    def read_parfiles(self,parfile_names):
        """ read the ParameterEnsemble realizations from par files.  Used to fill
        the ParameterEnsemble with realizations from PEST-type par files

        Parameters
        ----------
        parfile_names: list
            list of par files to load

        Note
        ----
        log transforms after loading according and possibly resets
        self.__istransformed

        """
        raise Exception("ParameterEnsemble.read_parfiles() is deprecated.  Use ParameterEnsemble.from_parfiles()")
        # for pfile in parfile_names:
        #     assert os.path.exists(pfile),"ParameterEnsemble.read_parfiles() error: " +\
        #                                  "file: {0} not found".format(pfile)
        #     df = read_parfile(pfile)
        #     self.loc[pfile] = df.loc[:,'parval1']
        # self.loc[:,:] = self.loc[:,:].astype(np.float64)

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
                warnings.warn("differences in scale detected, applying scale in par file")
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
                             format(','.join(diff)))
                blank_df = pd.DataFrame(index=df_all.index,columns=diff)

                df_all = pd.concat([df_all,blank_df],axis=1)

            diff = dset.difference(pset)
            if len(diff) > 0:
                warnings.warn("the following par file parameters are not in the control (being dropped):{0}".
                              format(','.join(diff)))
                df_all = df_all.loc[:, pst.par_names]

        return ParameterEnsemble.from_dataframe(df=df_all,pst=pst)


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
            warnings.warn("NaN in par ensemble")
        super(ParameterEnsemble,self).to_csv(*args,**kwargs)
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
        self.as_pyemu_matrix().to_binary(filename)
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

        if self.istransformed:
            self._back_transform(inplace=True)

        par_df = self.pst.parameter_data.loc[:,
                 ["parnme","parval1","scale","offset"]].copy()

        for real in self.index:
            par_file = "{0}{1}.par".format(prefix,real)
            par_df.loc[:,"parval1"] =self.loc[real,:]
            write_parfile(par_df,par_file)


    def add_base(self):
        """ add "base" control file values as a realization

        """
        if "base" in self.index:
            raise Exception("'base' already in index")
        self.loc["base",:] = self.pst.parameter_data.loc[self.columns,"parval1"]
        