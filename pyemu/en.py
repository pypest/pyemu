from __future__ import print_function, division
import os
from datetime import datetime
import copy
import math
import numpy as np
import pandas as pd

from pyemu.mat.mat_handler import get_common_elements,Matrix,Cov
from pyemu.pst.pst_utils import write_parfile,read_parfile

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
        self.__mean_values = mean_values

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
        x = self.copy().as_matrix().astype(np.float)
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

    def plot(self,*args,**kwargs):
        """placeholder overload of pandas.DataFrame.plot()

        Parameters
        ----------
        *args : list
            positional args to pass to pandas.DataFrame.plot()
        **kwargs : dict
            keyword args to pass to pandas.DataFrame.plot()

        Returns
        -------
        pandas.DataFrame.plot() return

        """
        if "marginals" in kwargs.keys():
            raise NotImplementedError()
        else:
            super(self,pd.DataFrame).plot(*args,**kwargs)


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
        mean_values = kwargs.pop("mean_values",df.mean(axis=0))
        e = cls(data=df,index=df.index,columns=df.columns,
                mean_values=mean_values,**kwargs)
        return e

    def reseed(self):
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
    def from_id_gaussian_draw(cls,oe,num_reals):
        """ this is an experiemental method to help speed up independent draws
        for a really large (>1E6) ensemble sizes.  WARNING: this constructor
        transforms the oe argument

        Parameters
        ----------
        oe : ObservationEnsemble
            an existing ObservationEnsemble instance to use for 'mean_values' and
            'columns' arguments
        num_reals : int
            number of realizations to draw

        Returns
        -------
            ObservationEnsemble : ObservationEnsemble

        """
        # set up some column names
        real_names = np.arange(num_reals,dtype=np.int64)
        arr = np.empty((num_reals,len(oe.pst.obs_names)))
        obs = oe.pst.observation_data
        stds = {name:1.0/obs.loc[name,"weight"] for name in oe.pst.nnz_obs_names}
        for i,oname in enumerate(oe.pst.obs_names):
            if oname in oe.pst.nnz_obs_names:
                arr[:,i] = np.random.normal(0.0,stds[oname],size=num_reals)
            else:
                arr[:,i] = 0.0
        df = pd.DataFrame(arr,index=real_names,columns=oe.pst.obs_names)
        df.loc[:,oe.pst.obs_names] += oe.pst.observation_data.obsval
        new_oe = cls.from_dataframe(pst=oe.pst,df=df)
        return new_oe


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

    @classmethod
    def from_gaussian_draw_homegrown(cls, pe, cov, num_reals=1):
        """ this is an experiemental method to help speed up draws
        for a really large (>1E6) ensemble sizes.  gets around the
        dataframe expansion-by-loc that is one col at a time.  Implements
        multivariate normal draws to get around the 32-bit lapack limitations
        in scipy/numpy

        Parameters
        ----------
        pe : ParameterEnsemble
            existing ParameterEnsemble used to get information
            needed to call ParameterEnsemble constructor
        cov : (pyemu.Cov)
            covariance matrix to use for drawing
        num_reals : int
            number of realizations to generate

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble

        Note
        ----
        this constructor transforms the pe argument!
        """

        s = datetime.now()
        print("{0} - starting home-grown multivariate draws".format(s))


        # set up some column names
        # real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
        real_names = np.arange(num_reals, dtype=np.int64)

        if not pe.istransformed:
            pe._transform()
        # make sure everything is cool WRT ordering
        if pe.names != cov.row_names:
            common_names = get_common_elements(pe.names,
                                               cov.row_names)
            vals = pe.mean_values.loc[common_names]
            cov = cov.get(common_names)
            pass
        else:
            vals = pe.mean_values
            common_names = pe.names

        #generate standard normal vectors
        snv = np.random.randn(num_reals,cov.shape[0])

        #jwhite - 18-dec-17: the cholesky version is giving significantly diff
        #results compared to eigen solve, so turning this off for now - need to
        #learn more about this...
        use_chol = False
        if use_chol:
            a = np.linalg.cholesky(cov.as_2d)

        else:
            #decompose...
            v, w = np.linalg.eigh(cov.as_2d)

            #form projection matrix
            a = np.dot(w,np.diag(np.sqrt(v)))

        #project...
        reals = []
        for vec in snv:
            real = vals + np.dot(a,vec)
            reals.append(real)

        df = pd.DataFrame(reals,columns=common_names,index=real_names)
        istransformed = pe.pst.parameter_data.loc[common_names, "partrans"] == "log"
        #print("back transforming")
        df.loc[:, istransformed] = 10.0 ** df.loc[:, istransformed]

        # replace the realizations for fixed parameters with the original
        # parval1 in the control file
        #print("handling fixed pars")
        pe.pst.parameter_data.index = pe.pst.parameter_data.parnme
        fixed_vals = pe.pst.parameter_data.loc[pe.fixed_indexer, "parval1"]
        for fname, fval in zip(fixed_vals.index, fixed_vals.values):
            # if fname not in df.columns:
            #    continue
            #print(fname)
            df.loc[:, fname] = fval

        # print("apply tied")
        new_pe = cls.from_dataframe(pst=pe.pst, df=df)
        # ParameterEnsemble.apply_tied(new_pe)


        e = datetime.now()
        print("{0} - done...took {1}".format(e,(e-s).total_seconds()))

        return new_pe

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
    def from_uniform_draw(cls,pe,num_reals):
        """ this is an experiemental method to help speed up uniform draws
        for a really large (>1E6) ensemble sizes.  WARNING: this constructor
        transforms the pe argument

        Parameters
        ----------
        pe : ParameterEnsemble
            existing ParameterEnsemble used to get information
            needed to call ParameterEnsemble constructor
        num_reals : int
            number of realizations to generate

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble

        """
        if not pe.istransformed:
            pe._transform()
        ub = pe.ubnd
        lb = pe.lbnd
        # set up some column names
        #real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
        real_names = np.arange(num_reals,dtype=np.int64)
        arr = np.empty((num_reals,len(pe.names)))
        for i,pname in enumerate(pe.names):
            if pname in pe.adj_names:
                arr[:,i] = np.random.uniform(lb[pname],
                                                      ub[pname],
                                                      size=num_reals)
            else:
                arr[:,i] = np.zeros((num_reals)) + \
                                    pe.pst.parameter_data.\
                                         loc[pname,"parval1"]
        print("back transforming")
        istransformed = pe.pst.parameter_data.loc[:,"partrans"] == "log"
        df = pd.DataFrame(arr,index=real_names,columns=pe.pst.par_names)
        df.loc[:,istransformed] = 10.0**df.loc[:,istransformed]

        new_pe = cls.from_dataframe(pst=pe.pst,df=pd.DataFrame(data=arr,columns=pe.names))
        #new_pe._applied_tied()
        return new_pe

    @classmethod
    def from_gaussian_draw(cls,pe,cov,num_reals=1):
        """ this is an experiemental method to help speed up draws
        for a really large (>1E6) ensemble sizes.  gets around the
        dataframe expansion-by-loc that is one col at a time.

        Parameters
        ----------
        pe : ParameterEnsemble
            existing ParameterEnsemble used to get information
            needed to call ParameterEnsemble constructor
        cov : (pyemu.Cov)
            covariance matrix to use for drawing
        num_reals : int
            number of realizations to generate

        Returns
        -------
        ParameterEnsemble : ParameterEnsemble

        Note
        ----
        this constructor transforms the pe argument!

        """

        # set up some column names
        #real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
        real_names = np.arange(num_reals,dtype=np.int64)

        if not pe.istransformed:
            pe._transform()
        # make sure everything is cool WRT ordering
        if pe.names != cov.row_names:
            common_names = get_common_elements(pe.names,
                                               cov.row_names)
            vals = pe.mean_values.loc[common_names]
            cov = cov.get(common_names)
            pass
        else:
            vals = pe.mean_values
            common_names = pe.names

        if cov.isdiagonal:
            print("making diagonal cov draws")
            arr = np.zeros((num_reals,len(pe.names)))
            stds = {pname:std for pname,std in zip(common_names,np.sqrt(cov.x.flatten()))}
            means = {pname:val for pname,val in zip(common_names,vals)}
            for i,pname in enumerate(pe.names):
                if pname in pe.pst.adj_par_names:
                    s = stds[pname]
                    v = means[pname]
                    arr[:,i] = np.random.normal(means[pname],stds[pname],
                                                size=num_reals)
                else:
                    arr[:,i] = means[pname]

            df = pd.DataFrame(data=arr,columns=common_names,index=real_names)
        else:

            #vals = pe.mean_values
            print("making full cov draws")
            df = pd.DataFrame(data=np.random.multivariate_normal(vals, cov.as_2d,num_reals),
                              columns = common_names,index=real_names)
            #print(df.shape,cov.shape)

        istransformed = pe.pst.parameter_data.loc[common_names,"partrans"] == "log"
        print("back transforming")
        df.loc[:,istransformed] = 10.0**df.loc[:,istransformed]

        # replace the realizations for fixed parameters with the original
        # parval1 in the control file
        print("handling fixed pars")
        pe.pst.parameter_data.index = pe.pst.parameter_data.parnme
        fixed_vals = pe.pst.parameter_data.loc[pe.fixed_indexer,"parval1"]
        for fname,fval in zip(fixed_vals.index,fixed_vals.values):
            #if fname not in df.columns:
            #    continue
            print(fname)
            df.loc[:,fname] = fval

        #print("apply tied")
        new_pe = cls.from_dataframe(pst=pe.pst,df=df)
        #ParameterEnsemble.apply_tied(new_pe)
        return new_pe

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
        ub = self.ubnd
        lb = self.lbnd
        for id in self.index:
            mx_diff = (self.loc[id,:] - ub) / ub
            mn_diff = (lb - self.loc[id,:]) / lb

            # if this real has a violation
            mx = max(mx_diff.max(),mn_diff.max())
            if mx > 1.0:
                scale_factor = 1.0 / mx
                self.loc[id,:] *= scale_factor

            mx = ub - self.loc[id,:]
            mn = lb - self.loc[id,:]
            print(mx.loc[mx<0.0])
            print(mn.loc[mn>0.0])
            if (ub - self.loc[id,:]).min() < 0.0 or\
                            (lb - self.loc[id,:]).max() > 0.0:
                raise Exception()

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

        ub = self.ubnd
        lb = self.lbnd
        for iname,name in enumerate(self.columns):
            self.loc[self.loc[:,name] > ub[name],name] = ub[name].copy() * (1.0 + self.bound_tol)
            self.loc[self.loc[:,name] < lb[name],name] = lb[name].copy() * (1.0 - self.bound_tol)

    def read_parfiles_prefix(self,prefix):
        """ thin wrapper around read_parfiles using the pnulpar prefix concept.  Used to
        fill ParameterEnsemble from PEST-type par files

        Parameters
        ----------
        prefix : str
            the par file prefix

        """
        pfile_count = 1
        parfile_names = []
        while True:
            pfile_name = prefix +"{0:d}.par".format(pfile_count)
            if not os.path.exists(pfile_name):
                break
            parfile_names.append(pfile_name)
            pfile_count += 1

        if len(parfile_names) == 0:
            raise Exception("ParameterEnsemble.read_parfiles_prefix() error: " + \
                            "no parfiles found with prefix {0}".format(prefix))

        return self.read_parfiles(parfile_names)


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
        for pfile in parfile_names:
            assert os.path.exists(pfile),"ParameterEnsemble.read_parfiles() error: " +\
                                         "file: {0} not found".format(pfile)
            df = read_parfile(pfile)
            self.loc[pfile] = df.loc[:,'parval1']
        self.loc[:,:] = self.loc[:,:].astype(np.float64)


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
        if self.istransformed:
            self._back_transform(inplace=True)
        super(ParameterEnsemble,self).to_csv(*args,**kwargs)

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