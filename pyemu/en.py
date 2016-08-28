from __future__ import print_function, division
import os
import copy
import math
import numpy as np
import pandas as pd

from pyemu.mat.mat_handler import get_common_elements,Matrix
from pyemu.pst.pst_utils import write_parfile,read_parfile

class Ensemble(pd.DataFrame):
    """ a pandas.DataFrame derived type to store
        ensembles of parameters and/or observations

        requires: columns and mean_values kwargs
    """
    def __init__(self,*args,**kwargs):

        assert "columns" in kwargs.keys(),"ensemble requires 'columns' kwarg"
        #assert "index" in kwargs.keys(),"ensemble requires 'index' kwarg"

        mean_values = kwargs.pop("mean_values",None)
        super(Ensemble,self).__init__(*args,**kwargs)

        if mean_values is None:
            raise Exception("Ensemble requires 'mean_values' kwarg")
        self.__mean_values = mean_values

    def as_pyemu_matrix(self):
        x = self.copy().as_matrix()
        return Matrix(x=x,row_names=list(self.index),
                      col_names=list(self.columns))

    def draw(self,cov,num_reals=1):
        """ draw random realizations from a multivariate
            Gaussian distribution

        Parameters:
        ----------
            cov: a Cov instance
                covariance structure to draw from
            num_reals: int
                number of realizations to generate
        Returns:
        -------
            None
        """
        # set up some column names
        real_names = ["{0:d}".format(i)
                      for i in range(num_reals)]

        # make sure everything is cool WRT ordering
        if self.names != cov.row_names:
            common_names = get_common_elements(self.names,
                                               cov.row_names)
            vals = self.mean_values.loc[common_names]
            cov = cov.get(common_names)
            pass
        else:
            vals = self.mean_values
            common_names = self.names

        # generate random numbers
        val_array = np.random.multivariate_normal(vals, cov.as_2d,num_reals)

        self.loc[:,:] = np.NaN
        self.dropna(inplace=True)

        # this sucks - can only set by enlargement one row at a time
        for rname,vals in zip(real_names,val_array):
            self.loc[rname,common_names] = vals
            # set NaNs to mean_values
            idx = pd.isnull(self.loc[rname,:])
            self.loc[rname,idx] = self.mean_values[idx]

    def enforce(self):
        raise Exception("Ensemble.enforce() must overloaded by derived types")

    def plot(self,*args,**kwargs):
        if "marginals" in kwargs.keys():
            raise NotImplementedError()
        else:
            super(self,pd.DataFrame).plot(*args,**kwargs)


    def __sub__(self,other):
        diff = super(Ensemble,self).__sub__(other)
        return Ensemble.from_dataframe(df=diff)

    @classmethod
    def from_dataframe(cls,**kwargs):
        df = kwargs.pop("df")
        assert isinstance(df,pd.DataFrame)
        mean_values = kwargs.pop("mean_values",df.mean(axis=1))
        e = cls(data=df,index=df.index,columns=df.columns,
                mean_values=mean_values,**kwargs)
        return e

    def copy(self):
        df = super(Ensemble,self).copy()
        return type(self).from_dataframe(df=df)

class ObservationEnsemble(Ensemble):
    """ Ensemble derived type observation noise

        Parameters:
        ----------
            pst : Pst instance

        Note:
        ----
            Does not generate realizations for observations with zero weight
            uses the obsnme attribute of Pst.observation_data from column names
    """

    def __init__(self,pst,**kwargs):
        kwargs["columns"] = pst.observation_data.obsnme
        kwargs["mean_values"] = pst.observation_data.obsval
        super(ObservationEnsemble,self).__init__(**kwargs)
        self.pst = pst
        self.pst.observation_data.index = self.pst.observation_data.obsnme

    def copy(self):
        df = super(Ensemble,self).copy()
        return type(self).from_dataframe(df=df,pst=self.pst.get())

    @property
    def names(self):
        return self.pst.nnz_obs_names


    @property
    def mean_values(self):
        """ zeros
        """
        vals = self.pst.observation_data.obsval.copy()
        vals.loc[self.names] = 0.0
        return vals


    def draw(self,cov,num_reals):
        super(ObservationEnsemble,self).draw(cov,num_reals)
        self.loc[:,self.names] += self.pst.observation_data.obsval


class ParameterEnsemble(Ensemble):
    """ Ensemble derived type for parameters
        implements bounds enforcement, log10 transformation,
        fixed parameters and null-space projection

    Parameters:
    ----------
        pst : pyemu.Pst instance
        bound_tol : float
            fractional amount to reset bounds transgression within the bound.
            This has been shown to be very useful for the subsequent recalibration
            because it moves parameters off of their bounds, so they are not treated as frozen in
            the upgrade calculations. defaults to 0.0

    Note:
    ----
        uses the parnme attribute of Pst.parameter_data from column names
        and uses the parval1 attribute of Pst.parameter_data as mean values

    """

    def __init__(self,pst,istransformed=False,**kwargs):
        kwargs["columns"] = pst.parameter_data.parnme
        kwargs["mean_values"] = pst.parameter_data.parval1

        super(ParameterEnsemble,self).__init__(**kwargs)
        # a flag for current log transform status
        self.__istransformed = bool(istransformed)
        self.pst = pst
        if "tied" in self.pst.parameter_data.partrans:
            raise NotImplementedError("ParameterEnsemble does not " +\
                                      "support tied parameters")
        self.pst.parameter_data.index = self.pst.parameter_data.parnme
        self.bound_tol = kwargs.get("bound_tol",0.0)

    def copy(self):
        df = super(Ensemble,self).copy()
        pe = ParameterEnsemble.from_dataframe(df=df,pst=self.pst.get())
        pe.__istransformed = self.istransformed
        return pe

    @property
    def istransformed(self):
        return copy.copy(self.__istransformed)

    @property
    def mean_values(self):
        """ the mean value vector while respecting log transform
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
       return list(self.pst.parameter_data.parnme)

    @property
    def adj_names(self):
        return list(self.pst.parameter_data.parnme.loc[~self.fixed_indexer])

    @property
    def ubnd(self):
        """ the upper bound vector while respecting log transform"""
        if not self.istransformed:
            return self.pst.parameter_data.parubnd.copy()
        else:
            #ub = (self.pst.parameter_data.parubnd *
            #      self.pst.parameter_data.scale) +\
            #      self.pst.parameter_data.offset
            ub = self.pst.parameter_data.parubnd.copy()
            ub[self.log_indexer] = np.log10(ub[self.log_indexer])
            return ub

    @property
    def lbnd(self):
        """ the lower bound vector while respecting log transform"""
        if not self.istransformed:
            return self.pst.parameter_data.parlbnd.copy()
        else:
            #lb = (self.pst.parameter_data.parlbnd * \
            #      self.pst.parameter_data.scale) + \
            #      self.pst.parameter_data.offset
            lb = self.pst.parameter_data.parlbnd.copy()
            lb[self.log_indexer] = np.log10(lb[self.log_indexer])
            return lb

    @property
    def log_indexer(self):
        """ indexer for log transform"""
        istransformed = self.pst.parameter_data.partrans == "log"
        return istransformed.values

    @property
    def fixed_indexer(self):
        """ indexer for fixed status"""
        isfixed = self.pst.parameter_data.partrans == "fixed"
        return isfixed.values


    # def plot(self,*args,**kwargs):
    #     if self.istransformed:
    #         self._back_transform(inplace=True)
    #     super(ParameterEnsemble,self).plot(*args,**kwargs)

    def draw(self,cov,num_reals=1,how="normal"):
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
                self.loc[:,fname] = fval
        istransformed = self.pst.parameter_data.loc[:,"partrans"] == "log"
        self.loc[:,istransformed] = 10.0**self.loc[:,istransformed]
        self.__istransformed = False

    def _draw_uniform(self,num_reals=1):
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



    def _back_transform(self,inplace=True):
        """ remove log10 transformation from ensemble
        Parameters:
        ----------
            inplace: (boolean) back transform self in place
        Returns:
        -------
            if not inplace, ParameterEnsemble, otherwise None

        Note:
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
        """ perform log10 transformation for ensemble
        Parameters:
        ----------
            inplace: (boolean) transform self in place
        Returns:
        -------
            if not inplace, ParameterEnsemble, otherwise None

        Note:
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



    def project(self,projection_matrix,inplace=True,log=None,enforce=True):
        """ project the ensemble
        Parameters:
        ----------
            projection_matrix: (pyemu.Matrix) projection operator - must already respect log transform

            inplace: (boolean) project self or return a new ParameterEnsemble instance

            log: (pyemu.la.logger instance) for logging progress

            enforce: (bool) parameter bound enforcement flag (True)

        Returns:
        -------
            if not inplace, ParameterEnsemble, otherwise None

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
            pdiff = np.dot(projection_matrix.x,
                           (self.loc[real,common_names] - base)\
                           .as_matrix())

            # lb_fac = np.abs(pdiff)/((base+pdiff)-self.lbnd)
            # lb_fac[pdiff>0.0] = 1.0
            #
            # ub_fac = np.abs(pdiff)/(self.ubnd-(base+pdiff))
            # ub_fac[pdiff<=0.0] = 1.0
            #
            # factor = max(lb_fac.max(),
            #              ub_fac.max())

            if inplace:
                self.loc[real,common_names] = base + pdiff
            else:
                new_en.loc[real,common_names] = base + pdiff

            if log is not None:
                log("projecting realization {0}".format(real))
        if not inplace:
            if enforce:
                new_en.enforce()
            new_en.loc[:,istransformed] = 10.0**new_en.loc[:,istransformed]
            new_en.__istransformed = False

            #new_en._back_transform()
            return new_en

        if enforce:
            self.enforce()
        self.loc[:,istransformed] = 10.0**self.loc[:,istransformed]
        self.__istransformed = False

    def enforce(self):
        """ enforce parameter bounds on the ensemble

        """
        ub = self.ubnd
        lb = self.lbnd
        for iname,name in enumerate(self.columns):
            self.loc[self.loc[:,name] > ub[name],name] = ub[name].copy() * (1.0 + self.bound_tol)
            #print(self.ubnd[name],self.loc[:,name])
            self.loc[self.loc[:,name] < lb[name],name] = lb[name].copy() * (1.0 - self.bound_tol)
            #print(self.lbnd[name],self.loc[:,name])


    def read_parfiles_prefix(self,prefix):
        """ thin wrapper around read_parfiles using the pnulpar prefix concept

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
        """ read the ensemble from par files

        Parameters:
        ----------
            parfile_names: (list[str]) list of par files to load

        Note:
        ----
            log transforms after loading according and possibly resets self.__istransformed

        """
        par_dfs = []
        for pfile in parfile_names:
            assert os.path.exists(pfile),"ParameterEnsemble.read_parfiles() error: " +\
                                         "file: {0} not found".format(pfile)
            df = read_parfile(pfile)
            self.loc[pfile] = df.loc[:,'parval1']
        self.loc[:,:] = self.loc[:,:].astype(np.float64)
        #if self.istransformed:
        #    self.__istransformed = False
        #self._transform(inplace=True)


    def to_parfiles(self,prefix):
        """
            write the parameter ensemble to pest-style parameter files

        Parameters:
        ----------
            prefix: (str) file prefix for par files

        Note:
        ----
            this function back-transforms before writing

        """

        if self.istransformed:
            self._back_transform(inplace=True)

        par_df = self.pst.parameter_data.loc[:,
                 ["parnme","parval1","scale","offset"]].copy()

        for real in self.index:
            par_file = prefix+real+".par"
            par_df.loc[:,"parval1"] =self.loc[real,:]
            write_parfile(par_df,par_file)