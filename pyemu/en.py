from __future__ import print_function, division
import os
import copy
import math
import numpy as np
import pandas as pd

from pyemu.mat.mat_handler import get_common_elements,Matrix
from pyemu.pst.pst_utils import write_parfile,read_parfile

SEED = 358183147 #from random.org on 5 Dec 2016
#print("setting random seed")
np.random.seed(SEED)

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
        x = self.copy().as_matrix().astype(np.float)
        return Matrix(x=x,row_names=list(self.index),
                      col_names=list(self.columns))

    def drop(self,arg):
        df = super(Ensemble,self).drop(arg)
        return type(self)(data=df,pst=self.pst)

    def dropna(self,*args,**kwargs):
        df = super(Ensemble,self).dropna(*args,**kwargs)
        return type(self)(data=df,pst=self.pst)

    def draw(self,cov,num_reals=1,names=None):
        """ draw random realizations from a multivariate
            Gaussian distribution

        Parameters:
        ----------
            cov: a Cov instance
                covariance structure to draw from
            num_reals: int
                number of realizations to generate
            names : list of names to draw for.  If None, values all names
                    are drawn
        Returns:
        -------
            None
        """
        # set up some realization names
        #real_names = ["{0:d}".format(i)
        #              for i in range(num_reals)]
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
            #for mu,std in zip(vals,np.sqrt(cov.x)):
            #    val_array.append(np.random.normal(mu,std,size=num_reals))
            #val_array = np.array(val_array).transpose()
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
        mean_values = kwargs.pop("mean_values",df.mean(axis=0))
        e = cls(data=df,index=df.index,columns=df.columns,
                mean_values=mean_values,**kwargs)
        return e

    def reseed(self):
         np.random.seed(SEED)

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
        super(ObservationEnsemble,self).draw(cov,num_reals,
                                             names=self.pst.nnz_obs_names)
        self.loc[:,self.names] += self.pst.observation_data.obsval

    @property
    def nonzero(self):
        df = self.loc[:,self.pst.nnz_obs_names]
        return ObservationEnsemble.from_dataframe(df=df,
                        pst=self.pst.get(obs_names=self.pst.nnz_obs_names))

    @classmethod
    def from_id_gaussian_draw(cls,oe,num_reals):
        """ this is an experiemental method to help speed up independent draws
        for a really large (>1E6) ensemble sizes.  WARNING: this constructor
        transforms the oe argument
        :param oe: ObservationEnsemble instance
        "param pst: Pst instance
        :param num_reals: number of realizations to generate
        :return: ObservationEnsemble
        """        # set up some column names
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
        weights = self.pst.observation_data.loc[self.names,"weight"]
        obsval = self.pst.observation_data.loc[self.names,"obsval"]
        phi_vec = []
        for idx in self.index.values:
            phi = ((self.loc[idx,self.names] - obsval)**2 * weights).sum()
            phi_vec.append(phi)
        return pd.DataFrame({"phi":phi_vec},index=self.index)

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
        if "tied" in list(self.pst.parameter_data.partrans.values):
            #raise NotImplementedError("ParameterEnsemble does not " +\
            #                          "support tied parameters")
            import warnings
            warnings.warn("tied parameters are treated as fixed in "+\
                         "ParameterEnsemble")
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
        #isfixed = self.pst.parameter_data.partrans == "fixed"
        isfixed = self.pst.parameter_data.partrans.\
            apply(lambda x : x in ["fixed","tied"])
        return isfixed.values


    # def plot(self,*args,**kwargs):
    #     if self.istransformed:
    #         self._back_transform(inplace=True)
    #     super(ParameterEnsemble,self).plot(*args,**kwargs)

    def draw(self,cov,num_reals=1,how="normal",enforce_bounds=None):
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

    # def _applied_tied(self):
    #     if self.pst is None:
    #         return
    #     if self.pst.tied is None:
    #         return
    #     ParameterEnsemble.apply_tied(self)

    # @staticmethod
    # def apply_tied(pe):
    #     if pe.pst is None:
    #         return
    #     if pe.pst.tied is None:
    #         return
    #     par = pe.pst.parameter_data
    #     tied =pe.pst.tied
    #     for pname,ptied in zip(tied.parnme,tied.partied):
    #         pval, tval = par.loc[pname,"parval1"],par.loc[ptied,"parval1"]
    #         tied_ratio = pval / tval
    #         if tval == 0.0:
    #             tied_ratio = pval
    #         #rvals = pe.loc[:,ptied]
    #         pe.loc[:,pname] = pe.loc[:,ptied] * tied_ratio
    #         #tvals = pe.loc[:,pname]
    #         #print(rvals/tvals)



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


    @classmethod
    def from_uniform_draw(cls,pe,num_reals):
        """ this is an experiemental method to help speed up uniform draws
        for a really large (>1E6) ensemble sizes.  WARNING: this constructor
        transforms the pe argument
        :param pe: ParameterEnsemble instance
        :param num_reals: number of realizations to generate
        :return: ParameterEnsemble
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
        dataframe expansion-by-loc that is one col at a time.  WARNING:
        this constructor transforms the pe argument!
        :param pe: ParameterEnsemble instance
        "param cov: Covariance instance
        :param num_reals: number of realizations to generate
        :return: ParameterEnsemble
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



    def project(self,projection_matrix,inplace=True,log=None,
                enforce_bounds="reset"):
        """ project the ensemble
        Parameters:
        ----------
            projection_matrix: (pyemu.Matrix) projection operator - must already respect log transform

            inplace: (boolean) project self or return a new ParameterEnsemble instance

            log: (pyemu.la.logger instance) for logging progress

            enforce_bounds: (str) parameter bound enforcement flag.  'drop' removes
             offending realizations, 'reset' resets offending values)

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
            pdiff = self.loc[real,common_names] - base
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
            new_en.enforce(enforce_bounds)
            new_en.loc[:,istransformed] = 10.0**new_en.loc[:,istransformed]
            new_en.__istransformed = False

            #new_en._back_transform()
            return new_en

        self.enforce(enforce_bounds)
        self.loc[:,istransformed] = 10.0**self.loc[:,istransformed]
        self.__istransformed = False

    def enforce(self,enforce_bounds="reset"):
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


    def to_csv(self,*args,**kwargs):
        if self.istransformed:
            self._back_transform(inplace=True)
        super(ParameterEnsemble,self).to_csv(*args,**kwargs)

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
            par_file = "{0}{1}.par".format(prefix,real)
            par_df.loc[:,"parval1"] =self.loc[real,:]
            write_parfile(par_df,par_file)