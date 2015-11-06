from __future__ import print_function, division
import os
import copy
import math
import numpy as np
import pandas as pd

from pyemu.mat.mat_handler import get_common_elements
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


    @property
    def names(self):
        return list(self.mean_values.index)


    def draw(self,cov,num_reals=1):
        """ draw random realizations from a multivariate
            Gaussian distribution

        :param cov: a Cov instance
        :param num_reals: number of realizatios to generate
        :return: None
        """
        # set up some column names
        real_names = ["{0:d}".format(i+1)
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


    @staticmethod
    def test():
        raise NotImplementedError()



class ParameterEnsemble(Ensemble):
    """ Ensemble derived type for parameters
        implements bounds enforcement, log10 transformation,
        fixed parameters and null-space projection

    Parameters:
    ----------
        pst : pyemu.Pst instance
        todo: tied parameters

    Note:
    ----
        uses the parnme attribute of Pst.parameter_data from column names
        and uses the parval1 attribute of Pst.parameter_data as mean values

    """

    def __init__(self,pst,islog=False,**kwargs):
        kwargs["columns"] = pst.parameter_data.parnme
        kwargs["mean_values"] = pst.parameter_data.parval1


        super(ParameterEnsemble,self).__init__(**kwargs)
        # a flag for current log transform status
        self.__islog = bool(islog)
        self.pst = pst
        if "tied" in self.pst.parameter_data.partrans:
            raise NotImplementedError("ParameterEnsemble does not " +\
                                      "support tied parameters")
        self.pst.parameter_data.index = self.pst.parameter_data.parnme


    @property
    def islog(self):
        return copy.copy(self.__islog)

    @property
    def mean_values(self):
        """ the mean value vector while respecting log transform
        """
        vals = self.pst.parameter_data.parval1.copy()
        if self.islog:
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
        """ the lower bound vector while respecting log transform"""
        ub = self.pst.parameter_data.parubnd.copy()
        if self.islog:
            ub[self.log_indexer] = np.log10(ub[self.log_indexer])
        return ub

    @property
    def lbnd(self):
        """ the lower boubd vectir while respecting log transform"""
        lb = self.pst.parameter_data.parlbnd.copy()
        if self.islog:
            lb[self.log_indexer] = np.log10(lb[self.log_indexer])
        return lb

    @property
    def log_indexer(self):
        """ indexer for log transform"""
        islog = self.pst.parameter_data.partrans == "log"
        return islog.values

    @property
    def fixed_indexer(self):
        """ indexer for fixed status"""
        isfixed = self.pst.parameter_data.partrans == "fixed"
        return isfixed.values

    def draw(self,cov,num_reals=1):
        if not self.islog:
            self._transform()
        super(ParameterEnsemble,self).draw(cov,num_reals=num_reals)
        self._back_transform()

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
        if not self.islog:
            raise Exception("ParameterEnsemble already back transformed")

        islog = self.pst.parameter_data.loc[:,"partrans"] == "log"
        if inplace:
            self.loc[:,islog] = 10.0**(self.loc[:,islog])
            self.__islog = False
        else:
            vals = self.pst.parameter_data.parval1.copy()
            new_en = ParameterEnsemble(pst=self.pst.get(),data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=vals,islog=False)
            new_en.loc[:,islog] = 10.0**(self.loc[:,islog])
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
        if self.islog:
            raise Exception("ParameterEnsemble already transformed")

        islog = self.pst.parameter_data.loc[:,"partrans"] == "log"
        if inplace:
            #self.loc[:,islog] = np.log10(self.loc[:,islog])
            self.loc[:,islog] = self.loc[:,islog].applymap(lambda x: math.log10(x))

            self.__islog = True
        else:
            vals = self.pst.parameter_data.parval1.copy()
            new_en = ParameterEnsemble(pst=self.pst.get(),data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=vals,islog=True)
            new_en.loc[:,islog] = self.loc[:,islog].applymap(lambda x: math.log10(x))
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

        if not self.islog:
            self._transform()

        #make sure everything is cool WRT ordering
        common_names = get_common_elements(self.adj_names,
                                                 projection_matrix.row_names)
        base = self.mean_values.loc[common_names]
        projection_matrix = projection_matrix.get(common_names,common_names)



        if not inplace:
            new_en = ParameterEnsemble(pst=self.pst.get(),data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=self.mean_values.copy(),islog=self.islog)

        for real in self.index:
            if log is not None:
                log("projecting realization " + real)


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
                new_en.loc[real,common_names] = base +  pdiff

            if log is not None:
                log("projecting realization " + real)
        if not inplace:
            if enforce:
                new_en.enforce()
            new_en._back_transform()
            return new_en

        if enforce:
            self.enforce()
        self._back_transform()

    def enforce(self):
        """ enforce parameter bounds on the ensemble

        """
        ub = self.ubnd
        lb = self.lbnd
        for iname,name in enumerate(self.columns):
            self.loc[self.loc[:,name] > ub[name],name] = ub[name].copy()
            #print(self.ubnd[name],self.loc[:,name])
            self.loc[self.loc[:,name] < lb[name],name] = lb[name].copy()
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
            log transforms after loading according and possibly resets self.__islog

        """
        par_dfs = []
        for pfile in parfile_names:
            assert os.path.exists(pfile),"ParameterEnsemble.read_parfiles() error: " +\
                                         "file: {0} not found".format(pfile)
            df = read_parfile(pfile)
            self.loc[pfile] = df.loc[:,'parval1']
        self.loc[:,:] = self.loc[:,:].astype(np.float64)
        #if self.islog:
        #    self.__islog = False
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

        if self.islog:
            self._back_transform(inplace=True)

        par_df = self.pst.parameter_data.loc[:,
                 ["parnme","parval1","scale","offset"]].copy()

        for real in self.index:
            par_file = prefix+real+".par"
            par_df.loc[:,"parval1"] =self.loc[real,:]
            write_parfile(par_df,par_file)