from __future__ import print_function, division
import numpy as np
import pandas as pd

from pyemu.mat.mat_handler import get_common_elements

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
        real_names = ["realization_{0:08d}".format(i)
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

        todo: tied parameters

    """

    def __init__(self,pst,**kwargs):
        kwargs["columns"] = pst.parameter_data.parnme
        kwargs["mean_values"] = pst.parameter_data.parval1
        super(ParameterEnsemble,self).__init__(**kwargs)
        self.pst = pst
        if "tied" in self.pst.parameter_data.partrans:
            raise NotImplementedError("ParameterEnsemble does not " +\
                                      "support tied parameters")
        self.pst.parameter_data.index = self.pst.parameter_data.parnme
        self.__mean_values = None

    @property
    def mean_values(self):
        if self.__mean_values is None:
            vals = self.pst.parameter_data.parval1.copy()
            vals[self.log_indexer] = np.log10(vals[self.log_indexer])
            self.__mean_values = vals
        return self.__mean_values

    @property
    def names(self):
       return list(self.pst.parameter_data.parnme)

    @property
    def adj_names(self):
        return list(self.pst.parameter_data.parnme.loc[~self.fixed_indexer])

    @property
    def ubnd(self):
        if self.__ubnd is None:
            ub = self.pst.parameter_data.parubnd.copy()
            ub[self.log_indexer] = np.log10(ub[self.log_indexer])
            self.__ubnd = ub
        return self.__ubnd


    @property
    def lbnd(self):
        if self.__lbnd is None:
            lb = self.pst.parameter_data.parlbnd.copy()
            lb[self.log_indexer] = np.log10(lb[self.log_indexer])
            self.__lbnd = lb
        return self.__lbnd


    @property
    def log_indexer(self):
        islog = self.pst.parameter_data.partrans == "log"
        return islog.values


    @property
    def fixed_indexer(self):
        isfixed = self.pst.parameter_data.partrans == "fixed"
        return isfixed.values

    def back_transform(self,inplace=True):
        """ remove log10 transformation from ensemble
        :param inplace: back transform the self in place
        :return: if not inplace, ParameterEnsemble, otherwise None
        """
        islog = self.pst.parameter_data.loc[:,"partrans"] == "log"
        if inplace:
            self.loc[:,islog] = 10.0**(self.loc[:,islog])
        else:
            vals = self.pst.parameter_data.parval1.copy()
            new_en = Ensemble(data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=vals)
            new_en.loc[:,islog] = 10.0**(self.loc[:,islog])
            return new_en

    def project(self,projection_matrix,inplace=True,log=None):
        """ project the ensemble
        :param projection_matrix: Matrix instance
        :param inplace: operate on self
        :param log: a logger instance
        :return: if not inplace, ParameterEnsemble, otherwise None
        """
        # check that everything is cool WRT order
        if self.adj_names != projection_matrix.row_names:
            common_names = get_common_elements(self.adj_names,
                                                     projection_matrix.row_names)
            base = self.mean_values.loc[common_names]
            projection_matrix = projection_matrix.get(common_names,common_names)
        else:
            base = self.mean_values
            common_names = self.adj_names

        if not inplace:
            vals = self.pst.parameter_data.parval1.copy()
            new_en = Ensemble(data=self.loc[:,:].copy(),
                              columns=self.columns,
                              mean_values=vals)

        for real in self.index:
            if log is not None:
                log("projecting realization " + real)

            this = self.loc[real,common_names]
            pdiff = (this - base).as_matrix()
            if inplace:
                self.loc[real,common_names] = base + np.dot(projection_matrix.x,pdiff)
            else:
                new_en.loc[real,common_names] = base + np.dot(projection_matrix.x,pdiff)

            if log is not None:
                log("projecting realization " + real)
        if not inplace:
            return new_en

    def enforce(self):
        """ enforce parameter bounds on the ensemble
        :return: None
        """
        for name in self.columns:
            #print(self.loc[:,name])
            self.loc[self.loc[:,name] > self.ubnd[name],name] = self.ubnd[name]
            #print(self.ubnd[name],self.loc[:,name])
            self.loc[self.loc[:,name] < self.lbnd[name],name] = self.lbnd[name]
            #print(self.lbnd[name],self.loc[:,name])


    def read_parfiles(self,prefix):
        raise NotImplementedError()


    def to_parfiles(self,prefix):
        raise NotImplementedError()


