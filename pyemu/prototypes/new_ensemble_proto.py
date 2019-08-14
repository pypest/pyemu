import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyemu


class Loc(object):
    def __init__(self,ensemble):
        self._ensemble = ensemble

    def __getitem__(self,item):
        return type(self._ensemble)(self._ensemble.pst, 
            self._ensemble._df.loc[item],
            istransformed=self._ensemble.istransformed)

    def __setitem__(self,idx,value):
        self._ensemble._df.loc[idx] = value



class Iloc(object):
    def __init__(self,ensemble):
        self._ensemble = ensemble

    def __getitem__(self,item):
        return type(self._ensemble)(self._ensemble.pst, 
            self._ensemble._df.iloc[item],
            istransformed=self._ensemble.istransformed)

    def __setitem__(self,idx,value):
        self._ensemble._df.iloc[idx] = value


class Ensemble(object):
    def __init__(self,pst,df,istransformed=False):
        self._df = df
        self.pst = pst
        self._istransformed = istransformed
        self.loc = Loc(self)
        self.iloc = Iloc(self)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    def __sub__(self,other):
        try:
            return self._df - other
        except:
            return self._df - other._df

    def __mul__(self,other):
        try:
            return self._df * other
        except:
            return self._df * other._df

    def __add__(self,other):
        try:
            return self._df + other
        except:
            return self._df + other._df

    def __pow__(self, pow):
        return self._df ** pow



    def copy(self):
        return type(self)(pst=self.pst.get(),
                          df=self._df.copy(),
                          istransformed=self.istransformed)

    @property
    def istransformed(self):
        return copy.deepcopy(self._istransformed)

    def _transform(self):
        if self.transformed:
            return
        self._transformed = True
        return

    def _back_transform(self):
        if not self.transformed:
            return
        self._transformed = False
        return

    def __getattr__(self,item):
        if item == "loc":
            return self.loc[item]
        elif item == "iloc":
            return self.iloc[item]
        elif item in set(dir(self)):
            return getattr(self,item)
        elif item in set(dir(self._df)):
            lhs = self._df.__getattr__(item)
            if type(lhs) == type(self._df):
                return type(self)(pst=pst,df=lhs,istransformed=self.istransformed)
            else:
                return lhs
        else:
            raise AttributeError("Ensemble error: the following item was not" +\
                                 "found in Ensemble or DataFrame attributes:{0}".format(item))
        return

    def plot(self,*args,**kwargs):
        self._df.plot(*args,**kwargs)

    @classmethod
    def from_binary(cls, pst, filename):
        df = pyemu.Matrix.from_binary(filename).to_dataframe()
        return cls(pst=pst, df=df)

    @classmethod
    def from_csv(cls, pst, filename, *args, **kwargs):
        if "index_col" not in kwargs:
            kwargs["index_col"] = 0
        df = pd.read_csv(filename,*args,**kwargs)
        return cls(pst=pst, df=df)

    def to_csv(self,filename,*args,**kwargs):
        retrans = False
        if self.istransformed:
            self._back_transform()
            retrans = True
        if self.isnull().values.any():
            warnings.warn("NaN in ensemble",PyemuWarning)
        self._df.to_csv(filename,*args,**kwargs)
        if retrans:
            self._transform()

    def to_binary(self,filename):
        retrans = False
        if self.istransformed:
            self._back_transform()
            retrans = True
        if self.isnull().values.any():
            warnings.warn("NaN in ensemble",PyemuWarning)
        pyemu.Matrix.from_dataframe(self._df).to_coo(filename)
        if retrans:
            self._transform()


    @staticmethod
    def _gaussian_draw(cov,mean_values,num_reals,grouper=None):

        # make sure all cov names are found in mean_values
        cov_names = set(cov.row_names)
        mv_names = set(mean_values.index.values)
        missing = cov_names - mv_names
        if len(missing) > 0:
            raise Exception("Ensemble._gaussian_draw() error: the following cov names are not in "
                            "mean_values: {0}".format(','.join(missing)))
        if cov.isdiagonal:
            stds = {name: std for name, std in zip(cov.row_names, np.sqrt(cov.x.flatten()))}
            snv = np.random.randn(num_reals, mean_values.shape[0])
            reals = np.zeros_like(snv)
            for i, name in enumerate(mean_values.index):
                if name in cov_names:
                    reals[:, i] = (snv[:, i] * stds[name]) + mean_values.loc[name]
                else:
                   reals[:, i] = mean_values.loc[name]
        else:
            reals = np.zeros((num_reals, mean_values.shape[0]))
            for i,v in enumerate(mean_values.values):
               reals[:,i] = v
            cov_map = {n:i for n,i in zip(cov.row_names,np.arange(cov.shape[0]))}
            mv_map = {n: i for n, i in zip(mean_values.index, np.arange(mean_values.shape[0]))}
            if grouper is not None:

                for grp_name,names in grouper.items():
                    idxs = [mv_map[name] for name in names]
                    snv = np.random.randn(num_reals, len(names))
                    cov_grp = cov.get(names)
                    if len(names) == 1:
                        std = np.sqrt(cov_grp.x)
                        reals[:, idxs] = mean_values.loc[names].values[0] + (snv * std)
                    else:
                        try:
                            cov_grp.inv
                        except:
                            covname = "trouble_{0}.cov".format(grp_name)
                            cov_grp.to_ascii(covname)
                            raise Exception("error inverting cov for group '{0}'," + \
                                            "saved trouble cov to {1}".
                                            format(grp_name, covname))


                    a = Ensemble._get_eigen_projection_matrix(cov_grp.as_2d)

                    # process each realization
                    group_mean_values = mean_values.loc[names]
                    for i in range(num_reals):
                        reals[i, idxs] = group_mean_values + np.dot(a, snv[i, :])

            else:
                snv = np.random.randn(num_reals, cov.shape[0])
                a = Ensemble._get_eigen_projection_matrix(cov.as_2d)
                cov_mean_values = mean_values.loc[cov.row_names].values
                idxs = [mv_map[name] for name in cov.row_names]
                for i in range(num_reals):
                    reals[i, idxs] = cov_mean_values + np.dot(a, snv[i, :])

        df = pd.DataFrame(reals,columns=mean_values.index.values)
        return df

    @staticmethod
    def _get_eigen_projection_matrix(x):
        # eigen factorization
        v, w = np.linalg.eigh(x)

        # check for near zero eig values
        for i in range(v.shape[0]):
            if v[i] > 1.0e-10:
                pass
            else:
                print("near zero eigen value found", v[i], \
                      "at index", i, " of ", v.shape[0])
                v[i] = 0.0

        # form the projection matrix
        vsqrt = np.sqrt(v)
        vsqrt[i+1:] = 0.0
        v = np.diag(vsqrt)
        a = np.dot(w, v)

        return a

    @staticmethod
    def _uniform_draw(upper_bound,lower_bound,num_reals):
        pass

    @staticmethod
    def _triangular_draw(mean_values,upper_bound_lower_bound,num_reals):
        pass

    @classmethod
    def from_binary(cls, pst, filename):
        """instantiate an ensemble from a PEST-type binary file

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
        df = pyemu.Matrix.from_binary(filename).to_dataframe()
        return cls(pst=pst,df=df)


class ObservationEnsemble(Ensemble):
    def __init__(self,pst,df,istransformed=False):
        super(ObservationEnsemble,self).__init__(pst,df,istransformed)

    @classmethod
    def from_gaussian_draw(cls,pst,cov=None,num_reals=100,by_groups=True):
        if cov is None:
            cov = pyemu.Cov.from_observation_data(pst)
        obs = pst.observation_data
        mean_values = obs.obsval.copy()
        mean_values.loc[pst.nnz_obs_names] = 0.0
        # only draw for non-zero weights, get a new cov
        nz_cov = cov.get(pst.nnz_obs_names)

        grouper = None
        if not cov.isdiagonal and by_groups:
            nz_obs = obs.loc[pst.nnz_obs_names,:].copy()
            grouper = nz_obs.groupby("obgnme").groups
            for grp in grouper.keys():
                grouper[grp] = list(grouper[grp])
        df = Ensemble._gaussian_draw(cov=nz_cov,mean_values=mean_values,
                                     num_reals=num_reals,grouper=grouper)
        return cls(pst,df,istransformed=False)


class ParameterEnsemble(Ensemble):
    def __init__(self,pst,df,istransformed=False):
        super(ParameterEnsemble,self).__init__(pst,df,istransformed)

    @classmethod
    def from_gaussian_draw(cls,pst,cov=None,num_reals=100,by_groups=True):
        if cov is None:
            cov = pyemu.Cov.from_parameter_data(pst)
        par = pst.parameter_data
        li = par.partrans == "log"
        mean_values = par.parval1.copy()
        mean_values.loc[li] = mean_values.loc[li].apply(np.log10)
        grouper = None
        if not cov.isdiagonal and by_groups:
            adj_par = par.loc[pst.adj_par_names,:]
            grouper = adj_par.groupby("pargp").groups
            for grp in grouper.keys():
                grouper[grp] = list(grouper[grp])
        df = Ensemble._gaussian_draw(cov=cov,mean_values=mean_values,
                                     num_reals=num_reals,grouper=grouper)
        df.loc[:,li] = 10.0**df.loc[:,li]
        return cls(pst,df,istransformed=False)


    def _back_transform(self):

        if not self.istransformed:
            return

        li = self.pst.parameter_data.loc[:,"partrans"] == "log"

        self.loc[:,li] = 10.0**(self._df.loc[:,li])
        self.loc[:,:] = (self.loc[:,:] -\
                         self.pst.parameter_data.offset)/\
                         self.pst.parameter_data.scale

        self._istransformed = False

    def _transform(self):

        if self.istransformed:
            return

        li = self.pst.parameter_data.loc[:,"partrans"] == "log"

        self.loc[:,:] = (self.loc[:,:] * self.pst.parameter_data.scale) +\
                         self.pst.parameter_data.offset
        self.loc[:, li] = self.loc[:, li].apply(np.log10)
        self._istransformed = True



def par_gauss_draw_consistency_test():

    pst = pyemu.Pst(os.path.join("..","..","autotest","pst","pest.pst"))
    pst.parameter_data.loc[pst.par_names[3::2],"partrans"] = "fixed"
    pst.parameter_data.loc[pst.par_names[0],"pargp"] = "test"

    num_reals = 10000

    pe1 = ParameterEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    sigma_range = 4
    cov = pyemu.Cov.from_parameter_data(pst,sigma_range=sigma_range).to_2d()
    pe2 = ParameterEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals)
    pe3 = ParameterEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals,by_groups=False)

    pst.add_transform_columns()
    theo_mean = pst.parameter_data.parval1_trans
    adj_par = pst.parameter_data.loc[pst.adj_par_names,:].copy()
    ub,lb = adj_par.parubnd_trans,adj_par.parlbnd_trans
    theo_std = ((ub - lb) / sigma_range)

    for pe in [pe1,pe2,pe3]:
        assert pe.shape[0] == num_reals
        assert pe.shape[1] == pst.npar
        pe._transform()
        d = (pe.mean() - theo_mean).apply(np.abs)
        assert d.max() < 0.01
        d = (pe.loc[:,pst.adj_par_names].std() - theo_std)
        assert d.max() < 0.01

        # ensemble should be transformed so now lets test the I/O
        pe_org = pe.copy()

        pe.to_binary("test.jcb")
        pe = ParameterEnsemble.from_binary(pst=pst, filename="test.jcb")
        pe._transform()
        pe._df.index = pe.index.map(np.int)
        d = (pe - pe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10, d.max().sort_values(ascending=False)

        pe.to_csv("test.csv")
        pe = ParameterEnsemble.from_csv(pst=pst,filename="test.csv")
        pe._transform()
        d = (pe - pe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10,d.max().sort_values(ascending=False)

def obs_gauss_draw_consistency_test():

    pst = pyemu.Pst(os.path.join("..","..","autotest","pst","pest.pst"))

    num_reals = 10000

    oe1 = ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    cov = pyemu.Cov.from_observation_data(pst).to_2d()
    oe2 = ObservationEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals)
    oe3 = ObservationEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals,by_groups=False)

    pst.add_transform_columns()
    theo_mean = pst.observation_data.obsval.copy()
    theo_mean.loc[pst.nnz_obs_names] = 0.0
    theo_std = 1.0 / pst.observation_data.loc[pst.nnz_obs_names,"weight"]

    for oe in [oe1,oe2,oe3]:
        assert oe.shape[0] == num_reals
        assert oe.shape[1] == pst.nobs
        d = (oe.mean() - theo_mean).apply(np.abs)
        assert d.max() < 0.01,d.sort_values()
        d = (oe.loc[:,pst.nnz_obs_names].std() - theo_std)
        assert d.max() < 0.01

        # ensemble should be transformed so now lets test the I/O
        oe_org = oe.copy()

        oe.to_binary("test.jcb")
        oe = ObservationEnsemble.from_binary(pst=pst, filename="test.jcb")
        oe._df.index = oe.index.map(np.int)
        d = (oe - oe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10, d.max().sort_values(ascending=False)

        oe.to_csv("test.csv")
        oe = ObservationEnsemble.from_csv(pst=pst,filename="test.csv")
        d = (oe - oe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10,d.max().sort_values(ascending=False)


if __name__ == "__main__":
    #par_gauss_draw_consistency_test()
    obs_gauss_draw_consistency_test()
