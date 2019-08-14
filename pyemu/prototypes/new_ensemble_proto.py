import os
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

class Iloc(object):
    def __init__(self,ensemble):
        self._ensemble = ensemble

    def __getitem__(self,item):
        return type(self._ensemble)(self._ensemble.pst, 
            self._ensemble._df.iloc[item],
            istransformed=self._ensemble.istransformed)



class Ensemble(object):
    def __init__(self,pst,df,istransformed=False):
        self._df = df
        self._istransformed = istransformed
        self.loc = Loc(self)
        self.iloc = Iloc(self)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    @property
    def transformed(self):
        return copy.deepcopy(self._transformed)
    

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
            print(item,type(lhs))
            if type(lhs) == type(self._df):
                return MyDf(lhs)
            else:
                return lhs
        else:
            raise AttributeError("Ensemble error: {0} not found in Ensemble or DataFrame attributes".format(item))
        return 

    def some_method(self,stuff):
        print("made it")
        return stuff

    def plot(self,*args,**kwargs):
        self._df.plot(*args,**kwargs)

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
                #else:
                #    reals[:, i] = mean_values.loc[name]
        else:
            reals = np.zeros((num_reals, mean_values.shape[0]))
            # for i,v in enumerate(mean_values.values):
            #    reals[:,i] = v
            # cov_map = {n:i for n,i in zip(cov.row_names,np.arange(cov.shape[0]))}
            mv_map = {n: i for n, i in zip(mean_values.index, np.arange(mean_values.shape[0]))}
            if grouper is not None:

                for grp_name,names in grouper.items():
                    print(grp_name)
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

    


class ParameterEnsemble(Ensemble):
    def __init__(self,pst,df,istransformed=False):
        super(ParameterEnsemble,self).__init__(pst,df,istransformed)

    @classmethod
    def from_gaussian_draw(cls,pst,cov=None,num_reals=100,group_chunks=True):

        if cov is None:
            cov = pyemu.Cov.from_parameter_data(pst)

        par = pst.parameter_data
        li = par.partrans == "log"
        mean_values = par.parval1.copy()
        mean_values.loc[li] = mean_values.loc[li].apply(np.log10)

        grouper = None
        if not cov.isdiagonal and group_chunks:

            tf = set(["tied","fixed"])
            adj_par = par.loc[par.partrans.apply(lambda x: x not in tf),:]
            grouper = adj_par.groupby("pargp").groups
            for grp in grouper.keys():
                grouper[grp] = list(grouper[grp])



        df = Ensemble._gaussian_draw(cov=cov,mean_values=mean_values,
                                     num_reals=num_reals,grouper=grouper)
        df.loc[:,li] = 10.0**df.loc[:,li]
        return cls(pst,df,istransformed=False)



# df = pd.DataFrame(np.random.random(100))
# pst = "1"
# mdf = ParameterEnsemble(pst,df)
# print(type(mdf.drop(0)))
# print(mdf.some_method("i"))
# mdf.to_csv("test")
pst = pyemu.Pst(os.path.join("..","..","autotest","pst","pest.pst"))
pst.parameter_data.loc[pst.par_names[3:],"partrans"] = "fixed"
pst.parameter_data.loc[pst.par_names[0],"pargp"] = "test"

pe = ParameterEnsemble.from_gaussian_draw(pst)
print(pe.shape,pst.npar)
print(pe)

cov = pyemu.Cov.from_parameter_data(pst).to_2d()
pe = ParameterEnsemble.from_gaussian_draw(pst,cov=cov)
print(pe.shape,pst.npar)
print(pe)

pe = ParameterEnsemble.from_gaussian_draw(pst,cov=cov,group_chunks=False)
print(pe.shape,pst.npar)
print(pe)

