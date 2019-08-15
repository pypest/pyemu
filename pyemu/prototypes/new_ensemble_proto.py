import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyemu

SEED = 358183147 #from random.org on 5 Dec 2016

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

    def reseed(self):
        np.random.seed(SEED)

    def copy(self):
        return type(self)(pst=self.pst.get(),
                          df=self._df.copy(),
                          istransformed=self.istransformed)

    @property
    def istransformed(self):
        return copy.deepcopy(self._istransformed)

    def transform(self):
        if self.transformed:
            return
        self._transformed = True
        return

    def back_transform(self):
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
            elif "DataFrame" in str(lhs):
                warnings.warn("return type uncaught, losing Ensemble type, returing DataFrame",pyemu.PyemuWarning)
                print("return type uncaught, losing Ensemble type, returing DataFrame")
                return lhs
            else:
                return lhs
        else:
            raise AttributeError("Ensemble error: the following item was not" +\
                                 "found in Ensemble or DataFrame attributes:{0}".format(item))
        return

        #def plot(self,*args,**kwargs):
            #self._df.plot(*args,**kwargs)
    def plot(self, bins=10, facecolor='0.5', plot_cols=None,
             filename="ensemble.pdf", func_dict=None,
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
        pyemu.plot_utils.ensemble_helper(self, bins=bins, facecolor=facecolor, plot_cols=plot_cols,
                        filename=filename)

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
            self.back_transform()
            retrans = True
        if self.isnull().values.any():
            warnings.warn("NaN in ensemble",PyemuWarning)
        self._df.to_csv(filename,*args,**kwargs)
        if retrans:
            self._transform()

    def to_binary(self,filename):
        retrans = False
        if self.istransformed:
            self.back_transform()
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

    def get_deviations(self,center_on=None):

        mean_vec = self.mean()
        if center_on is not None:
            if center_on not in self.index:
                raise Exception("'center_on' realization {0} not found".format(center_on))
            mean_vec = self._df.loc[center_on,:].copy()

        df = self._df.copy()
        for col in df.columns:
            df.loc[:,col] -= mean_vec[col]
        return type(self)(pst=self.pst,df=df,istransformed=self.istransformed)

    def as_pyemu_matrix(self,typ=pyemu.Matrix):
        return typ.from_dataframe(self._df)

    def covariance_matrix(self,localizer=None,center_on=None):
        devs = self.get_deviations(center_on=center_on).as_pyemu_matrix
        devs *= (1.0 / np.sqrt(float(self.shape[0] - 1.0)))

        if localizer is not None:
            delta = delta.T * delta
            return delta.hadamard_product(localizer)

        return delta.T * delta

    def dropna(self, *args, **kwargs):
        df = self._df.dropna(*args,**kwargs)
        return type(self)(pst=self.pst,df=df,istransformed=self.istransformed)


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

    @property
    def phi_vector(self):
        """property decorated method to get a vector of L2 norm (phi)
        for the realizations.  The ObservationEnsemble.pst.weights can be
        updated prior to calling this method to evaluate new weighting strategies

        Return
        ------
        pandas.DataFrame : pandas.DataFrame

        """
        cols = self._df.columns
        pst = self.pst
        weights = self.pst.observation_data.loc[cols, "weight"]
        obsval = self.pst.observation_data.loc[cols, "obsval"]
        phi_vec = []
        for idx in self._df.index.values:
            simval = self._df.loc[idx, cols]
            phi = (((simval - obsval) * weights) ** 2).sum()
            phi_vec.append(phi)
        return pd.Series(data=phi_vec, index=self.index)

    def add_base(self):
        if "base" in self.index:
            raise Exception("'base' already in ensemble")
        self.loc["base",:] = self.pst.observation_data.loc[self.columns,"obsval"]

    @property
    def nonzero(self):
        df = self._df.loc[:, self.pst.nnz_obs_names]
        return ObservationEnsemble(pst=self.pst.get(obs_names=self.pst.nnz_obs_names),df=df)

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

    def back_transform(self):

        if not self.istransformed:
            return
        li = self.pst.parameter_data.loc[:,"partrans"] == "log"
        self.loc[:,li] = 10.0**(self._df.loc[:,li])
        #self.loc[:,:] = (self.loc[:,:] -\
        #                 self.pst.parameter_data.offset)/\
        #                 self.pst.parameter_data.scale
        self._istransformed = False

    def transform(self):

        if self.istransformed:
            return
        li = self.pst.parameter_data.loc[:,"partrans"] == "log"
        #self.loc[:,:] = (self.loc[:,:] * self.pst.parameter_data.scale) +\
        #                 self.pst.parameter_data.offset
        self.loc[:, li] = self.loc[:, li].apply(np.log10)
        self._istransformed = True

    def add_base(self):
        if "base" in self.index:
            raise Exception("'base' realization already in ensemble")
        if self.istransformed:
            self.pst.add_transform_columns()
            self.loc["base", :] = self.pst.parameter_data.loc[self.columns, "parval1_trans"]
        else:
            self.loc["base",:] = self.pst.parameter_data.loc[self.columns,"parval1"]

    @property
    def adj_names(self):
        """ Get the names of adjustable parameters in the ParameterEnsemble

        Returns
        -------
        list : list
            adjustable parameter names

        """
        return self.pst.parameter_data.parnme.loc[~self.fixed_indexer].to_list()

    @property
    def ubnd(self):
        """ the upper bound vector while respecting current log transform status

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
        """ the lower bound vector while respecting current log transform status

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
        # isfixed = self.pst.parameter_data.partrans == "fixed"
        tf = set(["tied","fixed"])
        isfixed = self.pst.parameter_data.partrans. \
            apply(lambda x: x in tf)
        return isfixed.values

    def project(self,projection_matrix,center_on=None,og=None,
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


        self.transform()

        #make sure everything is cool WRT ordering

        base = self._df.mean()
        if center_on is not None:
            if center_on not in self._df.index:
                raise Exception("'center_on' realization {0} not found".format(center_on))
            base = self._df.loc[center_on,:].copy()
        names = list(base.index)
        projection_matrix = projection_matrix.get(names,names)

        new_en = self.copy()

        for real in new_en.index:
            if log is not None:
                log("projecting realization {0}".format(real))

            # null space projection of difference vector
            pdiff = self.loc[real,:] - base
            pdiff = np.dot(projection_matrix.x,pdiff.values)
            new_en.loc[real,common_names] = base + pdiff

            if log is not None:
                log("projecting realization {0}".format(real))

        new_en.enforce(enforce_bounds)
        new_en.loc[:,istransformed] = 10.0**new_en.loc[:,istransformed]
        new_en.__istransformed = False

        #new_en._back_transform()
        return new_en

    def enforce(self,how="reset",bound_tol=0.0):
        """ entry point for bounds enforcement.  This gets called for the
        draw method(s), so users shouldn't need to call this

        Parameters
        ----------
        enforce_bounds : str
            can be 'reset' to reset offending values or 'drop' to drop
            offending realizations

        """

        if how.lower().strip() == "reset":
            self._enforce_reset(bound_tol=bound_tol)
        elif how.lower().strip() == "drop":
            self._enforce_drop(bound_tol=bound_tol)
        elif how.lower().strip() == "scale":
            self._enforce_scale(bound_tol=bound_tol)
        else:
            raise Exception("unrecognized enforce_bounds arg:"+\
                            "{0}, should be 'reset' or 'drop'".\
                            format(enforce_bounds))

    def _enforce_scale(self, bound_tol):
        self.back_transform()
        ub = self.ubnd * (1.0 - bound_tol)
        lb = self.lbnd * (1.0 + bound_tol)
        base_vals = self.pst.parameter_data.loc[self._df.columns,"parval1"].copy()
        ub_dist = (ub - base_vals).apply(np.abs)
        lb_dist = (base_vals - lb).apply(np.abs)

        if ub_dist.min() <= 0.0:
            raise Exception("Ensemble._enforce_scale() error: the following parameter" +\
                            "are at or over ubnd: {0}".format(ub_dist.loc[ub_dist<=0.0].index.values))
        if lb_dist.min() <= 0.0:
            raise Exception("Ensemble._enforce_scale() error: the following parameter" +\
                            "are at or under lbnd: {0}".format(lb_dist.loc[ub_dist<=0.0].index.values))
        for ridx in self._df.index:
            real = self._df.loc[ridx,:]
            real_dist = (real - base_vals).apply(np.abs)
            out_ubnd = (real - ub)
            out_ubnd = out_ubnd.loc[out_ubnd>0.0]
            ubnd_facs = ub_dist.loc[out_ubnd.index] / real_dist.loc[out_ubnd.index]

            out_lbnd = (lb - real)
            out_lbnd = out_lbnd.loc[out_lbnd > 0.0]
            lbnd_facs = lb_dist.loc[out_lbnd.index] / real_dist.loc[out_lbnd.index]

            if ubnd_facs.shape[0] == 0 and lbnd_facs.shape[0] == 0:
                continue
            lmin = 1.0
            umin = 1.0
            sign = np.ones((self.pst.npar_adj))
            sign[real.loc[self.pst.adj_par_names] < base_vals.loc[self.pst.adj_par_names]] = -1.0
            if ubnd_facs.shape[0] > 0:
                umin = ubnd_facs.min()
                umin_idx = ubnd_facs.idxmin()
                print("enforce_scale ubnd controlling parameter, scale factor, " + \
                      "current value for realization {0}: {1}, {2}, {3}". \
                      format(ridx, umin_idx, umin, real.loc[umin_idx]))

            if lbnd_facs.shape[0] > 0:
                lmin = lbnd_facs.min()
                lmin_idx = lbnd_facs.idxmin()
                print("enforce_scale lbnd controlling parameter, scale factor, " + \
                      "current value for realization {0}: {1}, {2}. {3}". \
                      format(ridx, lmin_idx, lmin, real.loc[lmin_idx]))
            min_fac = min(umin, lmin)

            self._df.loc[ridx,:] = base_vals + (sign * real_dist * min_fac)




    def _enforce_drop(self, bound_tol):
        """ enforce parameter bounds on the ensemble by dropping
        violating realizations

        Notes:
            with a large (realistic) number of parameters, the
            probability that any one parameter is out of
            bounds is large, meaning most realization will
            be dropped.

        """
        ub = self.ubnd * (1.0 - bound_tol)
        lb = self.lbnd * (1.0 + bound_tol)
        drop = []
        for ridx in self._df.index:
            #mx = (ub - self.loc[id,:]).min()
            #mn = (lb - self.loc[id,:]).max()
            if (ub - self._df.loc[ridx,:]).min() < 0.0 or\
                            (lb - self._df.loc[ridx,:]).max() > 0.0:
                drop.append(ridx)
        self.loc[drop,:] = np.NaN
        self.dropna(inplace=True)

    def _enforce_reset(self, bound_tol):
        """enforce parameter bounds on the ensemble by resetting
        violating vals to bound
        """

        ub = (self.ubnd * (1.0 - bound_tol)).to_dict()
        lb = (self.lbnd * (1.0 + bound_tol)).to_dict()

        val_arr = self._df.values
        for iname, name in enumerate(self.columns):
            val_arr[val_arr[:,iname] > ub[name],iname] = ub[name]
            val_arr[val_arr[:, iname] < lb[name],iname] = lb[name]


def add_base_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))
    num_reals = 10
    pe = ParameterEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    oe = ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    pet = pe.copy()
    pet.transform()
    pe.add_base()
    pet.add_base()
    assert "base" in pe.index
    assert "base" in pet.index
    p = pe.loc["base",:]
    d = (pst.parameter_data.parval1 - pe.loc["base",:]).apply(np.abs)
    pst.add_transform_columns()
    d = (pst.parameter_data.parval1_trans - pet.loc["base", :]).apply(np.abs)
    assert d.max() == 0.0
    try:
        pe.add_base()
    except:
        pass
    else:
        raise Exception("should have failed")


    oe.add_base()
    d = (pst.observation_data.loc[oe.columns,"obsval"] - oe.loc["base",:]).apply(np.abs)
    assert d.max() == 0
    try:
        oe.add_base()
    except:
        pass
    else:
        raise Exception("should have failed")

def nz_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))
    num_reals = 10
    oe = ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    assert oe.shape[1] == pst.nobs
    oe_nz = oe.nonzero
    assert oe_nz.shape[1] == pst.nnz_obs
    assert list(oe_nz.columns.values) == pst.nnz_obs_names

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
        pe.transform()
        d = (pe.mean() - theo_mean).apply(np.abs)
        assert d.max() < 0.01
        d = (pe.loc[:,pst.adj_par_names].std() - theo_std)
        assert d.max() < 0.01

        # ensemble should be transformed so now lets test the I/O
        pe_org = pe.copy()

        pe.to_binary("test.jcb")
        pe = ParameterEnsemble.from_binary(pst=pst, filename="test.jcb")
        pe.transform()
        pe._df.index = pe.index.map(np.int)
        d = (pe - pe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10, d.max().sort_values(ascending=False)

        pe.to_csv("test.csv")
        pe = ParameterEnsemble.from_csv(pst=pst,filename="test.csv")
        pe.transform()
        d = (pe - pe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10,d.max().sort_values(ascending=False)

def obs_gauss_draw_consistency_test():

    pst = pyemu.Pst(os.path.join("..","..","autotest","pst","pest.pst"))

    num_reals = 10000

    oe1 = ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    cov = pyemu.Cov.from_observation_data(pst).to_2d()
    oe2 = ObservationEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals)
    oe3 = ObservationEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals,by_groups=False)

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

def phi_vector_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))
    num_reals = 10
    oe1 = ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pv = oe1.phi_vector

    for real in oe1.index:
        pst.res.loc[oe1.columns,"modelled"] = oe1.loc[real,:]
        d = np.abs(pst.phi - pv.loc[real])
        assert d < 1.0e-10

def deviations_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))
    num_reals = 10
    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    oe = ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe_devs = pe.get_deviations()
    oe_devs = oe.get_deviations()
    pe.add_base()
    pe_base_devs = pe.get_deviations(center_on="base")
    s = pe_base_devs.loc["base",:].apply(np.abs).sum()
    assert s == 0.0
    pe.transform()
    pe_base_devs = pe.get_deviations(center_on="base")
    s = pe_base_devs.loc["base", :].apply(np.abs).sum()
    assert s == 0.0

    oe.add_base()
    oe_base_devs = oe.get_deviations(center_on="base")
    s = oe_base_devs.loc["base", :].apply(np.abs).sum()
    assert s == 0.0

def as_pyemu_matrix_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))
    num_reals = 10
    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe.add_base()
    oe = ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    oe.add_base()

    pe_mat = pe.as_pyemu_matrix()
    assert type(pe_mat) == pyemu.Matrix
    assert pe_mat.shape == pe.shape
    pe._df.index = pe._df.index.map(str)
    d = (pe_mat.to_dataframe() - pe._df).apply(np.abs).values.sum()
    assert d == 0.0

    oe_mat = oe.as_pyemu_matrix(typ=pyemu.Cov)
    assert type(oe_mat) == pyemu.Cov
    assert oe_mat.shape == oe.shape
    oe._df.index = oe._df.index.map(str)
    d = (oe_mat.to_dataframe() - oe._df).apply(np.abs).values.sum()
    assert d == 0.0


def dropna_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))
    num_reals = 10
    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe.iloc[::3,:] = np.NaN
    ped = pe.dropna()
    assert type(ped) == ParameterEnsemble
    assert ped.shape == pe._df.dropna().shape

def enforce_test():
    pst = pyemu.Pst(os.path.join("..", "..", "autotest", "pst", "pest.pst"))

    # make sure sanity check is working
    num_reals = 10
    broke_pst = pst.get()
    broke_pst.parameter_data.loc[:, "parval1"] = broke_pst.parameter_data.parubnd
    pe = ParameterEnsemble.from_gaussian_draw(broke_pst, num_reals=num_reals)
    try:
        pe.enforce(how="scale")
    except:
        pass
    else:
        raise Exception("should have failed")
    broke_pst.parameter_data.loc[:, "parval1"] = broke_pst.parameter_data.parlbnd
    pe = ParameterEnsemble.from_gaussian_draw(broke_pst, num_reals=num_reals)
    try:
        pe.enforce(how="scale")
    except:
        pass
    else:
        raise Exception("should have failed")

    # check that all pars at parval1 values don't change
    num_reals = 1
    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[:, :] = pst.parameter_data.parval1.values
    pe._df.loc[0, pst.par_names[0]] = pst.parameter_data.parlbnd.loc[pst.par_names[0]] * 0.5
    pe.enforce(how="scale")
    assert (pe.loc[0,pst.par_names[1:]] - pst.parameter_data.loc[pst.par_names[1:], "parval1"]).apply(np.abs).max() == 0

    # now check that all pars are in bounds

    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe.enforce(how="scale")
    for ridx in pe._df.index:
        real = pe._df.loc[ridx,pst.adj_par_names]
        ub_diff = pe.ubnd - real
        assert ub_diff.min() >= 0.0
        lb_diff = real - pe.lbnd
        assert lb_diff.min() >= 0.0


    return

    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[0, :] += pst.parameter_data.parubnd
    pe.enforce(how="scale")

    pe = ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[0,:] += pst.parameter_data.parubnd
    pe.enforce()
    assert (pe._df.loc[0,:] - pst.parameter_data.parubnd).apply(np.abs).sum() == 0.0


    pe._df.loc[0, :] += pst.parameter_data.parubnd
    pe._df.loc[1:,:] = pst.parameter_data.parval1.values
    pe.enforce(how="drop")
    assert pe.shape[0] == num_reals - 1





def projection_test():
    pass

if __name__ == "__main__":
    #par_gauss_draw_consistency_test()
    #obs_gauss_draw_consistency_test()
    #phi_vector_test()
    #add_base_test()
    #nz_test()
    #deviations_test()
    #as_pyemu_matrix_test()
    #dropna_test()
    enforce_test()
