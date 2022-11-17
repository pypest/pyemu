import os
import copy
import warnings
import numpy as np
import pandas as pd

import pyemu
from .pyemu_warnings import PyemuWarning

SEED = 358183147  # from random.org on 5 Dec 2016
np.random.seed(SEED)


class Loc(object):
    """thin wrapper around `pandas.DataFrame.loc` to make sure returned type
    is `Ensemble` (instead of `pandas.DataFrame)`

    Args:
        ensemble (`pyemu.Ensemble`): an ensemble instance

    Note:
        Users do not need to mess with this class - it is added
        to each `Ensemble` instance

    """

    def __init__(self, ensemble):
        self._ensemble = ensemble

    def __getitem__(self, item):
        return type(self._ensemble)(
            self._ensemble.pst,
            self._ensemble._df.loc[item],
            istransformed=self._ensemble.istransformed,
        )

    def __setitem__(self, idx, value):
        self._ensemble._df.loc[idx] = value


class Iloc(object):
    """thin wrapper around `pandas.DataFrame.iloc` to make sure returned type
    is `Ensemble` (instead of `pandas.DataFrame)`

    Args:
        ensemble (`pyemu.Ensemble`): an ensemble instance

    Note:
        Users do not need to mess with this class - it is added
        to each `Ensemble` instance

    """

    def __init__(self, ensemble):
        self._ensemble = ensemble

    def __getitem__(self, item):
        return type(self._ensemble)(
            self._ensemble.pst,
            self._ensemble._df.iloc[item],
            istransformed=self._ensemble.istransformed,
        )

    def __setitem__(self, idx, value):
        self._ensemble._df.iloc[idx] = value


class Ensemble(object):
    """based class for handling ensembles of numeric values

    Args:
        pst (`pyemu.Pst`): a control file instance
        df (`pandas.DataFrame`): a pandas dataframe.  Columns
            should be parameter/observation names.  Index is
            treated as realization names
        istransformed (`bool`): flag to indicate parameter values
            are in log space.  Not used for `ObservationEnsemble`

    Example::

        pst = pyemu.Pst("my.pst")
        pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst)

    """

    def __init__(self, pst, df, istransformed=False):
        self._df = df
        """`pandas.DataFrame`: the underlying dataframe that stores the realized values"""
        self.pst = pst
        """`pyemu.Pst`: control file instance"""
        self._istransformed = istransformed
        self.loc = Loc(self)
        self.iloc = Iloc(self)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    def __sub__(self, other):
        try:
            return self._df - other
        except:
            return self._df - other._df

    def __mul__(self, other):
        try:
            return self._df * other
        except:
            return self._df * other._df

    def __truediv__(self, other):
        try:
            return self._df / other
        except:
            return self._df / other._df

    def __add__(self, other):
        try:
            return self._df + other
        except:
            return self._df + other._df

    def __pow__(self, pow):
        return self._df ** pow

    @staticmethod
    def reseed():
        """reset the `numpy.random.seed`

        Note:
            reseeds using the pyemu.en.SEED global variable

            The pyemu.en.SEED value is set as the numpy.random.seed on import, so
            make sure you know what you are doing if you call this method...

        """
        np.random.seed(SEED)

    def copy(self):
        """get a copy of `Ensemble`

        Returns:
            `Ensemble`: copy of this `Ensemble`

        Note:
            copies both `Ensemble.pst` and `Ensemble._df`: can be expensive

        """
        return type(self)(
            pst=self.pst.get(), df=self._df.copy(), istransformed=self.istransformed
        )

    @property
    def istransformed(self):
        """the parameter transformation status

        Returns:
            `bool`: flag to indicate whether or not the `ParameterEnsemble` is
            transformed with respect to log_{10}.  Not used for (and has no effect
            on) `ObservationEnsemble`.

        Note:
            parameter transformation status is only related to log_{10} and does not
            include the effects of `scale` and/or `offset`

        """
        return copy.deepcopy(self._istransformed)

    def transform(self):
        """transform parameters with respect to `partrans` value.

        Note:
            operates in place (None is returned).

            Parameter transform is only related to log_{10} and does not
            include the effects of `scale` and/or `offset`

            `Ensemble.transform() is only provided for inheritance purposes.
            It only changes the `Ensemble._transformed` flag

        """
        if self.istransformed:
            return
        self._transformed = True
        return

    def back_transform(self):
        """back transform parameters with respect to `partrans` value.

        Note:
            operates in place (None is returned).

            Parameter transform is only related to log_{10} and does not
            include the effects of `scale` and/or `offset`

            `Ensemble.back_transform() is only provided for inheritance purposes.
            It only changes the `Ensemble._transformed` flag

        """
        if not self.istransformed:
            return
        self._transformed = False
        return

    def __getattr__(self, item):
        if item == "loc":
            return self.loc[item]
        elif item == "iloc":
            return self.iloc[item]
        elif item == "index":
            return self._df.index
        elif item == "columns":
            return self._df.columns
        elif item in set(dir(self)):
            return getattr(self, item)
        elif item in set(dir(self._df)):
            lhs = self._df.__getattr__(item)
            if type(lhs) == type(self._df):
                return type(self)(
                    pst=self.pst, df=lhs, istransformed=self.istransformed
                )
            elif "DataFrame" in str(lhs):
                warnings.warn(
                    "return type uncaught, losing Ensemble type, returning DataFrame",
                    PyemuWarning,
                )
                print("return type uncaught, losing Ensemble type, returning DataFrame")
                return lhs
            else:
                return lhs
        else:
            raise AttributeError(
                "Ensemble error: the following item was not"
                + "found in Ensemble or DataFrame attributes:{0}".format(item)
            )
        return

        # def plot(self,*args,**kwargs):
        # self._df.plot(*args,**kwargs)

    def plot(
        self,
        bins=10,
        facecolor="0.5",
        plot_cols=None,
        filename="ensemble.pdf",
        func_dict=None,
        **kwargs
    ):
        """plot ensemble histograms to multipage pdf

        Args:
            bins (`int`): number of bins for the histograms
            facecolor (`str`): matplotlib color (e.g. `r`,`g`, etc)
            plot_cols ([`str`]): list of subset of ensemble columns to plot.
                If None, all are plotted. Default is None
            filename (`str`): multipage pdf filename. Default is "ensemble.pdf"
            func_dict (`dict`): a dict of functions to apply to specific
                columns. For example: {"par1": np.log10}
            **kwargs (`dict`): addkeyword args to pass to `pyemu.plot_utils.ensemble_helper()`


        Example::

            pst = pyemu.Pst("my.pst")
            pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst)
            pe.transform() # plot log space (if needed)
            pe.plot(bins=30)

        """
        pyemu.plot_utils.ensemble_helper(
            self, bins=bins, facecolor=facecolor, plot_cols=plot_cols, filename=filename
        )

    @classmethod
    def from_binary(cls, pst, filename):
        """create an `Ensemble` from a PEST-style binary file

        Args:
            pst (`pyemu.Pst`): a control file instance
            filename (`str`): filename containing binary ensemble

        Returns:
            `Ensemble`: the ensemble loaded from the binary file

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_binary("obs.jcb")


        """

        df = pyemu.Matrix.from_binary(filename).to_dataframe()
        return cls(pst=pst, df=df)

    @classmethod
    def from_csv(cls, pst, filename, *args, **kwargs):
        """create an `Ensemble` from a CSV file

        Args:
            pst (`pyemu.Pst`): a control file instance
            filename (`str`): filename containing CSV ensemble
            *args ([`object`]: positional arguments to pass to
                `pandas.read_csv()`.
            **kwargs ({`str`:`object`}): keyword arguments to pass
                to `pandas.read_csv()`.
        Returns:
            `Ensemble`
        Note:
            uses `pandas.read_csv()` to load numeric values from
            CSV file

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_csv("obs.csv")

        """

        if "index_col" not in kwargs:
            kwargs["index_col"] = 0
        df = pd.read_csv(filename, *args, **kwargs)
        return cls(pst=pst, df=df)

    def to_csv(self, filename, *args, **kwargs):
        """write `Ensemble` to a CSV file

        Args:
            filename (`str`): file to write
            *args ([`object`]: positional arguments to pass to
                `pandas.DataFrame.to_csv()`.
            **kwargs ({`str`:`object`}): keyword arguments to pass
                to `pandas.DataFrame.to_csv()`.

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst)
            oe.to_csv("obs.csv")

        Note:
            back transforms `ParameterEnsemble` before writing so that
            values are in arithmetic space

        """
        retrans = False
        if self.istransformed:
            self.back_transform()
            retrans = True
        if self._df.isnull().values.any():
            warnings.warn("NaN in ensemble", PyemuWarning)
        self._df.to_csv(filename, *args, **kwargs)
        if retrans:
            self.transform()

    def to_binary(self, filename):
        """write `Ensemble` to a PEST-style binary file

        Args:
            filename (`str`): file to write

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst)
            oe.to_binary("obs.jcb")

        Note:
            back transforms `ParameterEnsemble` before writing so that
            values are in arithmetic space

        """

        retrans = False
        if self.istransformed:
            self.back_transform()
            retrans = True
        if self._df.isnull().values.any():
            warnings.warn("NaN in ensemble", PyemuWarning)
        pyemu.Matrix.from_dataframe(self._df).to_coo(filename)
        if retrans:
            self.transform()

    def to_dense(self, filename):
        """write `Ensemble` to a dense-format binary file

        Args:
            filename (`str`): file to write

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst)
            oe.to_dense("obs.bin")

        Note:
            back transforms `ParameterEnsemble` before writing so that
            values are in arithmatic space

        """

        retrans = False
        if self.istransformed:
            self.back_transform()
            retrans = True
        if self._df.isnull().values.any():
            warnings.warn("NaN in ensemble", PyemuWarning)
        pyemu.Matrix.write_dense(
            filename,
            self._df.index.tolist(),
            self._df.columns.tolist(),
            self._df.values,
        )
        if retrans:
            self.transform()

    @classmethod
    def from_dataframe(cls, pst, df, istransformed=False):
        warnings.warn(
            "Ensemble.from_dataframe() is deprecated and has been "
            "replaced with the standard constructor, which takes"
            "the same arguments"
        )
        return cls(pst=pst, df=df, istransformed=istransformed)

    @staticmethod
    def _gaussian_draw(
        cov, mean_values, num_reals, grouper=None, fill=True, factor="eigen"
    ):

        factor = factor.lower()
        if factor not in ["eigen", "svd"]:
            raise Exception(
                "Ensemble._gaussian_draw() error: unrecognized"
                + "'factor': {0}".format(factor)
            )
        # make sure all cov names are found in mean_values
        cov_names = set(cov.row_names)
        mv_names = set(mean_values.index.values)
        missing = cov_names - mv_names
        if len(missing) > 0:
            raise Exception(
                "Ensemble._gaussian_draw() error: the following cov names are not in "
                "mean_values: {0}".format(",".join(missing))
            )
        if cov.isdiagonal:
            stds = {
                name: std for name, std in zip(cov.row_names, np.sqrt(cov.x.flatten()))
            }
            snv = np.random.randn(num_reals, mean_values.shape[0])
            reals = np.zeros_like(snv)
            reals[:, :] = np.NaN
            for i, name in enumerate(mean_values.index):
                if name in cov_names:
                    reals[:, i] = (snv[:, i] * stds[name]) + mean_values.loc[name]
                elif fill:
                    reals[:, i] = mean_values.loc[name]
        else:
            reals = np.zeros((num_reals, mean_values.shape[0]))
            reals[:, :] = np.NaN
            if fill:
                for i, v in enumerate(mean_values.values):
                    reals[:, i] = v
            #cov_map = {n: i for n, i in zip(cov.row_names, np.arange(cov.shape[0]))}
            mv_map = {
                n: i for n, i in zip(mean_values.index, np.arange(mean_values.shape[0]))
            }
            if grouper is not None:

                for grp_name, names in grouper.items():
                    print("drawing from group", grp_name)
                    # reorder names to be in cov matrix order
                    # cnames = names
                    cnames = []
                    snames = set(names)
                    for n in cov.names:
                        if n in snames:
                            cnames.append(n)
                    names = None
                    snames = None
                    idxs = [mv_map[name] for name in cnames]
                    snv = np.random.randn(num_reals, len(cnames))
                    cov_grp = cov.get(cnames)
                    if len(cnames) == 1:
                        std = np.sqrt(cov_grp.x)
                        reals[:, idxs] = mean_values.loc[cnames].values[0] + (snv * std)
                    else:
                        if factor == "eigen":
                            try:
                                cov_grp.inv
                            except:
                                covname = "trouble_{0}.cov".format(grp_name)
                                cov_grp.to_ascii(covname)
                                raise Exception(
                                    "error inverting cov for group '{0}',"
                                    + "saved trouble cov to {1}".format(
                                        grp_name, covname
                                    )
                                )

                            a, i = Ensemble._get_eigen_projection_matrix(cov_grp.as_2d)
                        elif factor == "svd":
                            a, i = Ensemble._get_svd_projection_matrix(cov_grp.as_2d)
                            snv[:, i:] = 0.0
                        # process each realization
                        group_mean_values = mean_values.loc[cnames]
                        for i in range(num_reals):
                            reals[i, idxs] = group_mean_values + np.dot(a, snv[i, :])

            else:
                snv = np.random.randn(num_reals, cov.shape[0])
                if factor == "eigen":
                    a, i = Ensemble._get_eigen_projection_matrix(cov.as_2d)
                elif factor == "svd":
                    a, i = Ensemble._get_svd_projection_matrix(cov.as_2d)
                    snv[:, i:] = 0.0
                cov_mean_values = mean_values.loc[cov.row_names].values
                idxs = [mv_map[name] for name in cov.row_names]
                for i in range(num_reals):
                    reals[i, idxs] = cov_mean_values + np.dot(a, snv[i, :])
        df = pd.DataFrame(reals, columns=mean_values.index.values)
        df.dropna(inplace=True, axis=1)
        return df

    @staticmethod
    def _get_svd_projection_matrix(x, maxsing=None, eigthresh=1.0e-7):
        if x.shape[0] != x.shape[1]:
            raise Exception("matrix not square")
        u, s, v = np.linalg.svd(x, full_matrices=True)
        v = v.transpose()

        if maxsing is None:
            maxsing = pyemu.Matrix.get_maxsing_from_s(s, eigthresh=eigthresh)
        u = u[:, :maxsing]
        s = s[:maxsing]
        v = v[:, :maxsing]

        # fill in full size svd component matrices
        s_full = np.zeros(x.shape)
        s_full[: s.shape[0], : s.shape[1]] = np.sqrt(
            s
        )  # sqrt since sing vals are eigvals**2
        v_full = np.zeros_like(s_full)
        v_full[: v.shape[0], : v.shape[1]] = v
        # form the projection matrix
        proj = np.dot(v_full, s_full)
        return proj, maxsing

    @staticmethod
    def _get_eigen_projection_matrix(x):
        # eigen factorization
        v, w = np.linalg.eigh(x)

        # check for near zero eig values
        for i in range(v.shape[0]):
            if v[i] > 1.0e-10:
                pass
            else:
                print(
                    "near zero eigen value found",
                    v[i],
                    "at index",
                    i,
                    " of ",
                    v.shape[0],
                )
                v[i] = 0.0

        # form the projection matrix
        vsqrt = np.sqrt(v)
        vsqrt[i + 1 :] = 0.0
        v = np.diag(vsqrt)
        a = np.dot(w, v)

        return a, i

    def get_deviations(self, center_on=None):
        """get the deviations of the realizations around a certain
        point in ensemble space

        Args:
            center_on (`str`, optional): a realization name to use as the centering
                point in ensemble space.  If `None`, the mean vector is
                treated as the centering point.  Default is None

        Returns:
            `Ensemble`: an ensemble of deviations around the centering point

        Note:
            deviations are the Euclidean distances from the `center_on` value to
            realized values for each column

            `center_on=None` yields the classic ensemble smoother/ensemble Kalman
            filter deviations from the mean vector

            Deviations respect log-transformation status.

        Example::

            pst = pyemu.Pst("my.pst")
            oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst)
            oe.add_base()
            oe_dev = oe.get_deviations(center_on="base")
            oe.to_csv("obs_base_devs.csv")

        """

        retrans = False
        if not self.istransformed:
            self.transform()
            retrans = True
        mean_vec = self.mean()
        if center_on is not None:
            if center_on not in self.index:
                raise Exception(
                    "'center_on' realization {0} not found".format(center_on)
                )
            mean_vec = self._df.loc[center_on, :].copy()

        df = self._df.copy()
        for col in df.columns:
            df.loc[:, col] -= mean_vec[col]
        if retrans:
            self.back_transform()
        return type(self)(pst=self.pst, df=df, istransformed=self.istransformed)

    def as_pyemu_matrix(self, typ=None):
        """get a `pyemu.Matrix` instance of `Ensemble`

        Args:
            typ (`pyemu.Matrix` or `pyemu.Cov`): the type of matrix to return.
                Default is `pyemu.Matrix`

        Returns:
            `pyemu.Matrix`: a matrix instance

        Example::

            oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst,num_reals=100)
            dev_mat = oe.get_deviations().as_pyemu_matrix(typ=pyemu.Cov)
            obscov = dev_mat.T * dev_mat

        """
        if typ is None:
            typ = pyemu.Matrix
        return typ.from_dataframe(self._df)

    def covariance_matrix(self, localizer=None, center_on=None):
        """get a empirical covariance matrix implied by the
        correlations between realizations

        Args:
            localizer (`pyemu.Matrix`, optional): a matrix to localize covariates
                in the resulting covariance matrix.  Default is None
            center_on (`str`, optional): a realization name to use as the centering
                point in ensemble space.  If `None`, the mean vector is
                treated as the centering point.  Default is None

        Returns:
            `pyemu.Cov`: the empirical (and optionally localized) covariance matrix

        """

        devs = self.get_deviations(center_on=center_on).as_pyemu_matrix()
        devs *= 1.0 / np.sqrt(float(self.shape[0] - 1.0))

        if localizer is not None:
            devs = devs.T * devs
            return devs.hadamard_product(localizer)

        return pyemu.Cov((devs.T * devs).x, names=devs.col_names)

    def dropna(self, *args, **kwargs):
        """override of `pandas.DataFrame.dropna()`

        Args:
            *args ([`object`]: positional arguments to pass to
                `pandas.DataFrame.dropna()`.
            **kwargs ({`str`:`object`}): keyword arguments to pass
                to `pandas.DataFrame.dropna()`.

        """
        df = self._df.dropna(*args, **kwargs)
        return type(self)(pst=self.pst, df=df, istransformed=self.istransformed)


class ObservationEnsemble(Ensemble):
    """Observation noise ensemble in the PEST(++) realm

    Args:
        pst (`pyemu.Pst`): a control file instance
        df (`pandas.DataFrame`): a pandas dataframe.  Columns
            should be observation names.  Index is
            treated as realization names
        istransformed (`bool`): flag to indicate parameter values
            are in log space.  Not used for `ObservationEnsemble`

    Example::

        pst = pyemu.Pst("my.pst")
        oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst)

    """

    def __init__(self, pst, df, istransformed=False):
        super(ObservationEnsemble, self).__init__(pst, df, istransformed)

    @classmethod
    def from_gaussian_draw(
        cls, pst, cov=None, num_reals=100, by_groups=True, fill=False, factor="eigen"
    ):
        """generate an `ObservationEnsemble` from a (multivariate) gaussian
        distribution

        Args:
            pst (`pyemu.Pst`): a control file instance.
            cov (`pyemu.Cov`): a covariance matrix describing the second
                moment of the gaussian distribution.  If None, `cov` is
                generated from the non-zero-weighted observation weights in `pst`.
                Only observations listed in `cov` are sampled.  Other observations are
                assigned the `obsval` value from `pst`.
            num_reals (`int`): number of stochastic realizations to generate.  Default
                is 100
            by_groups (`bool`): flag to generate realzations be observation group.  This
                assumes no correlation (covariates) between observation groups.
            fill (`bool`): flag to fill in zero-weighted observations with control file
                values.  Default is False.
            factor (`str`): how to factorize `cov` to form the projectin matrix.  Can
                be "eigen" or "svd". The "eigen" option is default and is faster.  But
                for (nearly) singular cov matrices (such as those generated empirically
                from ensembles), "svd" is the only way.  Ignored for diagonal `cov`.

        Returns:
            `ObservationEnsemble`: the realized `ObservationEnsemble` instance

        Note:
            Only observations named in `cov` are sampled. Additional, `cov` is processed prior
            to sampling to only include non-zero-weighted observations depending on the value of `fill`.
            So users must take care to make sure observations have been assigned non-zero weights even if `cov`
            is being passed

            The default `cov` is generated from `pyemu.Cov.from_observation_data`, which assumes
            observation noise standard deviations are the inverse of the weights listed in `pst`

        Example::

            pst = pyemu.Pst("my.pst")
            # the easiest way - just relying on weights in pst
            oe1 = pyemu.ObservationEnsemble.from_gaussian_draw(pst)

            # generate the cov explicitly
            cov = pyemu.Cov.from_observation_data(pst)
            oe2 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,cov=cov)

            # give all but one observation zero weight.  This will
            # result in an oe with only one randomly sampled observation noise
            # vector since the cov is processed to remove any zero-weighted
            # observations before sampling
            pst.observation_data.loc[pst.nnz_obs_names[1:],"weight] = 0.0
            oe3 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,cov=cov)

        """
        if cov is None:
            cov = pyemu.Cov.from_observation_data(pst)
        obs = pst.observation_data
        mean_values = obs.obsval.copy()
        if len(pst.nnz_obs_names) == 0:
            warnings.warn("ObservationEnsemble.from_gaussian_draw(): all zero weights",PyemuWarning)
        # only draw for non-zero weights, get a new cov
        if not fill:
            nz_cov = cov.get(pst.nnz_obs_names)
        else:
            nz_cov = cov.copy()

        grouper = None
        if not cov.isdiagonal and by_groups:
            nz_obs = obs.loc[pst.nnz_obs_names, :].copy()
            grouper = nz_obs.groupby("obgnme").groups
            for grp in grouper.keys():
                grouper[grp] = list(grouper[grp])
        df = Ensemble._gaussian_draw(
            cov=nz_cov,
            mean_values=mean_values,
            num_reals=num_reals,
            grouper=grouper,
            fill=fill,
            factor=factor,
        )
        if fill:
            df.loc[:, pst.zero_weight_obs_names] = pst.observation_data.loc[
                pst.zero_weight_obs_names, "obsval"
            ].values
        return cls(pst, df, istransformed=False)

    @property
    def phi_vector(self):
        """vector of L2 norm (phi) for the realizations (rows) of `Ensemble`.

        Returns:
            `pandas.Series`: series of realization name (`Ensemble.index`) and phi values

        Note:
            The ObservationEnsemble.pst.weights can be updated prior to calling
            this method to evaluate new weighting strategies

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
        """add the control file `obsval` values as a realization

        Note:
            replaces the last realization with the current `ObservationEnsemble.pst.observation_data.obsval` values
            as a new realization named "base"

            the PEST++ enemble tools will add this realization also if you dont wanna fool with it here...

        """
        if "base" in self.index:
            raise Exception("'base' already in ensemble")
        self._df = self._df.iloc[:-1, :]
        self._df.loc["base", :] = self.pst.observation_data.loc[self.columns, "obsval"]

    @property
    def nonzero(self):
        """get a new `ObservationEnsemble` of just non-zero weighted observations

        Returns:
            `ObservationEnsemble`: non-zero weighted observation ensemble.

        Note:
            The `pst` attribute of the returned `ObservationEnsemble` also only includes
            non-zero weighted observations (and is therefore not valid for running
            with PEST or PEST++)

        """
        df = self._df.loc[:, self.pst.nnz_obs_names]
        return ObservationEnsemble(
            pst=self.pst.get(obs_names=self.pst.nnz_obs_names), df=df
        )


class ParameterEnsemble(Ensemble):
    """Parameter ensembles in the PEST(++) realm

    Args:
        pst (`pyemu.Pst`): a control file instance
        df (`pandas.DataFrame`): a pandas dataframe.  Columns
            should be parameter names.  Index is
            treated as realization names
        istransformed (`bool`): flag to indicate parameter values
            are in log space (if `partrans` is "log" in `pst`)

    Example::

        pst = pyemu.Pst("my.pst")
        pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst)

    """

    def __init__(self, pst, df, istransformed=False):
        super(ParameterEnsemble, self).__init__(pst, df, istransformed)

    @classmethod
    def from_gaussian_draw(
        cls, pst, cov=None, num_reals=100, by_groups=True, fill=True, factor="eigen"
    ):
        """generate a `ParameterEnsemble` from a (multivariate) (log) gaussian
        distribution

        Args:
            pst (`pyemu.Pst`): a control file instance.
            cov (`pyemu.Cov`): a covariance matrix describing the second
                moment of the gaussian distribution.  If None, `cov` is
                generated from the bounds of the adjustable parameters in `pst`.
                the (log) width of the bounds is assumed to represent a multiple of
                the parameter standard deviation (this is the `sigma_range` argument
                that can be passed to `pyemu.Cov.from_parameter_data`).
            num_reals (`int`): number of stochastic realizations to generate.  Default
                is 100
            by_groups (`bool`): flag to generate realizations be parameter group.  This
                assumes no correlation (covariates) between parameter groups.  For large
                numbers of parameters, this help prevent memories but is slower.
            fill (`bool`): flag to fill in fixed and/or tied parameters with control file
                values.  Default is True.
            factor (`str`): how to factorize `cov` to form the projection matrix.  Can
                be "eigen" or "svd". The "eigen" option is default and is faster.  But
                for (nearly) singular cov matrices (such as those generated empirically
                from ensembles), "svd" is the only way.  Ignored for diagonal `cov`.

        Returns:
            `ParameterEnsemble`: the parameter ensemble realized from the gaussian
            distribution

        Note:

            Only parameters named in `cov` are sampled. Missing parameters are assigned values of
            `pst.parameter_data.parval1` along the corresponding columns of `ParameterEnsemble`
            according to the value of `fill`.

            The default `cov` is generated from `pyemu.Cov.from_observation_data`, which assumes
            parameter bounds in `ParameterEnsemble.pst` represent some multiple of parameter
            standard deviations.  Additionally, the default Cov only includes adjustable
            parameters (`partrans` not "tied" or "fixed").

            "tied" parameters are not sampled.

        Example::

            pst = pyemu.Pst("my.pst")
            # the easiest way - just relying on weights in pst
            pe1 = pyemu.ParameterEnsemble.from_gaussian_draw(pst)

            # generate the cov explicitly with a sigma_range
            cov = pyemu.Cov.from_parameter_data(pst,sigma_range=6)
            [e2 = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov=cov)

        """
        if cov is None:
            cov = pyemu.Cov.from_parameter_data(pst)
        par = pst.parameter_data
        li = par.partrans == "log"
        mean_values = par.parval1.copy()
        mean_values.loc[li] = mean_values.loc[li].apply(np.log10)
        if len(pst.adj_par_names) == 0:
            warnings.warn("ParameterEnsemble.from_gaussian_draw(): no adj pars", PyemuWarning)
        grouper = None
        if not cov.isdiagonal and by_groups:
            adj_par = par.loc[pst.adj_par_names, :]
            grouper = adj_par.groupby("pargp").groups
            for grp in grouper.keys():
                grouper[grp] = list(grouper[grp])
        df = Ensemble._gaussian_draw(
            cov=cov,
            mean_values=mean_values,
            num_reals=num_reals,
            grouper=grouper,
            fill=fill,
        )
        df.loc[:, li] = 10.0 ** df.loc[:, li]
        return cls(pst, df, istransformed=False)

    @classmethod
    def from_triangular_draw(cls, pst, num_reals=100, fill=True):
        """generate a `ParameterEnsemble` from a (multivariate) (log) triangular distribution

        Args:
            pst (`pyemu.Pst`): a control file instance
            num_reals (`int`, optional): number of realizations to generate.  Default is 100
            fill (`bool`): flag to fill in fixed and/or tied parameters with control file
                values.  Default is True.

        Returns:
            `ParameterEnsemble`: a parameter ensemble drawn from the multivariate (log) triangular
            distribution defined by the parameter upper and lower bounds and initial parameter
            values in `pst`

        Note:
            respects transformation status in `pst`: fixed and tied parameters are not realized,
            log-transformed parameters are drawn in log space.  The returned `ParameterEnsemble`
            is back transformed (not in log space)

            uses numpy.random.triangular

        Example::

            pst = pyemu.Pst("my.pst")
            pe = pyemu.ParameterEnsemble.from_triangular_draw(pst)
            pe.to_csv("my_tri_pe.csv")

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
        arr[:, :] = np.NaN
        adj_par_names = set(pst.adj_par_names)
        for i, pname in enumerate(pst.parameter_data.parnme):
            # print(pname, lb[pname], ub[pname])
            if pname in adj_par_names:
                arr[:, i] = np.random.triangular(
                    lb[pname], pv[pname], ub[pname], size=num_reals
                )
            elif fill:
                arr[:, i] = (
                    np.zeros((num_reals)) + pst.parameter_data.loc[pname, "parval1"]
                )

        df = pd.DataFrame(arr, index=real_names, columns=pst.par_names)
        df.dropna(inplace=True, axis=1)
        df.loc[:, li] = 10.0 ** df.loc[:, li]
        new_pe = cls(pst=pst, df=df)
        return new_pe

    @classmethod
    def from_uniform_draw(cls, pst, num_reals, fill=True):
        """generate a `ParameterEnsemble` from a (multivariate) (log) uniform
        distribution

        Args:
            pst (`pyemu.Pst`): a control file instance
            num_reals (`int`, optional): number of realizations to generate.  Default is 100
            fill (`bool`): flag to fill in fixed and/or tied parameters with control file
                values.  Default is True.

        Returns:
            `ParameterEnsemble`: a parameter ensemble drawn from the multivariate (log) uniform
            distribution defined by the parameter upper and lower bounds `pst`

        Note:
            respects transformation status in `pst`: fixed and tied parameters are not realized,
            log-transformed parameters are drawn in log space.  The returned `ParameterEnsemble`
            is back transformed (not in log space)

            uses numpy.random.uniform

        Example::

            pst = pyemu.Pst("my.pst")
            pe = pyemu.ParameterEnsemble.from_uniform_draw(pst)
            pe.to_csv("my_uni_pe.csv")


        """

        li = pst.parameter_data.partrans == "log"
        ub = pst.parameter_data.parubnd.copy()
        ub.loc[li] = ub.loc[li].apply(np.log10)
        ub = ub.to_dict()
        lb = pst.parameter_data.parlbnd.copy()
        lb.loc[li] = lb.loc[li].apply(np.log10)
        lb = lb.to_dict()

        real_names = np.arange(num_reals, dtype=np.int64)
        arr = np.empty((num_reals, len(ub)))
        arr[:, :] = np.NaN
        adj_par_names = set(pst.adj_par_names)
        if len(adj_par_names) == 0:
            warnings.warn("ParameterEnsemble.from_uniform_draw(): no adj pars",PyemuWarning)
        for i, pname in enumerate(pst.parameter_data.parnme):
            # print(pname,lb[pname],ub[pname])
            if pname in adj_par_names:
                arr[:, i] = np.random.uniform(lb[pname], ub[pname], size=num_reals)
            elif fill:
                arr[:, i] = (
                    np.zeros((num_reals)) + pst.parameter_data.loc[pname, "parval1"]
                )

        df = pd.DataFrame(arr, index=real_names, columns=pst.par_names)
        df.dropna(inplace=True, axis=1)
        df.loc[:, li] = 10.0 ** df.loc[:, li]

        new_pe = cls(pst=pst, df=df)
        return new_pe

    @classmethod
    def from_mixed_draws(
        cls,
        pst,
        how_dict,
        default="gaussian",
        num_reals=100,
        cov=None,
        sigma_range=6,
        enforce_bounds=True,
        partial=False,
        fill=True,
    ):
        """generate a `ParameterEnsemble` using a mixture of
        distributions.  Available distributions include (log) "uniform", (log) "triangular",
        and (log) "gaussian". log transformation is respected.

        Args:
            pst (`pyemu.Pst`): a control file
            how_dict (`dict`): a dictionary of parameter name keys and
                "how" values, where "how" can be "uniform","triangular", or "gaussian".
            default (`str`): the default distribution to use for parameter not listed
                in how_dict.  Default is "gaussian".
            num_reals (`int`): number of realizations to draw.  Default is 100.
            cov (`pyemu.Cov`): an optional Cov instance to use for drawing from gaussian distribution.
                If None, and "gaussian" is listed in `how_dict` (and/or `default`), then a diagonal
                covariance matrix is constructed from the parameter bounds in `pst` (with `sigma_range`).
                Default is None.
            sigma_range (`float`): the number of standard deviations implied by the parameter bounds in the pst.
                Only used if "gaussian" is in `how_dict` (and/or `default`) and `cov` is None.  Default is 6.
            enforce_bounds (`bool`): flag to enforce parameter bounds in resulting `ParameterEnsemble`. Only
                matters if "gaussian" is in values of `how_dict`.  Default is True.
            partial (`bool`): flag to allow a partial ensemble (not all pars included).  If True, parameters
                not name in `how_dict` will be sampled using the distribution named as `default`.
                Default is `False`.
            fill (`bool`): flag to fill in fixed and/or tied parameters with control file
                values.  Default is True.

        Example::

            pst = pyemu.Pst("pest.pst")
            # uniform for the fist 10 pars
            how_dict = {p:"uniform" for p in pst.adj_par_names[:10]}
            pe = pyemu.ParameterEnsemble(pst,how_dict=how_dict)
            pe.to_csv("my_mixed_pe.csv")

        """

        # error checking
        accept = {"uniform", "triangular", "gaussian"}
        assert (
            default in accept
        ), "ParameterEnsemble.from_mixed_draw() error: 'default' must be in {0}".format(
            accept
        )
        par_org = pst.parameter_data.copy()
        pset = set(pst.adj_par_names)
        hset = set(how_dict.keys())
        missing = pset.difference(hset)
        if not partial and len(missing) > 0:
            print(
                "{0} par names missing in how_dict, these parameters will be sampled using {1} (the 'default')".format(
                    len(missing), default
                )
            )
            for m in missing:
                how_dict[m] = default
        missing = hset.difference(pset)
        assert len(missing) == 0, (
            "ParameterEnsemble.from_mixed_draws() error: the following par names are not in "
            + " in the pst: {0}".format(",".join(missing))
        )

        unknown_draw = []
        for pname, how in how_dict.items():
            if how not in accept:
                unknown_draw.append("{0}:{1}".format(pname, how))
        if len(unknown_draw) > 0:
            raise Exception(
                "ParameterEnsemble.from_mixed_draws() error: the following hows are not recognized:{0}".format(
                    ",".join(unknown_draw)
                )
            )

        # work out 'how' grouping
        how_groups = {how: [] for how in accept}
        for pname, how in how_dict.items():
            how_groups[how].append(pname)

        # gaussian
        pes = []
        if len(how_groups["gaussian"]) > 0:
            gset = set(how_groups["gaussian"])
            par_gaussian = par_org.loc[gset, :]
            # par_gaussian.sort_values(by="parnme", inplace=True)
            par_gaussian.sort_index(inplace=True)
            pst.parameter_data = par_gaussian

            if cov is not None:
                cset = set(cov.row_names)
                # gset = set(how_groups["gaussian"])
                diff = gset.difference(cset)
                assert len(diff) == 0, (
                    "ParameterEnsemble.from_mixed_draws() error: the 'cov' is not compatible with "
                    + " the parameters listed as 'gaussian' in how_dict, the following are not in the cov:{0}".format(
                        ",".join(diff)
                    )
                )
            else:

                cov = pyemu.Cov.from_parameter_data(pst, sigma_range=sigma_range)
            pe_gauss = ParameterEnsemble.from_gaussian_draw(
                pst, cov, num_reals=num_reals
            )
            pes.append(pe_gauss)

        if len(how_groups["uniform"]) > 0:
            par_uniform = par_org.loc[how_groups["uniform"], :]
            # par_uniform.sort_values(by="parnme",inplace=True)
            par_uniform.sort_index(inplace=True)
            pst.parameter_data = par_uniform
            pe_uniform = ParameterEnsemble.from_uniform_draw(pst, num_reals=num_reals)
            pes.append(pe_uniform)

        if len(how_groups["triangular"]) > 0:
            par_tri = par_org.loc[how_groups["triangular"], :]
            # par_tri.sort_values(by="parnme", inplace=True)
            par_tri.sort_index(inplace=True)
            pst.parameter_data = par_tri
            pe_tri = ParameterEnsemble.from_triangular_draw(pst, num_reals=num_reals)
            pes.append(pe_tri)

        df = pd.DataFrame(index=np.arange(num_reals), columns=par_org.parnme.values)

        df.loc[:, :] = np.NaN
        if fill:
            fixed_tied = par_org.loc[
                par_org.partrans.apply(lambda x: x in ["fixed", "tied"]), "parval1"
            ].to_dict()
            for p, v in fixed_tied.items():
                df.loc[:, p] = v

        for pe in pes:
            df.loc[pe.index, pe.columns] = pe

        # this dropna covers both "fill" and "partial"
        df = df.dropna(axis=1)

        pst.parameter_data = par_org
        pe = ParameterEnsemble(df=df, pst=pst)
        if enforce_bounds:
            pe.enforce()
        return pe

    @classmethod
    def from_parfiles(cls, pst, parfile_names, real_names=None):
        """create a parameter ensemble from PEST-style parameter value files.
        Accepts parfiles with less than the parameters in the control
        (get NaNs in the ensemble) or extra parameters in the
        parfiles (get dropped)

        Args:
            pst (`pyemu.Pst`): control file instance
            parfile_names (`[str`]): par file names
            real_names (`str`): optional list of realization names.
                If None, a single integer counter is used

        Returns:
            `ParameterEnsemble`: parameter ensemble loaded from par files


        """
        if isinstance(pst, str):
            pst = pyemu.Pst(pst)
        dfs = {}
        if real_names is not None:
            assert len(real_names) == len(parfile_names)
        else:
            real_names = np.arange(len(parfile_names))

        for rname, pfile in zip(real_names, parfile_names):
            assert os.path.exists(pfile), (
                "ParameterEnsemble.from_parfiles() error: "
                + "file: {0} not found".format(pfile)
            )
            df = pyemu.pst_utils.read_parfile(pfile)
            # check for scale differences - I don't who is dumb enough
            # to change scale between par files and pst...
            diff = df.scale - pst.parameter_data.scale
            if diff.apply(np.abs).sum() > 0.0:
                warnings.warn(
                    "differences in scale detected, applying scale in par file",
                    PyemuWarning,
                )
                # df.loc[:,"parval1"] *= df.scale

            dfs[rname] = df.parval1.values

        df_all = pd.DataFrame(data=dfs).T
        df_all.columns = df.index

        if len(pst.par_names) != df_all.shape[1]:
            # if len(pst.par_names) < df_all.shape[1]:
            #    raise Exception("pst is not compatible with par files")
            pset = set(pst.par_names)
            dset = set(df_all.columns)
            diff = pset.difference(dset)
            if len(diff) > 0:
                warnings.warn(
                    "the following parameters are not in the par files (getting NaNs) :{0}".format(
                        ",".join(diff)
                    ),
                    PyemuWarning,
                )
                blank_df = pd.DataFrame(index=df_all.index, columns=diff)

                df_all = pd.concat([df_all, blank_df], axis=1)

            diff = dset.difference(pset)
            if len(diff) > 0:
                warnings.warn(
                    "the following par file parameters are not in the control (being dropped):{0}".format(
                        ",".join(diff)
                    ),
                    PyemuWarning,
                )
                df_all = df_all.loc[:, pst.par_names]

        return ParameterEnsemble(pst=pst, df=df_all)

    def back_transform(self):
        """back transform parameters with respect to `partrans` value.

        Note:
            operates in place (None is returned).

            Parameter transform is only related to log_{10} and does not
            include the effects of `scale` and/or `offset`

        """
        if not self.istransformed:
            return
        li = self.pst.parameter_data.loc[:, "partrans"] == "log"
        self.loc[:, li] = 10.0 ** (self._df.loc[:, li])
        # self.loc[:,:] = (self.loc[:,:] -\
        #                 self.pst.parameter_data.offset)/\
        #                 self.pst.parameter_data.scale
        self._istransformed = False

    def transform(self):
        """transform parameters with respect to `partrans` value.

        Note:
            operates in place (None is returned).

            Parameter transform is only related to log_{10} and does not
            include the effects of `scale` and/or `offset`

        """
        if self.istransformed:
            return
        li = self.pst.parameter_data.loc[:, "partrans"] == "log"
        df = self._df
        # self.loc[:,:] = (self.loc[:,:] * self.pst.parameter_data.scale) +\
        #                 self.pst.parameter_data.offset
        df.loc[:, li] = df.loc[:, li].apply(np.log10)
        self._istransformed = True

    def add_base(self):
        """add the control file `obsval` values as a realization

        Note:
            replaces the last realization with the current `ParameterEnsemble.pst.parameter_data.parval1` values
            as a new realization named "base"

            The PEST++ ensemble tools will add this realization also if you dont wanna fool with it here...

        """
        if "base" in self.index:
            raise Exception("'base' realization already in ensemble")
        self._df = self._df.iloc[:-1, :]
        if self.istransformed:
            self.pst.add_transform_columns()
            self.loc["base", :] = self.pst.parameter_data.loc[
                self.columns, "parval1_trans"
            ]
        else:
            self.loc["base", :] = self.pst.parameter_data.loc[self.columns, "parval1"]

    @property
    def adj_names(self):
        """the names of adjustable parameters in `ParameterEnsemble`

        Returns:
            [`str`]: adjustable parameter names

        """
        return self.pst.parameter_data.parnme.loc[~self.fixed_indexer].to_list()

    @property
    def ubnd(self):
        """the upper bound vector while respecting current log transform status

        Returns:
            `pandas.Series`: (log-transformed) upper parameter bounds listed in
            `ParameterEnsemble.pst.parameter_data.parubnd`

        """
        if not self.istransformed:
            return self.pst.parameter_data.parubnd.copy()
        else:
            ub = self.pst.parameter_data.parubnd.copy()
            ub[self.log_indexer] = np.log10(ub[self.log_indexer])
            return ub

    @property
    def lbnd(self):
        """the lower bound vector while respecting current log transform status

        Returns:
            `pandas.Series`: (log-transformed) lower parameter bounds listed in
            `ParameterEnsemble.pst.parameter_data.parlbnd`

        """
        if not self.istransformed:
            return self.pst.parameter_data.parlbnd.copy()
        else:
            lb = self.pst.parameter_data.parlbnd.copy()
            lb[self.log_indexer] = np.log10(lb[self.log_indexer])
            return lb

    @property
    def log_indexer(self):
        """boolean indexer for log transform

        Returns:
            `numpy.ndarray(bool)`: boolean array indicating which parameters are log
            transformed

        """
        istransformed = self.pst.parameter_data.partrans == "log"
        return istransformed.values

    @property
    def fixed_indexer(self):
        """boolean indexer for non-adjustable parameters

        Returns:
            `numpy.ndarray(bool)`: boolean array indicating which parameters have
            `partrans` equal to "log" or "fixed"

        """
        # isfixed = self.pst.parameter_data.partrans == "fixed"
        tf = set(["tied", "fixed"])
        isfixed = self.pst.parameter_data.partrans.apply(lambda x: x in tf)
        return isfixed.values

    def project(
        self, projection_matrix, center_on=None, log=None, enforce_bounds="reset"
    ):
        """project the ensemble using the null-space Monte Carlo method

        Args:
            projection_matrix (`pyemu.Matrix`): null-space projection operator.
            center_on (`str`): the name of the realization to use as the centering
                point for the null-space differening operation.  If `center_on` is `None`,
                the `ParameterEnsemble` mean vector is used.  Default is `None`
            log (`pyemu.Logger`, optional): for logging progress
            enforce_bounds (`str`): parameter bound enforcement option to pass to
                `ParameterEnsemble.enforce()`.  Valid options are `reset`, `drop`,
                `scale` or `None`.  Default is `reset`.

        Returns:
            `ParameterEnsemble`: untransformed, null-space projected ensemble.

        Example::

            ev = pyemu.ErrVar(jco="my.jco") #assumes my.pst exists
            pe = pyemu.ParameterEnsemble.from_gaussian_draw(ev.pst)
            pe_proj = pe.project(ev.get_null_proj(maxsing=25))
            pe_proj.to_csv("proj_par.csv")

        """

        retrans = False
        if not self.istransformed:
            self.transform()
            retrans = True

        # base = self._df.mean()
        self.pst.add_transform_columns()
        base = self.pst.parameter_data.parval1_trans
        if center_on is not None:
            if isinstance(center_on, pd.Series):
                base = center_on
            elif center_on in self._df.index:
                base = self._df.loc[center_on, :].copy()
            elif isinstance(center_on, "str"):
                try:
                    base = pyemu.pst_utils.read_parfile(center_on)
                except:
                    raise Exception(
                        "'center_on' arg not found in index and couldnt be loaded as a '.par' file"
                    )
            else:
                raise Exception(
                    "error processing 'center_on' arg.  should be realization names, par file, or series"
                )
        names = list(base.index)
        projection_matrix = projection_matrix.get(names, names)

        new_en = self.copy()

        for real in new_en.index:
            if log is not None:
                log("projecting realization {0}".format(real))

            # null space projection of difference vector
            pdiff = self._df.loc[real, names] - base
            pdiff = np.dot(projection_matrix.x, pdiff.values)
            new_en.loc[real, names] = base + pdiff

            if log is not None:
                log("projecting realization {0}".format(real))

        new_en.enforce(enforce_bounds)

        new_en.back_transform()
        if retrans:
            self.transform()
        return new_en

    def enforce(self, how="reset", bound_tol=0.0):
        """entry point for bounds enforcement.

        Args:
            enforce_bounds (`str`): can be 'reset' to reset offending values or 'drop' to drop
                offending realizations.  Default is "reset"

        Note:
            In very high dimensions, the "drop" and "scale" `how` types will result in either very few realizations
            or very short realizations.

        Example::

            pst = pyemu.Pst("my.pst")
            pe = pyemu.ParameterEnsemble.from_gaussian_draw()
            pe.enforce(how="scale")
            pe.to_csv("par.csv")


        """

        if how.lower().strip() == "reset":
            self._enforce_reset(bound_tol=bound_tol)
        elif how.lower().strip() == "drop":
            self._enforce_drop(bound_tol=bound_tol)
        elif how.lower().strip() == "scale":
            self._enforce_scale(bound_tol=bound_tol)
        else:
            raise Exception(
                "unrecognized enforce_bounds arg:"
                + "{0}, should be 'reset' or 'drop'".format(enforce_bounds)
            )

    def _enforce_scale(self, bound_tol):
        retrans = False
        if self.istransformed:
            self.back_transform()
            retrans = True
        ub = self.ubnd * (1.0 - bound_tol)
        lb = self.lbnd * (1.0 + bound_tol)
        base_vals = self.pst.parameter_data.loc[self._df.columns, "parval1"].copy()
        ub_dist = (ub - base_vals).apply(np.abs)
        lb_dist = (base_vals - lb).apply(np.abs)

        if ub_dist.min() <= 0.0:
            raise Exception(
                "Ensemble._enforce_scale() error: the following parameter"
                + "are at or over ubnd: {0}".format(
                    ub_dist.loc[ub_dist <= 0.0].index.values
                )
            )
        if lb_dist.min() <= 0.0:
            raise Exception(
                "Ensemble._enforce_scale() error: the following parameter"
                + "are at or under lbnd: {0}".format(
                    lb_dist.loc[ub_dist <= 0.0].index.values
                )
            )
        for ridx in self._df.index:
            real = self._df.loc[ridx, :]
            real_dist = (real - base_vals).apply(np.abs)
            out_ubnd = real - ub
            out_ubnd = out_ubnd.loc[out_ubnd > 0.0]
            ubnd_facs = ub_dist.loc[out_ubnd.index] / real_dist.loc[out_ubnd.index]

            out_lbnd = lb - real
            out_lbnd = out_lbnd.loc[out_lbnd > 0.0]
            lbnd_facs = lb_dist.loc[out_lbnd.index] / real_dist.loc[out_lbnd.index]

            if ubnd_facs.shape[0] == 0 and lbnd_facs.shape[0] == 0:
                continue
            lmin = 1.0
            umin = 1.0
            sign = np.ones((self.pst.npar_adj))
            sign[
                real.loc[self.pst.adj_par_names] < base_vals.loc[self.pst.adj_par_names]
            ] = -1.0
            if ubnd_facs.shape[0] > 0:
                umin = ubnd_facs.min()
                umin_idx = ubnd_facs.idxmin()
                print(
                    "enforce_scale ubnd controlling parameter, scale factor, "
                    + "current value for realization {0}: {1}, {2}, {3}".format(
                        ridx, umin_idx, umin, real.loc[umin_idx]
                    )
                )

            if lbnd_facs.shape[0] > 0:
                lmin = lbnd_facs.min()
                lmin_idx = lbnd_facs.idxmin()
                print(
                    "enforce_scale lbnd controlling parameter, scale factor, "
                    + "current value for realization {0}: {1}, {2}. {3}".format(
                        ridx, lmin_idx, lmin, real.loc[lmin_idx]
                    )
                )
            min_fac = min(umin, lmin)

            self._df.loc[ridx, :] = base_vals + (sign * real_dist * min_fac)

        if retrans:
            self.transform()

    def _enforce_drop(self, bound_tol):
        """enforce parameter bounds on the ensemble by dropping
        violating realizations

        Note:
            with a large (realistic) number of parameters, the
            probability that any one parameter is out of
            bounds is large, meaning most realization will
            be dropped.

        """
        ub = self.ubnd * (1.0 - bound_tol)
        lb = self.lbnd * (1.0 + bound_tol)
        drop = []
        for ridx in self._df.index:
            # mx = (ub - self.loc[id,:]).min()
            # mn = (lb - self.loc[id,:]).max()
            if (ub - self._df.loc[ridx, :]).min() < 0.0 or (
                lb - self._df.loc[ridx, :]
            ).max() > 0.0:
                drop.append(ridx)
        self.loc[drop, :] = np.NaN
        self.dropna(inplace=True)

    def _enforce_reset(self, bound_tol):
        """enforce parameter bounds on the ensemble by resetting
        violating vals to bound
        """

        ub = (self.ubnd * (1.0 - bound_tol)).to_dict()
        lb = (self.lbnd * (1.0 + bound_tol)).to_dict()

        val_arr = self._df.values
        for iname, name in enumerate(self.columns):
            val_arr[val_arr[:, iname] > ub[name], iname] = ub[name]
            val_arr[val_arr[:, iname] < lb[name], iname] = lb[name]
