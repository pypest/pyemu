"""
Sparse precision emulator (SPE).

DSI factors the empirical covariance of the simulated outputs with an SVD.
SPE instead builds a sparse factor of their precision (inverse covariance)
from a sequence of regressions, then conditions on observations in closed form.
"""
import warnings
import numpy as np
import pandas as pd

from pyemu.en import ObservationEnsemble
from .base import Emulator


class SPE(Emulator):
    """
    Sparse precision emulator.

    The outputs are transformed, standardized, and ordered with the
    observations first.  Each output is regressed on a few of the outputs
    before it:

        z_i = sum_j b_ij z_j + e_i,    var(e_i) = d_i

    The lasso (BIC) picks which predecessors each output uses, then the b_ij
    are refit by least squares without the lasso shrinkage.  This gives the
    prior precision Q = L^T D^-1 L with L = I - B, and L (and Q) are sparse.
    L, d and Q are all in the transformed, standardized space.

    Putting the observations first means each forecast is regressed on the
    observations (never the other way round), and it makes conditioning
    simple.  With S the matrix that picks out the observations and R the
    diagonal matrix of their noise variances, the posterior precision is
    P = Q + S^T R^-1 S, and because the observations are the leading block,
    S^T R^-1 S only adds 1/r to the first nobs diagonal entries of Q.  Each
    posterior realization is a prior draw pulled toward one row of an
    obs+noise ensemble.

    Parameters
    ----------
    pst : Pst
        control file.  the observed outputs are the nonzero-weight
        observations.  only their names are kept.
    data : pandas.DataFrame or ObservationEnsemble
        prior ensemble of simulated outputs, realizations by outputs.
    transforms : list of dict, optional
        feature transforms, same format as the other emulators.
        row_wise_minmax is not supported.
    max_candidates : int, optional
        how many of the most correlated predecessors each output offers to the
        lasso.  this keeps the regressions cheap, and well posed when there
        are many more outputs than realizations.  default: a third of the
        number of realizations, held between 1 and realizations - 2.
    min_var : float, optional
        floor on the conditional variances d_i, in standardized units (a
        fraction of each output's prior variance).  it stops D^-1 blowing up
        when an output is almost perfectly predicted by its predecessors.
        default 1.0e-8.
    verbose : bool, optional
        echo log statements.  default False.
    """

    def __init__(self, pst=None, data=None, transforms=None, max_candidates=None,
                 min_var=1.0e-8, verbose=False):
        super().__init__(transforms=transforms, verbose=verbose)
        if pst is None:
            raise ValueError("SPE requires a pst")
        if data is None:
            raise ValueError("SPE requires a 'data' ensemble of simulated outputs")
        if transforms is not None and any(t.get("type") == "row_wise_minmax" for t in transforms):
            # row-wise scaling depends on the whole training realization, so it
            # cannot be applied to an obs+noise ensemble in condition()
            raise ValueError("SPE does not support the row_wise_minmax transform")
        if isinstance(data, ObservationEnsemble):
            data = data._df

        self.data = data.astype(float)
        self.max_candidates = max_candidates
        self.min_var = float(min_var)

        # keep only the observation names: a Pst does not pickle, so storing
        # it would break save()
        obs = pst.observation_data
        self.obs_names = [c for c in self.data.columns
                          if c in obs.index and obs.loc[c, "weight"] > 0.0]
        if len(self.obs_names) == 0:
            raise ValueError("no nonzero-weight observations of pst are in data")

    def fit(self):
        """
        Build the sparse precision factor from the prior ensemble.

        Sets output_names (fit order, observations first), active_obs (the
        observations that vary in the prior), mean and std (standardization),
        L, d and Q (standardized space), and num_predecessors (how many earlier
        outputs each output was regressed on).

        Returns
        -------
        SPE
            self, fitted.
        """
        # scipy and scikit-learn are optional pyemu dependencies
        import scipy.sparse as sp
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LassoLarsIC, LinearRegression

        # 1. transform
        if self.transforms is not None:
            data = self._fit_transformer_pipeline(self.data, self.transforms)
        else:
            data = self.data.copy()

        # 2. an observation that never varies in the prior carries no information
        obs_data = data.loc[:, self.obs_names]
        constant = obs_data.columns[obs_data.max() == obs_data.min()].tolist()
        if len(constant) > 0:
            self.logger.warn("SPE: these observations do not vary in the prior and are "
                             "not conditioned on: {0}".format(",".join(constant)))
        self.active_obs = [o for o in self.obs_names if o not in constant]
        if len(self.active_obs) == 0:
            raise ValueError("every observation is constant in the prior")

        # 3. observations first, then everything else (see the class docstring)
        self.output_names = self.active_obs + [c for c in data.columns if c not in self.active_obs]
        x = data.loc[:, self.output_names].values
        nreal, nout = x.shape
        if nreal < 4:
            raise ValueError("SPE needs at least 4 realizations, got {0}".format(nreal))

        # 4. standardize.  a constant output gets std 0 and its own value as the
        #    mean (a computed mean can be off in the last digit), so it comes
        #    back exactly constant; it is divided by 1 here only to avoid 0/0
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        constant_out = x.max(axis=0) == x.min(axis=0)
        self.mean[constant_out] = x[0, constant_out]
        self.std[constant_out] = 0.0
        z = (x - self.mean) / np.where(self.std > 0.0, self.std, 1.0)

        # 5. regress each output on its predecessors
        max_candidates = self.max_candidates
        if max_candidates is None:
            max_candidates = nreal // 3
        # the BIC needs more realizations than candidates
        max_candidates = max(1, min(max_candidates, nreal - 2))

        rows, cols, coefs = [], [], []
        # an output with no predecessors has e_i = z_i, so d_i = var(z_i) = 1
        d = np.ones(nout)
        for i in range(1, nout):
            z_i = z[:, i]
            if self.std[i] == 0.0:
                continue

            # offer the lasso only the most correlated predecessors
            corr = z[:, :i].T.dot(z_i) / (nreal - 1)
            candidates = np.argsort(-np.abs(corr))[:max_candidates]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                lasso = LassoLarsIC(criterion="bic", fit_intercept=False)
                lasso.fit(z[:, candidates], z_i)
            chosen = candidates[lasso.coef_ != 0.0]
            if len(chosen) == 0:
                continue

            # refit the chosen predecessors without the lasso shrinkage
            b = LinearRegression(fit_intercept=False).fit(z[:, chosen], z_i).coef_
            resid = z_i - z[:, chosen].dot(b)
            # unbiased residual variance: one degree of freedom went to the mean
            # removed in standardizing, and one to each coefficient
            d[i] = max(resid.dot(resid) / max(1, nreal - 1 - len(chosen)), self.min_var)

            rows.extend([i] * len(chosen))
            cols.extend(chosen.tolist())
            coefs.extend(b.tolist())

        # 6. Q = L^T D^-1 L
        B = sp.csr_matrix((coefs, (rows, cols)), shape=(nout, nout))
        self.L = sp.identity(nout, format="csr") - B
        self.d = d
        self.Q = self.L.T @ sp.diags(1.0 / d) @ self.L
        self.num_predecessors = pd.Series(np.diff(B.indptr), index=self.output_names)
        self.fitted = True
        return self

    def _latent_to_z(self, e):
        """latent standard normal values (outputs by realizations) -> standardized
        outputs: solve L z = D^1/2 e"""
        import scipy.sparse.linalg as spla
        return spla.spsolve_triangular(self.L, e * np.sqrt(self.d)[:, None],
                                       lower=True, unit_diagonal=True)

    def _to_dataframe(self, z, index=None):
        """standardized outputs by realizations -> realizations by outputs, original units"""
        df = pd.DataFrame(z.T * self.std + self.mean, columns=self.output_names, index=index)
        if self.transforms is not None:
            df = self.transformer_pipeline.inverse(df)
        return df.loc[:, self.data.columns]

    def predict(self, pvals):
        """
        Map latent standard normal values e to outputs: z = L^-1 D^1/2 e, then
        un-standardize and inverse-transform.  standard normal pvals give draws
        from the emulated prior.

        Parameters
        ----------
        pvals : pandas.Series, pandas.DataFrame or numpy.ndarray
            one standard normal value per output per realization, by position
            in output_names order (observations first); column names are not
            used.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            outputs in original units, columns in the order of data.
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before prediction")
        single = np.ndim(pvals) == 1
        index = pvals.index if isinstance(pvals, pd.DataFrame) else None
        e = np.atleast_2d(np.asarray(pvals, dtype=float))
        if e.shape[1] != len(self.output_names):
            raise ValueError("pvals must have {0} values per realization, got {1}".format(
                len(self.output_names), e.shape[1]))
        df = self._to_dataframe(self._latent_to_z(e.T), index=index)
        if single:
            return df.iloc[0]
        return df

    def prior_ensemble(self, num_reals=100, seed=None):
        """
        Draw from the emulated prior.

        Parameters
        ----------
        num_reals : int, optional
            number of realizations to draw.  default 100.
        seed : int, optional
            seed for the latent standard normal draws.

        Returns
        -------
        pandas.DataFrame
            realizations by outputs, original units, columns in the order of data.
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before drawing")
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((len(self.output_names), num_reals))
        return self._to_dataframe(self._latent_to_z(e))

    def condition(self, obs_plus_noise, seed=None):
        """
        Condition the emulated outputs on an obs+noise ensemble.

        Parameters
        ----------
        obs_plus_noise : pandas.DataFrame or ObservationEnsemble
            observed values plus noise, realizations by observations, at least
            2 rows (for example ObservationEnsemble.from_gaussian_draw(pst)).
            it needs a column for every nonzero-weight observation; other
            columns are ignored.  the noise variance of each observation (R) is
            taken from this ensemble, not from the pst weights.

            a single vector of observed values is not accepted: it carries no
            noise, so there is nothing to take the noise variance from.  to
            condition on just the observed values in pst, draw the obs+noise
            ensemble from the weights first:

                oe = ObservationEnsemble.from_gaussian_draw(pst, num_reals=100)
                post = spe.condition(oe)
        seed : int, optional
            seed for the prior draws.

        Returns
        -------
        pandas.DataFrame
            one posterior realization per row of obs_plus_noise, original
            units, columns in the order of data.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        if not self.fitted:
            raise ValueError("SPE must be fitted before conditioning")
        if isinstance(obs_plus_noise, ObservationEnsemble):
            obs_plus_noise = obs_plus_noise._df
        if not isinstance(obs_plus_noise, pd.DataFrame) or obs_plus_noise.shape[0] < 2:
            raise ValueError("obs_plus_noise must be a DataFrame or ObservationEnsemble "
                             "with at least 2 rows")
        missing = [o for o in self.active_obs if o not in obs_plus_noise.columns]
        if len(missing) > 0:
            raise ValueError("obs_plus_noise is missing observations: {0}".format(",".join(missing)))

        nobs = len(self.active_obs)
        nout = len(self.output_names)
        nreal = obs_plus_noise.shape[0]

        # 1. into the transformed, standardized training space.  the transforms
        #    are given every output column (the non-observations at their prior
        #    mean) since some transforms need all of them
        y = pd.DataFrame([self.data.mean()] * nreal, index=obs_plus_noise.index)
        y.loc[:, self.active_obs] = obs_plus_noise.loc[:, self.active_obs].astype(float).values
        if self.transforms is not None:
            y = self.transformer_pipeline.transform(y)
        y = (y.loc[:, self.active_obs].values - self.mean[:nobs]) / self.std[:nobs]

        # 2. noise variance of each observation (the diagonal of R), from the
        #    ensemble itself, in standardized units
        r = y.var(axis=0, ddof=1)
        if (r <= 0.0).any():
            raise ValueError("obs_plus_noise has no noise for some observations")

        # 3. posterior precision P = Q + S^T R^-1 S.  the observations are first,
        #    so S^T R^-1 S just adds 1/r to the first nobs diagonal entries of Q
        noise_precision = np.zeros(nout)
        noise_precision[:nobs] = 1.0 / r
        P = (self.Q + sp.diags(noise_precision)).tocsc()

        # 4. pull one prior draw toward each row of obs_plus_noise:
        #    z = z_prior + P^-1 S^T R^-1 (y - S z_prior)
        rng = np.random.default_rng(seed)
        z_prior = self._latent_to_z(rng.standard_normal((nout, nreal)))
        misfit = np.zeros((nout, nreal))   # S^T R^-1 (y - S z_prior)
        misfit[:nobs] = (y.T - z_prior[:nobs]) / r[:, None]
        z_post = z_prior + spla.spsolve(P, misfit)

        return self._to_dataframe(z_post, index=obs_plus_noise.index)

    def prepare_pestpp(self, t_d, pst=None, verbose=False, **kwargs):
        """not supported: SPE conditions in closed form with condition()"""
        raise NotImplementedError("SPE conditions in closed form - use condition() rather than "
                                  "prepare_pestpp()")
