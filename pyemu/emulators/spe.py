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

    The outputs are standardized and ordered with the observations first.  Each
    output is regressed on a few of the outputs before it:

        z_i = sum_j b_ij z_j + e_i,    var(e_i) = d_i

    which gives the prior precision Q = L^T D^-1 L with L = I - B.  The lasso
    picks which predecessors each output uses, so L (and Q) are sparse.

    Conditioning on observations adds the noise precision to the observation
    part of the diagonal:  P = Q + S^T R^-1 S.  Each posterior realization is a
    prior draw pulled toward one row of an obs+noise ensemble.  condition()
    needs that ensemble, not a single vector of observed values; see
    condition() for the one-line work-around.

    Parameters
    ----------
    pst : Pst
        control file.  the observed outputs are the nonzero-weight observations.
    data : pandas.DataFrame or ObservationEnsemble
        prior ensemble of simulated outputs, realizations by outputs.
    transforms : list of dict, optional
        feature transforms, same format as the other emulators.
    max_candidates : int, optional
        how many of the most correlated predecessors each output offers to the
        lasso.  default: a third of the number of realizations.
    min_var : float, optional
        floor on the conditional variances d_i.  default 1.0e-8.
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
        if isinstance(data, ObservationEnsemble):
            data = data._df
        if transforms is not None:
            self._validate_transforms(transforms)

        self.pst = pst
        self.data = data.astype(float).copy()
        self.max_candidates = max_candidates
        self.min_var = float(min_var)

        obs = pst.observation_data
        self.obs_names = [c for c in self.data.columns
                          if c in obs.index and obs.loc[c, "weight"] > 0.0]
        if len(self.obs_names) == 0:
            raise ValueError("no nonzero-weight observations of pst are in data")

    def fit(self):
        """
        Build the sparse precision factor from the prior ensemble.

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
        obs_std = data.loc[:, self.obs_names].std()
        constant = obs_std.index[obs_std == 0.0].tolist()
        if len(constant) > 0:
            self.logger.warn("SPE: these observations do not vary in the prior and are "
                             "not conditioned on: {0}".format(",".join(constant)))
        self.active_obs = [o for o in self.obs_names if o not in constant]
        if len(self.active_obs) == 0:
            raise ValueError("every observation is constant in the prior")

        # 3. observations first, then everything else
        self.output_names = self.active_obs + [c for c in data.columns if c not in self.active_obs]
        x = data.loc[:, self.output_names].values
        nreal, nout = x.shape
        if nreal < 4:
            raise ValueError("SPE needs at least 4 realizations, got {0}".format(nreal))

        # 4. standardize
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        self.std[self.std == 0.0] = 1.0
        z = (x - self.mean) / self.std

        # 5. regress each output on its predecessors
        max_candidates = self.max_candidates
        if max_candidates is None:
            max_candidates = nreal // 3
        max_candidates = max(1, min(max_candidates, nreal - 2))

        rows, cols, coefs = [], [], []
        d = np.zeros(nout)
        for i in range(nout):
            y = z[:, i]
            d[i] = max(y.var(ddof=1), self.min_var)
            if i == 0 or y.var() == 0.0:
                continue

            # offer the lasso only the most correlated predecessors
            corr = z[:, :i].T.dot(y) / (nreal - 1)
            candidates = np.argsort(-np.abs(corr))[:max_candidates]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                lasso = LassoLarsIC(criterion="bic", fit_intercept=False)
                lasso.fit(z[:, candidates], y)
            chosen = candidates[lasso.coef_ != 0.0]
            if len(chosen) == 0:
                continue

            # refit the chosen predecessors without the lasso shrinkage
            b = LinearRegression(fit_intercept=False).fit(z[:, chosen], y).coef_
            resid = y - z[:, chosen].dot(b)
            d[i] = max(resid.dot(resid) / max(1, nreal - 1 - len(chosen)), self.min_var)

            rows.extend([i] * len(chosen))
            cols.extend(chosen.tolist())
            coefs.extend(b.tolist())

        # 6. Q = L^T D^-1 L
        B = sp.csr_matrix((coefs, (rows, cols)), shape=(nout, nout))
        self.L = (sp.identity(nout, format="csr") - B).tocsr()
        self.d = d
        self.Q = (self.L.T @ sp.diags(1.0 / d) @ self.L).tocsc()
        self.num_predecessors = pd.Series(np.diff(B.indptr), index=self.output_names)
        self.fitted = True
        return self

    def _draw_prior(self, nreal, rng):
        """standardized prior draws, outputs by realizations: solve L z = D^1/2 e"""
        import scipy.sparse.linalg as spla
        e =rng.standard_normal((len(self.output_names), nreal)) * np.sqrt(self.d)[:, None]
        return spla.spsolve_triangular(self.L, e, lower=True, unit_diagonal=True)

    def _to_dataframe(self, z, index=None):
        """standardized outputs by realizations -> realizations by outputs, original units"""
        df = pd.DataFrame(z.T * self.std + self.mean, columns=self.output_names, index=index)
        if self.transforms is not None:
            df = self.transformer_pipeline.inverse(df)
        return df.loc[:, self.data.columns]

    def predict(self, pvals):
        """
        Map latent standard normal values to outputs: z = L^-1 D^1/2 e.

        Parameters
        ----------
        pvals : pandas.Series, pandas.DataFrame or numpy.ndarray
            one value per output (in output_names order) per realization.

        Returns
        -------
        pandas.Series or pandas.DataFrame
        """
        import scipy.sparse.linalg as spla
        if not self.fitted:
            raise ValueError("SPE must be fitted before prediction")
        single = np.ndim(pvals) == 1
        index = pvals.index if isinstance(pvals, pd.DataFrame) else None
        e = np.atleast_2d(np.asarray(pvals, dtype=float))
        if e.shape[1] != len(self.output_names):
            raise ValueError("pvals must have {0} values per realization, got {1}".format(
                len(self.output_names), e.shape[1]))
        z = spla.spsolve_triangular(self.L, (e * np.sqrt(self.d)).T, lower=True, unit_diagonal=True)
        df = self._to_dataframe(z.reshape(len(self.output_names), -1), index=index)
        if single:
            return df.iloc[0]
        return df

    def prior_ensemble(self, num_reals=100, seed=None):
        """
        Draw from the emulated prior.

        Returns
        -------
        ObservationEnsemble
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before drawing")
        z = self._draw_prior(num_reals, np.random.default_rng(seed))
        return ObservationEnsemble(self.pst, self._to_dataframe(z))

    def condition(self, obs_plus_noise, seed=None):
        """
        Condition the emulated outputs on an obs+noise ensemble.

        Parameters
        ----------
        obs_plus_noise : pandas.DataFrame or ObservationEnsemble
            observed values plus noise, realizations by observations (for
            example ObservationEnsemble.from_gaussian_draw(pst)).  it needs a
            column for every nonzero-weight observation; other columns are
            ignored.  the noise variance of each observation is taken from
            this ensemble.

            a single vector of observed values is not accepted: it carries no
            noise, so there is nothing to take the noise variance from (a
            pandas.Series raises an error, and a one-row DataFrame gives an
            all-NaN posterior).  to condition on just the observed values in
            pst, draw the obs+noise ensemble from the weights first:

                oe = ObservationEnsemble.from_gaussian_draw(pst, num_reals=100)
                post = spe.condition(oe)
        seed : int, optional
            seed for the prior draws.

        Returns
        -------
        ObservationEnsemble
            one posterior realization per row of obs_plus_noise.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        if not self.fitted:
            raise ValueError("SPE must be fitted before conditioning")
        if isinstance(obs_plus_noise, ObservationEnsemble):
            obs_plus_noise = obs_plus_noise._df
        missing = [o for o in self.active_obs if o not in obs_plus_noise.columns]
        if len(missing) > 0:
            raise ValueError("obs_plus_noise is missing observations: {0}".format(",".join(missing)))

        nobs = len(self.active_obs)
        nout = len(self.output_names)
        nreal = obs_plus_noise.shape[0]

        # 1. into the standardized training space
        y = obs_plus_noise.loc[:, self.active_obs].astype(float)
        if self.transforms is not None:
            y = self.transformer_pipeline.transform(y)
        y = (y.values - self.mean[:nobs]) / self.std[:nobs]   # realizations by observations

        # 2. noise variance of each observation, from the ensemble itself
        r = y.var(axis=0, ddof=1)
        if (r <= 0.0).any():
            raise ValueError("obs_plus_noise has no noise for some observations")

        # 3. posterior precision: the observations are first, so S^T R^-1 S
        #    just adds 1/r to the first nobs diagonal entries of Q
        noise_precision = np.zeros(nout)
        noise_precision[:nobs] = 1.0 / r
        P = (self.Q + sp.diags(noise_precision)).tocsc()
        P_solver = spla.splu(P)

        # 4. pull one prior draw toward each row of obs_plus_noise:
        #    z = z_prior + P^-1 S^T R^-1 (y - S z_prior)
        z_prior = self._draw_prior(nreal, np.random.default_rng(seed))
        rhs = np.zeros((nout, nreal))
        rhs[:nobs] = (y.T - z_prior[:nobs]) / r[:, None]
        z_post = z_prior + P_solver.solve(rhs)

        return ObservationEnsemble(self.pst, self._to_dataframe(z_post, index=obs_plus_noise.index))

    def prepare_pestpp(self, t_d, pst=None, verbose=False, **kwargs):
        """not supported: SPE conditions in closed form with condition()"""
        raise NotImplementedError("SPE conditions in closed form - use condition() rather than "
                                  "prepare_pestpp()")
