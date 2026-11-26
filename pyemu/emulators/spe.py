"""
Sparse precision emulator (SPE).

DSI factors the empirical covariance of the simulated outputs with an SVD.
SPE instead splits the outputs into observations (o) and forecasts (f) and
models them in two pieces,

    p(f, o) = p(f | o) p(o)

each built from lasso regressions, then conditions on observations in
closed form.
"""
import warnings
import numpy as np
import pandas as pd

from pyemu.en import ObservationEnsemble
from .base import Emulator


def spe_file_forward_run(emu_file="spe.pickle", input_file="spe_pars.csv",
                         output_file="spe_sim_vals.csv"):
    """
    PEST++ forward run, one realization: read the eps_o values from
    input_file, run predict() with eps_f = 0 and write the outputs (original
    units) to output_file.
    """
    spe = SPE.load(emu_file)
    pars = pd.read_csv(input_file, index_col=0)
    # eps_o in active_obs order, then eps_f = 0 (the forecasts' conditional mean)
    names = ["eps_" + name for name in spe.active_obs]
    eps_o = pars.loc[names, "parval1"].values
    eps = np.concatenate([eps_o, np.zeros(len(spe.forecast_names))])
    sim_vals = spe.predict(eps)
    sim_vals.index.name = "obsnme"
    sim_vals.name = "simval"
    sim_vals.to_csv(output_file, header=True)


def spe_runstore_forward_run(emu_file="spe.pickle", pst_name="spe"):
    """
    PEST++ forward run for the external run manager (pestpp-ies ... /e): read
    every realization's eps_o values from the run storage file {pst_name}.rns,
    run predict() on all of them at once (eps_f = 0) and write the outputs back.
    """
    from pyemu.utils.helpers import RunStor
    spe = SPE.load(emu_file)
    rns_file = "{0}.rns".format(pst_name)
    header, par_names, obs_names = RunStor.file_info(rns_file)
    rs = RunStor(rns_file)
    df = rs.get_data()   # realizations by parameters and observations

    # RunStor lower-cases names.  eps_o in active_obs order, then eps_f = 0
    names = ["eps_" + name.lower() for name in spe.active_obs]
    eps_o = df.loc[:, names].values
    eps = np.hstack([eps_o, np.zeros((eps_o.shape[0], len(spe.forecast_names)))])
    sim_vals = spe.predict(eps)
    sim_vals.columns = [c.lower() for c in sim_vals.columns]
    df.loc[:, obs_names] = sim_vals.loc[:, obs_names].values
    rs.update(df)


def _lasso_regression(x, y, max_candidates):
    """
    Regress y on a few columns of x.

    The max_candidates columns of x most correlated with y are offered to a
    lasso regression, which picks a few of them (how many is chosen by the
    Bayesian information criterion, BIC).  the picked columns are then refit
    by least squares without the lasso shrinkage.  x and y are standardized
    (mean 0, std 1), realizations (rows) by variables (columns).

    Returns
    -------
    chosen : numpy.ndarray
        column indices of x that were picked (can be empty)
    b : numpy.ndarray
        least squares coefficients of the chosen columns
    resid_var : float
        unbiased variance of the residual y - x[:, chosen] b
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoLarsIC, LinearRegression

    nreal = x.shape[0]
    corr = x.T.dot(y) / (nreal - 1)
    candidates = np.argsort(-np.abs(corr))[:max_candidates]
    # drop exact duplicate columns (the lasso solver fails on them).  a
    # normal score transform makes any two outputs with the same rank order
    # identical, so they do happen
    _, first = np.unique(x[:, candidates], axis=1, return_index=True)
    candidates = candidates[np.sort(first)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso = LassoLarsIC(criterion="bic", fit_intercept=False)
        lasso.fit(x[:, candidates], y)
    chosen = candidates[lasso.coef_ != 0.0]
    if len(chosen) == 0:
        return chosen, np.zeros(0), y.dot(y) / (nreal - 1)

    b = LinearRegression(fit_intercept=False).fit(x[:, chosen], y).coef_
    resid = y - x[:, chosen].dot(b)
    # one degree of freedom went to the mean removed in standardizing, and one
    # to each coefficient
    resid_var = resid.dot(resid) / max(1, nreal - 1 - len(chosen))
    return chosen, b, resid_var


def _sparse_precision_factor(z, max_candidates, min_var):
    """
    Sparse factor of the precision (inverse covariance) of the columns of z.

    Each column is regressed on the columns before it (j < i), at most
    max_candidates of them:

        z_i = sum_j b_ij z_j + e_i,    var(e_i) = d_i

    which gives precision = L^T D^-1 L with L = I - B unit lower triangular.
    The column order matters: put related columns near each other.

    Parameters
    ----------
    z : numpy.ndarray
        realizations (rows) by variables (columns), standardized (mean 0, std 1).
    max_candidates : int
        passed to _lasso_regression.
    min_var : float
        floor on d_i.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        unit lower triangular factor.
    d : numpy.ndarray
        conditional variances, the diagonal of D.
    """
    import scipy.sparse as sp

    nreal, ncol = z.shape
    rows, cols, coefs = [], [], []
    d = np.zeros(ncol)
    d[0] = max(z[:, 0].dot(z[:, 0]) / (nreal - 1), min_var)
    for i in range(1, ncol):
        chosen, b, resid_var = _lasso_regression(z[:, :i], z[:, i], max_candidates)
        d[i] = max(resid_var, min_var)
        rows.extend([i] * len(chosen))
        cols.extend(chosen.tolist())
        coefs.extend(b.tolist())

    B = sp.csr_matrix((coefs, (rows, cols)), shape=(ncol, ncol))
    L = sp.identity(ncol, format="csr") - B
    return L, d


class SPE(Emulator):
    """
    Sparse precision emulator.

    The outputs are transformed and standardized with the prior ensemble mean
    and std of each output (z = (x - mean) / std), then split into
    observations z_o and forecasts z_f.  the observations are the
    nonzero-weight observations in pst that vary in the prior ensemble.
    "forecasts" here means every other output: zero-weight observations,
    observations that are constant in the prior, and anything else in data.

    Observation model, p(o).  The observations are put in reverse
    Cuthill-McKee (RCM) order, which places strongly correlated observations
    next to each other, and each is regressed on the observations before it.
    This gives a sparse factor of the observation precision:

        Q_oo = C_oo^-1 = L_oo^T D_o^-1 L_oo

    Forecast model, p(f | o).  Each forecast is regressed on the
    observations only:

        z_f = K z_o + e_f,    var(e_f) = d_f

    K holds the regression coefficients (one row per forecast) and is sparse.
    The forecast residuals e_f are treated as independent of each other, so
    forecasts are correlated with each other only through the observations
    they share in K.  forecast to forecast correlation that the observations
    do not explain is not reproduced.

    Conditioning on an obs+noise ensemble y (transformed, then standardized
    with the same prior mean and std as z_o; see condition()):

        1. prior observation draw:  z_o = L_oo^-1 D_o^1/2 eps_o
        2. update:                  z_o* = z_o + (Q_oo + R^-1)^-1 R^-1 (y - z_o)
        3. forecasts:               z_f = K z_o* + d_f^1/2 eps_f
        4. back to original units:  x = inverse_transform(mean + std * z)

    eps_o and eps_f are standard normal draws.  step 2 is the usual
    perturbed-observation (ensemble smoother) update, solved exactly with
    sparse matrices.  R^-1 is the noise precision.  It is fitted from the
    obs+noise ensemble the same way as Q_oo, so correlated noise is handled
    and R^-1 stays sparse.

    Everything (Q_oo, K, R^-1, ...) is in the transformed space, standardized
    by the prior ensemble mean and std of each output.

    Instead of condition(), the eps_o values can be estimated with
    PESTPP-IES, the same way DSI's latent values are: see prepare_pestpp().

    Parameters
    ----------
    pst : Pst
        control file.  the observed outputs are the nonzero-weight
        observations.  their names, obsvals and noise std (the
        "standard_deviation" column where present, otherwise 1/weight) are
        kept; the Pst itself is not stored.
    data : pandas.DataFrame or ObservationEnsemble
        prior ensemble of simulated outputs, realizations by outputs.
    transforms : list of dict, optional
        feature transforms, same format as the other emulators.
        row_wise_minmax is not supported.
    max_candidates : int, optional
        how many of the most correlated predictors each regression offers to
        the lasso.  this keeps the regressions cheap, and well posed when there
        are many more outputs than realizations.  default: a third of the
        number of realizations.  given or default, it is held between 1 and
        realizations - 2 (the obs+noise ensemble size when fitting R^-1).
    min_var : float, optional
        floor on the regression residual variances: for D_o and d_f in units
        of the prior std, for R^-1 in units of the noise std.  it stops D^-1
        blowing up when an output is almost perfectly predicted.
        default 1.0e-8.
    corr_threshold : float, optional
        two observations are neighbors in the RCM ordering if the absolute
        value of their prior correlation is above this.  it only sets the
        ordering; it does not remove any correlation from the model.
        default 0.5.
    verbose : bool, optional
        echo log statements.  default False.
    """

    def __init__(self, pst=None, data=None, transforms=None, max_candidates=None,
                 min_var=1.0e-8, corr_threshold=0.5, verbose=False):
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
        self.min_var = min_var
        self.corr_threshold = corr_threshold

        # keep only the observation names, values and weights: a Pst does not
        # pickle, so storing it would break save()
        obs = pst.observation_data
        self.obs_names = [c for c in self.data.columns
                          if c in obs.index and obs.loc[c, "weight"] > 0.0]
        if len(self.obs_names) == 0:
            raise ValueError("no nonzero-weight observations of pst are in data")
        self.obsval = obs.loc[self.obs_names, "obsval"].astype(float)
        # noise std for condition() without obs_plus_noise: the
        # standard_deviation column where it is given, otherwise 1/weight
        self.noise_std = 1.0 / obs.loc[self.obs_names, "weight"].astype(float)
        if "standard_deviation" in obs.columns:
            given = obs.loc[self.obs_names, "standard_deviation"].astype(float)
            self.noise_std.loc[given.notna()] = given.loc[given.notna()]

    def _get_max_candidates(self, nreal):
        """max_candidates for an ensemble with nreal realizations"""
        max_candidates = self.max_candidates
        if max_candidates is None:
            max_candidates = nreal // 3
        # the BIC needs more realizations than candidates
        return max(1, min(max_candidates, nreal - 2))

    def fit(self):
        """
        Build the observation model and the forecast model from the prior ensemble.

        Sets:

        - active_obs: nonzero-weight observations in data that vary in the
          prior, in RCM order
        - forecast_names: all other outputs, in data column order
        - output_names: active_obs followed by forecast_names
        - mean, std: numpy arrays in output_names order, transformed units;
          std is 0 for outputs that are constant in the prior
        - L_oo, d_o, Q_oo: observation model, sparse, active_obs by
          active_obs; Q_oo = L_oo^T D_o^-1 L_oo
        - K: sparse, forecast_names (rows) by active_obs (columns)
        - d_f: forecast residual variances, in forecast_names order
        - num_predictors: pandas.Series indexed by output_names.  for an
          observation, how many earlier observations it was regressed on;
          for a forecast, how many observations

        Returns
        -------
        SPE
            self, fitted.
        """
        # scipy and scikit-learn are optional pyemu dependencies
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        # 1. transform
        if self.transforms is not None:
            data = self._fit_transformer_pipeline(self.data, self.transforms)
        else:
            data = self.data
        nreal = data.shape[0]
        if nreal < 4:
            raise ValueError("SPE needs at least 4 realizations, got {0}".format(nreal))

        # 2. an observation that never varies in the prior carries no
        #    information: it is not conditioned on and is carried with the
        #    forecasts (it stays at its constant value)
        obs_data = data.loc[:, self.obs_names]
        constant = obs_data.columns[obs_data.max() == obs_data.min()].tolist()
        if len(constant) > 0:
            self.logger.warn("SPE: these observations do not vary in the prior and are "
                             "not conditioned on: {0}".format(",".join(constant)))
        active_obs = [o for o in self.obs_names if o not in constant]
        if len(active_obs) == 0:
            raise ValueError("every observation is constant in the prior")

        # 3. RCM order for the observations: neighbors are observations whose
        #    prior correlation is above corr_threshold
        corr = data.loc[:, active_obs].corr().values
        neighbors = sp.csr_matrix(np.abs(corr) > self.corr_threshold)
        order = reverse_cuthill_mckee(neighbors, symmetric_mode=True)
        self.active_obs = [active_obs[i] for i in order]
        self.forecast_names = [c for c in data.columns if c not in self.active_obs]
        self.output_names = self.active_obs + self.forecast_names
        nobs = len(self.active_obs)

        # 4. standardize.  a constant output gets std 0 and its own value as the
        #    mean (a computed mean can be off in the last digit), so it comes
        #    back exactly constant; it is divided by 1 here only to avoid 0/0
        x = data.loc[:, self.output_names].values
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        constant_out = x.max(axis=0) == x.min(axis=0)
        self.mean[constant_out] = x[0, constant_out]
        self.std[constant_out] = 0.0
        z = (x - self.mean) / np.where(self.std > 0.0, self.std, 1.0)
        z_o = z[:, :nobs]
        z_f = z[:, nobs:]

        max_candidates = self._get_max_candidates(nreal)

        # 5. observation model: Q_oo = L_oo^T D_o^-1 L_oo
        self.L_oo, self.d_o = _sparse_precision_factor(z_o, max_candidates, self.min_var)
        self.Q_oo = self.L_oo.T @ sp.diags(1.0 / self.d_o) @ self.L_oo

        # 6. forecast model: regress each forecast on the observations.  a
        #    forecast that is constant in the prior keeps an empty row of K and
        #    d_f = 1; its std is 0, so it comes back exactly constant
        nfore = len(self.forecast_names)
        rows, cols, coefs = [], [], []
        self.d_f = np.ones(nfore)
        for j in range(nfore):
            if self.std[nobs + j] == 0.0:
                continue
            chosen, b, resid_var = _lasso_regression(z_o, z_f[:, j], max_candidates)
            self.d_f[j] = max(resid_var, self.min_var)
            rows.extend([j] * len(chosen))
            cols.extend(chosen.tolist())
            coefs.extend(b.tolist())
        self.K = sp.csr_matrix((coefs, (rows, cols)), shape=(nfore, nobs))

        # nonzeros per row of L_oo, minus the unit diagonal, are the predictors
        num_obs_pred = np.diff(self.L_oo.indptr) - 1
        num_fore_pred = np.diff(self.K.indptr)
        self.num_predictors = pd.Series(np.concatenate([num_obs_pred, num_fore_pred]),
                                        index=self.output_names)
        self.fitted = True
        return self

    def _draw_obs(self, eps_o):
        """standard normal values (observations by realizations) -> standardized
        observations: solve L_oo z_o = D_o^1/2 eps_o"""
        import scipy.sparse.linalg as spla
        return spla.spsolve_triangular(self.L_oo, eps_o * np.sqrt(self.d_o)[:, None],
                                       lower=True, unit_diagonal=True)

    def _to_dataframe(self, z, index=None):
        """standardized outputs by realizations -> realizations by outputs, original units"""
        df = pd.DataFrame(z.T * self.std + self.mean, columns=self.output_names, index=index)
        if self.transforms is not None:
            df = self.transformer_pipeline.inverse(df)
        return df.loc[:, self.data.columns]

    def predict(self, pvals):
        """
        Map standard normal values to outputs.  standard normal pvals give
        draws from the emulated prior.

        Parameters
        ----------
        pvals : pandas.Series, pandas.DataFrame or numpy.ndarray
            one standard normal value per output per realization, by position
            in output_names order: the first len(active_obs) are eps_o, the
            rest are eps_f.  column names are not used.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            outputs in original units, columns in the order of data.
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before prediction")
        single = np.ndim(pvals) == 1
        index = pvals.index if isinstance(pvals, pd.DataFrame) else None
        eps = np.atleast_2d(np.asarray(pvals, dtype=float)).T
        if eps.shape[0] != len(self.output_names):
            raise ValueError("pvals must have {0} values per realization, got {1}".format(
                len(self.output_names), eps.shape[0]))
        nobs = len(self.active_obs)
        z_o = self._draw_obs(eps[:nobs])
        z_f = self.K @ z_o + eps[nobs:] * np.sqrt(self.d_f)[:, None]
        df = self._to_dataframe(np.vstack([z_o, z_f]), index=index)
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
            seed for the standard normal draws.

        Returns
        -------
        pandas.DataFrame
            realizations by outputs, original units, columns in the order of data.
        """
        rng = np.random.default_rng(seed)
        return self.predict(rng.standard_normal((num_reals, len(self.output_names))))

    def _obs_plus_noise_to_z(self, obs_plus_noise):
        """obs+noise ensemble -> transformed, standardized observations,
        realizations by active_obs"""
        if isinstance(obs_plus_noise, ObservationEnsemble):
            obs_plus_noise = obs_plus_noise._df
        if not isinstance(obs_plus_noise, pd.DataFrame) or obs_plus_noise.shape[0] < 4:
            raise ValueError("obs_plus_noise must be a DataFrame or ObservationEnsemble "
                             "with at least 4 rows")
        missing = [o for o in self.active_obs if o not in obs_plus_noise.columns]
        if len(missing) > 0:
            raise ValueError("obs_plus_noise is missing observations: {0}".format(",".join(missing)))

        # the transforms are given every output column (the non-observations at
        # their prior mean) since some transforms need all of them
        nobs = len(self.active_obs)
        y = pd.DataFrame([self.data.mean()] * obs_plus_noise.shape[0], index=obs_plus_noise.index)
        y.loc[:, self.active_obs] = obs_plus_noise.loc[:, self.active_obs].astype(float).values
        if self.transforms is not None:
            y = self.transformer_pipeline.transform(y)
        return (y.loc[:, self.active_obs].values - self.mean[:nobs]) / self.std[:nobs]

    def noise_precision(self, obs_plus_noise):
        """
        Sparse noise precision R^-1 fitted from an obs+noise ensemble.

        The noise is each realization minus the ensemble mean.  It is fitted
        the same way as Q_oo (regressions in active_obs order), so correlated
        noise is captured; uncorrelated noise gives a diagonal R^-1.

        Parameters
        ----------
        obs_plus_noise : pandas.DataFrame or ObservationEnsemble
            see condition().

        Returns
        -------
        scipy.sparse.csr_matrix
            R^-1, active_obs by active_obs, transformed and standardized space.
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before fitting the noise")
        return self._noise_precision_from_z(self._obs_plus_noise_to_z(obs_plus_noise))

    def _noise_precision_from_z(self, y):
        """R^-1 from transformed, standardized obs+noise y (realizations by active_obs)"""
        import scipy.sparse as sp
        noise = y - y.mean(axis=0)
        noise_std = noise.std(axis=0, ddof=1)
        if (noise_std <= 0.0).any():
            # a normal_score transform clamps values outside the prior range,
            # which can also remove the noise
            raise ValueError("obs_plus_noise has no noise for some observations after "
                             "transforming (a normal_score transform clamps values outside "
                             "the prior range)")

        # fit on unit-variance noise, then scale back: R^-1 = S^-1 L^T D^-1 L S^-1
        max_candidates = self._get_max_candidates(noise.shape[0])
        L_r, d_r = _sparse_precision_factor(noise / noise_std, max_candidates, self.min_var)
        S_inv = sp.diags(1.0 / noise_std)
        return (S_inv @ L_r.T @ sp.diags(1.0 / d_r) @ L_r @ S_inv).tocsr()

    def condition(self, obs_plus_noise=None, num_reals=100, seed=None):
        """
        Condition the emulated outputs on an obs+noise ensemble.

        Parameters
        ----------
        obs_plus_noise : pandas.DataFrame or ObservationEnsemble, optional
            observed values plus noise, realizations by observations, at least
            4 rows (for example ObservationEnsemble.from_gaussian_draw(pst)).
            it needs a column for every observation in active_obs (the
            nonzero-weight observations that vary in the prior); other columns
            are ignored.  the noise precision R^-1 is fitted from this
            ensemble (see noise_precision()).  with a normal_score transform,
            values outside the prior range are clamped, which shrinks the noise.

            if omitted, it is drawn from the pst used to build the emulator:
            obsval plus uncorrelated Gaussian noise, in original units.  the
            noise std is the observation data "standard_deviation" column
            where it is present and not nan, otherwise 1/weight.

            a single vector of observed values is not accepted: it carries no
            noise, so there is nothing to fit R^-1 from.  omit obs_plus_noise
            to condition on the pst obsvals instead.
        num_reals : int, optional
            number of realizations to draw when obs_plus_noise is omitted.
            default 100.
        seed : int, optional
            seed for the noise draws (when obs_plus_noise is omitted) and the
            prior draws.

        Returns
        -------
        pandas.DataFrame
            one posterior realization per row of obs_plus_noise, original
            units, columns in the order of data.
        """
        import scipy.sparse.linalg as spla
        if not self.fitted:
            raise ValueError("SPE must be fitted before conditioning")
        rng = np.random.default_rng(seed)

        # 0. no ensemble given: obsval plus noise with std noise_std, from the pst
        if obs_plus_noise is None:
            noise = rng.standard_normal((num_reals, len(self.obs_names))) * self.noise_std.values
            obs_plus_noise = pd.DataFrame(self.obsval.values + noise, columns=self.obs_names)

        # y: transformed and standardized, realizations by active_obs
        y = self._obs_plus_noise_to_z(obs_plus_noise)
        R_inv = self._noise_precision_from_z(y)
        y = y.T   # now active_obs by realizations, like z_o
        nobs, nreal = y.shape
        nfore = len(self.forecast_names)

        eps_o = rng.standard_normal((nobs, nreal))
        eps_f = rng.standard_normal((nfore, nreal))

        # 1. prior observation draw: z_o = L_oo^-1 D_o^1/2 eps_o
        z_o = self._draw_obs(eps_o)

        # 2. update: z_o* = z_o + (Q_oo + R^-1)^-1 R^-1 (y - z_o)
        #    Q_oo + R^-1 is the posterior precision of z_o
        post_precision = (self.Q_oo + R_inv).tocsc()
        z_o_post = z_o + spla.spsolve(post_precision, R_inv @ (y - z_o))

        # 3. forecasts: z_f = K z_o* + d_f^1/2 eps_f
        z_f = self.K @ z_o_post + eps_f * np.sqrt(self.d_f)[:, None]

        # 4. back to original units
        return self._to_dataframe(np.vstack([z_o_post, z_f]), index=obs_plus_noise.index)

    def _get_emulator_parameters(self, pst=None):
        """
        PEST++ parameters: the eps_o, one standard normal value per active
        observation, named eps_<observation name>, in active_obs order.
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before calling prepare_pestpp")
        par_names = ["eps_{0}".format(name) for name in self.active_obs]
        df = pd.DataFrame(index=par_names)
        df["parnme"] = par_names
        df["parval1"] = 0.0         # prior mean of a standard normal value
        df["parlbnd"] = -4.0        # 4 standard deviations either side
        df["parubnd"] = 4.0
        df["pargp"] = "spe_pars"
        df["partrans"] = "none"
        return df

    def _get_emulator_observations(self, pst=None):
        """
        PEST++ observations: every output in data.  the nonzero-weight
        observations get their obsval and weight 1/noise_std; everything else
        gets the prior mean and weight 0.  if a pst is passed to
        prepare_pestpp(), its observation data replaces these values.
        """
        df = pd.DataFrame(index=self.data.columns)
        df["obsnme"] = self.data.columns
        df["obsval"] = self.data.mean(axis=0)
        df["weight"] = 0.0
        df.loc[self.obs_names, "obsval"] = self.obsval
        df.loc[self.obs_names, "weight"] = 1.0 / self.noise_std
        df["obgnme"] = "spe_obs"
        return df

    def _write_forward_run_script(self, filename, emu_file, input_file, output_file,
                                  class_name, pst_name=None):
        """forward_run.py: calls spe_file_forward_run(), or
        spe_runstore_forward_run() when prepare_pestpp(use_runstor=True)"""
        if getattr(self, "_use_runstor", False):
            call = "spe_runstore_forward_run(emu_file='{0}', pst_name='{1}')".format(
                emu_file, pst_name if pst_name is not None else "spe")
        else:
            call = "spe_file_forward_run('{0}', '{1}', '{2}')".format(
                emu_file, input_file, output_file)
        lines = [
            "from pyemu.emulators.spe import spe_file_forward_run, spe_runstore_forward_run",
            "",
            "if __name__ == '__main__':",
            "    " + call,
        ]
        with open(filename, "w") as f:
            for line in lines:
                f.write(line + "\n")

    def prepare_pestpp(self, t_d, pst=None, use_runstor=False, pst_name="spe", verbose=False):
        """
        Set up a PESTPP-IES interface for the emulator, like DSI.

        The parameters are the standard normal values eps_o, one per active
        observation (the nonzero-weight observations that vary in the prior),
        named eps_<observation name>, with prior mean 0 and bounds -4 and 4.
        PESTPP-IES takes the prior std of a parameter from its bounds,
        (upper - lower) / par_sigma_range, so par_sigma_range is set to 8 to
        make that std 1.  PESTPP-IES estimates the eps_o, and handles the
        obs+noise itself, instead of condition().

        The forecasts are run with eps_f = 0, so each forecast is its
        conditional mean given the observations, z_f = K z_o.  the forecast
        residual variance d_f is not in the PESTPP-IES posterior: the
        forecast spread comes only from the spread of the observations.

        Parameters
        ----------
        t_d : str
            template directory to write.  it is removed first if it exists.
        pst : Pst, optional
            control file whose observation data (obsval, weight, ...) and
            pestpp options are carried into the new control file.  if not
            given, the obsvals and weights (1/noise_std) kept from the pst used
            to build the emulator are used.
        use_runstor : bool, optional
            write a forward_run.py for the PEST++ external run manager
            (pestpp-ies <pst_name>.pst /e): every realization is run in one
            call through the run storage file, which is much faster than one
            python start per realization.  default False.
        pst_name : str, optional
            name (without .pst) the control file will be written as.  only
            used with use_runstor, to find <pst_name>.rns.  default "spe".
        verbose : bool, optional
            echo log statements.

        Returns
        -------
        Pst
            control file for the emulator, noptmax 0.  it is not written to t_d.
        """
        pst_obj = super().prepare_pestpp(t_d, pst=pst, verbose=verbose,
                                         tpl_filename="spe_pars.csv.tpl",
                                         input_filename="spe_pars.csv",
                                         ins_filename="spe_sim_vals.csv.ins",
                                         output_filename="spe_sim_vals.csv",
                                         emu_filename="spe.pickle",
                                         use_runstor=use_runstor,
                                         pst_name=pst_name)

        # prior std of every eps value: (4 - -4) / 8 = 1
        pst_obj.pestpp_options["par_sigma_range"] = 8
        return pst_obj
