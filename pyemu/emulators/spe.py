"""
sparse precision emulator (spe): a data-space emulator built on a sparse factor
of the precision of the simulated outputs, rather than on a truncated svd of
their covariance like dsi.
"""
import warnings
import numpy as np
import pandas as pd

from .base import Emulator


class SPE(Emulator):
    """
    sparse precision emulator.

    the outputs (observation equivalents and predictions) are standardized and
    put in an order with the observations first.  each output is lasso
    regressed on the outputs before it - only its most correlated predecessors
    are offered to the lasso - and the chosen ones are refit without shrinkage.
    that gives z_i = sum_j b_ij z_j + e_i with var(e_i) = d_i, so the prior
    precision is Q = L^T D^-1 L with L = I - B unit lower triangular and
    sparse.  the factor is full rank: there is no truncation and no ensemble
    size rank limit, the zeros in B stand in for it.

    conditioning picks the observed outputs out exactly, so the posterior
    precision P = Q + S^T R^-1 S is sparse and one sparse factorization gives
    the posterior mean, exact gaussian posterior realizations (perturbed prior
    draws plus perturbed observations) and the leverages (hat matrix diagonal)
    of the observations.

    the observed set is fixed at fit time because it sets the order of the
    factor - changing which observations have weight needs a refit.  a
    zero-weight observation that is in the observed set is simply left out of
    the conditioning.

    Parameters
    ----------
    pst : Pst or pandas.DataFrame, optional
        a control file, or its observation_data.  used to pick the observed
        outputs (nonzero weight) when obs_names is not given, and as the default
        observation data for condition().  when it is a Pst, condition() returns
        an ObservationEnsemble.
    data : pandas.DataFrame or ObservationEnsemble
        the prior ensemble of simulated outputs, realizations by outputs.
    transforms : list of dict, optional
        feature transforms, same format as the other emulators (e.g.
        [{"type": "normal_score"}]).  the noise is carried into the transformed
        space with the local slope of the transform at the observed value.
    obs_names : list of str, optional
        the outputs that will be observed.  default: the nonzero-weight
        observations of pst that are in data.
    max_candidates : int, optional
        how many of the most correlated predecessors each output offers to the
        lasso.  default a third of the ensemble size; it is held below the
        ensemble size minus one, where the regressions would interpolate.
    min_var : float, optional
        floor on the conditional variances d_i.  default 1.0e-8.
    verbose : bool, optional
        echo log statements.  default False.
    """

    def __init__(self, pst=None, data=None, transforms=None, obs_names=None,
                 max_candidates=None, min_var=1.0e-8, verbose=False):
        super().__init__(transforms=transforms, verbose=verbose)
        from ..en import ObservationEnsemble
        from ..pst.pst_handler import Pst
        if data is None:
            raise ValueError("SPE requires a 'data' ensemble of simulated outputs")
        if isinstance(data, ObservationEnsemble):
            data = data._df.copy()
        data = data.astype(float).copy()
        if not data.columns.is_unique:
            raise ValueError("data has duplicate column names")
        self.pst = pst if isinstance(pst, Pst) else None
        if isinstance(pst, Pst):
            self.observation_data = pst.observation_data.copy()
        elif isinstance(pst, pd.DataFrame):
            self.observation_data = pst.copy()
        elif pst is None:
            self.observation_data = None
        else:
            raise ValueError("pst must be a Pst, an observation_data DataFrame or None")
        if obs_names is None:
            if self.observation_data is None:
                raise ValueError("obs_names was not given and there is no pst to take the "
                                 "nonzero-weight observations from")
            od = self.observation_data
            obs_names = [c for c in data.columns if c in od.index and float(od.loc[c, "weight"]) > 0.0]
        obs_names = list(obs_names)
        if len(obs_names) == 0:
            raise ValueError("no observed outputs: obs_names is empty")
        if len(set(obs_names)) != len(obs_names):
            raise ValueError("obs_names has duplicates")
        missing = [o for o in obs_names if o not in data.columns]
        if len(missing) > 0:
            raise ValueError("obs_names not in data: {0}".format(",".join(missing[:5])))
        self.data = data
        self.obs_names = obs_names
        self.max_candidates = max_candidates
        self.min_var = float(min_var)
        if transforms is not None:
            self._validate_transforms(transforms)
        # diagnostics, set by fit() and condition()
        self.dropped_obs = []
        self.num_predecessors = None
        self.num_lasso_failures = 0
        self.post_mean = None
        self.leverage = None
        self.p_eff = None
        self.p_eff_group = None
        self.tau2 = None
        self.em_history = []
        self.em_converged = None

    def fit(self):
        """
        build the sparse precision factor from the training ensemble.

        Returns
        -------
        SPE
            self, fitted.
        """
        try:
            import scipy.sparse as sp
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import LassoLarsIC, LinearRegression
        except ImportError as e:
            raise ImportError("SPE requires scipy and scikit-learn; install them with pip") from e
        if self.transforms is not None:
            data_t = self._fit_transformer_pipeline(self.data, self.transforms)
        else:
            self.transformer_pipeline = None
            data_t = self.data.copy()
        self.data_transformed = data_t
        # an observed output that does not vary in the prior carries nothing to learn
        sd = data_t.loc[:, self.obs_names].std(ddof=1)
        self.dropped_obs = sd.index[sd.values <= 0.0].tolist()
        if len(self.dropped_obs) > 0:
            self.logger.warn("SPE: {0} observed outputs do not vary in the prior and are left out of "
                             "the conditioning: {1}".format(len(self.dropped_obs),
                                                           ",".join(self.dropped_obs[:5])))
        obs = [o for o in self.obs_names if o not in set(self.dropped_obs)]
        if len(obs) == 0:
            raise ValueError("SPE: every observed output is constant in the prior")
        order = obs + [c for c in data_t.columns if c not in set(obs)]
        x = data_t.loc[:, order].values
        ne, nd = x.shape
        if ne < 4:
            raise ValueError("SPE needs at least 4 realizations, got {0}".format(ne))
        self.output_names = order
        self.active_obs = obs
        self.nobs = len(obs)
        self.data_mean = x.mean(axis=0)
        self.data_std = x.std(axis=0, ddof=1)
        self.data_std[self.data_std <= 0.0] = 1.0
        z = (x - self.data_mean) / self.data_std
        m = self.max_candidates
        if m is None:
            m = ne // 3
        m = int(max(1, min(m, ne - 2)))
        rows, cols, vals = [], [], []
        d = np.zeros(nd)
        npred = np.zeros(nd, dtype=int)
        self.num_lasso_failures = 0
        for i in range(nd):
            y = z[:, i]
            if i == 0 or y.var() == 0.0:
                d[i] = max(y.var(ddof=1), self.min_var)
                continue
            corr = z[:, :i].T.dot(y) / (ne - 1)
            cand = np.argsort(-np.abs(corr))[:min(m, i)]
            xc = z[:, cand]
            sel = np.zeros(0, dtype=int)
            try:
                lasso = LassoLarsIC(criterion="bic", fit_intercept=False)
                with warnings.catch_warnings():
                    # lars warns when candidates are collinear (the same location in two
                    # outputs); it drops one and carries on
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lasso.fit(xc, y)
                sel = np.where(np.abs(lasso.coef_) > 0.0)[0]
            except Exception as e:
                self.num_lasso_failures += 1
                self.logger.statement("SPE: lasso failed for {0}: {1}".format(order[i], str(e)))
            if sel.shape[0] > 0:
                # relaxed lasso: the lasso picks the predecessors, plain least squares sets
                # their coefficients so they are not shrunk
                ols = LinearRegression(fit_intercept=False)
                ols.fit(xc[:, sel], y)
                b = ols.coef_
                resid = y - xc[:, sel].dot(b)
                # centred data: one degree of freedom goes to the mean
                dof = max(1, ne - 1 - sel.shape[0])
                d[i] = max(resid.dot(resid) / dof, self.min_var)
                rows.extend([i] * sel.shape[0])
                cols.extend(cand[sel].tolist())
                vals.extend(b.tolist())
                npred[i] = sel.shape[0]
            else:
                d[i] = max(y.var(ddof=1), self.min_var)
        if self.num_lasso_failures > 0:
            self.logger.warn("SPE: the lasso failed for {0} outputs, they were given no "
                             "predecessors".format(self.num_lasso_failures))
        bmat = sp.csr_matrix((vals, (rows, cols)), shape=(nd, nd))
        self.lmat = (sp.identity(nd, format="csr") - bmat).tocsr()
        self.cond_var = d
        self.qmat = (self.lmat.T @ sp.diags(1.0 / d) @ self.lmat).tocsc()
        self.num_predecessors = pd.Series(npred, index=order)
        self.fitted = True
        self.logger.statement("SPE fit: {0} outputs ({1} observed), {2} realizations, mean "
                              "predecessors {3:.1f}, max {4}".format(nd, self.nobs, ne, npred.mean(),
                                                                    npred.max()))
        return self

    def _draw_prior_z(self, num_reals, rng):
        """prior draws in standardized space, (outputs, num_reals): solve L z = D^1/2 e"""
        import scipy.sparse.linalg as spla
        e = rng.standard_normal((len(self.output_names), num_reals)) * np.sqrt(self.cond_var)[:, None]
        return spla.spsolve_triangular(self.lmat, e, lower=True, unit_diagonal=True)

    def _to_frame(self, z, index=None):
        """standardized (outputs, reals) -> realizations by outputs, in the original units"""
        df = pd.DataFrame((z.T * self.data_std) + self.data_mean, columns=self.output_names)
        if self.transformer_pipeline is not None:
            df = self.transformer_pipeline.inverse(df)
        if index is not None:
            df.index = index
        df.index.name = "realization"
        df = df.loc[:, [c for c in self.data.columns if c in df.columns]]
        return df

    def _as_ensemble(self, df):
        if self.pst is not None:
            from ..en import ObservationEnsemble
            return ObservationEnsemble(self.pst, df)
        return df

    def predict(self, pvals):
        """
        map latent white noise to outputs: z = L^-1 D^1/2 e, then back to the
        original units.  e is one standard normal value per output, so a
        standard normal pvals gives draws from the emulated prior.

        Parameters
        ----------
        pvals : pandas.Series, pandas.DataFrame or numpy.ndarray
            latent values, length (or columns) = number of outputs.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            the outputs, one row per realization.
        """
        import scipy.sparse.linalg as spla
        if not self.fitted:
            raise ValueError("SPE must be fitted before prediction")
        single = isinstance(pvals, pd.Series) or np.asarray(pvals).ndim == 1
        index = pvals.index if isinstance(pvals, pd.DataFrame) else None
        e = np.atleast_2d(np.asarray(pvals, dtype=float))
        if e.shape[1] != len(self.output_names):
            raise ValueError("pvals must have {0} values per realization, got {1}".format(
                len(self.output_names), e.shape[1]))
        z = spla.spsolve_triangular(self.lmat, (e * np.sqrt(self.cond_var)).T, lower=True, unit_diagonal=True)
        df = self._to_frame(np.atleast_2d(z.T).T if z.ndim == 1 else z, index=index)
        return df.iloc[0] if single else df

    def prior_ensemble(self, num_reals=100, seed=None):
        """
        draw from the emulated prior.

        Parameters
        ----------
        num_reals : int, optional
            number of realizations, default 100.
        seed : int, optional
            seed for the draws.

        Returns
        -------
        ObservationEnsemble or pandas.DataFrame
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before drawing")
        z = self._draw_prior_z(num_reals, np.random.default_rng(seed))
        return self._as_ensemble(self._to_frame(z))

    def _obs_inputs(self, observation_data, obsvals, noise_std, groups):
        """observed values, noise std and groups for the active observations, original units"""
        od = observation_data if observation_data is not None else self.observation_data
        on = self.active_obs
        if obsvals is None:
            if od is None:
                raise ValueError("no observed values: pass observation_data or obsvals")
            obsvals = od.loc[on, "obsval"].astype(float)
        if noise_std is None:
            if od is None:
                raise ValueError("no noise: pass observation_data or noise_std")
            if "standard_deviation" in od.columns and od.loc[on, "standard_deviation"].notna().all():
                noise_std = od.loc[on, "standard_deviation"].astype(float)
            else:
                w = od.loc[on, "weight"].astype(float)
                noise_std = pd.Series(np.where(w > 0.0, 1.0 / w.where(w > 0.0, 1.0), np.inf), index=on)
        if groups is None:
            if od is not None and "obgnme" in od.columns:
                groups = od.loc[on, "obgnme"]
            else:
                groups = pd.Series("all", index=on)
        for name, s in (("obsvals", obsvals), ("noise_std", noise_std), ("groups", groups)):
            miss = [o for o in on if o not in s.index]
            if len(miss) > 0:
                raise ValueError("{0} is missing observations: {1}".format(name, ",".join(miss[:5])))
        g = groups.loc[on]
        if g.isna().any():
            raise ValueError("groups has missing values for: {0}".format(",".join(g.index[g.isna()][:5])))
        y = obsvals.loc[on].values.astype(float)
        sd = noise_std.loc[on].values.astype(float)
        bad = (sd <= 0.0) | np.isnan(sd)
        if bad.any():
            raise ValueError("noise std must be positive for: {0}".format(",".join(np.array(on)[bad][:5])))
        return y, sd, g.values

    def _transformed_obs(self, y, sd):
        """observed values and noise std carried into the transformed space; the std is
        scaled by the local slope of the transform at the observed value"""
        if self.transformer_pipeline is None:
            return y, sd
        on = self.active_obs
        base = pd.DataFrame([self.data.loc[:, self.output_names].mean()] * 3).reset_index(drop=True)
        h = 1.0e-3 * np.maximum(self.data.loc[:, on].std(ddof=1).values, 1.0e-12)
        base.loc[0, on] = y
        base.loc[1, on] = y + h
        base.loc[2, on] = y - h
        t = self.transformer_pipeline.transform(base).loc[:, on].values
        slope = np.abs(t[1] - t[2]) / (2.0 * h)
        slope = np.where(np.isfinite(slope) & (slope > 0.0), slope, np.nan)
        if np.isnan(slope).any():
            fill = np.nanmedian(slope) if np.isfinite(np.nanmedian(slope)) else 1.0
            slope = np.where(np.isnan(slope), fill, slope)
        return t[0], sd * slope

    def _condition_z(self, yz, rz, active, num_reals, rng, want_samples=True, eps=None):
        """standardized space.  yz, rz: observed values and noise variances of the active
        observations; active: their positions (rows of S).  eps: noise draws
        (len(active), num_reals) for the perturbed observations, else drawn from rz"""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        nd = len(self.output_names)
        na = len(active)
        s = sp.csr_matrix((np.ones(na), (np.arange(na), active)), shape=(na, nd))
        p = (self.qmat + s.T @ sp.diags(1.0 / rz) @ s).tocsc()
        lu = spla.splu(p)
        mean = lu.solve(s.T @ (yz / rz))
        x = lu.solve(s.T.toarray())
        post_obs_var = np.array([x[active[k], k] for k in range(na)])
        lev = post_obs_var / rz
        samples = None
        if want_samples:
            if eps is None:
                eps = rng.standard_normal((na, num_reals)) * np.sqrt(rz)[:, None]
            zp = self._draw_prior_z(eps.shape[1], rng)
            rhs = s.T @ ((yz[:, None] + eps - zp[active, :]) / rz[:, None])
            samples = zp + lu.solve(rhs)
        return mean, samples, post_obs_var, lev

    def condition(self, observation_data=None, obsvals=None, noise_std=None, noise=None, groups=None,
                  estimate_noise=False, pool_count=5.0, max_em_iter=50, tol=1.0e-4, num_reals=100,
                  seed=None):
        """
        condition the emulated outputs on observed values.

        Parameters
        ----------
        observation_data : pandas.DataFrame, optional
            obsval, and the noise from a standard_deviation column or 1/weight,
            and the groups from obgnme.  default: the observation data of the
            pst given at construction.  observations with zero weight are left
            out of the conditioning.
        obsvals, noise_std, groups : pandas.Series, optional
            override the values, noise standard deviations and groups, indexed
            by observation name.
        noise : pandas.DataFrame, optional
            noise realizations (rows) by observation, as deviations from obsval.
            each posterior realization is conditioned on obsval plus one row
            instead of a gaussian draw from noise_std, and the posterior has as
            many realizations as noise has rows.
        estimate_noise : bool, optional
            estimate a structural excess variance tau2 per group on top of the
            given noise (a hierarchical em).  default False.  the m-step is
            tau2_g = mean over the group of (r^2 + posterior variance - noise
            variance); its fixed point is the maximum likelihood value when the
            noise is uniform within a group, otherwise it is a moment estimate.
            groups are pooled toward a common value on a scale-free basis
            (tau2 relative to the group's noise variance) by pool_count pseudo
            observations.  it can be slow to move away from zero when the
            observations are very informative; em_converged says whether it
            stopped on tol or on max_em_iter.
        num_reals : int, optional
            posterior realizations when noise is not given, default 100.
        seed : int, optional
            seed for the prior and noise draws.

        Returns
        -------
        ObservationEnsemble or pandas.DataFrame
            posterior realizations of every output.  the posterior mean,
            leverage, p_eff, p_eff_group, tau2, em_history and em_converged are
            stored on the emulator.
        """
        if not self.fitted:
            raise ValueError("SPE must be fitted before conditioning")
        on = self.active_obs
        y, sd_obs, g = self._obs_inputs(observation_data, obsvals, noise_std, groups)
        finite = np.isfinite(sd_obs)
        if not finite.any():
            raise ValueError("every observed output has zero weight")
        y_t, sd_t = self._transformed_obs(y, np.where(finite, sd_obs, 1.0))
        active = np.where(finite)[0]
        mu, ss = self.data_mean[:self.nobs][active], self.data_std[:self.nobs][active]
        yz = (y_t[active] - mu) / ss
        su2 = sd_t[active] ** 2
        ga = g[active]
        ug = list(pd.unique(ga))
        rng = np.random.default_rng(seed)
        tau2_rel = {k: 0.0 for k in ug}
        su2_g = {k: su2[ga == k].mean() for k in ug}
        self.em_history = []
        self.em_converged = None
        if estimate_noise:
            self.em_converged = False
            for _ in range(max_em_iter):
                tvec = np.array([tau2_rel[k] * su2_g[k] for k in ga])
                rz = (su2 + tvec) / ss ** 2
                mean, _, pvar, _ = self._condition_z(yz, rz, active, num_reals, rng, want_samples=False)
                r = (yz - mean[active]) * ss
                pv = pvar * ss ** 2
                raw, ng = {}, {}
                for k in ug:
                    idx = ga == k
                    # scale free: the excess relative to the group's noise variance
                    raw[k] = np.mean(r[idx] ** 2 + pv[idx] - su2[idx]) / su2_g[k]
                    ng[k] = idx.sum()
                pooled = sum(raw[k] * ng[k] for k in ug) / sum(ng[k] for k in ug)
                new = {k: max(0.0, (ng[k] * raw[k] + pool_count * pooled) / (ng[k] + pool_count)) for k in ug}
                change = max(abs(new[k] - tau2_rel[k]) / (1.0 + tau2_rel[k]) for k in ug)
                tau2_rel = new
                self.em_history.append({k: tau2_rel[k] * su2_g[k] for k in ug})
                if change < tol:
                    self.em_converged = True
                    break
            if not self.em_converged:
                self.logger.warn("SPE: the noise estimate stopped at max_em_iter={0} without "
                                 "converging".format(max_em_iter))
        tvec = np.array([tau2_rel[k] * su2_g[k] for k in ga])
        rz = (su2 + tvec) / ss ** 2
        eps = None
        if noise is not None:
            miss = [o for o in np.array(on)[active] if o not in noise.columns]
            if len(miss) > 0:
                raise ValueError("noise is missing observations: {0}".format(",".join(miss[:5])))
            nv = noise.loc[:, list(np.array(on)[active])].values.astype(float)
            if self.transformer_pipeline is not None:
                nv = nv * (sd_t[active] / np.where(finite, sd_obs, 1.0)[active])
            eps = nv.T / ss[:, None]
            # the given noise covers the user's noise; a structural excess is drawn on top
            eps = eps + rng.standard_normal(eps.shape) * (np.sqrt(tvec) / ss)[:, None]
        mean, samples, pvar, lev = self._condition_z(yz, rz, active, num_reals, rng, want_samples=True, eps=eps)
        index = noise.index if noise is not None else None
        post = self._to_frame(samples, index=index)
        self.post_mean = self._to_frame(mean[:, None]).iloc[0]
        an = list(np.array(on)[active])
        self.leverage = pd.Series(lev, index=an)
        self.p_eff = float(lev.sum())
        self.p_eff_group = pd.Series({k: float(lev[ga == k].sum()) for k in ug})
        self.tau2 = pd.Series({k: tau2_rel[k] * su2_g[k] for k in ug})
        return self._as_ensemble(post)

    def prepare_pestpp(self, t_d, pst=None, verbose=False, **kwargs):
        """not supported: SPE conditions in closed form with condition(), so there is
        no forward run for pestpp-ies to drive."""
        raise NotImplementedError("SPE conditions in closed form - use condition() rather than "
                                  "prepare_pestpp()")
