"""
Partial Least Squares (PLS) regression emulator.

PLS finds a small set of latent factors that maximise covariance between an
input matrix X (parameters) and an output matrix Y (observations). It is a
natural fit for surrogate problems where the parameter dimension `d` is
larger than the training set size `n`, and where the outputs are correlated
multivariate quantities.

The class mirrors the surface area of the other pyemu emulators (``DSI``,
``DSIAE``, ``GPR``): same ``fit``/``predict`` shape, the same transformer
pipeline plumbing for optional input transforms, and the same
``prepare_pestpp`` hook inherited from :class:`Emulator` so a fitted
emulator can be used as a PEST++ forward run.
"""
from __future__ import print_function, division
import os
import warnings
import inspect
from typing import Optional, List, Union

import numpy as np
import pandas as pd

from .base import Emulator


# Threshold above which we warn the user that PLS may benefit from an
# external dimensionality-reduction step on the input side.
HIGH_D_WARN_THRESHOLD = 10_000


def pls_file_forward_run(emu_file="pls.pickle",
                         input_file="pls_pars.csv",
                         output_file="pls_sim_vals.csv"):
    """File-based forward-run helper for a fitted PLS emulator.

    Loads the pickled emulator, reads parameter values from ``input_file``,
    calls ``emu.predict``, and writes the resulting observation values to
    ``output_file``. Used when ``prepare_pestpp(use_runstor=False)``.
    """
    import os
    import pandas as pd
    import traceback

    try:
        try:
            from pyemu.emulators import PLS
        except ImportError:
            raise ImportError("pyemu.emulators.PLS could not be imported")

        emu = PLS.load(emu_file)

        if not os.path.exists(input_file):
            raise FileNotFoundError("Input file {0} not found".format(input_file))

        input_df = pd.read_csv(input_file, index_col=0)
        if "parval1" in input_df.columns:
            inputs = input_df["parval1"].to_frame().T
        else:
            inputs = input_df

        pred = emu.predict(inputs)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[0]
        pred.name = "simval"
        pred.to_csv(output_file, header=True)
    except Exception as e:
        print("Error in pls_file_forward_run: {0}".format(e))
        traceback.print_exc()
        raise e


def pls_runstore_forward_run(ws='.', pst_name="pls", emu_file="emulator.pkl"):
    """Runstor-based forward-run helper for a fitted PLS emulator.

    PESTPP-IES in panther / external run-manager mode (``/e``) reads/writes
    realisations through a binary RunStor (``{pst_name}.rns``) rather than via
    CSV files. The plain file-based helper never sees the rns and so the
    obs columns stay zero-filled -- that's the failure mode this function
    fixes. Mirror of :func:`pyemu.utils.helpers.dsi_runstore_forward_run`.

    NOTE: this function's source is embedded verbatim in the generated
    forward_run.py (via inspect.getsource), so it must stay ASCII-only --
    on Windows the script is read back with whatever the locale encoding
    is, and non-ASCII bytes break the UTF-8 source parse.
    """
    import os
    import pandas as pd
    import traceback
    from pyemu.utils.helpers import RunStor

    try:
        try:
            from pyemu.emulators import PLS
        except ImportError:
            raise ImportError("pyemu.emulators.PLS could not be imported")

        emu = PLS.load(os.path.join(ws, emu_file))

        fname = os.path.join(ws, "{0}.rns".format(pst_name))
        if not os.path.exists(fname):
            # Fall back to a few common names PESTPP-IES might have written.
            for alt in ("pls.rns", "dsi.rns"):
                cand = os.path.join(ws, alt)
                if os.path.exists(cand):
                    fname = cand
                    break
        header, par_names, obs_names = RunStor.file_info(fname)
        rs = RunStor(fname)
        df = rs.get_data()

        # Pass through the predictor -- _coerce_to_input_df silently drops
        # any extra columns and validates the required input_names.
        pvals = df.loc[:, [p for p in par_names if p in df.columns]]
        simvals = emu.predict(pvals)
        if not isinstance(simvals, pd.DataFrame):
            simvals = simvals.to_frame().T
        simvals.index = df.index

        # Only update obs columns the emulator actually predicts; leave any
        # extras (e.g. obs that aren't outputs of this PLS) untouched.
        common_obs = [o for o in obs_names if o in simvals.columns]
        if common_obs:
            df.loc[:, common_obs] = simvals.loc[:, common_obs].values

        rs.update(df)
    except Exception as e:
        print("Error in pls_runstore_forward_run: {0}".format(e))
        traceback.print_exc()
        raise e


class PLS(Emulator):
    """
    Partial Least Squares regression emulator.

    Parameters
    ----------
    pst : Pst, optional
        PEST control-file object. Source for ``observation_data`` during
        ``prepare_pestpp``; also used to infer ``input_names`` /
        ``output_names`` from ``data`` when those are not provided.
    data : pandas.DataFrame
        Joint training DataFrame containing both the input columns (named in
        ``input_names``) and the output columns (named in ``output_names``).
        Columns in ``data`` outside those two name lists are ignored. The
        caller is responsible for any non-zero-weight subsetting or other
        preprocessing — PLS treats whatever columns it is told to.
    input_names : list of str, optional
        Columns in ``data`` that are the regression inputs (parameters). May
        be omitted — see ``output_names`` for inference rules.
    output_names : list of str, optional
        Columns in ``data`` that are the regression outputs (observations).
        Resolution rules:

        * Both lists passed — used as-is.
        * Only one passed — the other is ``set(data.columns) - set(passed)``.
        * Neither passed (requires ``pst``) — if ``data`` contains both pst
          pars and pst obs, pars are inputs and obs are outputs; if ``data``
          contains only pst obs, nonzero-weight obs are inputs and
          zero-weight obs are outputs (the DSI-style "obs-as-pars" setup).
    transforms : list of dict, optional
        Feature transformations applied via the base-class transformer
        pipeline. Same format as :class:`DSIAE`.
    n_components : int, optional
        Number of PLS latent factors. If ``None`` (default), the value is
        chosen by k-fold cross-validation on the training data.
    cv_folds : int, default 5
        Number of folds used when ``n_components`` is selected by CV.
    parameter_reducer : sklearn-style transformer, optional
        Optional dimension-reducer fit on the input matrix before PLS (e.g.
        ``sklearn.decomposition.PCA`` or
        ``sklearn.random_projection.GaussianRandomProjection``). Must
        implement ``fit_transform`` and ``transform``. If left as ``None``
        and the input dimension exceeds :data:`HIGH_D_WARN_THRESHOLD`, a
        warning is emitted suggesting one — but PLS is still trained on the
        full input.
    verbose : bool, default False
        Enable verbose logging.
    """

    def __init__(self,
                 pst=None,
                 data: Optional[pd.DataFrame] = None,
                 input_names: Optional[List[str]] = None,
                 output_names: Optional[List[str]] = None,
                 transforms: Optional[List[dict]] = None,
                 n_components: Optional[int] = None,
                 cv_folds: int = 5,
                 parameter_reducer=None,
                 verbose: bool = False) -> None:
        super().__init__(verbose=verbose)

        self.observation_data = pst.observation_data.copy() if pst is not None else None
        # Cache parameter_data so _get_emulator_parameters can recover the
        # original parval1/parlbnd/parubnd/pargp/partrans/scale/offset even
        # when prepare_pestpp is later called without a ``pst`` kwarg.
        self.parameter_data = pst.parameter_data.copy() if pst is not None else None

        if data is None:
            raise ValueError("PLS requires a 'data' DataFrame")

        # Name resolution rules:
        # - both passed:   use as-is.
        # - neither + pst: infer both via pst-data overlap (mixed) or weight
        #                  split (obs-only). See _infer_io_from_pst.
        # - one passed:    derive the other as data.columns minus the passed
        #                  set. No pst needed because it's pure set-diff on
        #                  column names.
        if input_names is None and output_names is None:
            input_names, output_names = self._infer_io_from_pst(pst, data)
        elif input_names is not None and output_names is None:
            in_set = set(str(c) for c in input_names)
            output_names = [c for c in data.columns if str(c) not in in_set]
            self.logger.statement(
                "inferred output_names as data.columns - input_names ({0} cols)".format(
                    len(output_names)))
        elif output_names is not None and input_names is None:
            out_set = set(str(c) for c in output_names)
            input_names = [c for c in data.columns if str(c) not in out_set]
            self.logger.statement(
                "inferred input_names as data.columns - output_names ({0} cols)".format(
                    len(input_names)))

        if len(input_names) == 0:
            raise ValueError("PLS requires a non-empty 'input_names' list")
        if len(output_names) == 0:
            raise ValueError("PLS requires a non-empty 'output_names' list")

        missing_in = [c for c in input_names if c not in data.columns]
        missing_out = [c for c in output_names if c not in data.columns]
        if missing_in:
            raise ValueError("input_names not in data: {0}".format(missing_in))
        if missing_out:
            raise ValueError("output_names not in data: {0}".format(missing_out))

        self.data = data.astype(float).copy()
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.transforms = transforms
        self.n_components = n_components
        self.cv_folds = int(cv_folds)
        self.parameter_reducer = parameter_reducer

        self.pls_ = None
        self._fitted_reducer = None

        self.data_transformed = self._prepare_training_data()

    def _infer_io_from_pst(self, pst, data):
        """Infer ``(input_names, output_names)`` from a Pst + data overlap.

        Two cases:

        1. ``data`` contains both pst pars and pst obs columns — pars are
           inputs, obs are outputs.
        2. ``data`` contains only pst obs columns — nonzero-weight obs are
           inputs, zero-weight obs are outputs (the "obs-as-pars" / DSI-style
           setup where prior par draws are recorded as obs).

        Raises ``ValueError`` if neither case is satisfied or either side of
        the split is empty.
        """
        if pst is None:
            raise ValueError(
                "input_names / output_names were not provided and no pst was "
                "passed to infer them from")
        if not hasattr(pst, "parameter_data") or not hasattr(pst, "observation_data"):
            raise ValueError("pst must have parameter_data and observation_data")

        data_cols = [str(c) for c in data.columns]
        par_set = set(str(p) for p in pst.parameter_data.index)
        obs_set = set(str(o) for o in pst.observation_data.index)
        pars_in_data = [c for c in data_cols if c in par_set]
        obs_in_data = [c for c in data_cols if c in obs_set]

        if len(pars_in_data) == 0 and len(obs_in_data) == 0:
            raise ValueError(
                "no data columns matched pst parameter_data or observation_data; "
                "cannot infer input_names/output_names")

        if len(pars_in_data) == 0:
            # Case 2: obs-only. Split by weight.
            wts = pst.observation_data["weight"].astype(float)
            nz_in_data = [c for c in obs_in_data if float(wts.loc[c]) != 0.0]
            zw_in_data = [c for c in obs_in_data if float(wts.loc[c]) == 0.0]
            assert len(nz_in_data) > 0, (
                "obs-only inference: no nonzero-weight obs in data to use as "
                "input_names")
            assert len(zw_in_data) > 0, (
                "obs-only inference: no zero-weight obs in data to use as "
                "output_names")
            self.logger.statement(
                "inferred input_names from nonzero-weight obs ({0} cols), "
                "output_names from zero-weight obs ({1} cols)".format(
                    len(nz_in_data), len(zw_in_data)))
            return nz_in_data, zw_in_data

        # Case 1: mixed pars + obs. pars -> inputs, obs -> outputs.
        assert len(pars_in_data) > 0, (
            "mixed inference: no pst pars in data to use as input_names")
        assert len(obs_in_data) > 0, (
            "mixed inference: no pst obs in data to use as output_names")
        self.logger.statement(
            "inferred input_names from pst pars ({0} cols), "
            "output_names from pst obs ({1} cols)".format(
                len(pars_in_data), len(obs_in_data)))
        return pars_in_data, obs_in_data

    def _prepare_training_data(self) -> pd.DataFrame:
        """Apply optional feature transforms (or a no-op pipeline) to ``data``."""
        if self.data is None:
            raise ValueError("No data stored in the emulator")

        self.logger.statement("applying feature transforms")
        if self.transforms is not None:
            self.data_transformed = self._fit_transformer_pipeline(self.data, self.transforms)
        else:
            from .transformers import AutobotsAssemble
            self.transformer_pipeline = AutobotsAssemble(self.data.copy())
            self.data_transformed = self.data.copy()
        return self.data_transformed

    def _split_xy(self, df: pd.DataFrame):
        """Split a joint frame into input/output numpy arrays."""
        X = df.loc[:, self.input_names].values.astype(float)
        Y = df.loc[:, self.output_names].values.astype(float)
        return X, Y

    def _maybe_warn_high_d(self, X: np.ndarray) -> None:
        if X.shape[1] > HIGH_D_WARN_THRESHOLD and self.parameter_reducer is None:
            warnings.warn(
                "PLS input dimension ({0}) exceeds {1} and no parameter_reducer "
                "was provided; consider passing sklearn.decomposition.PCA or "
                "sklearn.random_projection.GaussianRandomProjection".format(
                    X.shape[1], HIGH_D_WARN_THRESHOLD),
                stacklevel=2,
            )

    def _apply_parameter_reducer(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Fit (or apply) the optional dimensionality reducer on the input matrix."""
        if self.parameter_reducer is None:
            return X
        if fit:
            X_reduced = self.parameter_reducer.fit_transform(X)
            self._fitted_reducer = self.parameter_reducer
            self.logger.statement(
                "applied parameter_reducer: {0} features -> {1}".format(
                    X.shape[1], X_reduced.shape[1]))
            return X_reduced
        if self._fitted_reducer is None:
            raise RuntimeError(
                "parameter_reducer requested but not fit; call .fit() first")
        return self._fitted_reducer.transform(X)

    @staticmethod
    def _log_anchors(max_k: int) -> list:
        """Decade-anchored coarse grid: 1, 2, 5, 10, 20, 50, 100, ... <= max_k.

        Always includes ``max_k`` so the coarse pass can detect a curve that's
        still descending at the upper bound.
        """
        anchors = []
        for dec in range(0, 10):
            stop = False
            for m in (1, 2, 5):
                v = m * (10 ** dec)
                if v > max_k:
                    stop = True
                    break
                anchors.append(v)
            if stop:
                break
        if max_k not in anchors:
            anchors.append(max_k)
        return sorted(set(anchors))

    def _pick_components_cv(self, X: np.ndarray, Y: np.ndarray) -> int:
        """Return the ``n_components`` that minimises k-fold CV RMSE on Y.

        Uses a single coarse pass on log-spaced anchors
        ``{1, 2, 5, 10, 20, 50, 100, ...}`` capped at ``max_k`` (always
        including ``max_k`` itself). No local refinement — within-anchor
        differences in CV RMSE are typically smaller than the fold-to-fold
        noise, so the extra fits don't reliably improve the pick.

        Self-state ``self._cv_scores`` (``{k: rmse}``) is populated for
        inspection / tests.
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=min(self.cv_folds, X.shape[0]),
                   shuffle=True, random_state=0)
        # sklearn requires n_components <= min(n_samples, n_features) for each
        # PLSRegression fit, so cap by the smallest *train* fold, not the full
        # X — otherwise k > min_train_size raises mid-CV.
        splits = list(kf.split(X))
        min_train = min(len(train_idx) for train_idx, _ in splits)
        max_k = max(1, min(min_train, X.shape[1], Y.shape[1]))

        scores: dict = {}
        anchors = self._log_anchors(max_k)
        for k in anchors:
            errs = []
            for train_idx, val_idx in splits:
                pls = PLSRegression(n_components=k)
                pls.fit(X[train_idx], Y[train_idx])
                pred = pls.predict(X[val_idx])
                errs.append(np.sqrt(np.mean((pred - Y[val_idx]) ** 2)))
            mean_rmse = float(np.mean(errs))
            scores[k] = mean_rmse
            self.logger.statement(
                "cv n_components={0}: rmse={1:.4g}".format(k, mean_rmse))

        best_k = min(scores, key=lambda kk: scores[kk])
        best_rmse = scores[best_k]
        self._cv_scores = scores
        self.logger.statement(
            "cv selected n_components={0} (rmse={1:.4g}); {2} anchors "
            "evaluated in [1,{3}]".format(best_k, best_rmse, len(scores), max_k))
        return best_k

    def fit(self) -> "PLS":
        """Fit the PLS regression on the (optionally transformed) training data."""
        from sklearn.cross_decomposition import PLSRegression

        if self.data_transformed is None:
            self.data_transformed = self._prepare_training_data()

        X, Y = self._split_xy(self.data_transformed)
        self._maybe_warn_high_d(X)
        X = self._apply_parameter_reducer(X, fit=True)

        if self.n_components is None:
            self.logger.statement("selecting n_components by {0}-fold CV".format(self.cv_folds))
            self.n_components = self._pick_components_cv(X, Y)

        max_k = max(1, min(X.shape[0] - 1, X.shape[1], Y.shape[1]))
        if self.n_components > max_k:
            warnings.warn(
                "n_components={0} exceeds max valid value {1}; clipping".format(
                    self.n_components, max_k))
            self.n_components = max_k

        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, Y)
        self.fitted = True
        self.logger.statement(
            "fitted PLS: {0} inputs x {1} outputs, n_components={2}".format(
                X.shape[1], Y.shape[1], self.n_components))
        return self

    def encode(self, X: Union[pd.DataFrame, np.ndarray, pd.Series]) -> pd.DataFrame:
        """Project new inputs into PLS latent space (X-scores)."""
        if not self.fitted:
            raise ValueError("Emulator must be fitted before encoding")

        df = self._coerce_to_input_df(X)
        X_t = self.transformer_pipeline.transform(df) if self.transforms is not None else df
        X_arr = X_t.loc[:, self.input_names].values.astype(float)
        X_arr = self._apply_parameter_reducer(X_arr, fit=False)
        scores = self.pls_.transform(X_arr)
        return pd.DataFrame(
            scores,
            index=df.index,
            columns=["pls_{0}".format(i) for i in range(scores.shape[1])],
        )

    def predict(self, pvals: Union[pd.DataFrame, pd.Series, np.ndarray]):
        """Predict outputs from input parameter values.

        Returns a Series for a single-row input (matching the DSIAE/DSI
        convention used by the PEST++ forward-run helper) and a DataFrame
        for multi-row input.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")

        df = self._coerce_to_input_df(pvals)
        X_t = self.transformer_pipeline.transform(df) if self.transforms is not None else df
        X_arr = X_t.loc[:, self.input_names].values.astype(float)
        X_arr = self._apply_parameter_reducer(X_arr, fit=False)
        Y_hat = self.pls_.predict(X_arr)

        Y_df = pd.DataFrame(Y_hat, index=df.index, columns=self.output_names)
        if self.transforms is not None:
            Y_df = self.transformer_pipeline.inverse(Y_df)

        if Y_df.shape[0] == 1:
            out = Y_df.iloc[0]
            out.index.name = "obsnme"
            out.name = "obsval"
            return out
        return Y_df

    def _coerce_to_input_df(self, pvals) -> pd.DataFrame:
        """Coerce arbitrary input forms to a DataFrame restricted to ``self.input_names``.

        Extra columns (or extra Series entries) are silently dropped — callers
        can pass a full par+obs frame, a controller dump, or anything wider
        than the emulator's input set without pre-slicing. Missing input
        columns raise a clear ``KeyError`` instead of pandas' default
        ``KeyError: '[…] not in index'`` (which can list dozens of names).
        Numpy arrays must have exactly ``len(input_names)`` columns since they
        carry no column metadata.
        """
        if isinstance(pvals, pd.Series):
            df = pvals.to_frame().T
        elif isinstance(pvals, pd.DataFrame):
            df = pvals
        elif isinstance(pvals, np.ndarray):
            arr = pvals.reshape(1, -1) if pvals.ndim == 1 else pvals
            if arr.shape[1] != len(self.input_names):
                raise ValueError(
                    "ndarray input has {0} columns but emulator expects {1} "
                    "input_names; pass a DataFrame/Series so columns can be "
                    "selected by name".format(arr.shape[1], len(self.input_names)))
            return pd.DataFrame(arr, columns=self.input_names)
        else:
            raise TypeError(
                "pvals must be a pandas DataFrame, Series, or numpy array")

        missing = [c for c in self.input_names if c not in df.columns]
        if missing:
            preview = missing[:5] + (["..."] if len(missing) > 5 else [])
            raise KeyError(
                "predict input is missing {0} of {1} required input_names "
                "(e.g. {2})".format(len(missing), len(self.input_names), preview))
        return df.loc[:, self.input_names]

    def _get_emulator_parameters(self, pst=None) -> pd.DataFrame:
        """Parameter table consumed by the base-class ``prepare_pestpp`` machinery.

        Source priority:

        1. Explicit ``pst`` arg passed to ``prepare_pestpp``.
        2. ``self.parameter_data`` cached at construction (when ``pst`` was
           supplied to ``__init__``).
        3. Last-ditch synthesized rows from training-data column statistics.
        """
        if pst is not None and hasattr(pst, "parameter_data"):
            src = pst.parameter_data
        elif self.parameter_data is not None:
            src = self.parameter_data
        else:
            src = None

        if src is not None:
            valid = [n for n in self.input_names if n in src.index]
            if len(valid) != len(self.input_names):
                missing = sorted(set(self.input_names) - set(valid))
                self.logger.statement(
                    "warning: {0} input_names missing from parameter_data: {1}".format(
                        len(missing), missing[:3]))
            par_df = src.loc[valid].copy()
            par_df["parnme"] = par_df.index
            return par_df

        train_X = self.data.loc[:, self.input_names]
        df = pd.DataFrame(index=self.input_names)
        df["parnme"] = self.input_names
        df["parval1"] = train_X.mean(axis=0).values
        df["parlbnd"] = train_X.min(axis=0).values
        df["parubnd"] = train_X.max(axis=0).values
        df["pargp"] = "pls_pars"
        df["partrans"] = "none"
        return df

    def _get_emulator_observations(self, pst=None) -> pd.DataFrame:
        """Observation table consumed by the base-class ``prepare_pestpp`` machinery."""
        if pst is not None and hasattr(pst, "observation_data"):
            valid = [n for n in self.output_names if n in pst.observation_data.index]
            if valid:
                obs_df = pst.observation_data.loc[valid].copy()
                obs_df["obsnme"] = obs_df.index
                return obs_df

        if self.observation_data is not None:
            valid = [n for n in self.output_names if n in self.observation_data.index]
            if valid:
                obs_df = self.observation_data.loc[valid].copy()
                obs_df["obsnme"] = obs_df.index
                return obs_df

        train_Y = self.data.loc[:, self.output_names]
        df = pd.DataFrame(index=self.output_names)
        df["obsnme"] = self.output_names
        df["obsval"] = train_Y.mean(axis=0).values
        df["weight"] = 1.0
        df["obgnme"] = "pls_pred"
        return df

    def _write_forward_run_script(self, filename, emu_file, input_file,
                                  output_file, class_name, pst_name=None):
        """Write the PEST++ forward-run script for a fitted PLS emulator.

        When ``self._use_runstor`` is set (via ``prepare_pestpp(use_runstor=True)``),
        the script invokes :func:`pls_runstore_forward_run` so it can be driven
        by PESTPP-IES's panther/external run-manager mode (the ``/e`` flag),
        which routes I/O through the binary RunStor instead of CSV files.
        Otherwise it invokes :func:`pls_file_forward_run` (CSV in / CSV out).
        """
        use_runstor = getattr(self, "_use_runstor", False)
        if use_runstor:
            target_func = "pls_runstore_forward_run"
            call_args = "emu_file='{0}'".format(emu_file)
            if pst_name is not None:
                call_args += ", pst_name='{0}'".format(pst_name)
        else:
            target_func = "pls_file_forward_run"
            call_args = "'{0}', '{1}', '{2}'".format(emu_file, input_file, output_file)

        lines = [
            "import sys",
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "import traceback",
            "import pickle",
            "",
            "sys.path.append(os.getcwd())",
            "",
        ]
        for func in (pls_file_forward_run, pls_runstore_forward_run):
            lines.append("# Source for {0}".format(func.__name__))
            lines.append(inspect.getsource(func))
            lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    {0}({1})".format(target_func, call_args))

        # utf-8 regardless of platform: python parses the generated script as
        # utf-8, but a bare open() on windows writes cp1252
        with open(filename, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
