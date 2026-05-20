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
    """Forward-run helper used by the PEST++ template for a fitted PLS emulator.

    Loads the pickled emulator, reads parameter values from ``input_file``,
    calls ``emu.predict``, and writes the resulting observation values to
    ``output_file``. Injected verbatim into the generated forward-run script
    by :func:`PLS._write_forward_run_script`.
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


class PLS(Emulator):
    """
    Partial Least Squares regression emulator.

    Parameters
    ----------
    pst : Pst, optional
        PEST control-file object. Used only as a default source for
        ``observation_data`` during ``prepare_pestpp``.
    data : pandas.DataFrame
        Joint training DataFrame containing both the input columns (named in
        ``input_names``) and the output columns (named in ``output_names``).
        Columns in ``data`` outside those two name lists are ignored. The
        caller is responsible for any non-zero-weight subsetting or other
        preprocessing — PLS treats whatever columns it is told to.
    input_names : list of str
        Columns in ``data`` that are the regression inputs (parameters).
    output_names : list of str
        Columns in ``data`` that are the regression outputs (observations).
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

        if data is None:
            raise ValueError("PLS requires a 'data' DataFrame")
        if input_names is None or len(input_names) == 0:
            raise ValueError("PLS requires a non-empty 'input_names' list")
        if output_names is None or len(output_names) == 0:
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

    def _pick_components_cv(self, X: np.ndarray, Y: np.ndarray) -> int:
        """Return the n_components that minimises k-fold CV RMSE on Y."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold

        max_k = max(1, min(X.shape[0] - 1, X.shape[1], Y.shape[1]))
        kf = KFold(n_splits=min(self.cv_folds, X.shape[0]),
                   shuffle=True, random_state=0)

        best_k, best_rmse = 1, float("inf")
        for k in range(1, max_k + 1):
            errs = []
            for train_idx, val_idx in kf.split(X):
                pls = PLSRegression(n_components=k)
                pls.fit(X[train_idx], Y[train_idx])
                pred = pls.predict(X[val_idx])
                errs.append(np.sqrt(np.mean((pred - Y[val_idx]) ** 2)))
            mean_rmse = float(np.mean(errs))
            if mean_rmse < best_rmse:
                best_rmse, best_k = mean_rmse, k
            self.logger.statement(
                "cv n_components={0}: rmse={1:.4g}".format(k, mean_rmse))
        self.logger.statement(
            "cv selected n_components={0} (rmse={1:.4g})".format(best_k, best_rmse))
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
        """Coerce arbitrary input forms to a DataFrame indexed by self.input_names."""
        if isinstance(pvals, pd.Series):
            return pvals.to_frame().T.loc[:, self.input_names]
        if isinstance(pvals, np.ndarray):
            arr = pvals.reshape(1, -1) if pvals.ndim == 1 else pvals
            return pd.DataFrame(arr, columns=self.input_names)
        if isinstance(pvals, pd.DataFrame):
            return pvals.loc[:, self.input_names]
        raise TypeError(
            "pvals must be a pandas DataFrame, Series, or numpy array")

    def _get_emulator_parameters(self, pst=None) -> pd.DataFrame:
        """Parameter table consumed by the base-class ``prepare_pestpp`` machinery."""
        if pst is not None and hasattr(pst, "parameter_data"):
            valid = [n for n in self.input_names if n in pst.parameter_data.index]
            if len(valid) != len(self.input_names):
                missing = sorted(set(self.input_names) - set(valid))
                self.logger.statement(
                    "warning: {0} input_names missing from pst.parameter_data: {1}".format(
                        len(missing), missing[:3]))
            par_df = pst.parameter_data.loc[valid].copy()
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
        """Write the PEST++ forward-run script for a fitted PLS emulator."""
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
            "# Source for pls_file_forward_run",
            inspect.getsource(pls_file_forward_run),
            "",
            'if __name__ == "__main__":',
            "    pls_file_forward_run({0})".format(call_args),
        ]
        with open(filename, "w") as f:
            for line in lines:
                f.write(line + "\n")
