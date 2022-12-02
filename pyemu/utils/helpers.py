"""High-level functions to help perform complex tasks

"""

from __future__ import print_function, division
import os
import multiprocessing as mp
import warnings
from datetime import datetime
import platform
import struct
import shutil
import copy
from ast import literal_eval
import traceback
import re
import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 100
from ..pyemu_warnings import PyemuWarning


try:
    import flopy
except:
    pass

import pyemu
from pyemu.utils.os_utils import run, start_workers


class Trie:
    """Regex::Trie in Python. Creates a Trie out of a list of words. The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union."""
    # after https://gist.github.com/EricDuminil/8faabc2f3de82b24e5a371b6dc0fd1e0
    def __init__(self):
        self.data = {}

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[''] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())

def autocorrelated_draw(pst,struct_dict,time_distance_col="distance",num_reals=100,verbose=True,
                        enforce_bounds=False, draw_ineq=False):
    """construct an autocorrelated observation noise ensemble from covariance matrices
        implied by geostatistical structure(s).

        Args:
            pst (`pyemu.Pst`): a control file (or the name of control file).  The
                information in the `* observation data` dataframe is used extensively,
                including weight, standard_deviation (if present), upper_bound/lower_bound (if present).
            time_distance_col (str): the column in `* observation_data` the represnts the distance in time
            for each observation listed in `struct_dict`

            struct_dict (`dict`): a dict of GeoStruct (or structure file), and list of
                observation names.
            num_reals (`int`, optional): number of realizations to draw.  Default is 100

            verbose (`bool`, optional): flag to control output to stdout.  Default is True.
                flag for stdout.
            enforce_bounds (`bool`, optional): flag to enforce `lower_bound` and `upper_bound` if
                these are present in `* observation data`.  Default is False
            draw_ineq (`bool`, optional): flag to generate noise realizations for inequality observations.
                If False, noise will not be added inequality observations in the ensemble.  Default is False


        Returns
            **pyemu.ObservationEnsemble**: the realized noise ensemble added to the observation values in the
                control file.

        Note:
            The variance of each observation is used to scale the resulting geostatistical
            covariance matrix (as defined by the weight or optional standard deviation.
            Therefore, the sill of the geostatistical structures
            in `struct_dict` should be 1.0

        Example::

            pst = pyemu.Pst("my.pst")
            #assuming there is only one timeseries of observations
            # and they are spaced one time unit apart
            pst.observation_data.loc[:,"distance"] = np.arange(pst.nobs)
            v = pyemu.geostats.ExpVario(a=10) #units of `a` are time units
            gs = pyemu.geostats.Geostruct(variograms=v)
            sd = {gs:["obs1","obs2",""obs3]}
            oe = pyemu.helpers.autocorrelated_draws(pst,struct_dict=sd}
            oe.to_csv("my_oe.csv")


        """

    #check that the required time metadata is appropriate
    passed_names = []
    nz_names = pst.nnz_obs_names
    [passed_names.extend(obs) for gs,obs in struct_dict.items()]
    missing = list(set(passed_names) - set(nz_names))
    if len(missing) > 0:
        raise Exception("the following obs in struct_dict were not found in the nz obs names"+str(missing))
    time_cols = ["time","datetime","distance"]
    obs = pst.observation_data
    if time_distance_col not in obs.columns:
        raise Exception("time_distance_col missing")
    dvals = obs.loc[passed_names,time_distance_col]
    pobs = obs.loc[passed_names,:]
    isna = pobs.loc[pd.isna(dvals),"obsnme"]
    if isna.shape[0] > 0:
        raise Exception("the following struct dict observations have NaN for time_distance_col: {0}".format(str(isna)))
    if verbose:
        print("--> getting full diagonal cov matrix")
    fcov = pyemu.Cov.from_observation_data(pst)
    fcov_dict = {o:np.sqrt(fcov.x[i]) for i,o in enumerate(fcov.names)}
    if verbose:
        print("-->draw full obs en from diagonal cov")
    full_oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst,fcov,num_reals=num_reals,fill=True)
    for gs,onames in struct_dict.items():
        if verbose:
            print("-->processing cov matrix for {0} items with gs {1}".format(len(onames),gs))
        dvals = obs.loc[onames,time_distance_col].values
        gcov = gs.covariance_matrix(dvals,np.ones(len(onames)),names=onames)
        if verbose:
            print("...scaling rows and cols")
        for i,name in enumerate(gcov.names):
            gcov.x[:,i] *= fcov_dict[name]
            gcov.x[i, :] *= fcov_dict[name]
        if verbose:
            print("...draw")
        oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst,gcov,num_reals=num_reals,fill=True,by_groups=False)
        oe = oe.loc[:,gcov.names]
        full_oe.loc[:,gcov.names] = oe._df.values

    if enforce_bounds:
        if verbose:
            print("-->enforcing bounds")
        ub_dict = {o:1e300 for o in full_oe.columns}
        if "upper_bound" in pst.observation_data.columns:
            ub_dict.update(pst.observation_data.upper_bound.fillna(1.0e300).to_dict())

        lb_dict = {o:-1e300 for o in full_oe.columns}
        if "lower_bound" in pst.observation_data.columns:
            lb_dict.update(pst.observation_data.lower_bound.fillna(-1.0e200).to_dict())
        allvals = full_oe.values
        for i,name in enumerate(full_oe.columns):
            #print("before:",name,ub_dict[name],full_oe.loc[:,name].max(),lb_dict[name],full_oe.loc[:,name].min())
            #vals = full_oe.loc[:,name].values
            vals = allvals[:,i]
            vals[vals>ub_dict[name]] = ub_dict[name]
            vals[vals < lb_dict[name]] = lb_dict[name]
            #full_oe.loc[:,name] = vals#oe.loc[:,name].apply(lambda x: min(x,ub_dict[name])).apply(lambda x: max(x,lb_dict[name]))
            allvals[:,i] = vals
            #print("...after:", name, ub_dict[name],full_oe.loc[:, name].max(),  lb_dict[name], full_oe.loc[:, name].min(), )

    if not draw_ineq:
        obs = pst.observation_data
        lt_tags = pst.get_constraint_tags("lt")
        lt_onames = [oname for oname,ogrp in zip(obs.obsnme,obs.obgnme) if True in [True if str(ogrp).startswith(tag) else False for tag in lt_tags]  ]
        if verbose:
            print("--> less than ineq obs:",lt_onames)
        lt_dict = obs.loc[lt_onames,"obsval"].to_dict()
        for n,v in lt_dict.items():
            full_oe.loc[:,n] = v
        obs = pst.observation_data
        gt_tags = pst.get_constraint_tags("gt")
        gt_onames = [oname for oname, ogrp in zip(obs.obsnme, obs.obgnme) if
                     True in [True if str(ogrp).startswith(tag) else False for tag in gt_tags]]
        if verbose:
            print("--> greater than ineq obs:", gt_onames)
        gt_dict = obs.loc[gt_onames, "obsval"].to_dict()
        for n, v in gt_dict.items():
            full_oe.loc[:, n] = v
    return full_oe














def geostatistical_draws(
    pst, struct_dict, num_reals=100, sigma_range=4, verbose=True,
        scale_offset=True, subset=None
):
    """construct a parameter ensemble from a prior covariance matrix
    implied by geostatistical structure(s) and parameter bounds.

    Args:
        pst (`pyemu.Pst`): a control file (or the name of control file).  The
            parameter bounds in `pst` are used to define the variance of each
            parameter group.
        struct_dict (`dict`): a dict of GeoStruct (or structure file), and list of
            pilot point template files pairs. If the values in the dict are
            `pd.DataFrames`, then they must have an 'x','y', and 'parnme' column.
            If the filename ends in '.csv', then a pd.DataFrame is loaded,
            otherwise a pilot points file is loaded.
        num_reals (`int`, optional): number of realizations to draw.  Default is 100
        sigma_range (`float`): a float representing the number of standard deviations
            implied by parameter bounds. Default is 4.0, which implies 95% confidence parameter bounds.
        verbose (`bool`, optional): flag to control output to stdout.  Default is True.
            flag for stdout.
        scale_offset (`bool`,optional): flag to apply scale and offset to parameter bounds
            when calculating variances - this is passed through to `pyemu.Cov.from_parameter_data()`.
            Default is True.
        subset (`array-like`, optional): list, array, set or pandas index defining subset of paramters
            for draw.

    Returns
        **pyemu.ParameterEnsemble**: the realized parameter ensemble.

    Note:
        Parameters are realized by parameter group.

        The variance of each parameter is used to scale the resulting geostatistical
        covariance matrix Therefore, the sill of the geostatistical structures
        in `struct_dict` should be 1.0

    Example::

        pst = pyemu.Pst("my.pst")
        sd = {"struct.dat":["hkpp.dat.tpl","vka.dat.tpl"]}
        pe = pyemu.helpers.geostatistical_draws(pst,struct_dict=sd}
        pe.to_csv("my_pe.csv")


    """

    if isinstance(pst, str):
        pst = pyemu.Pst(pst)
    assert isinstance(pst, pyemu.Pst), "pst arg must be a Pst instance, not {0}".format(
        type(pst)
    )
    if verbose:
        print("building diagonal cov")
    if subset is not None:
        if subset.empty or subset.intersection(pst.par_names).empty:
            warnings.warn(
                "Empty subset passed to draw method, or no intersecting pars "
                "with pst...\nwill draw full cov", PyemuWarning
            )
            subset = None
    full_cov = pyemu.Cov.from_parameter_data(
        pst, sigma_range=sigma_range, scale_offset=scale_offset,
        subset=subset
    )
    full_cov_dict = {n: float(v) for n, v in zip(full_cov.col_names, full_cov.x)}

    # par_org = pst.parameter_data.copy  # not sure about the need or function of this line? (BH)
    par = pst.parameter_data
    par_ens = []
    pars_in_cov = set()
    keys = list(struct_dict.keys())
    keys.sort()

    for gs in keys:
        items = struct_dict[gs]
        if verbose:
            print("processing ", gs)
        if isinstance(gs, str):
            gss = pyemu.geostats.read_struct_file(gs)
            if isinstance(gss, list):
                warnings.warn(
                    "using first geostat structure in file {0}".format(gs), PyemuWarning
                )
                gs = gss[0]
            else:
                gs = gss
        if gs.sill != 1.0:
            warnings.warn("GeoStruct {0} sill != 1.0 - this is bad!".format(gs.name))
        if not isinstance(items, list):
            items = [items]
        # items.sort()
        for iitem, item in enumerate(items):
            if isinstance(item, str):
                assert os.path.exists(item), "file {0} not found".format(item)
                if item.lower().endswith(".tpl"):
                    df = pyemu.pp_utils.pp_tpl_to_dataframe(item)
                elif item.lower.endswith(".csv"):
                    df = pd.read_csv(item)
            else:
                df = item
            if df.shape[0] < 2:
                continue
            if "pargp" in df.columns:
                if verbose:
                    print("working on pargroups {0}".format(df.pargp.unique().tolist()))
            for req in ["x", "y", "parnme"]:
                if req not in df.columns:
                    raise Exception("{0} is not in the columns".format(req))
            missing = df.loc[~df.parnme.isin(par.parnme), "parnme"]
            if len(missing) > 0:
                warnings.warn(
                    "the following parameters are not "
                    + "in the control file: {0}".format(",".join(missing)),
                    PyemuWarning,
                )
                df = df.loc[~df.parnme.isin(missing)]
            if df.shape[0] == 0:
                warnings.warn(
                    "geostatistical_draws(): empty parameter df at position {0} items for geostruct {1}, skipping...".format(
                        iitem, gs
                    )
                )
                continue
            if "zone" not in df.columns:
                df.loc[:, "zone"] = 1
            zones = df.zone.unique()
            aset = set(pst.adj_par_names)
            for zone in zones:
                df_zone = df.loc[df.zone == zone, :].copy()
                df_zone = df_zone.loc[df_zone.parnme.isin(aset), :]
                if df_zone.shape[0] == 0:
                    warnings.warn(
                        "all parameters in zone {0} tied and/or fixed, skipping...".format(
                            zone
                        ),
                        PyemuWarning,
                    )
                    continue

                # df_zone.sort_values(by="parnme",inplace=True)
                df_zone.sort_index(inplace=True)
                if verbose:
                    print("build cov matrix")
                cov = gs.covariance_matrix(df_zone.x, df_zone.y, df_zone.parnme)
                if verbose:
                    print("done")

                if verbose:
                    print("getting diag var cov", df_zone.shape[0])

                # tpl_var = max([full_cov_dict[pn] for pn in df_zone.parnme])
                if verbose:
                    print("scaling full cov by diag var cov")

                # for i in range(cov.shape[0]):
                #     cov.x[i, :] *= tpl_var
                for i, name in enumerate(cov.row_names):
                    # print(name,full_cov_dict[name])
                    cov.x[:, i] *= np.sqrt(full_cov_dict[name])
                    cov.x[i, :] *= np.sqrt(full_cov_dict[name])
                    cov.x[i, i] = full_cov_dict[name]
                # no fixed values here
                pe = pyemu.ParameterEnsemble.from_gaussian_draw(
                    pst=pst, cov=cov, num_reals=num_reals, by_groups=False, fill=False
                )
                par_ens.append(pe._df)
                pars_in_cov.update(set(pe.columns))

    if verbose:
        print("adding remaining parameters to diagonal")
    fset = set(full_cov.row_names)
    diff = list(fset.difference(pars_in_cov))
    if len(diff) > 0:
        name_dict = {name: i for i, name in enumerate(full_cov.row_names)}
        vec = np.atleast_2d(np.array([full_cov.x[name_dict[d]] for d in diff]))
        cov = pyemu.Cov(x=vec, names=diff, isdiagonal=True)
        # cov = full_cov.get(diff,diff)
        # here we fill in the fixed values
        pe = pyemu.ParameterEnsemble.from_gaussian_draw(
            pst, cov, num_reals=num_reals, fill=False
        )
        par_ens.append(pe._df)
    par_ens = pd.concat(par_ens, axis=1)
    par_ens = pyemu.ParameterEnsemble(pst=pst, df=par_ens)
    return par_ens


def geostatistical_prior_builder(
    pst, struct_dict, sigma_range=4, verbose=False, scale_offset=False
):
    """construct a full prior covariance matrix using geostastical structures
    and parameter bounds information.

    Args:
        pst (`pyemu.Pst`): a control file instance (or the name of control file)
        struct_dict (`dict`): a dict of GeoStruct (or structure file), and list of
            pilot point template files pairs. If the values in the dict are
            `pd.DataFrame` instances, then they must have an 'x','y', and 'parnme' column.
            If the filename ends in '.csv', then a pd.DataFrame is loaded,
            otherwise a pilot points file is loaded.
        sigma_range (`float`): a float representing the number of standard deviations
            implied by parameter bounds. Default is 4.0, which implies 95% confidence parameter bounds.
        verbose (`bool`, optional): flag to control output to stdout.  Default is True.
            flag for stdout.
        scale_offset (`bool`): a flag to apply scale and offset to parameter upper and lower bounds
            before applying log transform.  Passed to pyemu.Cov.from_parameter_data().  Default
            is False

    Returns:
        **pyemu.Cov**: a covariance matrix that includes all adjustable parameters in the control
        file.

    Note:
        The covariance of parameters associated with geostatistical structures is defined
        as a mixture of GeoStruct and bounds.  That is, the GeoStruct is used to construct a
        pyemu.Cov, then the rows and columns of the pyemu.Cov block are scaled by the uncertainty implied by the bounds and
        sigma_range. Most users will want to sill of the geostruct to sum to 1.0 so that the resulting
        covariance matrices have variance proportional to the parameter bounds. Sounds complicated...

        If the number of parameters exceeds about 20,000 this function may use all available memory
        then crash your computer.  In these high-dimensional cases, you probably dont need the prior
        covariance matrix itself, but rather an ensemble of paraaeter realizations.  In this case,
        please use the `geostatistical_draws()` function.

    Example::

        pst = pyemu.Pst("my.pst")
        sd = {"struct.dat":["hkpp.dat.tpl","vka.dat.tpl"]}
        cov = pyemu.helpers.geostatistical_prior_builder(pst,struct_dict=sd}
        cov.to_binary("prior.jcb")

    """

    if isinstance(pst, str):
        pst = pyemu.Pst(pst)
    assert isinstance(pst, pyemu.Pst), "pst arg must be a Pst instance, not {0}".format(
        type(pst)
    )
    if verbose:
        print("building diagonal cov")
    full_cov = pyemu.Cov.from_parameter_data(
        pst, sigma_range=sigma_range, scale_offset=scale_offset
    )

    full_cov_dict = {n: float(v) for n, v in zip(full_cov.col_names, full_cov.x)}
    # full_cov = None
    par = pst.parameter_data
    for gs, items in struct_dict.items():
        if verbose:
            print("processing ", gs)
        if isinstance(gs, str):
            gss = pyemu.geostats.read_struct_file(gs)
            if isinstance(gss, list):
                warnings.warn(
                    "using first geostat structure in file {0}".format(gs), PyemuWarning
                )
                gs = gss[0]
            else:
                gs = gss
        if gs.sill != 1.0:
            warnings.warn(
                "geostatistical_prior_builder() warning: geostruct sill != 1.0, user beware!"
            )
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if isinstance(item, str):
                assert os.path.exists(item), "file {0} not found".format(item)
                if item.lower().endswith(".tpl"):
                    df = pyemu.pp_utils.pp_tpl_to_dataframe(item)
                elif item.lower.endswith(".csv"):
                    df = pd.read_csv(item)
            else:
                df = item
            for req in ["x", "y", "parnme"]:
                if req not in df.columns:
                    raise Exception("{0} is not in the columns".format(req))
            missing = df.loc[df.parnme.apply(lambda x: x not in par.parnme), "parnme"]
            if len(missing) > 0:
                warnings.warn(
                    "the following parameters are not "
                    + "in the control file: {0}".format(",".join(missing)),
                    PyemuWarning,
                )
                df = df.loc[df.parnme.apply(lambda x: x not in missing)]
            if "zone" not in df.columns:
                df.loc[:, "zone"] = 1
            zones = df.zone.unique()
            aset = set(pst.adj_par_names)
            for zone in zones:
                df_zone = df.loc[df.zone == zone, :].copy()
                df_zone = df_zone.loc[df_zone.parnme.apply(lambda x: x in aset), :]
                if df_zone.shape[0] == 0:
                    warnings.warn(
                        "all parameters in zone {0} tied and/or fixed, skipping...".format(
                            zone
                        ),
                        PyemuWarning,
                    )
                    continue
                # df_zone.sort_values(by="parnme",inplace=True)
                df_zone.sort_index(inplace=True)
                if verbose:
                    print("build cov matrix")
                cov = gs.covariance_matrix(df_zone.x, df_zone.y, df_zone.parnme)
                if verbose:
                    print("done")
                # find the variance in the diagonal cov
                if verbose:
                    print("getting diag var cov", df_zone.shape[0])

                # tpl_var = max([full_cov_dict[pn] for pn in df_zone.parnme])

                if verbose:
                    print("scaling full cov by diag var cov")

                # cov *= tpl_var
                for i, name in enumerate(cov.row_names):
                    cov.x[:, i] *= np.sqrt(full_cov_dict[name])
                    cov.x[i, :] *= np.sqrt(full_cov_dict[name])
                    cov.x[i, i] = full_cov_dict[name]
                if verbose:
                    print("test for inversion")
                try:
                    ci = cov.inv
                except:
                    df_zone.to_csv("prior_builder_crash.csv")
                    raise Exception("error inverting cov {0}".format(cov.row_names[:3]))

                if verbose:
                    print("replace in full cov")
                full_cov.replace(cov)
                # d = np.diag(full_cov.x)
                # idx = np.argwhere(d==0.0)
                # for i in idx:
                #     print(full_cov.names[i])
    return full_cov


def _rmse(v1, v2):
    """return root mean squared error between v1 and v2

    Args:
        v1 (iterable): one vector
        v2 (iterable): another vector

    Returns:
        **float**: root mean squared error of v1,v2

    """

    return np.sqrt(np.mean(np.square(v1 - v2)))


def calc_observation_ensemble_quantiles(
    ens, pst, quantiles, subset_obsnames=None, subset_obsgroups=None
):
    """Given an observation ensemble, and requested quantiles, this function calculates the requested
       quantile point-by-point in the ensemble. This resulting set of values does not, however, correspond
       to a single realization in the ensemble. So, this function finds the minimum weighted squared
       distance to the quantile and labels it in the ensemble. Also indicates which realizations
       correspond to the selected quantiles.

    Args:
        ens (pandas DataFrame): DataFrame read from an observation
        pst (pyemy.Pst object) - needed to obtain observation weights
        quantiles (iterable): quantiles ranging from 0-1.0 for which results requested
        subset_obsnames (iterable): list of observation names to include in calculations
        subset_obsgroups (iterable): list of observation groups to include in calculations

    Returns:
        tuple containing

        - **pandas DataFrame**: same ens object that was input but with quantile realizations
                                appended as new rows labelled with 'q_#' where '#' is the slected quantile
        - **dict**: dictionary with keys being quantiles and values being realizations
                    corresponding to each realization
    """
    # TODO: handle zero weights due to PDC

    quantile_idx = {}
    # make sure quantiles and subset names and groups are lists
    if not isinstance(quantiles, list):
        quantiles = list(quantiles)
    if not isinstance(subset_obsnames, list) and subset_obsnames is not None:
        subset_obsnames = list(subset_obsnames)
    if not isinstance(subset_obsgroups, list) and subset_obsgroups is not None:
        subset_obsgroups = list(subset_obsgroups)

    if "real_name" in ens.columns:
        ens.set_index("real_name")
    # if 'base' real was lost, then the index is of type int. needs to be string later so set here
    ens.index = [str(i) for i in ens.index]
    if not isinstance(pst, pyemu.Pst):
        raise Exception("pst object must be of type pyemu.Pst")

    # get the observation data
    obs = pst.observation_data.copy()

    # confirm that the indices and weights line up
    if False in np.unique(ens.columns == obs.index):
        raise Exception("ens and pst observation names do not align")

    # deal with any subsetting of observations that isn't handled through weights

    trimnames = obs.index.values
    if subset_obsgroups is not None and subset_obsnames is not None:
        raise Exception(
            "can only specify information in one of subset_obsnames of subset_obsgroups. not both"
        )

    if subset_obsnames is not None:
        trimnames = subset_obsnames
        if len(set(trimnames) - set(obs.index.values)) != 0:
            raise Exception(
                "the following names in subset_obsnames are not in the ensemble:\n"
                + ["{}\n".format(i) for i in (set(trimnames) - set(obs.index.values))]
            )

    if subset_obsgroups is not None:
        if len((set(subset_obsgroups) - set(pst.obs_groups))) != 0:
            raise Exception(
                "the following groups in subset_obsgroups are not in pst:\n"
                + [
                    "{}\n".format(i)
                    for i in (set(subset_obsgroups) - set(pst.obs_groups))
                ]
            )

        trimnames = obs.loc[obs.obgnme.isin(subset_obsgroups)].obsnme.tolist()
        if len((set(trimnames) - set(obs.index.values))) != 0:
            raise Exception(
                "the following names in subset_obsnames are not in the ensemble:\n"
                + ["{}\n".format(i) for i in (set(trimnames) - set(obs.index.values))]
            )
    # trim the data to subsets (or complete )
    ens_eval = ens[trimnames].copy()
    weights = obs.loc[trimnames].weight.values

    for cq in quantiles:
        # calculate the point-wise quantile values
        qfit = np.quantile(ens_eval, cq, axis=0)
        # calculate the weighted distance between all reals and the desired quantile
        qreal = np.argmin(
            np.linalg.norm([(i - qfit) * weights for i in ens_eval.values], axis=1)
        )
        quantile_idx["q{}".format(cq)] = qreal
        ens = ens.append(ens.iloc[qreal])
        idx = ens.index.values
        idx[-1] = "q{}".format(cq)
        ens.set_index(idx, inplace=True)

    return ens, quantile_idx


def calc_rmse_ensemble(ens, pst, bygroups=True, subset_realizations=None):
    """
    DEPRECATED -->please see pyemu.utils.metrics.calc_metric_ensemble()
    Calculates RMSE (without weights) to quantify fit to observations for ensemble members

    Args:
        ens (pandas DataFrame): DataFrame read from an observation
        pst (pyemy.Pst object) - needed to obtain observation weights
        bygroups (Bool): Flag to summarize by groups or not. Defaults to True.
        subset_realizations (iterable, optional): Subset of realizations for which
                to report RMSE. Defaults to None which returns all realizations.

    Returns:
        **pandas.DataFrame**: rows are realizations. Columns are groups. Content is RMSE
    """

    raise Exception(
        "this is deprecated-->please see pyemu.utils.metrics.calc_metric_ensemble()"
    )


def _condition_on_par_knowledge(cov, var_knowledge_dict):
    """experimental function to condition a covariance matrix with the variances of new information.

    Args:
        cov (`pyemu.Cov`): prior covariance matrix
        var_knowledge_dict (`dict`): a dictionary of covariance entries and variances

    Returns:
        **pyemu.Cov**: the conditional covariance matrix

    """

    missing = []
    for name in var_knowledge_dict.keys():
        if name not in cov.row_names:
            missing.append(name)
    if len(missing):
        raise Exception(
            "var knowledge dict entries not found: {0}".format(",".join(missing))
        )
    if cov.isdiagonal:
        raise Exception(
            "_condition_on_par_knowledge(): cov is diagonal, simply update the par variances"
        )
    know_names = list(var_knowledge_dict.keys())
    know_names.sort()
    know_cross_cov = cov.get(cov.row_names, know_names)

    know_cov = cov.get(know_names, know_names)
    # add the par knowledge to the diagonal of know_cov
    for i, name in enumerate(know_names):
        know_cov.x[i, i] += var_knowledge_dict[name]

    # kalman gain
    k_gain = know_cross_cov * know_cov.inv
    # selection matrix
    h = k_gain.zero2d.T
    know_dict = {n: i for i, n in enumerate(know_names)}
    for i, name in enumerate(cov.row_names):
        if name in know_dict:
            h.x[know_dict[name], i] = 1.0

    prod = k_gain * h
    conditional_cov = (prod.identity - prod) * cov
    return conditional_cov


def kl_setup(
    num_eig,
    sr,
    struct,
    prefixes,
    factors_file="kl_factors.dat",
    islog=True,
    basis_file=None,
    tpl_dir=".",
):
    """setup a karhuenen-Loeve based parameterization for a given
    geostatistical structure.

    Args:
        num_eig (`int`): the number of basis vectors to retain in the
            reduced basis
        sr (`flopy.reference.SpatialReference`): a spatial reference instance
        struct (`str`): a PEST-style structure file.  Can also be a
            `pyemu.geostats.Geostruct` instance.
        prefixes ([`str`]): a list of parameter prefixes to generate KL
            parameterization for.
        factors_file (`str`, optional): name of the PEST-style interpolation
            factors file to write (can be processed with FAC2REAL).
            Default is "kl_factors.dat".
        islog (`bool`, optional): flag to indicate if the parameters are log transformed.
            Default is True
        basis_file (`str`, optional): the name of the PEST-style binary (e.g. jco)
            file to write the reduced basis vectors to.  Default is None (not saved).
        tpl_dir (`str`, optional): the directory to write the resulting
            template files to.  Default is "." (current directory).

    Returns:
        **pandas.DataFrame**: a dataframe of parameter information.

    Note:
        This is the companion function to `helpers.apply_kl()`

    Example::

        m = flopy.modflow.Modflow.load("mymodel.nam")
        prefixes = ["hk","vka","ss"]
        df = pyemu.helpers.kl_setup(10,m.sr,"struct.dat",prefixes)

    """

    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    assert isinstance(sr, pyemu.helpers.SpatialReference)
    # for name,array in array_dict.items():
    #     assert isinstance(array,np.ndarray)
    #     assert array.shape[0] == sr.nrow
    #     assert array.shape[1] == sr.ncol
    #     assert len(name) + len(str(num_eig)) <= 12,"name too long:{0}".\
    #         format(name)

    if isinstance(struct, str):
        assert os.path.exists(struct)
        gs = pyemu.utils.read_struct_file(struct)
    else:
        gs = struct
    names = []
    for i in range(sr.nrow):
        names.extend(["i{0:04d}j{1:04d}".format(i, j) for j in range(sr.ncol)])

    cov = gs.covariance_matrix(
        sr.xcentergrid.flatten(), sr.ycentergrid.flatten(), names=names
    )

    eig_names = ["eig_{0:04d}".format(i) for i in range(cov.shape[0])]
    trunc_basis = cov.u
    trunc_basis.col_names = eig_names
    # trunc_basis.col_names = [""]
    if basis_file is not None:
        trunc_basis.to_binary(basis_file)
    trunc_basis = trunc_basis[:, :num_eig]
    eig_names = eig_names[:num_eig]

    pp_df = pd.DataFrame({"name": eig_names}, index=eig_names)
    pp_df.loc[:, "x"] = -1.0 * sr.ncol
    pp_df.loc[:, "y"] = -1.0 * sr.nrow
    pp_df.loc[:, "zone"] = -999
    pp_df.loc[:, "parval1"] = 1.0
    pyemu.pp_utils.write_pp_file(os.path.join("temp.dat"), pp_df)

    _eigen_basis_to_factor_file(
        sr.nrow, sr.ncol, trunc_basis, factors_file=factors_file, islog=islog
    )
    dfs = []
    for prefix in prefixes:
        tpl_file = os.path.join(tpl_dir, "{0}.dat_kl.tpl".format(prefix))
        df = pyemu.pp_utils.pilot_points_to_tpl("temp.dat", tpl_file, prefix)
        shutil.copy2("temp.dat", tpl_file.replace(".tpl", ""))
        df.loc[:, "tpl_file"] = tpl_file
        df.loc[:, "in_file"] = tpl_file.replace(".tpl", "")
        df.loc[:, "prefix"] = prefix
        df.loc[:, "pargp"] = "kl_{0}".format(prefix)
        dfs.append(df)
        # arr = pyemu.geostats.fac2real(df,factors_file=factors_file,out_file=None)
    df = pd.concat(dfs)
    df.loc[:, "parubnd"] = 10.0
    df.loc[:, "parlbnd"] = 0.1
    return pd.concat(dfs)

    # back_array_dict = {}
    # f = open(tpl_file,'w')
    # f.write("ptf ~\n")
    # f.write("name,org_val,new_val\n")
    # for name,array in array_dict.items():
    #     mname = name+"mean"
    #     f.write("{0},{1:20.8E},~   {2}    ~\n".format(mname,0.0,mname))
    #     #array -= array.mean()
    #     array_flat = pyemu.Matrix(x=np.atleast_2d(array.flatten()).transpose()
    #                               ,col_names=["flat"],row_names=names,
    #                               isdiagonal=False)
    #     factors = trunc_basis * array_flat
    #     enames = ["{0}{1:04d}".format(name,i) for i in range(num_eig)]
    #     for n,val in zip(enames,factors.x):
    #        f.write("{0},{1:20.8E},~    {0}    ~\n".format(n,val[0]))
    #     back_array_dict[name] = (factors.T * trunc_basis).x.reshape(array.shape)
    #     print(array_back)
    #     print(factors.shape)
    #
    # return back_array_dict


def _eigen_basis_to_factor_file(nrow, ncol, basis, factors_file, islog=True):
    assert nrow * ncol == basis.shape[0]
    with open(factors_file, "w") as f:
        f.write("junk.dat\n")
        f.write("junk.zone.dat\n")
        f.write("{0} {1}\n".format(ncol, nrow))
        f.write("{0}\n".format(basis.shape[1]))
        [f.write(name + "\n") for name in basis.col_names]
        t = 0
        if islog:
            t = 1
        for i in range(nrow * ncol):
            f.write("{0} {1} {2} {3:8.5e}".format(i + 1, t, basis.shape[1], 0.0))
            [
                f.write(" {0} {1:12.8g} ".format(i + 1, w))
                for i, w in enumerate(basis.x[i, :])
            ]
            f.write("\n")


def kl_apply(par_file, basis_file, par_to_file_dict, arr_shape):
    """Apply a KL parameterization transform from basis factors to model
    input arrays.

    Args:
        par_file (`str`): the csv file to get factor values from.  Must contain
            the following columns: "name", "new_val", "org_val"
        basis_file (`str`): the PEST-style binary file that contains the reduced
            basis
        par_to_file_dict (`dict`): a mapping from KL parameter prefixes to array
            file names.
        arr_shape (tuple): a length 2 tuple of number of rows and columns
            the resulting arrays should have.

        Note:
            This is the companion function to kl_setup.
            This function should be called during the forward run

    """
    df = pd.read_csv(par_file)
    assert "name" in df.columns
    assert "org_val" in df.columns
    assert "new_val" in df.columns

    df.loc[:, "prefix"] = df.name.apply(lambda x: x[:-4])
    for prefix in df.prefix.unique():
        assert prefix in par_to_file_dict.keys(), "missing prefix:{0}".format(prefix)
    basis = pyemu.Matrix.from_binary(basis_file)
    assert basis.shape[1] == arr_shape[0] * arr_shape[1]
    arr_min = 1.0e-10  # a temp hack

    # means = df.loc[df.name.apply(lambda x: x.endswith("mean")),:]
    # print(means)
    df = df.loc[df.name.apply(lambda x: not x.endswith("mean")), :]
    for prefix, filename in par_to_file_dict.items():
        factors = pyemu.Matrix.from_dataframe(df.loc[df.prefix == prefix, ["new_val"]])
        factors.autoalign = False
        basis_prefix = basis[: factors.shape[0], :]
        arr = (factors.T * basis_prefix).x.reshape(arr_shape)
        # arr += means.loc[means.prefix==prefix,"new_val"].values
        arr[arr < arr_min] = arr_min
        np.savetxt(filename, arr, fmt="%20.8E")


def zero_order_tikhonov(pst, parbounds=True, par_groups=None, reset=True):
    """setup preferred-value regularization in a pest control file.

    Args:
        pst (`pyemu.Pst`): the control file instance
        parbounds (`bool`, optional): flag to weight the new prior information
            equations according to parameter bound width - approx the KL
            transform. Default is True
        par_groups (`list`): a list of parameter groups to build PI equations for.
            If None, all adjustable parameters are used. Default is None
        reset (`bool`): a flag to remove any existing prior information equations
            in the control file.  Default is True

    Note:
        Operates in place.


    Example::

        pst = pyemu.Pst("my.pst")
        pyemu.helpers.zero_order_tikhonov(pst)
        pst.write("my_reg.pst")

    """

    if par_groups is None:
        par_groups = pst.par_groups

    pilbl, obgnme, weight, equation = [], [], [], []
    for idx, row in pst.parameter_data.iterrows():
        pt = row["partrans"].lower()
        try:
            pt = pt.decode()
        except:
            pass
        if pt not in ["tied", "fixed"] and row["pargp"] in par_groups:
            pilbl.append(row["parnme"])
            weight.append(1.0)
            ogp_name = "regul" + row["pargp"]
            obgnme.append(ogp_name[:12])
            parnme = row["parnme"]
            parval1 = row["parval1"]
            if pt == "log":
                parnme = "log(" + parnme + ")"
                parval1 = np.log10(parval1)
            eq = "1.0 * " + parnme + " ={0:15.6E}".format(parval1)
            equation.append(eq)

    if reset:
        pst.prior_information = pd.DataFrame(
            {"pilbl": pilbl, "equation": equation, "obgnme": obgnme, "weight": weight}
        )
    else:
        pi = pd.DataFrame(
            {"pilbl": pilbl, "equation": equation, "obgnme": obgnme, "weight": weight}
        )
        pst.prior_information = pst.prior_information.append(pi)
    if parbounds:
        _regweight_from_parbound(pst)
    if pst.control_data.pestmode == "estimation":
        pst.control_data.pestmode = "regularization"


def _regweight_from_parbound(pst):
    """sets regularization weights from parameter bounds
    which approximates the KL expansion.  Called by
    zero_order_tikhonov().

    Args:
        pst (`pyemu.Pst`): control file

    """

    pst.parameter_data.index = pst.parameter_data.parnme
    pst.prior_information.index = pst.prior_information.pilbl
    for idx, parnme in enumerate(pst.prior_information.pilbl):
        if parnme in pst.parameter_data.index:
            row = pst.parameter_data.loc[parnme, :]
            lbnd, ubnd = row["parlbnd"], row["parubnd"]
            if row["partrans"].lower() == "log":
                weight = 1.0 / (np.log10(ubnd) - np.log10(lbnd))
            else:
                weight = 1.0 / (ubnd - lbnd)
            pst.prior_information.loc[parnme, "weight"] = weight
        else:
            print(
                "prior information name does not correspond"
                + " to a parameter: "
                + str(parnme)
            )


def first_order_pearson_tikhonov(pst, cov, reset=True, abs_drop_tol=1.0e-3):
    """setup preferred-difference regularization from a covariance matrix.


    Args:
        pst (`pyemu.Pst`): the PEST control file
        cov (`pyemu.Cov`): a covariance matrix instance with
            some or all of the parameters listed in `pst`.
        reset (`bool`): a flag to remove any existing prior information equations
            in the control file.  Default is True
        abs_drop_tol (`float`, optional): tolerance to control how many pi equations
            are written. If the absolute value of the Pearson CC is less than
            abs_drop_tol, the prior information equation will not be included in
            the control file.

    Note:
        The weights on the prior information equations are the Pearson
        correlation coefficients implied by covariance matrix.

        Operates in place

    Example::

        pst = pyemu.Pst("my.pst")
        cov = pyemu.Cov.from_ascii("my.cov")
        pyemu.helpers.first_order_pearson_tikhonov(pst,cov)
        pst.write("my_reg.pst")

    """
    assert isinstance(cov, pyemu.Cov)
    print("getting CC matrix")
    cc_mat = cov.to_pearson()
    # print(pst.parameter_data.dtypes)
    try:
        ptrans = pst.parameter_data.partrans.apply(lambda x: x.decode()).to_dict()
    except:
        ptrans = pst.parameter_data.partrans.to_dict()
    pi_num = pst.prior_information.shape[0] + 1
    pilbl, obgnme, weight, equation = [], [], [], []
    sadj_names = set(pst.adj_par_names)
    print("processing")
    for i, iname in enumerate(cc_mat.row_names):
        if iname not in sadj_names:
            continue
        for j, jname in enumerate(cc_mat.row_names[i + 1 :]):
            if jname not in sadj_names:
                continue
            # print(i,iname,i+j+1,jname)
            cc = cc_mat.x[i, j + i + 1]
            if cc < abs_drop_tol:
                continue
            pilbl.append("pcc_{0}".format(pi_num))
            iiname = str(iname)
            if str(ptrans[iname]) == "log":
                iiname = "log(" + iname + ")"
            jjname = str(jname)
            if str(ptrans[jname]) == "log":
                jjname = "log(" + jname + ")"
            equation.append("1.0 * {0} - 1.0 * {1} = 0.0".format(iiname, jjname))
            weight.append(cc)
            obgnme.append("regul_cc")
            pi_num += 1
    df = pd.DataFrame(
        {"pilbl": pilbl, "equation": equation, "obgnme": obgnme, "weight": weight}
    )
    df.index = df.pilbl
    if reset:
        pst.prior_information = df
    else:
        pst.prior_information = pst.prior_information.append(df)

    if pst.control_data.pestmode == "estimation":
        pst.control_data.pestmode = "regularization"


def simple_tpl_from_pars(parnames, tplfilename="model.input.tpl"):
    """Make a simple template file from a list of parameter names.

    Args:
        parnames ([`str`]): list of parameter names to put in the
            new template file
        tplfilename (`str`): Name of the template file to create.  Default
            is "model.input.tpl"

    Note:
        Writes a file `tplfilename` with each parameter name in `parnames` on a line

    """
    with open(tplfilename, "w") as ofp:
        ofp.write("ptf ~\n")
        [ofp.write("~{0:^12}~\n".format(cname)) for cname in parnames]


def simple_ins_from_obs(obsnames, insfilename="model.output.ins"):
    """write a simple instruction file that reads the values named
     in obsnames in order, one per line from a model output file

    Args:
        obsnames (`str`): list of observation names to put in the
            new instruction file
        insfilename (`str`): the name of the instruction file to
            create. Default is "model.output.ins"

    Note:
        writes a file `insfilename` with each observation read off
        of a single line


    """
    with open(insfilename, "w") as ofp:
        ofp.write("pif ~\n")
        [ofp.write("l1 !{0}!\n".format(cob)) for cob in obsnames]


def pst_from_parnames_obsnames(
    parnames, obsnames, tplfilename="model.input.tpl", insfilename="model.output.ins"
):
    """Creates a Pst object from a list of parameter names and a list of observation names.

    Args:
        parnames (`str`): list of parameter names
        obsnames (`str`): list of observation names
        tplfilename (`str`): template filename. Default is  "model.input.tpl"
        insfilename (`str`): instruction filename. Default is "model.output.ins"

    Returns:
        `pyemu.Pst`: the generic control file

    Example::

        parnames = ["p1","p2"]
        obsnames = ["o1","o2"]
        pst = pyemu.helpers.pst_from_parnames_obsnames(parname,obsnames)


    """
    simple_tpl_from_pars(parnames, tplfilename)
    simple_ins_from_obs(obsnames, insfilename)

    modelinputfilename = tplfilename.replace(".tpl", "")
    modeloutputfilename = insfilename.replace(".ins", "")

    return pyemu.Pst.from_io_files(
        tplfilename, modelinputfilename, insfilename, modeloutputfilename
    )


def read_pestpp_runstorage(filename, irun=0, with_metadata=False):
    """read pars and obs from a specific run in a pest++ serialized
    run storage file (e.g. .rns/.rnj) into dataframes.

    Args:
        filename (`str`): the name of the run storage file
        irun (`int`): the run id to process. If 'all', then all runs are
            read. Default is 0
        with_metadata (`bool`): flag to return run stats and info txt as well

    Returns:
        tuple containing

        - **pandas.DataFrame**: parameter information
        - **pandas.DataFrame**: observation information
        - **pandas.DataFrame**: optionally run status and info txt.

    Note:

        This function can save you heaps of misery of your pest++ run
        died before writing output files...

    """

    header_dtype = np.dtype(
        [
            ("n_runs", np.int64),
            ("run_size", np.int64),
            ("p_name_size", np.int64),
            ("o_name_size", np.int64),
        ]
    )

    try:
        irun = int(irun)
    except:
        if irun.lower() == "all":
            irun = irun.lower()
        else:
            raise Exception(
                "unrecognized 'irun': should be int or 'all', not '{0}'".format(irun)
            )

    def status_str(r_status):
        if r_status == 0:
            return "not completed"
        if r_status == 1:
            return "completed"
        if r_status == -100:
            return "canceled"
        else:
            return "failed"

    assert os.path.exists(filename)
    f = open(filename, "rb")
    header = np.fromfile(f, dtype=header_dtype, count=1)
    p_name_size, o_name_size = header["p_name_size"][0], header["o_name_size"][0]
    par_names = (
        struct.unpack("{0}s".format(p_name_size), f.read(p_name_size))[0]
        .strip()
        .lower()
        .decode()
        .split("\0")[:-1]
    )
    obs_names = (
        struct.unpack("{0}s".format(o_name_size), f.read(o_name_size))[0]
        .strip()
        .lower()
        .decode()
        .split("\0")[:-1]
    )
    n_runs, run_size = header["n_runs"][0], header["run_size"][0]
    run_start = f.tell()

    def _read_run(irun):
        f.seek(run_start + (irun * run_size))
        r_status = np.fromfile(f, dtype=np.int8, count=1)
        info_txt = struct.unpack("41s", f.read(41))[0].strip().lower().decode()
        par_vals = np.fromfile(f, dtype=np.float64, count=len(par_names) + 1)[1:]
        obs_vals = np.fromfile(f, dtype=np.float64, count=len(obs_names) + 1)[:-1]
        par_df = pd.DataFrame({"parnme": par_names, "parval1": par_vals})

        par_df.index = par_df.pop("parnme")
        obs_df = pd.DataFrame({"obsnme": obs_names, "obsval": obs_vals})
        obs_df.index = obs_df.pop("obsnme")
        return r_status, info_txt, par_df, obs_df

    if irun == "all":
        par_dfs, obs_dfs = [], []
        r_stats, txts = [], []
        for irun in range(n_runs):
            # print(irun)
            r_status, info_txt, par_df, obs_df = _read_run(irun)
            par_dfs.append(par_df)
            obs_dfs.append(obs_df)
            r_stats.append(r_status)
            txts.append(info_txt)
        par_df = pd.concat(par_dfs, axis=1).T
        par_df.index = np.arange(n_runs)
        obs_df = pd.concat(obs_dfs, axis=1).T
        obs_df.index = np.arange(n_runs)
        meta_data = pd.DataFrame({"r_status": r_stats, "info_txt": txts})
        meta_data.loc[:, "status"] = meta_data.r_status.apply(status_str)

    else:
        assert irun <= n_runs
        r_status, info_txt, par_df, obs_df = _read_run(irun)
        meta_data = pd.DataFrame({"r_status": [r_status], "info_txt": [info_txt]})
        meta_data.loc[:, "status"] = meta_data.r_status.apply(status_str)
    f.close()
    if with_metadata:
        return par_df, obs_df, meta_data
    else:
        return par_df, obs_df


def jco_from_pestpp_runstorage(rnj_filename, pst_filename):
    """read pars and obs from a pest++ serialized run storage
    file (e.g., .rnj) and return jacobian matrix instance

    Args:
        rnj_filename (`str`): the name of the run storage file
        pst_filename (`str`): the name of the pst file

    Note:
        This can then be passed to Jco.to_binary or Jco.to_coo, etc., to write jco
        file in a subsequent step to avoid memory resource issues associated
        with very large problems.


    Returns:
        `pyemu.Jco`: a jacobian matrix constructed from the run results and
        pest control file information.

    """

    header_dtype = np.dtype(
        [
            ("n_runs", np.int64),
            ("run_size", np.int64),
            ("p_name_size", np.int64),
            ("o_name_size", np.int64),
        ]
    )

    pst = pyemu.Pst(pst_filename)
    par = pst.parameter_data
    log_pars = set(par.loc[par.partrans == "log", "parnme"].values)
    with open(rnj_filename, "rb") as f:
        header = np.fromfile(f, dtype=header_dtype, count=1)

    try:
        base_par, base_obs = read_pestpp_runstorage(rnj_filename, irun=0)
    except:
        raise Exception("couldn't get base run...")
    par = par.loc[base_par.index, :]
    li = base_par.index.map(lambda x: par.loc[x, "partrans"] == "log")
    base_par.loc[li] = base_par.loc[li].apply(np.log10)
    jco_cols = {}
    for irun in range(1, int(header["n_runs"])):
        par_df, obs_df = read_pestpp_runstorage(rnj_filename, irun=irun)
        par_df.loc[li] = par_df.loc[li].apply(np.log10)
        obs_diff = base_obs - obs_df
        par_diff = base_par - par_df
        # check only one non-zero element per col(par)
        if len(par_diff[par_diff.parval1 != 0]) > 1:
            raise Exception(
                "more than one par diff - looks like the file wasn't created during jco filling..."
            )
        parnme = par_diff[par_diff.parval1 != 0].index[0]
        parval = par_diff.parval1.loc[parnme]

        # derivatives
        jco_col = obs_diff / parval
        # some tracking, checks
        print("processing par {0}: {1}...".format(irun, parnme))
        print(
            "%nzsens: {0}%...".format(
                (jco_col[abs(jco_col.obsval) > 1e-8].shape[0] / jco_col.shape[0])
                * 100.0
            )
        )

        jco_cols[parnme] = jco_col.obsval

    jco_cols = pd.DataFrame.from_records(
        data=jco_cols, index=list(obs_diff.index.values)
    )

    jco_cols = pyemu.Jco.from_dataframe(jco_cols)

    # write # memory considerations important here for very large matrices - break into chunks...
    # jco_fnam = "{0}".format(filename[:-4]+".jco")
    # jco_cols.to_binary(filename=jco_fnam, droptol=None, chunk=None)

    return jco_cols


def parse_dir_for_io_files(d, prepend_path=False):
    """find template/input file pairs and instruction file/output file
    pairs by extension.

    Args:
        d (`str`): directory to search for interface files
        prepend_path (`bool`, optional): flag to prepend `d` to each file name.
            Default is False

    Returns:
        tuple containing

        - **[`str`]**: list of template files in d
        - **[`str`]**: list of input files in d
        - **[`str`]**: list of instruction files in d
        - **[`str`]**: list of output files in d

    Note:
        the return values from this function can be passed straight to
        `pyemu.Pst.from_io_files()` classmethod constructor.

        Assumes the template file names are <input_file>.tpl and instruction file names
        are <output_file>.ins.

    Example::

        files = pyemu.helpers.parse_dir_for_io_files("template",prepend_path=True)
        pst = pyemu.Pst.from_io_files(*files,pst_path=".")


    """

    files = os.listdir(d)
    tpl_files = [f for f in files if f.endswith(".tpl")]
    in_files = [f.replace(".tpl", "") for f in tpl_files]
    ins_files = [f for f in files if f.endswith(".ins")]
    out_files = [f.replace(".ins", "") for f in ins_files]
    if prepend_path:
        tpl_files = [os.path.join(d, item) for item in tpl_files]
        in_files = [os.path.join(d, item) for item in in_files]
        ins_files = [os.path.join(d, item) for item in ins_files]
        out_files = [os.path.join(d, item) for item in out_files]

    return tpl_files, in_files, ins_files, out_files


def pst_from_io_files(
    tpl_files, in_files, ins_files, out_files, pst_filename=None, pst_path=None
):
    """create a Pst instance from model interface files.

    Args:
        tpl_files ([`str`]): list of template file names
        in_files ([`str`]): list of model input file names (pairs with template files)
        ins_files ([`str`]): list of instruction file names
        out_files ([`str`]): list of model output file names (pairs with instruction files)
        pst_filename (`str`): name of control file to write.  If None, no file is written.
            Default is None
        pst_path (`str`): the path to append to the template_file and in_file in the control file.  If
            not None, then any existing path in front of the template or in file is split off
            and pst_path is prepended.  If python is being run in a directory other than where the control
            file will reside, it is useful to pass `pst_path` as `.`.  Default is None


    Returns:
        `Pst`: new control file instance with parameter and observation names
        found in `tpl_files` and `ins_files`, repsectively.

    Note:
        calls `pyemu.helpers.pst_from_io_files()`

        Assigns generic values for parameter info.  Tries to use INSCHEK
        to set somewhat meaningful observation values

        all file paths are relatively to where python is running.

    Example::

        tpl_files = ["my.tpl"]
        in_files = ["my.in"]
        ins_files = ["my.ins"]
        out_files = ["my.out"]
        pst = pyemu.Pst.from_io_files(tpl_files,in_files,ins_files,out_files)
        pst.control_data.noptmax = 0
        pst.write("my.pst)



    """
    par_names = set()
    if not isinstance(tpl_files, list):
        tpl_files = [tpl_files]
    if not isinstance(in_files, list):
        in_files = [in_files]
    assert len(in_files) == len(tpl_files), "len(in_files) != len(tpl_files)"

    for tpl_file in tpl_files:
        assert os.path.exists(tpl_file), "template file not found: " + str(tpl_file)
        # new_names = [name for name in pyemu.pst_utils.parse_tpl_file(tpl_file) if name not in par_names]
        # par_names.extend(new_names)
        new_names = pyemu.pst_utils.parse_tpl_file(tpl_file)
        par_names.update(new_names)

    if not isinstance(ins_files, list):
        ins_files = [ins_files]
    if not isinstance(out_files, list):
        out_files = [out_files]
    assert len(ins_files) == len(out_files), "len(out_files) != len(out_files)"

    obs_names = []
    for ins_file in ins_files:
        assert os.path.exists(ins_file), "instruction file not found: " + str(ins_file)
        obs_names.extend(pyemu.pst_utils.parse_ins_file(ins_file))

    new_pst = pyemu.pst_utils.generic_pst(list(par_names), list(obs_names))

    if "window" in platform.platform().lower() and pst_path == ".":
        pst_path = ""

    # new_pst.instruction_files = ins_files
    # new_pst.output_files = out_files
    new_pst.model_output_data = pd.DataFrame(
        {"pest_file": ins_files, "model_file": out_files}, index=ins_files
    )

    # try to run inschek to find the observtion values
    # do this here with full paths to files
    pyemu.pst_utils.try_process_output_pst(new_pst)

    if pst_path is not None:
        tpl_files = [
            os.path.join(pst_path, os.path.split(tpl_file)[-1])
            for tpl_file in tpl_files
        ]
        in_files = [
            os.path.join(pst_path, os.path.split(in_file)[-1]) for in_file in in_files
        ]
        # now set the true path location to instruction files and output files
        ins_files = [
            os.path.join(pst_path, os.path.split(ins_file)[-1])
            for ins_file in ins_files
        ]
        out_files = [
            os.path.join(pst_path, os.path.split(out_file)[-1])
            for out_file in out_files
        ]

    new_pst.model_input_data = pd.DataFrame(
        {"pest_file": tpl_files, "model_file": in_files}, index=tpl_files
    )

    new_pst.model_output_data = pd.DataFrame(
        {"pest_file": ins_files, "model_file": out_files}, index=ins_files
    )

    new_pst.try_parse_name_metadata()
    if pst_filename:
        new_pst.write(pst_filename)

    return new_pst


wildass_guess_par_bounds_dict = {
    "hk": [0.01, 100.0],
    "vka": [0.1, 10.0],
    "sy": [0.25, 1.75],
    "ss": [0.1, 10.0],
    "cond": [0.01, 100.0],
    "flux": [0.25, 1.75],
    "rech": [0.9, 1.1],
    "stage": [0.9, 1.1],
}


class PstFromFlopyModel(object):
    """a monster helper class to setup a complex PEST interface around
    an existing MODFLOW-2005-family model.


    Args:
        model (`flopy.mbase`): a loaded flopy model instance. If model is an str, it is treated as a
            MODFLOW nam file (requires org_model_ws)
        new_model_ws (`str`): a directory where the new version of MODFLOW input files and PEST(++)
            files will be written
        org_model_ws (`str`): directory to existing MODFLOW model files.  Required if model argument
            is an str.  Default is None
        pp_props ([[`str`,[`int`]]]): pilot point multiplier parameters for grid-based properties.
            A nested list of grid-scale model properties to parameterize using
            name, iterable pairs.  For 3D properties, the iterable is zero-based
            layer indices.  For example, ["lpf.hk",[0,1,2,]] would setup pilot point multiplier
            parameters for layer property file horizontal hydraulic conductivity for model
            layers 1,2, and 3.  For time-varying properties (e.g. recharge), the
            iterable is for zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
            would setup pilot point multiplier parameters for recharge for stress
            period 1,5,11,and 16.
        const_props ([[`str`,[`int`]]]): constant (uniform) multiplier parameters for grid-based properties.
            A nested list of grid-scale model properties to parameterize using
            name, iterable pairs.  For 3D properties, the iterable is zero-based
            layer indices.  For example, ["lpf.hk",[0,1,2,]] would setup constant (uniform) multiplier
            parameters for layer property file horizontal hydraulic conductivity for model
            layers 1,2, and 3.  For time-varying properties (e.g. recharge), the
            iterable is for zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
            would setup constant (uniform) multiplier parameters for recharge for stress
            period 1,5,11,and 16.
        temporal_list_props ([[`str`,[`int`]]]): list-type input stress-period level multiplier parameters.
            A nested list of list-type input elements to parameterize using
            name, iterable pairs.  The iterable is zero-based stress-period indices.
            For example, to setup multipliers for WEL flux and for RIV conductance,
            temporal_list_props = [["wel.flux",[0,1,2]],["riv.cond",None]] would setup
            multiplier parameters for well flux for stress periods 1,2 and 3 and
            would setup one single river conductance multiplier parameter that is applied
            to all stress periods
        spatial_list_props ([[`str`,[`int`]]]):  list-type input for spatial multiplier parameters.
            A nested list of list-type elements to parameterize using
            names (e.g. [["riv.cond",0],["wel.flux",1] to setup up cell-based parameters for
            each list-type element listed.  These multiplier parameters are applied across
            all stress periods.  For this to work, there must be the same number of entries
            for all stress periods.  If more than one list element of the same type is in a single
            cell, only one parameter is used to multiply all lists in the same cell.
        grid_props ([[`str`,[`int`]]]): grid-based (every active model cell) multiplier parameters.
            A nested list of grid-scale model properties to parameterize using
            name, iterable pairs.  For 3D properties, the iterable is zero-based
            layer indices (e.g., ["lpf.hk",[0,1,2,]] would setup a multiplier
            parameter for layer property file horizontal hydraulic conductivity for model
            layers 1,2, and 3 in every active model cell).  For time-varying properties (e.g. recharge), the
            iterable is for zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
            would setup grid-based multiplier parameters in every active model cell
            for recharge for stress period 1,5,11,and 16.
        sfr_pars (`bool`): setup parameters for the stream flow routing modflow package.
            If list is passed it defines the parameters to set up.
        sfr_temporal_pars (`bool`)
            flag to include stress-period level spatially-global multipler parameters in addition to
            the spatially-discrete `sfr_pars`.  Requires `sfr_pars` to be passed.  Default is False
        grid_geostruct (`pyemu.geostats.GeoStruct`): the geostatistical structure to build the prior parameter covariance matrix
            elements for grid-based parameters.  If None, a generic GeoStruct is created
            using an "a" parameter that is 10 times the max cell size.  Default is None
        pp_space (`int`): number of grid cells between pilot points.  If None, use the default
            in pyemu.pp_utils.setup_pilot_points_grid.  Default is None
        zone_props ([[`str`,[`int`]]]): zone-based multiplier parameters.
            A nested list of zone-based model properties to parameterize using
            name, iterable pairs.  For 3D properties, the iterable is zero-based
            layer indices (e.g., ["lpf.hk",[0,1,2,]] would setup a multiplier
            parameter for layer property file horizontal hydraulic conductivity for model
            layers 1,2, and 3 for unique zone values in the ibound array.
            For time-varying properties (e.g. recharge), the iterable is for
            zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
            would setup zone-based multiplier parameters for recharge for stress
            period 1,5,11,and 16.
        pp_geostruct (`pyemu.geostats.GeoStruct`): the geostatistical structure to use for building the prior parameter
            covariance matrix for pilot point parameters.  If None, a generic
            GeoStruct is created using pp_space and grid-spacing information.
            Default is None
        par_bounds_dict (`dict`): a dictionary of model property/boundary condition name, upper-lower bound pairs.
            For example, par_bounds_dict = {"hk":[0.01,100.0],"flux":[0.5,2.0]} would
            set the bounds for horizontal hydraulic conductivity to
            0.001 and 100.0 and set the bounds for flux parameters to 0.5 and
            2.0.  For parameters not found in par_bounds_dict,
            `pyemu.helpers.wildass_guess_par_bounds_dict` is
            used to set somewhat meaningful bounds.  Default is None
        temporal_list_geostruct (`pyemu.geostats.GeoStruct`): the geostastical struture to
            build the prior parameter covariance matrix
            for time-varying list-type multiplier parameters.  This GeoStruct
            express the time correlation so that the 'a' parameter is the length of
            time that boundary condition multiplier parameters are correlated across.
            If None, then a generic GeoStruct is created that uses an 'a' parameter
            of 3 stress periods.  Default is None
        spatial_list_geostruct (`pyemu.geostats.GeoStruct`): the geostastical struture to
            build the prior parameter covariance matrix
            for spatially-varying list-type multiplier parameters.
            If None, a generic GeoStruct is created using an "a" parameter that
            is 10 times the max cell size.  Default is None.
        remove_existing (`bool`): a flag to remove an existing new_model_ws directory.  If False and
            new_model_ws exists, an exception is raised.  If True and new_model_ws
            exists, the directory is destroyed - user beware! Default is False.
        k_zone_dict (`dict`):  a dictionary of zero-based layer index, zone array pairs.
            e.g. {lay: np.2darray}  Used to
            override using ibound zones for zone-based parameterization.  If None,
            use ibound values greater than zero as zones. Alternatively a dictionary of dictionaries
            can be passed to allow different zones to be defined for different parameters.
            e.g. {"upw.hk" {lay: np.2darray}, "extra.rc11" {lay: np.2darray}}
            or {"hk" {lay: np.2darray}, "rc11" {lay: np.2darray}}
        use_pp_zones (`bool`): a flag to use ibound zones (or k_zone_dict, see above) as pilot
             point zones.  If False, ibound values greater than zero are treated as
             a single zone for pilot points.  Default is False
        obssim_smp_pairs ([[`str`,`str`]]: a list of observed-simulated PEST-type SMP file
            pairs to get observations
            from and include in the control file.  Default is []
        external_tpl_in_pairs ([[`str`,`str`]]: a list of existing template file, model input
            file pairs to parse parameters
            from and include in the control file.  Default is []
        external_ins_out_pairs ([[`str`,`str`]]: a list of existing instruction file,
            model output file pairs to parse
            observations from and include in the control file.  Default is []
        extra_pre_cmds ([`str`]): a list of preprocessing commands to add to the forward_run.py script
            commands are executed with os.system() within forward_run.py. Default is None.
        redirect_forward_output (`bool`): flag for whether to redirect forward model output to text files (True) or
            allow model output to be directed to the screen (False).  Default is True
        extra_post_cmds ([`str`]): a list of post-processing commands to add to the forward_run.py script.
            Commands are executed with os.system() within forward_run.py. Default is None.
        tmp_files ([`str`]): a list of temporary files that should be removed at the start of the forward
            run script.  Default is [].
        model_exe_name (`str`): binary name to run modflow.  If None, a default from flopy is used,
            which is dangerous because of the non-standard binary names
            (e.g. MODFLOW-NWT_x64, MODFLOWNWT, mfnwt, etc). Default is None.
        build_prior (`bool`): flag to build prior covariance matrix. Default is True
        sfr_obs (`bool`): flag to include observations of flow and aquifer exchange from
            the sfr ASCII output file
        hfb_pars (`bool`): add HFB parameters.  uses pyemu.gw_utils.write_hfb_template().  the resulting
            HFB pars have parval1 equal to the values in the original file and use the
            spatial_list_geostruct to build geostatistical covariates between parameters
        kl_props ([[`str`,[`int`]]]): karhunen-loeve based multiplier parameters.
            A nested list of KL-based model properties to parameterize using
            name, iterable pairs.  For 3D properties, the iterable is zero-based
            layer indices (e.g., ["lpf.hk",[0,1,2,]] would setup a multiplier
            parameter for layer property file horizontal hydraulic conductivity for model
            layers 1,2, and 3 for unique zone values in the ibound array.
            For time-varying properties (e.g. recharge), the iterable is for
            zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
            would setup zone-based multiplier parameters for recharge for stress
            period 1,5,11,and 16.
        kl_num_eig (`int`): the number of KL-based eigenvector multiplier parameters to use for each
            KL parameter set. default is 100
        kl_geostruct (`pyemu.geostats.Geostruct`): the geostatistical structure
            to build the prior parameter covariance matrix
            elements for KL-based parameters.  If None, a generic GeoStruct is created
            using an "a" parameter that is 10 times the max cell size.  Default is None


    Note:

        Setup up multiplier parameters for an existing MODFLOW model.

        Does all kinds of coolness like building a
        meaningful prior, assigning somewhat meaningful parameter groups and
        bounds, writes a forward_run.py script with all the calls need to
        implement multiplier parameters, run MODFLOW and post-process.

        While this class does work, the newer `PstFrom` class is a more pythonic
        implementation

    """

    def __init__(
        self,
        model,
        new_model_ws,
        org_model_ws=None,
        pp_props=[],
        const_props=[],
        temporal_bc_props=[],
        temporal_list_props=[],
        grid_props=[],
        grid_geostruct=None,
        pp_space=None,
        zone_props=[],
        pp_geostruct=None,
        par_bounds_dict=None,
        sfr_pars=False,
        temporal_sfr_pars=False,
        temporal_list_geostruct=None,
        remove_existing=False,
        k_zone_dict=None,
        mflist_waterbudget=True,
        mfhyd=True,
        hds_kperk=[],
        use_pp_zones=False,
        obssim_smp_pairs=None,
        external_tpl_in_pairs=None,
        external_ins_out_pairs=None,
        extra_pre_cmds=None,
        extra_model_cmds=None,
        extra_post_cmds=None,
        redirect_forward_output=True,
        tmp_files=None,
        model_exe_name=None,
        build_prior=True,
        sfr_obs=False,
        spatial_bc_props=[],
        spatial_list_props=[],
        spatial_list_geostruct=None,
        hfb_pars=False,
        kl_props=None,
        kl_num_eig=100,
        kl_geostruct=None,
    ):
        dep_warn = (
            "\n`PstFromFlopyModel()` method is getting old and may not"
            "be kept in sync with changes to Flopy and MODFLOW.\n"
            "Perhaps consider looking at `pyemu.utils.PstFrom()`,"
            "which is (aiming to be) much more general,"
            "forward model independent, and generally kicks ass.\n"
            "Checkout: https://www.sciencedirect.com/science/article/abs/pii/S1364815221000657?via%3Dihub\n"
            "and https://github.com/pypest/pyemu_pestpp_workflow for more info."
        )
        warnings.warn(dep_warn, DeprecationWarning)

        self.logger = pyemu.logger.Logger("PstFromFlopyModel.log")
        self.log = self.logger.log

        self.logger.echo = True
        self.zn_suffix = "_zn"
        self.gr_suffix = "_gr"
        self.pp_suffix = "_pp"
        self.cn_suffix = "_cn"
        self.kl_suffix = "_kl"
        self.arr_org = "arr_org"
        self.arr_mlt = "arr_mlt"
        self.list_org = "list_org"
        self.list_mlt = "list_mlt"
        self.forward_run_file = "forward_run.py"

        self.remove_existing = remove_existing
        self.external_tpl_in_pairs = external_tpl_in_pairs
        self.external_ins_out_pairs = external_ins_out_pairs

        self._setup_model(model, org_model_ws, new_model_ws)
        self._add_external()

        self.arr_mult_dfs = []
        self.par_bounds_dict = par_bounds_dict
        self.pp_props = pp_props
        self.pp_space = pp_space
        self.pp_geostruct = pp_geostruct
        self.use_pp_zones = use_pp_zones

        self.const_props = const_props

        self.grid_props = grid_props
        self.grid_geostruct = grid_geostruct

        self.zone_props = zone_props

        self.kl_props = kl_props
        self.kl_geostruct = kl_geostruct
        self.kl_num_eig = kl_num_eig

        if len(temporal_bc_props) > 0:
            if len(temporal_list_props) > 0:
                self.logger.lraise(
                    "temporal_bc_props and temporal_list_props. "
                    + "temporal_bc_props is deprecated and replaced by temporal_list_props"
                )
            self.logger.warn(
                "temporal_bc_props is deprecated and replaced by temporal_list_props"
            )
            temporal_list_props = temporal_bc_props
        if len(spatial_bc_props) > 0:
            if len(spatial_list_props) > 0:
                self.logger.lraise(
                    "spatial_bc_props and spatial_list_props. "
                    + "spatial_bc_props is deprecated and replaced by spatial_list_props"
                )
            self.logger.warn(
                "spatial_bc_props is deprecated and replaced by spatial_list_props"
            )
            spatial_list_props = spatial_bc_props

        self.temporal_list_props = temporal_list_props
        self.temporal_list_geostruct = temporal_list_geostruct
        if self.temporal_list_geostruct is None:
            v = pyemu.geostats.ExpVario(
                contribution=1.0, a=180.0
            )  # 180 correlation length
            self.temporal_list_geostruct = pyemu.geostats.GeoStruct(
                variograms=v, name="temporal_list_geostruct"
            )

        self.spatial_list_props = spatial_list_props
        self.spatial_list_geostruct = spatial_list_geostruct
        if self.spatial_list_geostruct is None:
            dist = 10 * float(
                max(self.m.dis.delr.array.max(), self.m.dis.delc.array.max())
            )
            v = pyemu.geostats.ExpVario(contribution=1.0, a=dist)
            self.spatial_list_geostruct = pyemu.geostats.GeoStruct(
                variograms=v, name="spatial_list_geostruct"
            )

        self.obssim_smp_pairs = obssim_smp_pairs
        self.hds_kperk = hds_kperk
        self.sfr_obs = sfr_obs
        self.frun_pre_lines = []
        self.frun_model_lines = []
        self.frun_post_lines = []
        self.tmp_files = []
        self.extra_forward_imports = []
        if tmp_files is not None:
            if not isinstance(tmp_files, list):
                tmp_files = [tmp_files]
            self.tmp_files.extend(tmp_files)

        if k_zone_dict is None:
            self.k_zone_dict = {
                k: self.m.bas6.ibound[k].array for k in np.arange(self.m.nlay)
            }
        else:
            # check if k_zone_dict is a dictionary of dictionaries
            if np.all([isinstance(v, dict) for v in k_zone_dict.values()]):
                # loop over outer keys
                for par_key in k_zone_dict.keys():
                    for k, arr in k_zone_dict[par_key].items():
                        if k not in np.arange(self.m.nlay):
                            self.logger.lraise(
                                "k_zone_dict for par {1}, layer index not in nlay:{0}".format(
                                    k, par_key
                                )
                            )
                        if arr.shape != (self.m.nrow, self.m.ncol):
                            self.logger.lraise(
                                "k_zone_dict arr for k {0} for par{2} has wrong shape:{1}".format(
                                    k, arr.shape, par_key
                                )
                            )
            else:
                for k, arr in k_zone_dict.items():
                    if k not in np.arange(self.m.nlay):
                        self.logger.lraise(
                            "k_zone_dict layer index not in nlay:{0}".format(k)
                        )
                    if arr.shape != (self.m.nrow, self.m.ncol):
                        self.logger.lraise(
                            "k_zone_dict arr for k {0} has wrong shape:{1}".format(
                                k, arr.shape
                            )
                        )
            self.k_zone_dict = k_zone_dict

        # add any extra commands to the forward run lines

        for alist, ilist in zip(
            [self.frun_pre_lines, self.frun_model_lines, self.frun_post_lines],
            [extra_pre_cmds, extra_model_cmds, extra_post_cmds],
        ):
            if ilist is None:
                continue

            if not isinstance(ilist, list):
                ilist = [ilist]
            for cmd in ilist:
                self.logger.statement("forward_run line:{0}".format(cmd))
                alist.append("pyemu.os_utils.run('{0}')\n".format(cmd))

        # add the model call

        if model_exe_name is None:
            model_exe_name = self.m.exe_name
            self.logger.warn(
                "using flopy binary to execute the model:{0}".format(model)
            )
        if redirect_forward_output:
            line = "pyemu.os_utils.run('{0} {1} 1>{1}.stdout 2>{1}.stderr')".format(
                model_exe_name, self.m.namefile
            )
        else:
            line = "pyemu.os_utils.run('{0} {1} ')".format(
                model_exe_name, self.m.namefile
            )
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_model_lines.append(line)

        self.tpl_files, self.in_files = [], []
        self.ins_files, self.out_files = [], []
        self._setup_mult_dirs()

        self.mlt_files = []
        self.org_files = []
        self.m_files = []
        self.mlt_counter = {}
        self.par_dfs = {}
        self.mlt_dfs = []

        self._setup_list_pars()
        self._setup_array_pars()

        if not sfr_pars and temporal_sfr_pars:
            self.logger.lraise("use of `temporal_sfr_pars` requires `sfr_pars`")

        if sfr_pars:
            if isinstance(sfr_pars, str):
                sfr_pars = [sfr_pars]
            if isinstance(sfr_pars, list):
                self._setup_sfr_pars(sfr_pars, include_temporal_pars=temporal_sfr_pars)
            else:
                self._setup_sfr_pars(include_temporal_pars=temporal_sfr_pars)

        if hfb_pars:
            self._setup_hfb_pars()

        self.mflist_waterbudget = mflist_waterbudget
        self.mfhyd = mfhyd
        self._setup_observations()
        self.build_pst()
        if build_prior:
            self.parcov = self.build_prior()
        else:
            self.parcov = None
        self.log("saving intermediate _setup_<> dfs into {0}".format(self.m.model_ws))
        for tag, df in self.par_dfs.items():
            df.to_csv(
                os.path.join(
                    self.m.model_ws,
                    "_setup_par_{0}_{1}.csv".format(
                        tag.replace(" ", "_"), self.pst_name
                    ),
                )
            )
        for tag, df in self.obs_dfs.items():
            df.to_csv(
                os.path.join(
                    self.m.model_ws,
                    "_setup_obs_{0}_{1}.csv".format(
                        tag.replace(" ", "_"), self.pst_name
                    ),
                )
            )
        self.log("saving intermediate _setup_<> dfs into {0}".format(self.m.model_ws))
        warnings.warn(dep_warn, DeprecationWarning)
        self.logger.statement("all done")

    def _setup_sfr_obs(self):
        """setup sfr ASCII observations"""
        if not self.sfr_obs:
            return

        if self.m.sfr is None:
            self.logger.lraise("no sfr package found...")
        org_sfr_out_file = os.path.join(
            self.org_model_ws, "{0}.sfr.out".format(self.m.name)
        )
        if not os.path.exists(org_sfr_out_file):
            self.logger.lraise(
                "setup_sfr_obs() error: could not locate existing sfr out file: {0}".format(
                    org_sfr_out_file
                )
            )
        new_sfr_out_file = os.path.join(
            self.m.model_ws, os.path.split(org_sfr_out_file)[-1]
        )
        shutil.copy2(org_sfr_out_file, new_sfr_out_file)
        seg_group_dict = None
        if isinstance(self.sfr_obs, dict):
            seg_group_dict = self.sfr_obs

        df = pyemu.gw_utils.setup_sfr_obs(
            new_sfr_out_file,
            seg_group_dict=seg_group_dict,
            model=self.m,
            include_path=True,
        )
        if df is not None:
            self.obs_dfs["sfr"] = df
        self.frun_post_lines.append("pyemu.gw_utils.apply_sfr_obs()")

    def _setup_sfr_pars(self, par_cols=None, include_temporal_pars=None):
        """setup multiplier parameters for sfr segment data
        Adding support for reachinput (and isfropt = 1)"""
        assert self.m.sfr is not None, "can't find sfr package..."
        if isinstance(par_cols, str):
            par_cols = [par_cols]
        reach_pars = False  # default to False
        seg_pars = True
        par_dfs = {}
        df = pyemu.gw_utils.setup_sfr_seg_parameters(
            self.m, par_cols=par_cols, include_temporal_pars=include_temporal_pars
        )  # now just pass model
        # self.par_dfs["sfr"] = df
        if df.empty:
            warnings.warn("No sfr segment parameters have been set up", PyemuWarning)
            par_dfs["sfr"] = []
            seg_pars = False
        else:
            par_dfs["sfr"] = [df]  # may need df for both segs and reaches
            self.tpl_files.append("sfr_seg_pars.dat.tpl")
            self.in_files.append("sfr_seg_pars.dat")
            if include_temporal_pars:
                self.tpl_files.append("sfr_seg_temporal_pars.dat.tpl")
                self.in_files.append("sfr_seg_temporal_pars.dat")
        if self.m.sfr.reachinput:
            # if include_temporal_pars:
            #     raise NotImplementedError("temporal pars is not set up for reach data style")
            df = pyemu.gw_utils.setup_sfr_reach_parameters(self.m, par_cols=par_cols)
            if df.empty:
                warnings.warn("No sfr reach parameters have been set up", PyemuWarning)
            else:
                self.tpl_files.append("sfr_reach_pars.dat.tpl")
                self.in_files.append("sfr_reach_pars.dat")
                reach_pars = True
                par_dfs["sfr"].append(df)
        if len(par_dfs["sfr"]) > 0:
            self.par_dfs["sfr"] = pd.concat(par_dfs["sfr"])
            self.frun_pre_lines.append(
                "pyemu.gw_utils.apply_sfr_parameters(seg_pars={0}, reach_pars={1})".format(
                    seg_pars, reach_pars
                )
            )
        else:
            warnings.warn("No sfr parameters have been set up!", PyemuWarning)

    def _setup_hfb_pars(self):
        """setup non-mult parameters for hfb (yuck!)"""
        if self.m.hfb6 is None:
            self.logger.lraise("couldn't find hfb pak")
        tpl_file, df = pyemu.gw_utils.write_hfb_template(self.m)

        self.in_files.append(os.path.split(tpl_file.replace(".tpl", ""))[-1])
        self.tpl_files.append(os.path.split(tpl_file)[-1])
        self.par_dfs["hfb"] = df

    def _setup_mult_dirs(self):
        """setup the directories to use for multiplier parameterization.  Directories
        are make within the PstFromFlopyModel.m.model_ws directory

        """
        # setup dirs to hold the original and multiplier model input quantities
        set_dirs = []
        #        if len(self.pp_props) > 0 or len(self.zone_props) > 0 or \
        #                        len(self.grid_props) > 0:
        if (
            self.pp_props is not None
            or self.zone_props is not None
            or self.grid_props is not None
            or self.const_props is not None
            or self.kl_props is not None
        ):
            set_dirs.append(self.arr_org)
            set_dirs.append(self.arr_mlt)
        #       if len(self.bc_props) > 0:
        if len(self.temporal_list_props) > 0 or len(self.spatial_list_props) > 0:
            set_dirs.append(self.list_org)
        if len(self.spatial_list_props):
            set_dirs.append(self.list_mlt)

        for d in set_dirs:
            d = os.path.join(self.m.model_ws, d)
            self.log("setting up '{0}' dir".format(d))
            if os.path.exists(d):
                if self.remove_existing:
                    pyemu.os_utils._try_remove_existing(d)
                else:
                    raise Exception("dir '{0}' already exists".format(d))
            os.mkdir(d)
            self.log("setting up '{0}' dir".format(d))

    def _setup_model(self, model, org_model_ws, new_model_ws):
        """setup the flopy.mbase instance for use with multipler parameters.
        Changes model_ws, sets external_path and writes new MODFLOW input
        files

        """
        split_new_mws = [i for i in os.path.split(new_model_ws) if len(i) > 0]
        if len(split_new_mws) != 1:
            self.logger.lraise(
                "new_model_ws can only be 1 folder-level deep:{0}".format(
                    str(split_new_mws)
                )
            )

        if isinstance(model, str):
            self.log("loading flopy model")
            try:
                import flopy
            except:
                raise Exception("from_flopy_model() requires flopy")
            # prepare the flopy model
            self.org_model_ws = org_model_ws
            self.new_model_ws = new_model_ws
            self.m = flopy.modflow.Modflow.load(
                model, model_ws=org_model_ws, check=False, verbose=True, forgive=False
            )
            self.log("loading flopy model")
        else:
            self.m = model
            self.org_model_ws = str(self.m.model_ws)
            self.new_model_ws = new_model_ws
        try:
            self.sr = self.m.sr
        except AttributeError:  # if sr doesnt exist anymore!
            # assume that we have switched to model grid
            self.sr = SpatialReference.from_namfile(
                os.path.join(self.org_model_ws, self.m.namefile),
                delr=self.m.modelgrid.delr,
                delc=self.m.modelgrid.delc,
            )
        self.log("updating model attributes")
        self.m.array_free_format = True
        self.m.free_format_input = True
        self.m.external_path = "."
        self.log("updating model attributes")
        if os.path.exists(new_model_ws):
            if not self.remove_existing:
                self.logger.lraise("'new_model_ws' already exists")
            else:
                self.logger.warn("removing existing 'new_model_ws")
                pyemu.os_utils._try_remove_existing(new_model_ws)
        self.m.change_model_ws(new_model_ws, reset_external=True)
        self.m.exe_name = self.m.exe_name.replace(".exe", "")
        self.m.exe = self.m.version
        self.log("writing new modflow input files")
        self.m.write_input()
        self.log("writing new modflow input files")

    def _get_count(self, name):
        """get the latest counter for a certain parameter type."""
        if name not in self.mlt_counter:
            self.mlt_counter[name] = 1
            c = 0
        else:
            c = self.mlt_counter[name]
            self.mlt_counter[name] += 1
            # print(name,c)
        return c

    def _prep_mlt_arrays(self):
        """prepare multipler arrays.  Copies existing model input arrays and
        writes generic (ones) multiplier arrays

        """
        par_props = [
            self.pp_props,
            self.grid_props,
            self.zone_props,
            self.const_props,
            self.kl_props,
        ]
        par_suffixs = [
            self.pp_suffix,
            self.gr_suffix,
            self.zn_suffix,
            self.cn_suffix,
            self.kl_suffix,
        ]

        # Need to remove props and suffixes for which no info was provided (e.g. still None)
        del_idx = []
        for i, cp in enumerate(par_props):
            if cp is None:
                del_idx.append(i)
        for i in del_idx[::-1]:
            del par_props[i]
            del par_suffixs[i]

        mlt_dfs = []
        for par_prop, suffix in zip(par_props, par_suffixs):
            if len(par_prop) == 2:
                if not isinstance(par_prop[0], list):
                    par_prop = [par_prop]
            if len(par_prop) == 0:
                continue
            for pakattr, k_org in par_prop:
                attr_name = pakattr.split(".")[1]
                pak, attr = self._parse_pakattr(pakattr)
                ks = np.arange(self.m.nlay)
                if isinstance(attr, flopy.utils.Transient2d):
                    ks = np.arange(self.m.nper)
                try:
                    k_parse = self._parse_k(k_org, ks)
                except Exception as e:
                    self.logger.lraise("error parsing k {0}:{1}".format(k_org, str(e)))
                org, mlt, mod, layer = [], [], [], []
                c = self._get_count(attr_name)
                mlt_prefix = "{0}{1}".format(attr_name, c)
                mlt_name = os.path.join(
                    self.arr_mlt, "{0}.dat{1}".format(mlt_prefix, suffix)
                )
                for k in k_parse:
                    # horrible kludge to avoid passing int64 to flopy
                    # this gift may give again...
                    if type(k) is np.int64:
                        k = int(k)
                    if isinstance(attr, flopy.utils.Util2d):
                        fname = self._write_u2d(attr)

                        layer.append(k)
                    elif isinstance(attr, flopy.utils.Util3d):
                        fname = self._write_u2d(attr[k])
                        layer.append(k)
                    elif isinstance(attr, flopy.utils.Transient2d):
                        fname = self._write_u2d(attr.transient_2ds[k])
                        layer.append(0)  # big assumption here
                    mod.append(os.path.join(self.m.external_path, fname))
                    mlt.append(mlt_name)
                    org.append(os.path.join(self.arr_org, fname))
                df = pd.DataFrame(
                    {
                        "org_file": org,
                        "mlt_file": mlt,
                        "model_file": mod,
                        "layer": layer,
                    }
                )
                df.loc[:, "suffix"] = suffix
                df.loc[:, "prefix"] = mlt_prefix
                df.loc[:, "attr_name"] = attr_name
                mlt_dfs.append(df)
        if len(mlt_dfs) > 0:
            mlt_df = pd.concat(mlt_dfs, ignore_index=True)
            return mlt_df

    def _write_u2d(self, u2d):
        """write a flopy.utils.Util2D instance to an ASCII text file using the
        Util2D filename

        """
        filename = os.path.split(u2d.filename)[-1]
        np.savetxt(
            os.path.join(self.m.model_ws, self.arr_org, filename),
            u2d.array,
            fmt="%15.6E",
        )
        return filename

    def _write_const_tpl(self, name, tpl_file, zn_array):
        """write a template file a for a constant (uniform) multiplier parameter"""
        parnme = []
        with open(os.path.join(self.m.model_ws, tpl_file), "w") as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if zn_array[i, j] < 1:
                        pname = " 1.0  "
                    else:
                        pname = "{0}{1}".format(name, self.cn_suffix)
                        if len(pname) > 12:
                            self.logger.warn(
                                "zone pname too long for pest:{0}".format(pname)
                            )
                        parnme.append(pname)
                        pname = " ~   {0}    ~".format(pname)
                    f.write(pname)
                f.write("\n")
        df = pd.DataFrame({"parnme": parnme}, index=parnme)
        # df.loc[:,"pargp"] = "{0}{1}".format(self.cn_suffixname)
        df.loc[:, "pargp"] = "{0}_{1}".format(self.cn_suffix.replace("_", ""), name)
        df.loc[:, "tpl"] = tpl_file
        return df

    def _write_grid_tpl(self, name, tpl_file, zn_array):
        """write a template file a for grid-based multiplier parameters"""
        parnme, x, y = [], [], []
        with open(os.path.join(self.m.model_ws, tpl_file), "w") as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if zn_array[i, j] < 1:
                        pname = " 1.0 "
                    else:
                        pname = "{0}{1:03d}{2:03d}".format(name, i, j)
                        if len(pname) > 12:
                            self.logger.warn(
                                "grid pname too long for pest:{0}".format(pname)
                            )
                        parnme.append(pname)
                        pname = " ~     {0}   ~ ".format(pname)
                        x.append(self.sr.xcentergrid[i, j])
                        y.append(self.sr.ycentergrid[i, j])
                    f.write(pname)
                f.write("\n")
        df = pd.DataFrame({"parnme": parnme, "x": x, "y": y}, index=parnme)
        df.loc[:, "pargp"] = "{0}{1}".format(self.gr_suffix.replace("_", ""), name)
        df.loc[:, "tpl"] = tpl_file
        return df

    def _grid_prep(self):
        """prepare grid-based parameterizations"""
        if len(self.grid_props) == 0:
            return

        if self.grid_geostruct is None:
            self.logger.warn(
                "grid_geostruct is None,"
                " using ExpVario with contribution=1 and a=(max(delc,delr)*10"
            )
            dist = 10 * float(
                max(self.m.dis.delr.array.max(), self.m.dis.delc.array.max())
            )
            v = pyemu.geostats.ExpVario(contribution=1.0, a=dist)
            self.grid_geostruct = pyemu.geostats.GeoStruct(
                variograms=v, name="grid_geostruct", transform="log"
            )

    def _pp_prep(self, mlt_df):
        """prepare pilot point based parameterization"""
        if len(self.pp_props) == 0:
            return
        if self.pp_space is None:
            self.logger.warn("pp_space is None, using 10...\n")
            self.pp_space = 10
        if self.pp_geostruct is None:
            self.logger.warn(
                "pp_geostruct is None,"
                " using ExpVario with contribution=1 and a=(pp_space*max(delr,delc))"
            )
            pp_dist = self.pp_space * float(
                max(self.m.dis.delr.array.max(), self.m.dis.delc.array.max())
            )
            v = pyemu.geostats.ExpVario(contribution=1.0, a=pp_dist)
            self.pp_geostruct = pyemu.geostats.GeoStruct(
                variograms=v, name="pp_geostruct", transform="log"
            )

        pp_df = mlt_df.loc[mlt_df.suffix == self.pp_suffix, :]
        layers = pp_df.layer.unique()
        layers.sort()
        pp_dict = {
            l: list(pp_df.loc[pp_df.layer == l, "prefix"].unique()) for l in layers
        }
        # big assumption here - if prefix is listed more than once, use the lowest layer index
        pp_dict_sort = {}
        for i, l in enumerate(layers):
            p = set(pp_dict[l])
            pl = list(p)
            pl.sort()
            pp_dict_sort[l] = pl
            for ll in layers[i + 1 :]:
                pp = set(pp_dict[ll])
                d = list(pp - p)
                d.sort()
                pp_dict_sort[ll] = d
        pp_dict = pp_dict_sort

        pp_array_file = {p: m for p, m in zip(pp_df.prefix, pp_df.mlt_file)}
        self.logger.statement("pp_dict: {0}".format(str(pp_dict)))

        self.log("calling setup_pilot_point_grid()")
        if self.use_pp_zones:
            # check if k_zone_dict is a dictionary of dictionaries
            if np.all([isinstance(v, dict) for v in self.k_zone_dict.values()]):
                ib = {
                    p.split(".")[-1]: k_dict for p, k_dict in self.k_zone_dict.items()
                }
                for attr in pp_df.attr_name.unique():
                    if attr not in [p.split(".")[-1] for p in ib.keys()]:
                        if "general_zn" not in ib.keys():
                            warnings.warn(
                                "Dictionary of dictionaries passed as zones, {0} not in keys: {1}. "
                                "Will use ibound for zones".format(attr, ib.keys()),
                                PyemuWarning,
                            )
                        else:
                            self.logger.statement(
                                "Dictionary of dictionaries passed as pp zones, "
                                "using 'general_zn' for {0}".format(attr)
                            )
                    if "general_zn" not in ib.keys():
                        ib["general_zn"] = {
                            k: self.m.bas6.ibound[k].array for k in range(self.m.nlay)
                        }
            else:
                ib = {"general_zn": self.k_zone_dict}
        else:
            ib = {}
            for k in range(self.m.nlay):
                a = self.m.bas6.ibound[k].array.copy()
                a[a > 0] = 1
                ib[k] = a
            for k, i in ib.items():
                if np.any(i < 0):
                    u, c = np.unique(i[i > 0], return_counts=True)
                    counts = dict(zip(u, c))
                    mx = -1.0e10
                    imx = None
                    for u, c in counts.items():
                        if c > mx:
                            mx = c
                            imx = u
                    self.logger.warn(
                        "resetting negative ibound values for PP zone"
                        + "array in layer {0} : {1}".format(k + 1, u)
                    )
                    i[i < 0] = u
                ib[k] = i
            ib = {"general_zn": ib}
        pp_df = pyemu.pp_utils.setup_pilotpoints_grid(
            self.m,
            ibound=ib,
            use_ibound_zones=self.use_pp_zones,
            prefix_dict=pp_dict,
            every_n_cell=self.pp_space,
            pp_dir=self.m.model_ws,
            tpl_dir=self.m.model_ws,
            shapename=os.path.join(self.m.model_ws, "pp.shp"),
        )
        self.logger.statement(
            "{0} pilot point parameters created".format(pp_df.shape[0])
        )
        self.logger.statement(
            "pilot point 'pargp':{0}".format(",".join(pp_df.pargp.unique()))
        )
        self.log("calling setup_pilot_point_grid()")

        # calc factors for each layer
        pargp = pp_df.pargp.unique()
        pp_dfs_k = {}
        fac_files = {}
        pp_processed = set()
        pp_df.loc[:, "fac_file"] = np.NaN
        for pg in pargp:
            ks = pp_df.loc[pp_df.pargp == pg, "k"].unique()
            if len(ks) == 0:
                self.logger.lraise(
                    "something is wrong in fac calcs for par group {0}".format(pg)
                )
            if len(ks) == 1:
                if np.all(
                    [isinstance(v, dict) for v in ib.values()]
                ):  # check is dict of dicts
                    if np.any([pg.startswith(p) for p in ib.keys()]):
                        p = next(p for p in ib.keys() if pg.startswith(p))
                        # get dict relating to parameter prefix
                        ib_k = ib[p][ks[0]]
                    else:
                        p = "general_zn"
                        ib_k = ib[p][ks[0]]
                else:
                    ib_k = ib[ks[0]]
            if len(ks) != 1:  # TODO
                # self.logger.lraise("something is wrong in fac calcs for par group {0}".format(pg))
                self.logger.warn(
                    "multiple k values for {0},forming composite zone array...".format(
                        pg
                    )
                )
                ib_k = np.zeros((self.m.nrow, self.m.ncol))
                for k in ks:
                    t = ib["general_zn"][k].copy()
                    t[t < 1] = 0
                    ib_k[t > 0] = t[t > 0]
            k = int(ks[0])
            kattr_id = "{}_{}".format(k, p)
            kp_id = "{}_{}".format(k, pg)
            if kp_id not in pp_dfs_k.keys():
                self.log("calculating factors for p={0}, k={1}".format(pg, k))
                fac_file = os.path.join(self.m.model_ws, "pp_k{0}.fac".format(kattr_id))
                var_file = fac_file.replace(".fac", ".var.dat")
                pp_df_k = pp_df.loc[pp_df.pargp == pg]
                if kattr_id not in pp_processed:
                    self.logger.statement(
                        "saving krige variance file:{0}".format(var_file)
                    )
                    self.logger.statement(
                        "saving krige factors file:{0}".format(fac_file)
                    )
                    ok_pp = pyemu.geostats.OrdinaryKrige(self.pp_geostruct, pp_df_k)
                    ok_pp.calc_factors_grid(
                        self.sr,
                        var_filename=var_file,
                        zone_array=ib_k,
                        num_threads=10,
                    )
                    ok_pp.to_grid_factors_file(fac_file)
                    pp_processed.add(kattr_id)
                fac_files[kp_id] = fac_file
                self.log("calculating factors for p={0}, k={1}".format(pg, k))
                pp_dfs_k[kp_id] = pp_df_k

        for kp_id, fac_file in fac_files.items():
            k = int(kp_id.split("_")[0])
            pp_prefix = kp_id.split("_", 1)[-1]
            # pp_files = pp_df.pp_filename.unique()
            fac_file = os.path.split(fac_file)[-1]
            # pp_prefixes = pp_dict[k]
            # for pp_prefix in pp_prefixes:
            self.log("processing pp_prefix:{0}".format(pp_prefix))
            if pp_prefix not in pp_array_file.keys():
                self.logger.lraise(
                    "{0} not in self.pp_array_file.keys()".format(
                        pp_prefix, ",".join(pp_array_file.keys())
                    )
                )

            out_file = os.path.join(
                self.arr_mlt, os.path.split(pp_array_file[pp_prefix])[-1]
            )

            pp_files = pp_df.loc[
                pp_df.pp_filename.apply(
                    lambda x: os.path.split(x)[-1].split(".")[0]
                    == "{0}pp".format(pp_prefix)
                ),
                "pp_filename",
            ]
            if pp_files.unique().shape[0] != 1:
                self.logger.lraise(
                    "wrong number of pp_files found:{0}".format(",".join(pp_files))
                )
            pp_file = os.path.split(pp_files.iloc[0])[-1]
            pp_df.loc[pp_df.pargp == pp_prefix, "fac_file"] = fac_file
            pp_df.loc[pp_df.pargp == pp_prefix, "pp_file"] = pp_file
            pp_df.loc[pp_df.pargp == pp_prefix, "out_file"] = out_file

        pp_df.loc[:, "pargp"] = pp_df.pargp.apply(lambda x: "pp_{0}".format(x))
        out_files = mlt_df.loc[
            mlt_df.mlt_file.apply(lambda x: x.endswith(self.pp_suffix)), "mlt_file"
        ]
        # mlt_df.loc[:,"fac_file"] = np.NaN
        # mlt_df.loc[:,"pp_file"] = np.NaN
        for out_file in out_files:
            pp_df_pf = pp_df.loc[pp_df.out_file == out_file, :]
            fac_files = pp_df_pf.fac_file
            if fac_files.unique().shape[0] != 1:
                self.logger.lraise(
                    "wrong number of fac files:{0}".format(str(fac_files.unique()))
                )
            fac_file = fac_files.iloc[0]
            pp_files = pp_df_pf.pp_file
            if pp_files.unique().shape[0] != 1:
                self.logger.lraise(
                    "wrong number of pp files:{0}".format(str(pp_files.unique()))
                )
            pp_file = pp_files.iloc[0]
            mlt_df.loc[mlt_df.mlt_file == out_file, "fac_file"] = fac_file
            mlt_df.loc[mlt_df.mlt_file == out_file, "pp_file"] = pp_file
            mlt_df.loc[mlt_df.mlt_file == out_file, "pp_fill_value"] = 1.0
            mlt_df.loc[mlt_df.mlt_file == out_file, "pp_lower_limit"] = 1.0e-10
            mlt_df.loc[mlt_df.mlt_file == out_file, "pp_upper_limit"] = 1.0e10

        self.par_dfs[self.pp_suffix] = pp_df

        mlt_df.loc[mlt_df.suffix == self.pp_suffix, "tpl_file"] = np.NaN

    def _kl_prep(self, mlt_df):
        """prepare KL based parameterizations"""
        if len(self.kl_props) == 0:
            return

        if self.kl_geostruct is None:
            self.logger.warn(
                "kl_geostruct is None,"
                " using ExpVario with contribution=1 and a=(10.0*max(delr,delc))"
            )
            kl_dist = 10.0 * float(
                max(self.m.dis.delr.array.max(), self.m.dis.delc.array.max())
            )
            v = pyemu.geostats.ExpVario(contribution=1.0, a=kl_dist)
            self.kl_geostruct = pyemu.geostats.GeoStruct(
                variograms=v, name="kl_geostruct", transform="log"
            )

        kl_df = mlt_df.loc[mlt_df.suffix == self.kl_suffix, :]
        layers = kl_df.layer.unique()
        # kl_dict = {l:list(kl_df.loc[kl_df.layer==l,"prefix"].unique()) for l in layers}
        # big assumption here - if prefix is listed more than once, use the lowest layer index
        # for i,l in enumerate(layers):
        #    p = set(kl_dict[l])
        #    for ll in layers[i+1:]:
        #        pp = set(kl_dict[ll])
        #        d = pp - p
        #        kl_dict[ll] = list(d)
        kl_prefix = list(kl_df.loc[:, "prefix"])

        kl_array_file = {p: m for p, m in zip(kl_df.prefix, kl_df.mlt_file)}
        self.logger.statement("kl_prefix: {0}".format(str(kl_prefix)))

        fac_file = os.path.join(self.m.model_ws, "kl.fac")

        self.log("calling kl_setup() with factors file {0}".format(fac_file))

        kl_df = kl_setup(
            self.kl_num_eig,
            self.sr,
            self.kl_geostruct,
            kl_prefix,
            factors_file=fac_file,
            basis_file=fac_file + ".basis.jcb",
            tpl_dir=self.m.model_ws,
        )
        self.logger.statement("{0} kl parameters created".format(kl_df.shape[0]))
        self.logger.statement("kl 'pargp':{0}".format(",".join(kl_df.pargp.unique())))

        self.log("calling kl_setup() with factors file {0}".format(fac_file))
        kl_mlt_df = mlt_df.loc[mlt_df.suffix == self.kl_suffix]
        for prefix in kl_df.prefix.unique():
            prefix_df = kl_df.loc[kl_df.prefix == prefix, :]
            in_file = os.path.split(prefix_df.loc[:, "in_file"].iloc[0])[-1]
            assert prefix in mlt_df.prefix.values, "{0}:{1}".format(
                prefix, mlt_df.prefix
            )
            mlt_df.loc[mlt_df.prefix == prefix, "pp_file"] = in_file
            mlt_df.loc[mlt_df.prefix == prefix, "fac_file"] = os.path.split(fac_file)[
                -1
            ]
            mlt_df.loc[mlt_df.prefix == prefix, "pp_fill_value"] = 1.0
            mlt_df.loc[mlt_df.prefix == prefix, "pp_lower_limit"] = 1.0e-10
            mlt_df.loc[mlt_df.prefix == prefix, "pp_upper_limit"] = 1.0e10

        print(kl_mlt_df)
        mlt_df.loc[mlt_df.suffix == self.kl_suffix, "tpl_file"] = np.NaN
        self.par_dfs[self.kl_suffix] = kl_df
        # calc factors for each layer

    def _setup_array_pars(self):
        """main entry point for setting up array multipler parameters"""
        mlt_df = self._prep_mlt_arrays()
        if mlt_df is None:
            return
        mlt_df.loc[:, "tpl_file"] = mlt_df.mlt_file.apply(
            lambda x: os.path.split(x)[-1] + ".tpl"
        )
        # mlt_df.loc[mlt_df.tpl_file.apply(lambda x:pd.notnull(x.pp_file)),"tpl_file"] = np.NaN
        mlt_files = mlt_df.mlt_file.unique()
        # for suffix,tpl_file,layer,name in zip(self.mlt_df.suffix,
        #                                 self.mlt_df.tpl,self.mlt_df.layer,
        #                                     self.mlt_df.prefix):
        par_dfs = {}
        for mlt_file in mlt_files:
            suffixes = mlt_df.loc[mlt_df.mlt_file == mlt_file, "suffix"]
            if suffixes.unique().shape[0] != 1:
                self.logger.lraise("wrong number of suffixes for {0}".format(mlt_file))
            suffix = suffixes.iloc[0]

            tpl_files = mlt_df.loc[mlt_df.mlt_file == mlt_file, "tpl_file"]
            if tpl_files.unique().shape[0] != 1:
                self.logger.lraise("wrong number of tpl_files for {0}".format(mlt_file))
            tpl_file = tpl_files.iloc[0]
            layers = mlt_df.loc[mlt_df.mlt_file == mlt_file, "layer"]
            # if layers.unique().shape[0] != 1:
            #    self.logger.lraise("wrong number of layers for {0}"\
            #                       .format(mlt_file))
            layer = layers.iloc[0]
            names = mlt_df.loc[mlt_df.mlt_file == mlt_file, "prefix"]
            if names.unique().shape[0] != 1:
                self.logger.lraise("wrong number of names for {0}".format(mlt_file))
            name = names.iloc[0]
            attr_names = mlt_df.loc[mlt_df.mlt_file == mlt_file, "attr_name"]
            if attr_names.unique().shape[0] != 1:
                self.logger.lraise(
                    "wrong number of attr_names for {0}".format(mlt_file)
                )
            attr_name = attr_names.iloc[0]

            # ib = self.k_zone_dict[layer]
            df = None
            if suffix == self.cn_suffix:
                self.log("writing const tpl:{0}".format(tpl_file))
                # df = self.write_const_tpl(name,tpl_file,self.m.bas6.ibound[layer].array)
                try:
                    df = write_const_tpl(
                        name,
                        os.path.join(self.m.model_ws, tpl_file),
                        self.cn_suffix,
                        self.m.bas6.ibound[layer].array,
                        (self.m.nrow, self.m.ncol),
                        self.sr,
                    )
                except Exception as e:
                    self.logger.lraise(
                        "error writing const template: {0}".format(str(e))
                    )
                self.log("writing const tpl:{0}".format(tpl_file))

            elif suffix == self.gr_suffix:
                self.log("writing grid tpl:{0}".format(tpl_file))
                # df = self.write_grid_tpl(name,tpl_file,self.m.bas6.ibound[layer].array)
                try:
                    df = write_grid_tpl(
                        name,
                        os.path.join(self.m.model_ws, tpl_file),
                        self.gr_suffix,
                        self.m.bas6.ibound[layer].array,
                        (self.m.nrow, self.m.ncol),
                        self.sr,
                    )
                except Exception as e:
                    self.logger.lraise(
                        "error writing grid template: {0}".format(str(e))
                    )
                self.log("writing grid tpl:{0}".format(tpl_file))

            elif suffix == self.zn_suffix:
                self.log("writing zone tpl:{0}".format(tpl_file))
                if np.all(
                    [isinstance(v, dict) for v in self.k_zone_dict.values()]
                ):  # check is dict of dicts
                    if attr_name in [p.split(".")[-1] for p in self.k_zone_dict.keys()]:
                        k_zone_dict = next(
                            k_dict
                            for p, k_dict in self.k_zone_dict.items()
                            if p.split(".")[-1] == attr_name
                        )  # get dict relating to parameter prefix
                    else:
                        assert (
                            "general_zn" in self.k_zone_dict.keys()
                        ), "Neither {0} nor 'general_zn' are in k_zone_dict keys: {1}".format(
                            attr_name, self.k_zone_dict.keys()
                        )
                        k_zone_dict = self.k_zone_dict["general_zn"]
                else:
                    k_zone_dict = self.k_zone_dict
                # df = self.write_zone_tpl(self.m, name, tpl_file, self.k_zone_dict[layer], self.zn_suffix, self.logger)
                try:
                    df = write_zone_tpl(
                        name,
                        os.path.join(self.m.model_ws, tpl_file),
                        self.zn_suffix,
                        k_zone_dict[layer],
                        (self.m.nrow, self.m.ncol),
                        self.sr,
                    )
                except Exception as e:
                    self.logger.lraise(
                        "error writing zone template: {0}".format(str(e))
                    )
                self.log("writing zone tpl:{0}".format(tpl_file))

            if df is None:
                continue
            if suffix not in par_dfs:
                par_dfs[suffix] = [df]
            else:
                par_dfs[suffix].append(df)
        for suf, dfs in par_dfs.items():
            self.par_dfs[suf] = pd.concat(dfs)

        if self.pp_suffix in mlt_df.suffix.values:
            self.log("setting up pilot point process")
            self._pp_prep(mlt_df)
            self.log("setting up pilot point process")

        if self.gr_suffix in mlt_df.suffix.values:
            self.log("setting up grid process")
            self._grid_prep()
            self.log("setting up grid process")

        if self.kl_suffix in mlt_df.suffix.values:
            self.log("setting up kl process")
            self._kl_prep(mlt_df)
            self.log("setting up kl process")

        mlt_df.to_csv(os.path.join(self.m.model_ws, "arr_pars.csv"))
        ones = np.ones((self.m.nrow, self.m.ncol))
        for mlt_file in mlt_df.mlt_file.unique():
            self.log("save test mlt array {0}".format(mlt_file))
            np.savetxt(os.path.join(self.m.model_ws, mlt_file), ones, fmt="%15.6E")
            self.log("save test mlt array {0}".format(mlt_file))
            tpl_files = mlt_df.loc[mlt_df.mlt_file == mlt_file, "tpl_file"]
            if tpl_files.unique().shape[0] != 1:
                self.logger.lraise("wrong number of tpl_files for {0}".format(mlt_file))
            tpl_file = tpl_files.iloc[0]
            if pd.notnull(tpl_file):
                self.tpl_files.append(tpl_file)
                self.in_files.append(mlt_file)

        # for tpl_file,mlt_file in zip(mlt_df.tpl_file,mlt_df.mlt_file):
        #     if pd.isnull(tpl_file):
        #         continue
        #     self.tpl_files.append(tpl_file)
        #     self.in_files.append(mlt_file)

        os.chdir(self.m.model_ws)
        try:
            apply_array_pars()
        except Exception as e:
            os.chdir("..")
            self.logger.lraise(
                "error test running apply_array_pars():{0}".format(str(e))
            )
        os.chdir("..")
        line = "pyemu.helpers.apply_array_pars()\n"
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

    def _setup_observations(self):
        """main entry point for setting up observations"""
        obs_methods = [
            self._setup_water_budget_obs,
            self._setup_hyd,
            self._setup_smp,
            self._setup_hob,
            self._setup_hds,
            self._setup_sfr_obs,
        ]
        obs_types = [
            "mflist water budget obs",
            "hyd file",
            "external obs-sim smp files",
            "hob",
            "hds",
            "sfr",
        ]
        self.obs_dfs = {}
        for obs_method, obs_type in zip(obs_methods, obs_types):
            self.log("processing obs type {0}".format(obs_type))
            obs_method()
            self.log("processing obs type {0}".format(obs_type))

    def draw(self, num_reals=100, sigma_range=6, use_specsim=False, scale_offset=True):

        """draw from the geostatistically-implied parameter covariance matrix

        Args:
            num_reals (`int`): number of realizations to generate. Default is 100
            sigma_range (`float`): number of standard deviations represented by
                the parameter bounds.  Default is 6.
            use_specsim (`bool`): flag to use spectral simulation for grid-based
                parameters.  Requires a regular grid but is wicked fast.  Default is False
            scale_offset (`bool`, optional): flag to apply scale and offset to parameter
                bounds when calculating variances - this is passed through to
                `pyemu.Cov.from_parameter_data`.  Default is True.

        Note:
            operates on parameters by groups to avoid having to construct a very large
            covariance matrix for problems with more the 30K parameters.

            uses `helpers.geostatitical_draw()`

        Returns:
            `pyemu.ParameterEnsemble`: The realized parameter ensemble

        """

        self.log("drawing realizations")
        struct_dict = {}
        gr_par_pe = None
        if self.pp_suffix in self.par_dfs.keys():
            pp_df = self.par_dfs[self.pp_suffix]
            pp_dfs = []
            for pargp in pp_df.pargp.unique():
                gp_df = pp_df.loc[pp_df.pargp == pargp, :]
                p_df = gp_df.drop_duplicates(subset="parnme")
                pp_dfs.append(p_df)
            # pp_dfs = [pp_df.loc[pp_df.pargp==pargp,:].copy() for pargp in pp_df.pargp.unique()]
            struct_dict[self.pp_geostruct] = pp_dfs
        if self.gr_suffix in self.par_dfs.keys():
            gr_df = self.par_dfs[self.gr_suffix]

            if not use_specsim:
                gr_dfs = []
                for pargp in gr_df.pargp.unique():
                    gp_df = gr_df.loc[gr_df.pargp == pargp, :]
                    p_df = gp_df.drop_duplicates(subset="parnme")
                    gr_dfs.append(p_df)
                # gr_dfs = [gr_df.loc[gr_df.pargp==pargp,:].copy() for pargp in gr_df.pargp.unique()]
                struct_dict[self.grid_geostruct] = gr_dfs
            else:
                if not pyemu.geostats.SpecSim2d.grid_is_regular(
                    self.m.dis.delr.array, self.m.dis.delc.array
                ):
                    self.logger.lraise(
                        "draw() error: can't use spectral simulation with irregular grid"
                    )
                gr_df.loc[:, "i"] = gr_df.parnme.apply(lambda x: int(x[-6:-3]))
                gr_df.loc[:, "j"] = gr_df.parnme.apply(lambda x: int(x[-3:]))
                if gr_df.i.max() > self.m.nrow - 1 or gr_df.i.min() < 0:
                    self.logger.lraise(
                        "draw(): error parsing grid par names for 'i' index"
                    )
                if gr_df.j.max() > self.m.ncol - 1 or gr_df.j.min() < 0:
                    self.logger.lraise(
                        "draw(): error parsing grid par names for 'j' index"
                    )
                self.log("spectral simulation for grid-scale pars")
                ss = pyemu.geostats.SpecSim2d(
                    delx=self.m.dis.delr.array,
                    dely=self.m.dis.delc.array,
                    geostruct=self.grid_geostruct,
                )
                gr_par_pe = ss.grid_par_ensemble_helper(
                    pst=self.pst,
                    gr_df=gr_df,
                    num_reals=num_reals,
                    sigma_range=sigma_range,
                    logger=self.logger,
                )
                self.log("spectral simulation for grid-scale pars")
        if "temporal_list" in self.par_dfs.keys():
            bc_df = self.par_dfs["temporal_list"]
            bc_df.loc[:, "y"] = 0
            bc_df.loc[:, "x"] = bc_df.timedelta.apply(lambda x: x.days)
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp == pargp, :]
                p_df = gp_df.drop_duplicates(subset="parnme")
                # print(p_df)
                bc_dfs.append(p_df)
            # bc_dfs = [bc_df.loc[bc_df.pargp==pargp,:].copy() for pargp in bc_df.pargp.unique()]
            struct_dict[self.temporal_list_geostruct] = bc_dfs
        if "spatial_list" in self.par_dfs.keys():
            bc_df = self.par_dfs["spatial_list"]
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp == pargp, :]
                # p_df = gp_df.drop_duplicates(subset="parnme")
                # print(p_df)
                bc_dfs.append(gp_df)
            struct_dict[self.spatial_list_geostruct] = bc_dfs

        pe = geostatistical_draws(
            self.pst,
            struct_dict=struct_dict,
            num_reals=num_reals,
            sigma_range=sigma_range,
            scale_offset=scale_offset,
        )
        if gr_par_pe is not None:
            pe.loc[:, gr_par_pe.columns] = gr_par_pe.values
        self.log("drawing realizations")
        return pe

    def build_prior(
        self, fmt="ascii", filename=None, droptol=None, chunk=None, sigma_range=6
    ):
        """build and optionally save the prior parameter covariance matrix.

        Args:
            fmt (`str`, optional): the format to save the cov matrix.  Options are "ascii","binary","uncfile", "coo".
                Default is "ascii".  If "none" (lower case string, not None), then no file is created.
            filename (`str`, optional): the filename to save the prior cov matrix to.  If None, the name is formed using
                model nam_file name.  Default is None.
            droptol (`float`, optional): tolerance for dropping near-zero values when writing compressed binary.
                Default is None.
            chunk (`int`, optional): chunk size to write in a single pass - for binary only.  Default
                is None (no chunking).
            sigma_range (`float`): number of standard deviations represented by the parameter bounds.  Default
                is 6.

        Returns:
            `pyemu.Cov`: the full prior parameter covariance matrix, generated by processing parameters by
            groups

        """

        fmt = fmt.lower()
        acc_fmts = ["ascii", "binary", "uncfile", "none", "coo"]
        if fmt not in acc_fmts:
            self.logger.lraise(
                "unrecognized prior save 'fmt':{0}, options are: {1}".format(
                    fmt, ",".join(acc_fmts)
                )
            )

        self.log("building prior covariance matrix")
        struct_dict = {}
        if self.pp_suffix in self.par_dfs.keys():
            pp_df = self.par_dfs[self.pp_suffix]
            pp_dfs = []
            for pargp in pp_df.pargp.unique():
                gp_df = pp_df.loc[pp_df.pargp == pargp, :]
                p_df = gp_df.drop_duplicates(subset="parnme")
                pp_dfs.append(p_df)
            # pp_dfs = [pp_df.loc[pp_df.pargp==pargp,:].copy() for pargp in pp_df.pargp.unique()]
            struct_dict[self.pp_geostruct] = pp_dfs
        if self.gr_suffix in self.par_dfs.keys():
            gr_df = self.par_dfs[self.gr_suffix]
            gr_dfs = []
            for pargp in gr_df.pargp.unique():
                gp_df = gr_df.loc[gr_df.pargp == pargp, :]
                p_df = gp_df.drop_duplicates(subset="parnme")
                gr_dfs.append(p_df)
            # gr_dfs = [gr_df.loc[gr_df.pargp==pargp,:].copy() for pargp in gr_df.pargp.unique()]
            struct_dict[self.grid_geostruct] = gr_dfs
        if "temporal_list" in self.par_dfs.keys():
            bc_df = self.par_dfs["temporal_list"]
            bc_df.loc[:, "y"] = 0
            bc_df.loc[:, "x"] = bc_df.timedelta.apply(lambda x: x.days)
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp == pargp, :]
                p_df = gp_df.drop_duplicates(subset="parnme")
                # print(p_df)
                bc_dfs.append(p_df)
            # bc_dfs = [bc_df.loc[bc_df.pargp==pargp,:].copy() for pargp in bc_df.pargp.unique()]
            struct_dict[self.temporal_list_geostruct] = bc_dfs
        if "spatial_list" in self.par_dfs.keys():
            bc_df = self.par_dfs["spatial_list"]
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp == pargp, :]
                # p_df = gp_df.drop_duplicates(subset="parnme")
                # print(p_df)
                bc_dfs.append(gp_df)
            struct_dict[self.spatial_list_geostruct] = bc_dfs
        if "hfb" in self.par_dfs.keys():
            if self.spatial_list_geostruct in struct_dict.keys():
                struct_dict[self.spatial_list_geostruct].append(self.par_dfs["hfb"])
            else:
                struct_dict[self.spatial_list_geostruct] = [self.par_dfs["hfb"]]

        if "sfr" in self.par_dfs.keys():
            self.logger.warn("geospatial prior not implemented for SFR pars")

        if len(struct_dict) > 0:
            cov = pyemu.helpers.geostatistical_prior_builder(
                self.pst, struct_dict=struct_dict, sigma_range=sigma_range
            )
        else:
            cov = pyemu.Cov.from_parameter_data(self.pst, sigma_range=sigma_range)

        if filename is None:
            filename = os.path.join(self.m.model_ws, self.pst_name + ".prior.cov")
        if fmt != "none":
            self.logger.statement(
                "saving prior covariance matrix to file {0}".format(filename)
            )
        if fmt == "ascii":
            cov.to_ascii(filename)
        elif fmt == "binary":
            cov.to_binary(filename, droptol=droptol, chunk=chunk)
        elif fmt == "uncfile":
            cov.to_uncfile(filename)
        elif fmt == "coo":
            cov.to_coo(filename, droptol=droptol, chunk=chunk)
        self.log("building prior covariance matrix")
        return cov

    def build_pst(self, filename=None):
        """build the pest control file using the parameters and
        observations.

        Args:
            filename (`str`): the filename to save the contorl file to.  If None, the
                name if formed from the model namfile name.  Default is None.  The control
                is saved in the `PstFromFlopy.m.model_ws` directory.
        Note:

            calls pyemu.Pst.from_io_files

            calls PESTCHEK

        """
        self.logger.statement("changing dir in to {0}".format(self.m.model_ws))
        os.chdir(self.m.model_ws)
        tpl_files = copy.deepcopy(self.tpl_files)
        in_files = copy.deepcopy(self.in_files)
        try:
            files = os.listdir(".")
            new_tpl_files = [
                f for f in files if f.endswith(".tpl") and f not in tpl_files
            ]
            new_in_files = [f.replace(".tpl", "") for f in new_tpl_files]
            tpl_files.extend(new_tpl_files)
            in_files.extend(new_in_files)
            ins_files = [f for f in files if f.endswith(".ins")]
            out_files = [f.replace(".ins", "") for f in ins_files]
            for tpl_file, in_file in zip(tpl_files, in_files):
                if tpl_file not in self.tpl_files:
                    self.tpl_files.append(tpl_file)
                    self.in_files.append(in_file)

            for ins_file, out_file in zip(ins_files, out_files):
                if ins_file not in self.ins_files:
                    self.ins_files.append(ins_file)
                    self.out_files.append(out_file)
            self.log("instantiating control file from i/o files")
            self.logger.statement("tpl files: {0}".format(",".join(self.tpl_files)))
            self.logger.statement("ins files: {0}".format(",".join(self.ins_files)))
            pst = pyemu.Pst.from_io_files(
                tpl_files=self.tpl_files,
                in_files=self.in_files,
                ins_files=self.ins_files,
                out_files=self.out_files,
            )

            self.log("instantiating control file from i/o files")
        except Exception as e:
            os.chdir("..")
            self.logger.lraise("error build Pst:{0}".format(str(e)))
        os.chdir("..")
        # more customization here
        par = pst.parameter_data
        for name, df in self.par_dfs.items():
            if "parnme" not in df.columns:
                continue
            df.index = df.parnme
            for col in par.columns:
                if col in df.columns:
                    par.loc[df.parnme, col] = df.loc[:, col]

        par.loc[:, "parubnd"] = 10.0
        par.loc[:, "parlbnd"] = 0.1

        for name, df in self.par_dfs.items():
            if "parnme" not in df:
                continue
            df.index = df.parnme
            for col in ["parubnd", "parlbnd", "pargp"]:
                if col in df.columns:
                    par.loc[df.index, col] = df.loc[:, col]

        for tag, [lw, up] in wildass_guess_par_bounds_dict.items():
            par.loc[par.parnme.apply(lambda x: x.startswith(tag)), "parubnd"] = up
            par.loc[par.parnme.apply(lambda x: x.startswith(tag)), "parlbnd"] = lw

        if self.par_bounds_dict is not None:
            for tag, [lw, up] in self.par_bounds_dict.items():
                par.loc[par.parnme.apply(lambda x: x.startswith(tag)), "parubnd"] = up
                par.loc[par.parnme.apply(lambda x: x.startswith(tag)), "parlbnd"] = lw

        obs = pst.observation_data
        for name, df in self.obs_dfs.items():
            if "obsnme" not in df.columns:
                continue
            df.index = df.obsnme
            for col in df.columns:
                if col in obs.columns:
                    obs.loc[df.obsnme, col] = df.loc[:, col]

        self.pst_name = self.m.name + ".pst"
        pst.model_command = ["python forward_run.py"]
        pst.control_data.noptmax = 0
        self.log("writing forward_run.py")
        self.write_forward_run()
        self.log("writing forward_run.py")

        if filename is None:
            filename = os.path.join(self.m.model_ws, self.pst_name)
        self.logger.statement("writing pst {0}".format(filename))

        pst.write(filename)
        self.pst = pst

        self.log("running pestchek on {0}".format(self.pst_name))
        os.chdir(self.m.model_ws)
        try:
            pyemu.os_utils.run("pestchek {0} >pestchek.stdout".format(self.pst_name))
        except Exception as e:
            self.logger.warn("error running pestchek:{0}".format(str(e)))
        for line in open("pestchek.stdout"):
            self.logger.statement("pestcheck:{0}".format(line.strip()))
        os.chdir("..")
        self.log("running pestchek on {0}".format(self.pst_name))

    def _add_external(self):
        """add external (existing) template files and/or instruction files to the
        Pst instance

        """
        if self.external_tpl_in_pairs is not None:
            if not isinstance(self.external_tpl_in_pairs, list):
                external_tpl_in_pairs = [self.external_tpl_in_pairs]
            for tpl_file, in_file in self.external_tpl_in_pairs:
                if not os.path.exists(tpl_file):
                    self.logger.lraise(
                        "couldn't find external tpl file:{0}".format(tpl_file)
                    )
                self.logger.statement("external tpl:{0}".format(tpl_file))
                shutil.copy2(
                    tpl_file, os.path.join(self.m.model_ws, os.path.split(tpl_file)[-1])
                )
                if os.path.exists(in_file):
                    shutil.copy2(
                        in_file,
                        os.path.join(self.m.model_ws, os.path.split(in_file)[-1]),
                    )

        if self.external_ins_out_pairs is not None:
            if not isinstance(self.external_ins_out_pairs, list):
                external_ins_out_pairs = [self.external_ins_out_pairs]
            for ins_file, out_file in self.external_ins_out_pairs:
                if not os.path.exists(ins_file):
                    self.logger.lraise(
                        "couldn't find external ins file:{0}".format(ins_file)
                    )
                self.logger.statement("external ins:{0}".format(ins_file))
                shutil.copy2(
                    ins_file, os.path.join(self.m.model_ws, os.path.split(ins_file)[-1])
                )
                if os.path.exists(out_file):
                    shutil.copy2(
                        out_file,
                        os.path.join(self.m.model_ws, os.path.split(out_file)[-1]),
                    )
                    self.logger.warn(
                        "obs listed in {0} will have values listed in {1}".format(
                            ins_file, out_file
                        )
                    )
                else:
                    self.logger.warn("obs listed in {0} will have generic values")

    def write_forward_run(self):
        """write the forward run script forward_run.py

        Note:
            This method can be called repeatedly, especially after any
            changed to the pre- and/or post-processing routines.

        """
        with open(os.path.join(self.m.model_ws, self.forward_run_file), "w") as f:
            f.write(
                "import os\nimport multiprocessing as mp\nimport numpy as np"
                + "\nimport pandas as pd\nimport flopy\n"
            )
            f.write("import pyemu\n")
            f.write("def main():\n")
            f.write("\n")
            s = "    "
            for ex_imp in self.extra_forward_imports:
                f.write(s + "import {0}\n".format(ex_imp))
            for tmp_file in self.tmp_files:
                f.write(s + "try:\n")
                f.write(s + "   os.remove('{0}')\n".format(tmp_file))
                f.write(s + "except Exception as e:\n")
                f.write(
                    s + "   print('error removing tmp file:{0}')\n".format(tmp_file)
                )
            for line in self.frun_pre_lines:
                f.write(s + line + "\n")
            for line in self.frun_model_lines:
                f.write(s + line + "\n")
            for line in self.frun_post_lines:
                f.write(s + line + "\n")
            f.write("\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    mp.freeze_support()\n    main()\n\n")

    def _parse_k(self, k, vals):
        """parse the iterable from a property or boundary condition argument"""
        try:
            k = int(k)
        except:
            pass
        else:
            assert k in vals, "k {0} not in vals".format(k)
            return [k]
        if k is None:
            return vals
        else:
            try:
                k_vals = vals[k]
            except Exception as e:
                raise Exception("error slicing vals with {0}:{1}".format(k, str(e)))
            return k_vals

    def _parse_pakattr(self, pakattr):
        """parse package-iterable pairs from a property or boundary condition
        argument

        """

        raw = pakattr.lower().split(".")
        if len(raw) != 2:
            self.logger.lraise("pakattr is wrong:{0}".format(pakattr))
        pakname = raw[0]
        attrname = raw[1]
        pak = self.m.get_package(pakname)
        if pak is None:
            if pakname == "extra":
                self.logger.statement("'extra' pak detected:{0}".format(pakattr))
                ud = flopy.utils.Util3d(
                    self.m,
                    (self.m.nlay, self.m.nrow, self.m.ncol),
                    np.float32,
                    1.0,
                    attrname,
                )
                return "extra", ud

            self.logger.lraise("pak {0} not found".format(pakname))
        if hasattr(pak, attrname):
            attr = getattr(pak, attrname)
            return pak, attr
        elif hasattr(pak, "stress_period_data"):
            dtype = pak.stress_period_data.dtype
            if attrname not in dtype.names:
                self.logger.lraise(
                    "attr {0} not found in dtype.names for {1}.stress_period_data".format(
                        attrname, pakname
                    )
                )
            attr = pak.stress_period_data
            return pak, attr, attrname
        # elif hasattr(pak,'hfb_data'):
        #     dtype = pak.hfb_data.dtype
        #     if attrname not in dtype.names:
        #         self.logger.lraise('attr {0} not found in dtypes.names for {1}.hfb_data. Thanks for playing.'.\
        #                            format(attrname,pakname))
        #     attr = pak.hfb_data
        #     return pak, attr, attrname
        else:
            self.logger.lraise("unrecognized attr:{0}".format(attrname))

    def _setup_list_pars(self):
        """main entry point for setting up list multiplier
        parameters

        """
        tdf = self._setup_temporal_list_pars()
        sdf = self._setup_spatial_list_pars()
        if tdf is None and sdf is None:
            return
        os.chdir(self.m.model_ws)
        try:
            apply_list_pars()
        except Exception as e:
            os.chdir("..")
            self.logger.lraise(
                "error test running apply_list_pars():{0}".format(str(e))
            )
        os.chdir("..")
        line = "pyemu.helpers.apply_list_pars()\n"
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

    def _setup_temporal_list_pars(self):

        if len(self.temporal_list_props) == 0:
            return
        self.log("processing temporal_list_props")
        bc_filenames = []
        bc_cols = []
        bc_pak = []
        bc_k = []
        bc_dtype_names = []
        bc_parnme = []
        if len(self.temporal_list_props) == 2:
            if not isinstance(self.temporal_list_props[0], list):
                self.temporal_list_props = [self.temporal_list_props]
        for pakattr, k_org in self.temporal_list_props:
            pak, attr, col = self._parse_pakattr(pakattr)
            k_parse = self._parse_k(k_org, np.arange(self.m.nper))
            c = self._get_count(pakattr)
            for k in k_parse:
                bc_filenames.append(self._list_helper(k, pak, attr, col))
                bc_cols.append(col)
                pak_name = pak.name[0].lower()
                bc_pak.append(pak_name)
                bc_k.append(k)
                bc_dtype_names.append(",".join(attr.dtype.names))

                bc_parnme.append("{0}{1}_{2:03d}".format(pak_name, col, c))

        df = pd.DataFrame(
            {
                "filename": bc_filenames,
                "col": bc_cols,
                "kper": bc_k,
                "pak": bc_pak,
                "dtype_names": bc_dtype_names,
                "parnme": bc_parnme,
            }
        )
        tds = pd.to_timedelta(np.cumsum(self.m.dis.perlen.array), unit="d")
        dts = pd.to_datetime(self.m._start_datetime) + tds
        df.loc[:, "datetime"] = df.kper.apply(lambda x: dts[x])
        df.loc[:, "timedelta"] = df.kper.apply(lambda x: tds[x])
        df.loc[:, "val"] = 1.0
        # df.loc[:,"kper"] = df.kper.apply(np.int64)
        # df.loc[:,"parnme"] = df.apply(lambda x: "{0}{1}_{2:03d}".format(x.pak,x.col,x.kper),axis=1)
        df.loc[:, "tpl_str"] = df.parnme.apply(lambda x: "~   {0}   ~".format(x))
        df.loc[:, "list_org"] = self.list_org
        df.loc[:, "model_ext_path"] = self.m.external_path
        df.loc[:, "pargp"] = df.parnme.apply(lambda x: x.split("_")[0])
        names = [
            "filename",
            "dtype_names",
            "list_org",
            "model_ext_path",
            "col",
            "kper",
            "pak",
            "val",
        ]
        df.loc[:, names].to_csv(
            os.path.join(self.m.model_ws, "temporal_list_pars.dat"), sep=" "
        )
        df.loc[:, "val"] = df.tpl_str
        tpl_name = os.path.join(self.m.model_ws, "temporal_list_pars.dat.tpl")
        # f_tpl =  open(tpl_name,'w')
        # f_tpl.write("ptf ~\n")
        # f_tpl.flush()
        # df.loc[:,names].to_csv(f_tpl,sep=' ',quotechar=' ')
        # f_tpl.write("index ")
        # f_tpl.write(df.loc[:,names].to_string(index_names=True))
        # f_tpl.close()
        _write_df_tpl(
            tpl_name, df.loc[:, names], sep=" ", index_label="index", quotechar=" "
        )
        self.par_dfs["temporal_list"] = df

        self.log("processing temporal_list_props")
        return True

    def _setup_spatial_list_pars(self):

        if len(self.spatial_list_props) == 0:
            return
        self.log("processing spatial_list_props")

        bc_filenames = []
        bc_cols = []
        bc_pak = []
        bc_k = []
        bc_dtype_names = []
        bc_parnme = []
        if len(self.spatial_list_props) == 2:
            if not isinstance(self.spatial_list_props[0], list):
                self.spatial_list_props = [self.spatial_list_props]
        for pakattr, k_org in self.spatial_list_props:
            pak, attr, col = self._parse_pakattr(pakattr)
            k_parse = self._parse_k(k_org, np.arange(self.m.nlay))
            if len(k_parse) > 1:
                self.logger.lraise(
                    "spatial_list_pars error: each set of spatial list pars can only be applied "
                    + "to a single layer (e.g. [wel.flux,0].\n"
                    + "You passed [{0},{1}], implying broadcasting to layers {2}".format(
                        pakattr, k_org, k_parse
                    )
                )
            # # horrible special case for HFB since it cannot vary over time
            # if type(pak) != flopy.modflow.mfhfb.ModflowHfb:
            for k in range(self.m.nper):
                bc_filenames.append(self._list_helper(k, pak, attr, col))
                bc_cols.append(col)
                pak_name = pak.name[0].lower()
                bc_pak.append(pak_name)
                bc_k.append(k_parse[0])
                bc_dtype_names.append(",".join(attr.dtype.names))

        info_df = pd.DataFrame(
            {
                "filename": bc_filenames,
                "col": bc_cols,
                "k": bc_k,
                "pak": bc_pak,
                "dtype_names": bc_dtype_names,
            }
        )
        info_df.loc[:, "list_mlt"] = self.list_mlt
        info_df.loc[:, "list_org"] = self.list_org
        info_df.loc[:, "model_ext_path"] = self.m.external_path

        # check that all files for a given package have the same number of entries
        info_df.loc[:, "itmp"] = np.NaN
        pak_dfs = {}
        for pak in info_df.pak.unique():
            df_pak = info_df.loc[info_df.pak == pak, :]
            itmp = []
            for filename in df_pak.filename:
                names = df_pak.dtype_names.iloc[0].split(",")

                # mif pak != 'hfb6':
                fdf = pd.read_csv(
                    os.path.join(self.m.model_ws, filename),
                    delim_whitespace=True,
                    header=None,
                    names=names,
                )
                for c in ["k", "i", "j"]:
                    fdf.loc[:, c] -= 1
                # else:
                #     # need to navigate the HFB file to skip both comments and header line
                #     skiprows = sum(
                #         [1 if i.strip().startswith('#') else 0
                #          for i in open(os.path.join(self.m.model_ws, filename), 'r').readlines()]) + 1
                #     fdf = pd.read_csv(os.path.join(self.m.model_ws, filename),
                #                       delim_whitespace=True, header=None, names=names, skiprows=skiprows  ).dropna()
                #
                #     for c in ['k', 'irow1','icol1','irow2','icol2']:
                #         fdf.loc[:, c] -= 1

                itmp.append(fdf.shape[0])
                pak_dfs[pak] = fdf
            info_df.loc[info_df.pak == pak, "itmp"] = itmp
            if np.unique(np.array(itmp)).shape[0] != 1:
                info_df.to_csv("spatial_list_trouble.csv")
                self.logger.lraise(
                    "spatial_list_pars() error: must have same number of "
                    + "entries for every stress period for {0}".format(pak)
                )

        # make the pak dfs have unique model indices
        for pak, df in pak_dfs.items():
            # if pak != 'hfb6':
            df.loc[:, "idx"] = df.apply(
                lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}".format(x.k, x.i, x.j), axis=1
            )
            # else:
            #     df.loc[:, "idx"] = df.apply(lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}{2:04.0f}{2:04.0f}".format(x.k, x.irow1, x.icol1,
            #                                                                                                  x.irow2, x.icol2), axis=1)
            if df.idx.unique().shape[0] != df.shape[0]:
                self.logger.warn(
                    "duplicate entries in list pak {0}...collapsing".format(pak)
                )
                df.drop_duplicates(subset="idx", inplace=True)
            df.index = df.idx
            pak_dfs[pak] = df

        # write template files - find which cols are parameterized...
        par_dfs = []
        for pak, df in pak_dfs.items():
            pak_df = info_df.loc[info_df.pak == pak, :]
            # reset all non-index cols to 1.0
            for col in df.columns:
                if col not in [
                    "k",
                    "i",
                    "j",
                    "inode",
                    "irow1",
                    "icol1",
                    "irow2",
                    "icol2",
                ]:
                    df.loc[:, col] = 1.0
            in_file = os.path.join(self.list_mlt, pak + ".csv")
            tpl_file = os.path.join(pak + ".csv.tpl")
            # save an all "ones" mult df for testing
            df.to_csv(os.path.join(self.m.model_ws, in_file), sep=" ")
            parnme, pargp = [], []
            # if pak != 'hfb6':
            x = df.apply(
                lambda x: self.sr.xcentergrid[int(x.i), int(x.j)], axis=1
            ).values
            y = df.apply(
                lambda x: self.sr.ycentergrid[int(x.i), int(x.j)], axis=1
            ).values
            # else:
            #     # note -- for HFB6, only row and col for node 1
            #     x = df.apply(lambda x: self.m.sr.xcentergrid[int(x.irow1),int(x.icol1)],axis=1).values
            #     y = df.apply(lambda x: self.m.sr.ycentergrid[int(x.irow1),int(x.icol1)],axis=1).values

            for col in pak_df.col.unique():
                col_df = pak_df.loc[pak_df.col == col]
                k_vals = col_df.k.unique()
                npar = col_df.k.apply(lambda x: x in k_vals).shape[0]
                if npar == 0:
                    continue
                names = df.index.map(lambda x: "{0}{1}{2}".format(pak[0], col[0], x))

                df.loc[:, col] = names.map(lambda x: "~   {0}   ~".format(x))
                df.loc[df.k.apply(lambda x: x not in k_vals), col] = 1.0
                par_df = pd.DataFrame(
                    {"parnme": names, "x": x, "y": y, "k": df.k.values}, index=names
                )
                par_df = par_df.loc[par_df.k.apply(lambda x: x in k_vals)]
                if par_df.shape[0] == 0:
                    self.logger.lraise(
                        "no parameters found for spatial list k,pak,attr {0}, {1}, {2}".format(
                            k_vals, pak, col
                        )
                    )

                par_df.loc[:, "pargp"] = df.k.apply(
                    lambda x: "{0}{1}_k{2:02.0f}".format(pak, col, int(x))
                ).values

                par_df.loc[:, "tpl_file"] = tpl_file
                par_df.loc[:, "in_file"] = in_file
                par_dfs.append(par_df)

            # with open(os.path.join(self.m.model_ws,tpl_file),'w') as f:
            #    f.write("ptf ~\n")
            # f.flush()
            # df.to_csv(f)
            #    f.write("index ")
            #    f.write(df.to_string(index_names=False)+'\n')
            _write_df_tpl(
                os.path.join(self.m.model_ws, tpl_file),
                df,
                sep=" ",
                quotechar=" ",
                index_label="index",
            )
            self.tpl_files.append(tpl_file)
            self.in_files.append(in_file)

        par_df = pd.concat(par_dfs)
        self.par_dfs["spatial_list"] = par_df
        info_df.to_csv(os.path.join(self.m.model_ws, "spatial_list_pars.dat"), sep=" ")

        self.log("processing spatial_list_props")
        return True

    def _list_helper(self, k, pak, attr, col):
        """helper to setup list multiplier parameters for a given
        k, pak, attr set.

        """
        # special case for horrible HFB6 exception
        # if type(pak) == flopy.modflow.mfhfb.ModflowHfb:
        #     filename = pak.file_name[0]
        # else:
        filename = attr.get_filename(k)
        filename_model = os.path.join(self.m.external_path, filename)
        shutil.copy2(
            os.path.join(self.m.model_ws, filename_model),
            os.path.join(self.m.model_ws, self.list_org, filename),
        )
        return filename_model

    def _setup_hds(self):
        """setup modflow head save file observations for given kper (zero-based
        stress period index) and k (zero-based layer index) pairs using the
        kperk argument.

        """
        if self.hds_kperk is None or len(self.hds_kperk) == 0:
            return
        from .gw_utils import setup_hds_obs

        # if len(self.hds_kperk) == 2:
        #     try:
        #         if len(self.hds_kperk[0] == 2):
        #             pass
        #     except:
        #         self.hds_kperk = [self.hds_kperk]
        oc = self.m.get_package("OC")
        if oc is None:
            raise Exception("can't find OC package in model to setup hds grid obs")
        if not oc.savehead:
            raise Exception("OC not saving hds, can't setup grid obs")
        hds_unit = oc.iuhead
        hds_file = self.m.get_output(unit=hds_unit)
        assert os.path.exists(
            os.path.join(self.org_model_ws, hds_file)
        ), "couldn't find existing hds file {0} in org_model_ws".format(hds_file)
        shutil.copy2(
            os.path.join(self.org_model_ws, hds_file),
            os.path.join(self.m.model_ws, hds_file),
        )
        inact = None
        if self.m.lpf is not None:
            inact = self.m.lpf.hdry
        elif self.m.upw is not None:
            inact = self.m.upw.hdry
        if inact is None:
            skip = lambda x: np.NaN if x == self.m.bas6.hnoflo else x
        else:
            skip = lambda x: np.NaN if x == self.m.bas6.hnoflo or x == inact else x
        print(self.hds_kperk)
        frun_line, df = setup_hds_obs(
            os.path.join(self.m.model_ws, hds_file),
            kperk_pairs=self.hds_kperk,
            skip=skip,
        )
        self.obs_dfs["hds"] = df
        self.frun_post_lines.append(
            "pyemu.gw_utils.apply_hds_obs('{0}')".format(hds_file)
        )
        self.tmp_files.append(hds_file)

    def _setup_smp(self):
        """setup observations from PEST-style SMP file pairs"""
        if self.obssim_smp_pairs is None:
            return
        if len(self.obssim_smp_pairs) == 2:
            if isinstance(self.obssim_smp_pairs[0], str):
                self.obssim_smp_pairs = [self.obssim_smp_pairs]
        for obs_smp, sim_smp in self.obssim_smp_pairs:
            self.log("processing {0} and {1} smp files".format(obs_smp, sim_smp))
            if not os.path.exists(obs_smp):
                self.logger.lraise("couldn't find obs smp: {0}".format(obs_smp))
            if not os.path.exists(sim_smp):
                self.logger.lraise("couldn't find sim smp: {0}".format(sim_smp))
            new_obs_smp = os.path.join(self.m.model_ws, os.path.split(obs_smp)[-1])
            shutil.copy2(obs_smp, new_obs_smp)
            new_sim_smp = os.path.join(self.m.model_ws, os.path.split(sim_smp)[-1])
            shutil.copy2(sim_smp, new_sim_smp)
            pyemu.smp_utils.smp_to_ins(new_sim_smp)

    def _setup_hob(self):
        """setup observations from the MODFLOW HOB package"""

        if self.m.hob is None:
            return
        hob_out_unit = self.m.hob.iuhobsv
        new_hob_out_fname = os.path.join(
            self.m.model_ws,
            self.m.get_output_attribute(unit=hob_out_unit, attr="fname"),
        )
        org_hob_out_fname = os.path.join(
            self.org_model_ws,
            self.m.get_output_attribute(unit=hob_out_unit, attr="fname"),
        )

        if not os.path.exists(org_hob_out_fname):
            self.logger.warn(
                "could not find hob out file: {0}...skipping".format(hob_out_fname)
            )
            return
        shutil.copy2(org_hob_out_fname, new_hob_out_fname)
        hob_df = pyemu.gw_utils.modflow_hob_to_instruction_file(new_hob_out_fname)
        self.obs_dfs["hob"] = hob_df
        self.tmp_files.append(os.path.split(new_hob_out_fname)[-1])

    def _setup_hyd(self):
        """setup observations from the MODFLOW HYDMOD package"""
        if self.m.hyd is None:
            return
        if self.mfhyd:
            org_hyd_out = os.path.join(self.org_model_ws, self.m.name + ".hyd.bin")
            if not os.path.exists(org_hyd_out):
                self.logger.warn(
                    "can't find existing hyd out file:{0}...skipping".format(
                        org_hyd_out
                    )
                )
                return
            new_hyd_out = os.path.join(self.m.model_ws, os.path.split(org_hyd_out)[-1])
            shutil.copy2(org_hyd_out, new_hyd_out)
            df = pyemu.gw_utils.modflow_hydmod_to_instruction_file(new_hyd_out)
            df.loc[:, "obgnme"] = df.obsnme.apply(lambda x: "_".join(x.split("_")[:-1]))
            line = "pyemu.gw_utils.modflow_read_hydmod_file('{0}')".format(
                os.path.split(new_hyd_out)[-1]
            )
            self.logger.statement("forward_run line: {0}".format(line))
            self.frun_post_lines.append(line)
            self.obs_dfs["hyd"] = df
            self.tmp_files.append(os.path.split(new_hyd_out)[-1])

    def _setup_water_budget_obs(self):
        """setup observations from the MODFLOW list file for
        volume and flux water buget information

        """
        if self.mflist_waterbudget:
            org_listfile = os.path.join(self.org_model_ws, self.m.lst.file_name[0])
            if os.path.exists(org_listfile):
                shutil.copy2(
                    org_listfile, os.path.join(self.m.model_ws, self.m.lst.file_name[0])
                )
            else:
                self.logger.warn(
                    "can't find existing list file:{0}...skipping".format(org_listfile)
                )
                return
            list_file = os.path.join(self.m.model_ws, self.m.lst.file_name[0])
            flx_file = os.path.join(self.m.model_ws, "flux.dat")
            vol_file = os.path.join(self.m.model_ws, "vol.dat")
            df = pyemu.gw_utils.setup_mflist_budget_obs(
                list_file,
                flx_filename=flx_file,
                vol_filename=vol_file,
                start_datetime=self.m.start_datetime,
            )
            if df is not None:
                self.obs_dfs["wb"] = df
            # line = "try:\n    os.remove('{0}')\nexcept:\n    pass".format(os.path.split(list_file)[-1])
            # self.logger.statement("forward_run line:{0}".format(line))
            # self.frun_pre_lines.append(line)
            self.tmp_files.append(os.path.split(list_file)[-1])
            line = "pyemu.gw_utils.apply_mflist_budget_obs('{0}',flx_filename='{1}',vol_filename='{2}',start_datetime='{3}')".format(
                os.path.split(list_file)[-1],
                os.path.split(flx_file)[-1],
                os.path.split(vol_file)[-1],
                self.m.start_datetime,
            )
            self.logger.statement("forward_run line:{0}".format(line))
            self.frun_post_lines.append(line)


def apply_list_and_array_pars(arr_par_file="mult2model_info.csv", chunk_len=50):
    """Apply multiplier parameters to list and array style model files

    Args:
        arr_par_file (str):
        chunk_len (`int`): the number of files to process per multiprocessing
            chunk in appl_array_pars().  default is 50.

    Returns:

    Note:
        Used to implement the parameterization constructed by
        PstFrom during a forward run

        Should be added to the forward_run.py script; added programmatically
        by `PstFrom.build_pst()`
    """
    df = pd.read_csv(arr_par_file, index_col=0)
    if "operator" not in df.columns:
        df.loc[:, "operator"] = "m"
    df.loc[pd.isna(df.operator), "operator"] = "m"
    file_cols = df.columns.values[df.columns.str.contains("file")]
    for file_col in file_cols:
        df.loc[:, file_col] = df.loc[:, file_col].apply(
            lambda x: os.path.join(*x.replace("\\","/").split("/"))
            if isinstance(x,str) else x
        )
    arr_pars = df.loc[df.index_cols.isna()].copy()
    list_pars = df.loc[df.index_cols.notna()].copy()
    # extract lists from string in input df
    list_pars["index_cols"] = list_pars.index_cols.apply(literal_eval)
    list_pars["use_cols"] = list_pars.use_cols.apply(literal_eval)
    list_pars["lower_bound"] = list_pars.lower_bound.apply(literal_eval)
    list_pars["upper_bound"] = list_pars.upper_bound.apply(literal_eval)

    # TODO check use_cols is always present
    apply_genericlist_pars(list_pars, chunk_len=chunk_len)
    apply_array_pars(arr_pars, chunk_len=chunk_len)


def _process_chunk_fac2real(chunk, i):
    for args in chunk:
        pyemu.geostats.fac2real(**args)
    print("process", i, " processed ", len(chunk), "fac2real calls")


def _process_chunk_array_files(chunk, i, df):
    for model_file in chunk:
        _process_array_file(model_file, df)
    print("process", i, " processed ", len(chunk), "process_array_file calls")


def _process_array_file(model_file, df):
    if "operator" not in df.columns:
        df.loc[:, "operator"] = "m"
    # find all mults that need to be applied to this array
    df_mf = df.loc[df.model_file == model_file, :]
    results = []
    org_file = df_mf.org_file.unique()
    if org_file.shape[0] != 1:
        raise Exception("wrong number of org_files for {0}".format(model_file))
    org_arr = np.loadtxt(org_file[0])

    if "mlt_file" in df_mf.columns:
        for mlt, operator in zip(df_mf.mlt_file, df_mf.operator):
            if pd.isna(mlt):
                continue
            mlt_data = np.loadtxt(mlt)
            if org_arr.shape != mlt_data.shape:
                raise Exception(
                    "shape of org file {}:{} differs from mlt file {}:{}".format(
                        org_file, org_arr.shape, mlt, mlt_data.shape
                    )
                )
            if operator == "*" or operator.lower()[0] == "m":
                org_arr *= mlt_data
            elif operator == "+" or operator.lower()[0] == "a":
                org_arr += mlt_data
            else:
                raise Exception(
                    "unrecognized operator '{0}' for mlt file '{1}'".format(
                        operator, mlt
                    )
                )
        if "upper_bound" in df.columns:
            ub_vals = df_mf.upper_bound.value_counts().dropna().to_dict()
            if len(ub_vals) == 0:
                pass
            elif len(ub_vals) > 1:
                print(ub_vals)
                raise Exception("different upper bound values for {0}".format(org_file))
            else:
                ub = float(list(ub_vals.keys())[0])
                org_arr[org_arr > ub] = ub
        if "lower_bound" in df.columns:
            lb_vals = df_mf.lower_bound.value_counts().dropna().to_dict()
            if len(lb_vals) == 0:
                pass
            elif len(lb_vals) > 1:
                raise Exception("different lower bound values for {0}".format(org_file))
            else:
                lb = float(list(lb_vals.keys())[0])
                org_arr[org_arr < lb] = lb

    np.savetxt(model_file, np.atleast_2d(org_arr), fmt="%15.6E", delimiter="")


def apply_array_pars(arr_par="arr_pars.csv", arr_par_file=None, chunk_len=50):
    """a function to apply array-based multipler parameters.

    Args:
        arr_par (`str` or `pandas.DataFrame`): if type `str`,
        path to csv file detailing parameter array multipliers.
            This file can be written by PstFromFlopy.
        if type `pandas.DataFrame` is Dataframe with columns of
        ['mlt_file', 'model_file', 'org_file'] and optionally
        ['pp_file', 'fac_file'].
        chunk_len (`int`) : the number of files to process per chunk
            with multiprocessing - applies to both fac2real and process_
            input_files. Default is 50.

    Note:
        Used to implement the parameterization constructed by
        PstFromFlopyModel during a forward run

        This function should be added to the forward_run.py script but can
        be called on any correctly formatted csv

        This function using multiprocessing, spawning one process for each
        model input array (and optionally pp files).  This speeds up
        execution time considerably but means you need to make sure your
        forward run script uses the proper multiprocessing idioms for
        freeze support and main thread handling (`PstFrom` does this for you).

    """
    if arr_par_file is not None:
        warnings.warn(
            "`arr_par_file` argument is deprecated and replaced "
            "by arr_par. Method now support passing DataFrame as "
            "arr_par arg.",
            PyemuWarning,
        )
        arr_par = arr_par_file
    if isinstance(arr_par, str):
        df = pd.read_csv(arr_par, index_col=0)
    elif isinstance(arr_par, pd.DataFrame):
        df = arr_par
    else:
        raise TypeError(
            "`arr_par` argument must be filename string or "
            "Pandas DataFrame, "
            "type {0} passed".format(type(arr_par))
        )
    # for fname in df.model_file:
    #     try:
    #         os.remove(fname)
    #     except:
    #         print("error removing mult array:{0}".format(fname))

    if "pp_file" in df.columns:
        print("starting fac2real", datetime.now())
        pp_df = df.loc[
            df.pp_file.notna(),
            [
                "pp_file",
                "fac_file",
                "mlt_file",
                "pp_fill_value",
                "pp_lower_limit",
                "pp_upper_limit",
            ],
        ].rename(
            columns={
                "fac_file": "factors_file",
                "mlt_file": "out_file",
                "pp_fill_value": "fill_value",
                "pp_lower_limit": "lower_lim",
                "pp_upper_limit": "upper_lim",
            }
        )
        # don't need to process all (e.g. if const. mults apply across kper...)
        pp_args = pp_df.drop_duplicates().to_dict("records")
        num_ppargs = len(pp_args)
        num_chunk_floor = num_ppargs // chunk_len
        main_chunks = (
            np.array(pp_args)[: num_chunk_floor * chunk_len]
            .reshape([-1, chunk_len])
            .tolist()
        )
        remainder = np.array(pp_args)[num_chunk_floor * chunk_len :].tolist()
        chunks = main_chunks + [remainder]
        print("number of chunks to process:", len(chunks))
        if len(chunks) == 1:
            _process_chunk_fac2real(chunks[0], 0)
        else:
            pool = mp.Pool(processes=min(mp.cpu_count(), len(chunks), 60))
            x = [
                pool.apply_async(_process_chunk_fac2real, args=(chunk, i))
                for i, chunk in enumerate(chunks)
            ]
            [xx.get() for xx in x]
            pool.close()
            pool.join()
        # procs = []
        # for chunk in chunks:
        #     p = mp.Process(target=_process_chunk_fac2real, args=[chunk])
        #     p.start()
        #     procs.append(p)
        # for p in procs:
        #     p.join()

        print("finished fac2real", datetime.now())

    print("starting arr mlt", datetime.now())
    uniq = df.model_file.unique()  # unique model input files to be produced
    num_uniq = len(uniq)  # number of input files to be produced
    # number of files to send to each processor
    # lazy plitting the files to be processed into even chunks
    num_chunk_floor = num_uniq // chunk_len  # number of whole chunks
    main_chunks = (
        uniq[: num_chunk_floor * chunk_len].reshape([-1, chunk_len]).tolist()
    )  # the list of files broken down into chunks
    remainder = uniq[num_chunk_floor * chunk_len :].tolist()  # remaining files
    chunks = main_chunks + [remainder]
    print("number of chunks to process:", len(chunks))
    if len(chunks) == 1:
        _process_chunk_array_files(chunks[0], 0, df)
    # procs = []
    # for chunk in chunks:  # now only spawn processor for each chunk
    #     p = mp.Process(target=_process_chunk_model_files, args=[chunk, df])
    #     p.start()
    #     procs.append(p)
    # for p in procs:
    #     r = p.get(False)
    #     p.join()
    else:
        pool = mp.Pool(processes=min(mp.cpu_count(), len(chunks), 60))
        x = [
            pool.apply_async(_process_chunk_array_files, args=(chunk, i, df))
            for i, chunk in enumerate(chunks)
        ]
        [xx.get() for xx in x]
        pool.close()
        pool.join()
    print("finished arr mlt", datetime.now())


def apply_list_pars():
    """a function to apply boundary condition multiplier parameters.

    Note:
        Used to implement the parameterization constructed by
        PstFromFlopyModel during a forward run

        Requires either "temporal_list_pars.csv" or "spatial_list_pars.csv"

        Should be added to the forward_run.py script (called programmaticlly
        by the `PstFrom` forward run script)


    """
    temp_file = "temporal_list_pars.dat"
    spat_file = "spatial_list_pars.dat"

    temp_df, spat_df = None, None
    if os.path.exists(temp_file):
        temp_df = pd.read_csv(temp_file, delim_whitespace=True)
        temp_df.loc[:, "split_filename"] = temp_df.filename.apply(
            lambda x: os.path.split(x)[-1]
        )
        org_dir = temp_df.list_org.iloc[0]
        model_ext_path = temp_df.model_ext_path.iloc[0]
    if os.path.exists(spat_file):
        spat_df = pd.read_csv(spat_file, delim_whitespace=True)
        spat_df.loc[:, "split_filename"] = spat_df.filename.apply(
            lambda x: os.path.split(x)[-1]
        )
        mlt_dir = spat_df.list_mlt.iloc[0]
        org_dir = spat_df.list_org.iloc[0]
        model_ext_path = spat_df.model_ext_path.iloc[0]
    if temp_df is None and spat_df is None:
        raise Exception("apply_list_pars() - no key dfs found, nothing to do...")
    # load the spatial mult dfs
    sp_mlts = {}
    if spat_df is not None:

        for f in os.listdir(mlt_dir):
            pak = f.split(".")[0].lower()
            df = pd.read_csv(
                os.path.join(mlt_dir, f), index_col=0, delim_whitespace=True
            )
            # if pak != 'hfb6':
            df.index = df.apply(
                lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}".format(x.k, x.i, x.j), axis=1
            )
            # else:
            #     df.index = df.apply(lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}{2:04.0f}{2:04.0f}".format(x.k, x.irow1, x.icol1,
            #                                                                      x.irow2, x.icol2), axis = 1)
            if pak in sp_mlts.keys():
                raise Exception("duplicate multiplier csv for pak {0}".format(pak))
            if df.shape[0] == 0:
                raise Exception("empty dataframe for spatial list file: {0}".format(f))
            sp_mlts[pak] = df

    org_files = os.listdir(org_dir)
    # for fname in df.filename.unique():
    for fname in org_files:
        # need to get the PAK name to handle stupid horrible expceptions for HFB...
        # try:
        #     pakspat = sum([True if fname in i else False for i in spat_df.filename])
        #     if pakspat:
        #         pak = spat_df.loc[spat_df.filename.str.contains(fname)].pak.values[0]
        #     else:
        #         pak = 'notHFB'
        # except:
        #     pak = "notHFB"

        names = None
        if temp_df is not None and fname in temp_df.split_filename.values:
            temp_df_fname = temp_df.loc[temp_df.split_filename == fname, :]
            if temp_df_fname.shape[0] > 0:
                names = temp_df_fname.dtype_names.iloc[0].split(",")
        if spat_df is not None and fname in spat_df.split_filename.values:
            spat_df_fname = spat_df.loc[spat_df.split_filename == fname, :]
            if spat_df_fname.shape[0] > 0:
                names = spat_df_fname.dtype_names.iloc[0].split(",")
        if names is not None:

            df_list = pd.read_csv(
                os.path.join(org_dir, fname),
                delim_whitespace=True,
                header=None,
                names=names,
            )
            df_list.loc[:, "idx"] = df_list.apply(
                lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}".format(
                    x.k - 1, x.i - 1, x.j - 1
                ),
                axis=1,
            )

            df_list.index = df_list.idx
            pak_name = fname.split("_")[0].lower()
            if pak_name in sp_mlts:
                mlt_df = sp_mlts[pak_name]
                mlt_df_ri = mlt_df.reindex(df_list.index)
                for col in df_list.columns:
                    if col in [
                        "k",
                        "i",
                        "j",
                        "inode",
                        "irow1",
                        "icol1",
                        "irow2",
                        "icol2",
                        "idx",
                    ]:
                        continue
                    if col in mlt_df.columns:
                        # print(mlt_df.loc[mlt_df.index.duplicated(),:])
                        # print(df_list.loc[df_list.index.duplicated(),:])
                        df_list.loc[:, col] *= mlt_df_ri.loc[:, col].values

            if temp_df is not None and fname in temp_df.split_filename.values:
                temp_df_fname = temp_df.loc[temp_df.split_filename == fname, :]
                for col, val in zip(temp_df_fname.col, temp_df_fname.val):
                    df_list.loc[:, col] *= val
            fmts = ""
            for name in names:
                if name in ["i", "j", "k", "inode", "irow1", "icol1", "irow2", "icol2"]:
                    fmts += " %9d"
                else:
                    fmts += " %9G"
        np.savetxt(
            os.path.join(model_ext_path, fname), df_list.loc[:, names].values, fmt=fmts
        )


def calc_array_par_summary_stats(arr_par_file="mult2model_info.csv"):
    """read and generate summary statistics for the resulting model input arrays from
    applying array par multipliers

    Args:
        arr_par_file (`str`): the array multiplier key file

    Returns:
        pd.DataFrame: dataframe of summary stats for each model_file entry

    Note:
        This function uses an optional "zone_file" column in the `arr_par_file`. If multiple zones
        files are used, then zone arrays are aggregated to a single array

        "dif" values are original array values minus model input array values

        The outputs from the function can be used to monitor model input array
        changes that occur during PE/UQ analyses, especially when the parameters
        are multiplier types and the dimensionality is very high.

        Consider using `PstFrom.add_observations()` to setup obs for the csv file
        that this function writes.

    """
    df = pd.read_csv(arr_par_file, index_col=0)
    df = df.loc[df.index_cols.isna(), :].copy()
    if df.shape[0] == 0:
        return None
    file_cols = df.columns.values[df.columns.str.contains("file")]
    for file_col in file_cols:
        df.loc[:, file_col] = df.loc[:, file_col].apply(
            lambda x: os.path.join(*x.replace("\\", "/").split("/")) if isinstance(x, str) else x)
    model_input_files = df.model_file.unique()
    model_input_files.sort()
    records = dict()
    stat_dict = {
        "mean": np.nanmean,
        "stdev": np.nanstd,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
    }
    quantiles = [0.05, 0.25, 0.75, 0.95]
    for stat in stat_dict.keys():
        records[stat] = []
        records[stat + "_org"] = []
        records[stat + "_dif"] = []

    for q in quantiles:
        records["quantile_{0}".format(q)] = []
        records["quantile_{0}_org".format(q)] = []
        records["quantile_{0}_dif".format(q)] = []
    records["upper_bound"] = []
    records["lower_bound"] = []
    records["upper_bound_org"] = []
    records["lower_bound_org"] = []
    records["upper_bound_dif"] = []
    records["lower_bound_dif"] = []

    for model_input_file in model_input_files:

        arr = np.loadtxt(model_input_file)
        org_file = df.loc[df.model_file == model_input_file, "org_file"].values
        org_file = org_file[0]
        org_arr = np.loadtxt(org_file)
        if "zone_file" in df.columns:
            zone_file = (
                df.loc[df.model_file == model_input_file, "zone_file"].dropna().unique()
            )
            zone_arr = None
            if len(zone_file) > 1:
                zone_arr = np.zeros_like(arr)
                for zf in zone_file:
                    za = np.loadtxt(zf)
                    zone_arr[za != 0] = 1
            elif len(zone_file) == 1:
                zone_arr = np.loadtxt(zone_file[0])
            if zone_arr is not None:
                arr[zone_arr == 0] = np.NaN
                org_arr[zone_arr == 0] = np.NaN

        for stat, func in stat_dict.items():
            v = func(arr)
            records[stat].append(v)
            ov = func(org_arr)
            records[stat + "_org"].append(ov)
            records[stat + "_dif"].append(ov - v)
        for q in quantiles:
            v = np.nanquantile(arr, q)
            ov = np.nanquantile(org_arr, q)
            records["quantile_{0}".format(q)].append(v)
            records["quantile_{0}_org".format(q)].append(ov)
            records["quantile_{0}_dif".format(q)].append(ov - v)
        ub = df.loc[df.model_file == model_input_file, "upper_bound"].max()
        lb = df.loc[df.model_file == model_input_file, "lower_bound"].min()
        if pd.isna(ub):
            records["upper_bound"].append(0)
            records["upper_bound_org"].append(0)
            records["upper_bound_dif"].append(0)

        else:
            iarr = np.zeros_like(arr)
            iarr[arr == ub] = 1
            v = iarr.sum()
            iarr = np.zeros_like(arr)
            iarr[org_arr == ub] = 1
            ov = iarr.sum()
            records["upper_bound"].append(v)
            records["upper_bound_org"].append(ov)
            records["upper_bound_dif"].append(ov - v)

        if pd.isna(lb):
            records["lower_bound"].append(0)
            records["lower_bound_org"].append(0)
            records["lower_bound_dif"].append(0)

        else:
            iarr = np.zeros_like(arr)
            iarr[arr == lb] = 1
            v = iarr.sum()
            iarr = np.zeros_like(arr)
            iarr[org_arr == lb] = 1
            ov = iarr.sum()
            records["lower_bound"].append(v)
            records["lower_bound_org"].append(ov)
            records["lower_bound_dif"].append(ov - v)

    # scrub model input files
    model_input_files = [
        f.replace(".", "_").replace("\\", "_").replace("/", "_")
        for f in model_input_files
    ]
    df = pd.DataFrame(records, index=model_input_files)
    df.index.name = "model_file"
    df.to_csv("arr_par_summary.csv")
    return df


def apply_genericlist_pars(df, chunk_len=50):
    """a function to apply list style mult parameters

    Args:
        df (pandas.DataFrame): DataFrame that relates files containing
            multipliers to model input file names. Required columns include:
            {"model_file": file name of resulatant model input file,
            "org_file": file name of original file that multipliers act on,
            "fmt": format specifier for model input file (currently on 'free' supported),
            "sep": separator for model input file if 'free' formatted,
            "head_rows": Number of header rows to transfer from orig file to model file,
            "index_cols": list of columns (either indexes or strings) to be used to align mults, orig and model files,
            "use_cols": columns to mults act on,
            "upper_bound": ultimate upper bound for model input file parameter,
            "lower_bound": ultimate lower bound for model input file parameter}
        chunk_len (`int`): number of chunks for each multiprocessing instance to handle.
            Default is 50.

    Note:
        This function is called programmatically during the `PstFrom` forward run process


    """
    print("starting list mlt", datetime.now())
    uniq = df.model_file.unique()  # unique model input files to be produced
    num_uniq = len(uniq)  # number of input files to be produced
    # number of files to send to each processor
    # lazy plitting the files to be processed into even chunks
    num_chunk_floor = num_uniq // chunk_len  # number of whole chunks
    main_chunks = (
        uniq[: num_chunk_floor * chunk_len].reshape([-1, chunk_len]).tolist()
    )  # the list of files broken down into chunks
    remainder = uniq[num_chunk_floor * chunk_len :].tolist()  # remaining files
    chunks = main_chunks + [remainder]
    print("number of chunks to process:", len(chunks))
    if len(chunks) == 1:
        _process_chunk_list_files(chunks[0], 0, df)
    else:
        pool = mp.Pool(processes=min(mp.cpu_count(), len(chunks), 60))
        x = [
            pool.apply_async(_process_chunk_list_files, args=(chunk, i, df))
            for i, chunk in enumerate(chunks)
        ]
        [xx.get() for xx in x]
        pool.close()
        pool.join()
    print("finished list mlt", datetime.now())


def _process_chunk_list_files(chunk, i, df):
    for model_file in chunk:
        try:
            _process_list_file(model_file, df)
        except Exception as e:
            f"{e}: Issue processing model file {model_file}"
            raise e
    print("process", i, " processed ", len(chunk), "process_list_file calls")


def _process_list_file(model_file, df):

    # print("processing model file:", model_file)
    df_mf = df.loc[df.model_file == model_file, :].copy()
    # read data stored in org (mults act on this)
    org_file = df_mf.org_file.unique()
    if org_file.shape[0] != 1:
        raise Exception("wrong number of org_files for {0}".format(model_file))
    org_file = org_file[0]
    # print("org file:", org_file)
    notfree = df_mf.fmt[df_mf.fmt != "free"]
    if len(notfree) > 1:
        raise Exception(
            "too many different format specifiers for "
            "model file: {0}".format(model_file)
        )
    elif len(notfree) == 1:
        fmt = notfree.values[0]
    else:
        fmt = df_mf.fmt.values[-1]
    if fmt == "free":
        if df_mf.sep.dropna().nunique() > 1:
            raise Exception(
                "too many different sep specifiers for "
                "model file: {0}".format(model_file)
            )
        else:
            sep = df_mf.sep.dropna().values[-1]
    else:
        sep = None
    datastrtrow = df_mf.head_rows.values[-1]
    if fmt.lower() == "free" and sep == " ":
        delim_whitespace = True
    if datastrtrow > 0:
        with open(org_file, "r") as fp:
            storehead = [next(fp) for _ in range(datastrtrow)]
    else:
        storehead = []
    # work out if headers are used for index_cols
    # big assumption here that int type index cols will not be written as headers
    index_col_eg = df_mf.index_cols.iloc[-1][0]
    if isinstance(index_col_eg, str):
        # TODO: add test for model file with headers
        # index_cols can be from header str
        header = 0
        hheader = True
    elif isinstance(index_col_eg, int):
        # index_cols are column numbers in input file
        header = None
        hheader = None
        # actually do need index cols to be list of strings
        # to be compatible when the saved original file is read in.
        df_mf.loc[:, "index_cols"] = df_mf.index_cols.apply(
            lambda x: [str(i) for i in x]
        )

    # if writen by PstFrom this should always be comma delim - tidy
    org_data = pd.read_csv(org_file, skiprows=datastrtrow,
                           header=header, dtype='object')
    # mult columns will be string type, so to make sure they align
    org_data.columns = org_data.columns.astype(str)
    # print("org_data columns:", org_data.columns)
    # print("org_data shape:", org_data.shape)
    new_df = org_data.copy()
    for mlt in df_mf.itertuples():
        new_df.loc[:, mlt.index_cols] = new_df.loc[:, mlt.index_cols].apply(
            pd.to_numeric, errors='ignore', downcast='integer')
        try:
            new_df = new_df.reset_index().rename(
                columns={"index": "oidx"}
            ).set_index(
                mlt.index_cols
            )
            new_df = new_df.sort_index()
        except Exception as e:
            print(
                "error setting mlt index_cols: ",
                str(mlt.index_cols),
                " for new_df with cols: ",
                list(new_df.columns),
            )
            raise Exception("error setting mlt index_cols: " + str(e))

        if not hasattr(mlt, "mlt_file") or pd.isna(mlt.mlt_file):
            print("null mlt file for org_file '" + org_file + "', continuing...")
        else:
            mlts = pd.read_csv(mlt.mlt_file)
            # get mult index to align with org_data,
            # get mult index to align with org_data,
            # mult idxs will always be written zero based if int
            # if original model files is not zero based need to add 1
            add1 = int(mlt.zero_based == False)
            mlts.index = pd.MultiIndex.from_tuples(
                mlts.sidx.apply(
                    lambda x: [
                        add1 + int(xx) if xx.strip().isdigit() else xx.strip("'\" ")
                        for xx in x.strip("()").split(",")
                        if xx
                    ]
                ),
                names=mlt.index_cols,
            )
            if mlts.index.nlevels < 2:  # just in case only one index col is used
                mlts.index = mlts.index.get_level_values(0)
            common_idx = (
                new_df.index.intersection(mlts.index).sort_values().drop_duplicates()
            )
            mlt_cols = [str(col) for col in mlt.use_cols]
            operator = mlt.operator
            if operator == "*" or operator.lower()[0] == "m":
                new_df.loc[common_idx, mlt_cols] = \
                    new_df.loc[common_idx, mlt_cols].apply(
                        pd.to_numeric) * mlts.loc[common_idx, mlt_cols]
            elif operator == "+" or operator.lower()[0] == "a":
                new_df.loc[common_idx, mlt_cols] = \
                    new_df.loc[common_idx, mlt_cols].apply(
                        pd.to_numeric) + mlts.loc[common_idx, mlt_cols]
            else:
                raise Exception(
                    "unsupported operator '{0}' for mlt file '{1}'".format(
                        operator, mlt.mlt_file
                    )
                )
        # bring mult index back to columns AND re-order
        new_df = new_df.reset_index().set_index("oidx")[org_data.columns].sort_index()
    if "upper_bound" in df.columns:
        ub = df_mf.apply(
            lambda x: pd.Series({str(c): b for c, b in zip(x.use_cols, x.upper_bound)}),
            axis=1,
        ).max()
        if ub.notnull().any():
            for col, val in ub.items():
                numeric = new_df.loc[new_df[col].apply(np.isreal)]
                sel = numeric.loc[numeric[col] > val].index
                new_df.loc[sel, col] = val
    if "lower_bound" in df.columns:
        lb = df_mf.apply(
            lambda x: pd.Series({str(c): b for c, b in zip(x.use_cols, x.lower_bound)}),
            axis=1,
        ).min()
        if lb.notnull().any():
            for col, val in lb.items():
                numeric = new_df.loc[new_df[col].apply(np.isreal)]
                sel = numeric.loc[numeric[col] < val].index
                new_df.loc[sel, col] = val
    with open(model_file, "w") as fo:
        kwargs = {}
        if "win" in platform.platform().lower():
            kwargs = {"line_terminator": "\n"}
        if len(storehead) != 0:
            fo.write("\n".join(storehead))
            fo.flush()
        if fmt.lower() == "free":
            new_df.to_csv(fo, index=False, mode="a", sep=sep, header=hheader, **kwargs)
        else:
            np.savetxt(
                fo,
                np.atleast_2d(new_df.apply(pd.to_numeric, errors="ignore").values),
                fmt=fmt
            )


def write_const_tpl(name, tpl_file, suffix, zn_array=None, shape=None, longnames=False):
    """write a constant (uniform) template file for a 2-D array

    Args:
        name (`str`): the base parameter name
        tpl_file (`str`): the template file to write
        zn_array (`numpy.ndarray`, optional): an array used to skip inactive cells,
            and optionally get shape info.
        shape (`tuple`): tuple nrow and ncol.  Either `zn_array` or `shape`
            must be passed
        longnames (`bool`): flag to use longer names that exceed 12 chars in length.
            Default is False.

    Returns:
        `pandas.DataFrame`: a dataframe with parameter information

    Note:
        This function is used during the `PstFrom` setup process


    """

    if shape is None and zn_array is None:
        raise Exception("must pass either zn_array or shape")
    elif shape is None:
        shape = zn_array.shape

    parnme = []
    with open(tpl_file, "w") as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zn_array is not None and zn_array[i, j] < 1:
                    pname = " 1.0  "
                else:
                    if longnames:
                        pname = "const_{0}_{1}".format(name, suffix)
                    else:
                        pname = "{0}{1}".format(name, suffix)
                        if len(pname) > 12:
                            warnings.warn(
                                "zone pname too long for pest:{0}".format(pname)
                            )
                    parnme.append(pname)
                    pname = " ~   {0}    ~".format(pname)
                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme}, index=parnme)
    # df.loc[:,"pargp"] = "{0}{1}".format(self.cn_suffixname)
    df.loc[:, "pargp"] = "{0}_{1}".format(suffix.replace("_", ""), name)
    df.loc[:, "tpl"] = tpl_file
    return df


def write_grid_tpl(
    name,
    tpl_file,
    suffix,
    zn_array=None,
    shape=None,
    spatial_reference=None,
    longnames=False,
):
    """write a grid-based template file for a 2-D array

    Args:
        name (`str`): the base parameter name
        tpl_file (`str`): the template file to write - include path
        zn_array (`numpy.ndarray`, optional): zone array to identify
            inactive cells.  Default is None
        shape (`tuple`, optional): a length-two tuple of nrow and ncol.  Either
            `zn_array` or `shape` must be passed.
        spatial_reference (`flopy.utils.SpatialReference`): a spatial reference instance.
            If `longnames` is True, then `spatial_reference` is used to add spatial info
            to the parameter names.
        longnames (`bool`): flag to use longer names that exceed 12 chars in length.
            Default is False.

    Returns:
        `pandas.DataFrame`: a dataframe with parameter information

    Note:
        This function is used during the `PstFrom` setup process

    Example::

        pyemu.helpers.write_grid_tpl("hk_layer1","hk_Layer_1.ref.tpl","gr",
                                     zn_array=ib_layer_1,shape=(500,500))


    """

    if shape is None and zn_array is None:
        raise Exception("must pass either zn_array or shape")
    elif shape is None:
        shape = zn_array.shape

    parnme, x, y = [], [], []
    with open(tpl_file, "w") as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zn_array is not None and zn_array[i, j] < 1:
                    pname = " 1.0 "
                else:
                    if longnames:
                        pname = "{0}_i:{1}_j:{2}_{3}".format(name, i, j, suffix)
                        if spatial_reference is not None:
                            pname += "_x:{0:10.2E}_y:{1:10.2E}".format(
                                spatial_reference.xcentergrid[i, j],
                                spatial_reference.ycentergrid[i, j],
                            ).replace(" ", "")
                    else:
                        pname = "{0}{1:03d}{2:03d}".format(name, i, j)
                        if len(pname) > 12:
                            warnings.warn(
                                "grid pname too long for pest:{0}".format(pname)
                            )
                    parnme.append(pname)
                    pname = " ~     {0}   ~ ".format(pname)
                    if spatial_reference is not None:
                        x.append(spatial_reference.xcentergrid[i, j])
                        y.append(spatial_reference.ycentergrid[i, j])

                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme}, index=parnme)
    if spatial_reference is not None:
        df.loc[:, "x"] = x
        df.loc[:, "y"] = y
    df.loc[:, "pargp"] = "{0}_{1}".format(suffix.replace("_", ""), name)
    df.loc[:, "tpl"] = tpl_file
    return df


def write_zone_tpl(
    name,
    tpl_file,
    suffix="",
    zn_array=None,
    shape=None,
    longnames=False,
    fill_value="1.0",
):
    """write a zone-based template file for a 2-D array

    Args:
        name (`str`): the base parameter name
        tpl_file (`str`): the template file to write
        suffix (`str`): suffix to add to parameter names.  Only used if `longnames=True`
        zn_array (`numpy.ndarray`, optional): an array used to skip inactive cells,
            and optionally get shape info.  zn_array values less than 1 are given `fill_value`
        shape (`tuple`): tuple nrow and ncol.  Either `zn_array` or `shape`
            must be passed
        longnames (`bool`): flag to use longer names that exceed 12 chars in length.
            Default is False.
        fill_value (`str`): value to fill locations where `zn_array` is zero or less.
            Default is "1.0".

    Returns:
        `pandas.DataFrame`: a dataframe with parameter information

    Note:
        This function is used during the `PstFrom` setup process

    """

    if shape is None and zn_array is None:
        raise Exception("must pass either zn_array or shape")
    elif shape is None:
        shape = zn_array.shape

    parnme = []
    zone = []
    with open(tpl_file, "w") as f:
        f.write("ptf ~\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if zn_array is not None and zn_array[i, j] < 1:
                    pname = " {0}  ".format(fill_value)
                else:
                    zval = 1
                    if zn_array is not None:
                        zval = zn_array[i, j]
                    if longnames:
                        pname = "{0}_zone:{1}_{2}".format(name, zval, suffix)
                    else:

                        pname = "{0}_zn{1}".format(name, zval)
                        if len(pname) > 12:
                            warnings.warn(
                                "zone pname too long for pest:{0}".format(pname)
                            )
                    parnme.append(pname)
                    zone.append(zval)
                    pname = " ~   {0}    ~".format(pname)
                f.write(pname)
            f.write("\n")
    df = pd.DataFrame({"parnme": parnme, "zone": zone}, index=parnme)
    df.loc[:, "pargp"] = "{0}_{1}".format(suffix.replace("_", ""), name)
    return df


def build_jac_test_csv(pst, num_steps, par_names=None, forward=True):
    """build a dataframe of jactest inputs for use with pestpp-swp

    Args:
        pst (`pyemu.Pst`): existing control file
        num_steps (`int`): number of pertubation steps for each parameter
        par_names [`str`]: list of parameter names of pars to test.
            If None, all adjustable pars are used. Default is None
        forward (`bool`): flag to start with forward pertubations.
            Default is True

    Returns:
        `pandas.DataFrame`: the sequence of model runs to evaluate
        for the jactesting.


    """
    if isinstance(pst, str):
        pst = pyemu.Pst(pst)
    # pst.add_transform_columns()
    pst.build_increments()
    incr = pst.parameter_data.increment.to_dict()
    irow = 0
    par = pst.parameter_data
    if par_names is None:
        par_names = pst.adj_par_names
    total_runs = num_steps * len(par_names) + 1
    idx = ["base"]
    for par_name in par_names:
        idx.extend(["{0}_{1}".format(par_name, i) for i in range(num_steps)])
    df = pd.DataFrame(index=idx, columns=pst.par_names)
    li = par.partrans == "log"
    lbnd = par.parlbnd.copy()
    ubnd = par.parubnd.copy()
    lbnd.loc[li] = lbnd.loc[li].apply(np.log10)
    ubnd.loc[li] = ubnd.loc[li].apply(np.log10)
    lbnd = lbnd.to_dict()
    ubnd = ubnd.to_dict()

    org_vals = par.parval1.copy()
    org_vals.loc[li] = org_vals.loc[li].apply(np.log10)
    if forward:
        sign = 1.0
    else:
        sign = -1.0

    # base case goes in as first row, no perturbations
    df.loc["base", pst.par_names] = par.parval1.copy()
    irow = 1
    full_names = ["base"]
    for jcol, par_name in enumerate(par_names):
        org_val = org_vals.loc[par_name]
        last_val = org_val
        for step in range(num_steps):
            vals = org_vals.copy()
            i = incr[par_name]

            val = last_val + (sign * incr[par_name])
            if val > ubnd[par_name]:
                sign = -1.0
                val = org_val + (sign * incr[par_name])
                if val < lbnd[par_name]:
                    raise Exception("parameter {0} went out of bounds".format(par_name))
            elif val < lbnd[par_name]:
                sign = 1.0
                val = org_val + (sign * incr[par_name])
                if val > ubnd[par_name]:
                    raise Exception("parameter {0} went out of bounds".format(par_name))

            vals.loc[par_name] = val
            vals.loc[li] = 10 ** vals.loc[li]
            df.loc[idx[irow], pst.par_names] = vals
            full_names.append(
                "{0}_{1:<15.6E}".format(par_name, vals.loc[par_name]).strip()
            )

            irow += 1
            last_val = val
    df.index = full_names
    return df


def _write_df_tpl(filename, df, sep=",", tpl_marker="~", headerlines=None, **kwargs):
    """function write a pandas dataframe to a template file."""
    if "line_terminator" not in kwargs:
        if "win" in platform.platform().lower():
            kwargs["line_terminator"] = "\n"
    with open(filename, "w") as f:
        f.write("ptf {0}\n".format(tpl_marker))
        f.flush()
        if headerlines is not None:
            _add_headerlines(f, headerlines)
        df.to_csv(f, sep=sep, mode="a", **kwargs)


def _add_headerlines(f, headerlines):
    lc = 0
    for key in sorted(headerlines.keys()):
        if key > lc:
            lc += 1
            continue
            # TODO if we want to preserve mid-table comments,
            #  these lines might help - will also need to
            #  pass comment_char through so it can be
            #  used by the apply methods
            # to = key - lc
            # df.iloc[fr:to].to_csv(
            #     fp, sep=',', mode='a', header=hheader, # todo - presence of header may cause an issue with this
            #     **kwargs)
            # lc += to - fr
            # fr = to
        f.write(headerlines[key])
        f.flush()
        lc += 1


def setup_fake_forward_run(
    pst, new_pst_name, org_cwd=".", bak_suffix="._bak", new_cwd="."
):
    """setup a fake forward run for a pst.

    Args:
        pst (`pyemu.Pst`): existing control file
        new_pst_name (`str`): new control file to write
        org_cwd (`str`): existing working dir.  Default is "."
        bak_suffix (`str`, optional): suffix to add to existing
            model output files when making backup copies.
        new_cwd (`str`): new working dir.  Default is ".".

    Note:
        The fake forward run simply copies existing backup versions of
        model output files to the outfiles pest(pp) is looking
        for.  This is really a development option for debugging
        PEST++ issues.

    """

    if new_cwd != org_cwd and not os.path.exists(new_cwd):
        os.mkdir(new_cwd)
    pairs = {}

    for output_file in pst.output_files:
        org_pth = os.path.join(org_cwd, output_file)
        new_pth = os.path.join(new_cwd, os.path.split(output_file)[-1])
        assert os.path.exists(org_pth), org_pth
        shutil.copy2(org_pth, new_pth + bak_suffix)
        pairs[output_file] = os.path.split(output_file)[-1] + bak_suffix

    if new_cwd != org_cwd:
        for files in [pst.template_files, pst.instruction_files]:
            for f in files:
                raw = os.path.split(f)
                if len(raw[0]) == 0:
                    raw = raw[1:]
                if len(raw) > 1:
                    pth = os.path.join(*raw[:-1])
                    pth = os.path.join(new_cwd, pth)
                    if not os.path.exists(pth):
                        os.makedirs(pth)

                org_pth = os.path.join(org_cwd, f)
                new_pth = os.path.join(new_cwd, f)
                assert os.path.exists(org_pth), org_pth
                shutil.copy2(org_pth, new_pth)
        for f in pst.input_files:
            raw = os.path.split(f)
            if len(raw[0]) == 0:
                raw = raw[1:]
            if len(raw) > 1:
                pth = os.path.join(*raw[:-1])
                pth = os.path.join(new_cwd, pth)
                if not os.path.exists(pth):
                    os.makedirs(pth)

        for key, f in pst.pestpp_options.items():
            if not isinstance(f, str):
                continue
            raw = os.path.split(f)
            if len(raw[0]) == 0:
                raw = raw[1:]
            if len(raw) > 1:
                pth = os.path.join(*raw[:-1])
                pth = os.path.join(new_cwd, pth)
                if not os.path.exists(pth):
                    os.makedirs(pth)
            org_pth = os.path.join(org_cwd, f)
            new_pth = os.path.join(new_cwd, f)

            if os.path.exists(org_pth):
                shutil.copy2(org_pth, new_pth)

    with open(os.path.join(new_cwd, "fake_forward_run.py"), "w") as f:
        f.write("import os\nimport shutil\n")
        for org, bak in pairs.items():
            f.write("shutil.copy2(r'{0}',r'{1}')\n".format(bak, org))
    pst.model_command = "python fake_forward_run.py"
    pst.write(os.path.join(new_cwd, new_pst_name))

    return pst


def setup_temporal_diff_obs(
    pst,
    ins_file,
    out_file=None,
    include_zero_weight=False,
    include_path=False,
    sort_by_name=True,
    long_names=True,
    prefix="dif",
):
    """a helper function to setup difference-in-time observations based on an existing
    set of observations in an instruction file using the observation grouping in the
    control file

    Args:
        pst (`pyemu.Pst`): existing control file
        ins_file (`str`): an existing instruction file
        out_file (`str`, optional): an existing model output file that corresponds to
            the instruction file.  If None, `ins_file.replace(".ins","")` is used
        include_zero_weight (`bool`, optional): flag to include zero-weighted observations
            in the difference observation process.  Default is False so that only non-zero
            weighted observations are used.
        include_path (`bool`, optional): flag to setup the binary file processing in directory where the hds_file
            is located (if different from where python is running).  This is useful for setting up
            the process in separate directory for where python is running.
        sort_by_name (`bool`,optional): flag to sort observation names in each group prior to setting up
            the differencing.  The order of the observations matters for the differencing.  If False, then
            the control file order is used.  If observation names have a datetime suffix, make sure the format is
            year-month-day to use this sorting.  Default is True
        long_names (`bool`, optional): flag to use long, descriptive names by concating the two observation names
            that are being differenced.  This will produce names that are too long for tradtional PEST(_HP).
            Default is True.
        prefix (`str`, optional): prefix to prepend to observation names and group names.  Default is "dif".

    Returns:
        tuple containing

        - **str**: the forward run command to execute the binary file process during model runs.

        - **pandas.DataFrame**: a dataframe of observation information for use in the pest control file

    Note:
        This is the companion function of `helpers.apply_temporal_diff_obs()`.



    """
    if not os.path.exists(ins_file):
        raise Exception(
            "setup_temporal_diff_obs() error: ins_file '{0}' not found".format(ins_file)
        )
    # the ins routines will check for missing obs, etc
    try:
        ins = pyemu.pst_utils.InstructionFile(ins_file, pst)
    except Exception as e:
        raise Exception(
            "setup_temporal_diff_obs(): error processing instruction file: {0}".format(
                str(e)
            )
        )

    if out_file is None:
        out_file = ins_file.replace(".ins", "")

    # find obs groups from the obs names in the ins that have more than one observation
    # (cant diff single entry groups)
    obs = pst.observation_data
    if include_zero_weight:
        group_vc = pst.observation_data.loc[ins.obs_name_set, "obgnme"].value_counts()
    else:

        group_vc = obs.loc[
            obs.apply(lambda x: x.weight > 0 and x.obsnme in ins.obs_name_set, axis=1),
            "obgnme",
        ].value_counts()
    groups = list(group_vc.loc[group_vc > 1].index)
    if len(groups) == 0:
        raise Exception(
            "setup_temporal_diff_obs() error: no obs groups found "
            + "with more than one non-zero weighted obs"
        )

    # process each group
    diff_dfs = []
    for group in groups:
        # get a sub dataframe with non-zero weighted obs that are in this group and in the instruction file
        obs_group = obs.loc[obs.obgnme == group, :].copy()
        obs_group = obs_group.loc[
            obs_group.apply(
                lambda x: x.weight > 0 and x.obsnme in ins.obs_name_set, axis=1
            ),
            :,
        ]
        # sort if requested
        if sort_by_name:
            obs_group = obs_group.sort_values(by="obsnme", ascending=True)
        # the names starting with the first
        diff1 = obs_group.obsnme[:-1].values
        # the names ending with the last
        diff2 = obs_group.obsnme[1:].values
        # form a dataframe
        diff_df = pd.DataFrame({"diff1": diff1, "diff2": diff2})
        # build up some obs names
        if long_names:
            diff_df.loc[:, "obsnme"] = [
                "{0}_{1}__{2}".format(prefix, d1, d2) for d1, d2 in zip(diff1, diff2)
            ]
        else:
            diff_df.loc[:, "obsnme"] = [
                "{0}_{1}_{2}".format(prefix, group, c) for c in len(diff1)
            ]
        # set the obs names as the index (per usual)
        diff_df.index = diff_df.obsnme
        # set the group name for the diff obs
        diff_df.loc[:, "obgnme"] = "{0}_{1}".format(prefix, group)
        # set the weights using the standard prop of variance formula
        d1_std, d2_std = (
            1.0 / obs_group.weight[:-1].values,
            1.0 / obs_group.weight[1:].values,
        )
        diff_df.loc[:, "weight"] = 1.0 / (np.sqrt((d1_std ** 2) + (d2_std ** 2)))

        diff_dfs.append(diff_df)
    # concat all the diff dataframes
    diff_df = pd.concat(diff_dfs)

    # save the dataframe as a config file
    config_file = ins_file.replace(".ins", ".diff.config")

    f = open(config_file, "w")
    if include_path:
        # ins_path = os.path.split(ins_file)[0]
        # f = open(os.path.join(ins_path,config_file),'w')
        f.write(
            "{0},{1}\n".format(os.path.split(ins_file)[-1], os.path.split(out_file)[-1])
        )
        # diff_df.to_csv(os.path.join(ins_path,config_file))
    else:
        f.write("{0},{1}\n".format(ins_file, out_file))
        # diff_df.to_csv(os.path.join(config_file))

    f.flush()
    diff_df.to_csv(f, mode="a")
    f.flush()
    f.close()

    # write the instruction file
    diff_ins_file = config_file.replace(".config", ".processed.ins")
    with open(diff_ins_file, "w") as f:
        f.write("pif ~\n")
        f.write("l1 \n")
        for oname in diff_df.obsnme:
            f.write("l1 w w w !{0}! \n".format(oname))

    if include_path:
        config_file = os.path.split(config_file)[-1]
        diff_ins_file = os.path.split(diff_ins_file)[-1]

    # if the corresponding output file exists, try to run the routine
    if os.path.exists(out_file):
        if include_path:
            b_d = os.getcwd()
            ins_path = os.path.split(ins_file)[0]
            os.chdir(ins_path)
        # try:
        processed_df = apply_temporal_diff_obs(config_file=config_file)
        # except Exception as e:
        # if include_path:
        #     os.chdir(b_d)
        #

        # ok, now we can use the new instruction file to process the diff outputs
        ins = pyemu.pst_utils.InstructionFile(diff_ins_file)
        ins_pro_diff_df = ins.read_output_file(diff_ins_file.replace(".ins", ""))

        if include_path:
            os.chdir(b_d)
        print(ins_pro_diff_df)
        diff_df.loc[ins_pro_diff_df.index, "obsval"] = ins_pro_diff_df.obsval
    frun_line = "pyemu.helpers.apply_temporal_diff_obs('{0}')\n".format(config_file)
    return frun_line, diff_df


def apply_temporal_diff_obs(config_file):
    """process an instruction-output file pair and formulate difference observations.

    Args:
        config_file (`str`): configuration file written by `pyemu.helpers.setup_temporal_diff_obs`.

    Returns:
        diff_df (`pandas.DataFrame`) : processed difference observations

    Note:
        Writes `config_file.replace(".config",".processed")` output file that can be read
        with the instruction file that is created by `pyemu.helpers.setup_temporal_diff_obs()`.

        This is the companion function of `helpers.setup_setup_temporal_diff_obs()`.
    """

    if not os.path.exists(config_file):
        raise Exception(
            "apply_temporal_diff_obs() error: config_file '{0}' not found".format(
                config_file
            )
        )
    with open(config_file, "r") as f:
        line = f.readline().strip().split(",")
        ins_file, out_file = line[0], line[1]
        diff_df = pd.read_csv(f)
    if not os.path.exists(out_file):
        raise Exception(
            "apply_temporal_diff_obs() error: out_file '{0}' not found".format(out_file)
        )
    if not os.path.exists(ins_file):
        raise Exception(
            "apply_temporal_diff_obs() error: ins_file '{0}' not found".format(ins_file)
        )
    try:
        ins = pyemu.pst_utils.InstructionFile(ins_file)
    except Exception as e:
        raise Exception(
            "apply_temporal_diff_obs() error instantiating ins file: {0}".format(str(e))
        )
    try:
        out_df = ins.read_output_file(out_file)
    except Exception as e:
        raise Exception(
            "apply_temporal_diff_obs() error processing ins-out file pair: {0}".format(
                str(e)
            )
        )

    # make sure all the listed obs names in the diff_df are in the out_df
    diff_names = set(diff_df.diff1.to_list())
    diff_names.update(set(diff_df.diff2.to_list()))
    missing = diff_names - set(list(out_df.index.values))
    if len(missing) > 0:
        raise Exception(
            "apply_temporal_diff_obs() error: the following obs names in the config file "
            + "are not in the instruction file processed outputs :"
            + ",".join(missing)
        )
    diff_df.loc[:, "diff1_obsval"] = out_df.loc[diff_df.diff1.values, "obsval"].values
    diff_df.loc[:, "diff2_obsval"] = out_df.loc[diff_df.diff2.values, "obsval"].values
    diff_df.loc[:, "diff_obsval"] = diff_df.diff1_obsval - diff_df.diff2_obsval
    processed_name = config_file.replace(".config", ".processed")
    diff_df.loc[:, ["obsnme", "diff1_obsval", "diff2_obsval", "diff_obsval"]].to_csv(
        processed_name, sep=" ", index=False
    )
    return diff_df


# web address of spatial reference dot org
srefhttp = "https://spatialreference.org"


class SpatialReference(object):
    """
    a class to locate a structured model grid in x-y space.
    Lifted wholesale from Flopy, and preserved here...
    ...maybe slighlty over-engineered for here

    Args:

        delr (`numpy ndarray`): the model discretization delr vector (An array of spacings along a row)
        delc (`numpy ndarray`): the model discretization delc vector (An array of spacings along a column)
        lenuni (`int`): the length units flag from the discretization package. Default is 2.
        xul (`float`): The x coordinate of the upper left corner of the grid. Enter either xul and yul or xll and yll.
        yul (`float`): The y coordinate of the upper left corner of the grid. Enter either xul and yul or xll and yll.
        xll (`float`): The x coordinate of the lower left corner of the grid. Enter either xul and yul or xll and yll.
        yll (`float`): The y coordinate of the lower left corner of the grid. Enter either xul and yul or xll and yll.
        rotation (`float`): The counter-clockwise rotation (in degrees) of the grid
        proj4_str (`str`): a PROJ4 string that identifies the grid in space. warning: case sensitive!
        units (`string`): Units for the grid.  Must be either "feet" or "meters"
        epsg (`int`): EPSG code that identifies the grid in space. Can be used in lieu of
            proj4. PROJ4 attribute will auto-populate if there is an internet
            connection(via get_proj4 method).
            See https://www.epsg-registry.org/ or spatialreference.org
        length_multiplier (`float`): multiplier to convert model units to spatial reference units.
            delr and delc above will be multiplied by this value. (default=1.)


    """

    xul, yul = None, None
    xll, yll = None, None
    rotation = 0.0
    length_multiplier = 1.0
    origin_loc = "ul"  # or ll

    defaults = {
        "xul": None,
        "yul": None,
        "rotation": 0.0,
        "proj4_str": None,
        "units": None,
        "lenuni": 2,
        "length_multiplier": None,
        "source": "defaults",
    }

    lenuni_values = {"undefined": 0, "feet": 1, "meters": 2, "centimeters": 3}
    lenuni_text = {v: k for k, v in lenuni_values.items()}

    def __init__(
        self,
        delr=np.array([]),
        delc=np.array([]),
        lenuni=2,
        xul=None,
        yul=None,
        xll=None,
        yll=None,
        rotation=0.0,
        proj4_str=None,
        epsg=None,
        prj=None,
        units=None,
        length_multiplier=None,
        source=None,
    ):

        for delrc in [delr, delc]:
            if isinstance(delrc, float) or isinstance(delrc, int):
                msg = (
                    "delr and delcs must be an array or sequences equal in "
                    "length to the number of rows/columns."
                )
                raise TypeError(msg)

        self.delc = np.atleast_1d(np.array(delc)).astype(
            np.float64
        )  # * length_multiplier
        self.delr = np.atleast_1d(np.array(delr)).astype(
            np.float64
        )  # * length_multiplier

        if self.delr.sum() == 0 or self.delc.sum() == 0:
            if xll is None or yll is None:
                msg = (
                    "Warning: no grid spacing. "
                    "Lower-left corner offset calculation methods requires "
                    "arguments for delr and delc. Origin will be set to "
                    "upper-left"
                )
                warnings.warn(msg, PyemuWarning)
                xll, yll = None, None
                # xul, yul = None, None

        self._lenuni = lenuni
        self._proj4_str = proj4_str
        #
        self._epsg = epsg
        # if epsg is not None:
        #     self._proj4_str = getproj4(self._epsg)
        # self.prj = prj
        # self._wkt = None
        # self.crs = CRS(prj=prj, epsg=epsg)

        self.supported_units = ["feet", "meters"]
        self._units = units
        self._length_multiplier = length_multiplier
        self._reset()
        self.set_spatialreference(xul, yul, xll, yll, rotation)

    @property
    def xll(self):
        if self.origin_loc == "ll":
            xll = self._xll if self._xll is not None else 0.0
        elif self.origin_loc == "ul":
            # calculate coords for lower left corner
            xll = self._xul - (
                np.sin(self.theta) * self.yedge[0] * self.length_multiplier
            )
        return xll

    @property
    def yll(self):
        if self.origin_loc == "ll":
            yll = self._yll if self._yll is not None else 0.0
        elif self.origin_loc == "ul":
            # calculate coords for lower left corner
            yll = self._yul - (
                np.cos(self.theta) * self.yedge[0] * self.length_multiplier
            )
        return yll

    @property
    def xul(self):
        if self.origin_loc == "ll":
            # calculate coords for upper left corner
            xul = self._xll + (
                np.sin(self.theta) * self.yedge[0] * self.length_multiplier
            )
        if self.origin_loc == "ul":
            # calculate coords for lower left corner
            xul = self._xul if self._xul is not None else 0.0
        return xul

    @property
    def yul(self):
        if self.origin_loc == "ll":
            # calculate coords for upper left corner
            yul = self._yll + (
                np.cos(self.theta) * self.yedge[0] * self.length_multiplier
            )

        if self.origin_loc == "ul":
            # calculate coords for lower left corner
            yul = self._yul if self._yul is not None else 0.0
        return yul

    @property
    def proj4_str(self):
        proj4_str = None
        if self._proj4_str is not None:
            if "epsg" in self._proj4_str.lower():
                if "init" not in self._proj4_str.lower():
                    proj4_str = "+init=" + self._proj4_str
                else:
                    proj4_str = self._proj4_str
                # set the epsg if proj4 specifies it
                tmp = [i for i in self._proj4_str.split() if "epsg" in i.lower()]
                self._epsg = int(tmp[0].split(":")[1])
            else:
                proj4_str = self._proj4_str
        elif self.epsg is not None:
            proj4_str = "+init=epsg:{}".format(self.epsg)
        return proj4_str

    @property
    def epsg(self):
        # don't reset the proj4 string here
        # because proj4 attribute may already be populated
        # (with more details than getprj would return)
        # instead reset proj4 when epsg is set
        # (on init or setattr)
        return self._epsg

    # @property
    # def wkt(self):
    #     if self._wkt is None:
    #         if self.prj is not None:
    #             with open(self.prj) as src:
    #                 wkt = src.read()
    #         elif self.epsg is not None:
    #             wkt = getprj(self.epsg)
    #         else:
    #             return None
    #         return wkt
    #     else:
    #         return self._wkt

    @property
    def lenuni(self):
        return self._lenuni

    def _parse_units_from_proj4(self):
        units = None
        try:
            # need this because preserve_units doesn't seem to be
            # working for complex proj4 strings.  So if an
            # epsg code was passed, we have no choice, but if a
            # proj4 string was passed, we can just parse it
            proj_str = self.proj4_str
            # if "EPSG" in self.proj4_str.upper():
            #     import pyproj
            #
            #     crs = pyproj.Proj(self.proj4_str,
            #                       preserve_units=True,
            #                       errcheck=True)
            #     proj_str = crs.srs
            # else:
            #     proj_str = self.proj4_str
            # http://proj4.org/parameters.html#units
            # from proj4 source code
            # "us-ft", "0.304800609601219", "U.S. Surveyor's Foot",
            # "ft", "0.3048", "International Foot",
            if "units=m" in proj_str:
                units = "meters"
            elif (
                "units=ft" in proj_str
                or "units=us-ft" in proj_str
                or "to_meters:0.3048" in proj_str
            ):
                units = "feet"
            return units
        except:
            if self.proj4_str is not None:
                print("   could not parse units from {}".format(self.proj4_str))

    @property
    def units(self):
        if self._units is not None:
            units = self._units.lower()
        else:
            units = self._parse_units_from_proj4()
        if units is None:
            # print("warning: assuming SpatialReference units are meters")
            units = "meters"
        assert units in self.supported_units
        return units

    @property
    def length_multiplier(self):
        """
        Attempt to identify multiplier for converting from
        model units to sr units, defaulting to 1.
        """
        lm = None
        if self._length_multiplier is not None:
            lm = self._length_multiplier
        else:
            if self.model_length_units == "feet":
                if self.units == "meters":
                    lm = 0.3048
                elif self.units == "feet":
                    lm = 1.0
            elif self.model_length_units == "meters":
                if self.units == "feet":
                    lm = 1 / 0.3048
                elif self.units == "meters":
                    lm = 1.0
            elif self.model_length_units == "centimeters":
                if self.units == "meters":
                    lm = 1 / 100.0
                elif self.units == "feet":
                    lm = 1 / 30.48
            else:  # model units unspecified; default to 1
                lm = 1.0
        return lm

    @property
    def model_length_units(self):
        return self.lenuni_text[self.lenuni]

    @property
    def bounds(self):
        """
        Return bounding box in shapely order.
        """
        xmin, xmax, ymin, ymax = self.get_extent()
        return xmin, ymin, xmax, ymax

    @staticmethod
    def load(namefile=None, reffile="usgs.model.reference"):
        """
        Attempts to load spatial reference information from
        the following files (in order):
        1) usgs.model.reference
        2) NAM file (header comment)
        3) SpatialReference.default dictionary
        """
        reffile = os.path.join(os.path.split(namefile)[0], reffile)
        d = SpatialReference.read_usgs_model_reference_file(reffile)
        if d is not None:
            return d
        d = SpatialReference.attribs_from_namfile_header(namefile)
        if d is not None:
            return d
        else:
            return SpatialReference.defaults

    @staticmethod
    def attribs_from_namfile_header(namefile):
        # check for reference info in the nam file header
        d = SpatialReference.defaults.copy()
        d["source"] = "namfile"
        if namefile is None:
            return None
        header = []
        with open(namefile, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                header.extend(
                    line.strip().replace("#", "").replace(",", ";").split(";")
                )

        for item in header:
            if "xul" in item.lower():
                try:
                    d["xul"] = float(item.split(":")[1])
                except:
                    print("   could not parse xul " + "in {}".format(namefile))
            elif "yul" in item.lower():
                try:
                    d["yul"] = float(item.split(":")[1])
                except:
                    print("   could not parse yul " + "in {}".format(namefile))
            elif "rotation" in item.lower():
                try:
                    d["rotation"] = float(item.split(":")[1])
                except:
                    print("   could not parse rotation " + "in {}".format(namefile))
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ":".join(item.split(":")[1:]).strip()
                    if proj4_str.lower() == "none":
                        proj4_str = None
                    d["proj4_str"] = proj4_str
                except:
                    print("   could not parse proj4_str " + "in {}".format(namefile))
            elif "start" in item.lower():
                try:
                    d["start_datetime"] = item.split(":")[1].strip()
                except:
                    print("   could not parse start " + "in {}".format(namefile))

            # spatial reference length units
            elif "units" in item.lower():
                d["units"] = item.split(":")[1].strip()
            # model length units
            elif "lenuni" in item.lower():
                d["lenuni"] = int(item.split(":")[1].strip())
            # multiplier for converting from model length units to sr length units
            elif "length_multiplier" in item.lower():
                d["length_multiplier"] = float(item.split(":")[1].strip())
        return d

    @staticmethod
    def read_usgs_model_reference_file(reffile="usgs.model.reference"):
        """
        read spatial reference info from the usgs.model.reference file
        https://water.usgs.gov/ogw/policy/gw-model/modelers-setup.html
        """

        ITMUNI = {
            0: "undefined",
            1: "seconds",
            2: "minutes",
            3: "hours",
            4: "days",
            5: "years",
        }
        itmuni_values = {v: k for k, v in ITMUNI.items()}

        d = SpatialReference.defaults.copy()
        d["source"] = "usgs.model.reference"
        # discard default to avoid confusion with epsg code if entered
        d.pop("proj4_str")
        if os.path.exists(reffile):
            with open(reffile) as fref:
                for line in fref:
                    if len(line) > 1:
                        if line.strip()[0] != "#":
                            info = line.strip().split("#")[0].split()
                            if len(info) > 1:
                                d[info[0].lower()] = " ".join(info[1:])
            d["xul"] = float(d["xul"])
            d["yul"] = float(d["yul"])
            d["rotation"] = float(d["rotation"])

            # convert the model.reference text to a lenuni value
            # (these are the model length units)
            if "length_units" in d.keys():
                d["lenuni"] = SpatialReference.lenuni_values[d["length_units"]]
            if "time_units" in d.keys():
                d["itmuni"] = itmuni_values[d["time_units"]]
            if "start_date" in d.keys():
                start_datetime = d.pop("start_date")
                if "start_time" in d.keys():
                    start_datetime += " {}".format(d.pop("start_time"))
                d["start_datetime"] = start_datetime
            if "epsg" in d.keys():
                try:
                    d["epsg"] = int(d["epsg"])
                except Exception as e:
                    raise Exception("error reading epsg code from file:\n" + str(e))
            # this prioritizes epsg over proj4 if both are given
            # (otherwise 'proj4' entry will be dropped below)
            elif "proj4" in d.keys():
                d["proj4_str"] = d["proj4"]

            # drop any other items that aren't used in sr class
            d = {
                k: v
                for k, v in d.items()
                if k.lower() in SpatialReference.defaults.keys()
                or k.lower() in {"epsg", "start_datetime", "itmuni", "source"}
            }
            return d
        else:
            return None

    def __setattr__(self, key, value):
        reset = True
        if key == "delr":
            super(SpatialReference, self).__setattr__(
                "delr", np.atleast_1d(np.array(value))
            )
        elif key == "delc":
            super(SpatialReference, self).__setattr__(
                "delc", np.atleast_1d(np.array(value))
            )
        elif key == "xul":
            super(SpatialReference, self).__setattr__("_xul", float(value))
            self.origin_loc = "ul"
        elif key == "yul":
            super(SpatialReference, self).__setattr__("_yul", float(value))
            self.origin_loc = "ul"
        elif key == "xll":
            super(SpatialReference, self).__setattr__("_xll", float(value))
            self.origin_loc = "ll"
        elif key == "yll":
            super(SpatialReference, self).__setattr__("_yll", float(value))
            self.origin_loc = "ll"
        elif key == "length_multiplier":
            super(SpatialReference, self).__setattr__(
                "_length_multiplier", float(value)
            )
        elif key == "rotation":
            super(SpatialReference, self).__setattr__("rotation", float(value))
        elif key == "lenuni":
            super(SpatialReference, self).__setattr__("_lenuni", int(value))
        elif key == "units":
            value = value.lower()
            assert value in self.supported_units
            super(SpatialReference, self).__setattr__("_units", value)
        elif key == "proj4_str":
            super(SpatialReference, self).__setattr__("_proj4_str", value)
            # reset the units and epsg
            units = self._parse_units_from_proj4()
            if units is not None:
                self._units = units
            self._epsg = None
        elif key == "epsg":
            super(SpatialReference, self).__setattr__("_epsg", value)
            # reset the units and proj4
            # self._units = None
            # self._proj4_str = getproj4(self._epsg)
            # self.crs = crs(epsg=value)
        elif key == "prj":
            super(SpatialReference, self).__setattr__("prj", value)
            # translation to proj4 strings in crs class not robust yet
            # leave units and proj4 alone for now.
            # self.crs = CRS(prj=value, epsg=self.epsg)
        else:
            super(SpatialReference, self).__setattr__(key, value)
            reset = False
        if reset:
            self._reset()

    def reset(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return

    def _reset(self):
        self._xgrid = None
        self._ygrid = None
        self._ycentergrid = None
        self._xcentergrid = None
        self._vertices = None
        return

    @property
    def nrow(self):
        return self.delc.shape[0]

    @property
    def ncol(self):
        return self.delr.shape[0]

    def __eq__(self, other):
        if not isinstance(other, SpatialReference):
            return False
        if other.xul != self.xul:
            return False
        if other.yul != self.yul:
            return False
        if other.rotation != self.rotation:
            return False
        if other.proj4_str != self.proj4_str:
            return False
        return True

    @classmethod
    def from_namfile(cls, namefile, delr=np.array([]), delc=np.array([])):
        if delr is None or delc is None:
            warnings.warn(
                "One or both of grid spacing information "
                "missing,\n    required for most pyemu methods "
                "that use sr,\n    can be passed later if desired "
                "(e.g. sr.delr = row spacing)",
                PyemuWarning,
            )
        attribs = SpatialReference.attribs_from_namfile_header(namefile)
        attribs["delr"] = delr
        attribs["delc"] = delc
        try:
            attribs.pop("start_datetime")
        except:
            print("   could not remove start_datetime")
        return SpatialReference(**attribs)

    @classmethod
    def from_gridspec(cls, gridspec_file, lenuni=0):
        f = open(gridspec_file, "r")
        raw = f.readline().strip().split()
        nrow = int(raw[0])
        ncol = int(raw[1])
        raw = f.readline().strip().split()
        xul, yul, rot = float(raw[0]), float(raw[1]), float(raw[2])
        delr = []
        j = 0
        while j < ncol:
            raw = f.readline().strip().split()
            for r in raw:
                if "*" in r:
                    rraw = r.split("*")
                    for n in range(int(rraw[0])):
                        delr.append(float(rraw[1]))
                        j += 1
                else:
                    delr.append(float(r))
                    j += 1
        delc = []
        i = 0
        while i < nrow:
            raw = f.readline().strip().split()
            for r in raw:
                if "*" in r:
                    rraw = r.split("*")
                    for n in range(int(rraw[0])):
                        delc.append(float(rraw[1]))
                        i += 1
                else:
                    delc.append(float(r))
                    i += 1
        f.close()
        return cls(
            np.array(delr), np.array(delc), lenuni, xul=xul, yul=yul, rotation=rot
        )

    @property
    def attribute_dict(self):
        return {
            "xul": self.xul,
            "yul": self.yul,
            "rotation": self.rotation,
            "proj4_str": self.proj4_str,
        }

    def set_spatialreference(
        self, xul=None, yul=None, xll=None, yll=None, rotation=0.0
    ):
        """
        set spatial reference - can be called from model instance

        """
        if xul is not None and xll is not None:
            msg = (
                "Both xul and xll entered. Please enter either xul, yul or " "xll, yll."
            )
            raise ValueError(msg)
        if yul is not None and yll is not None:
            msg = (
                "Both yul and yll entered. Please enter either xul, yul or " "xll, yll."
            )
            raise ValueError(msg)
        # set the origin priority based on the left corner specified
        # (the other left corner will be calculated).  If none are specified
        # then default to upper left
        if xul is None and yul is None and xll is None and yll is None:
            self.origin_loc = "ul"
            xul = 0.0
            yul = self.delc.sum()
        elif xll is not None:
            self.origin_loc = "ll"
        else:
            self.origin_loc = "ul"

        self.rotation = rotation
        self._xll = xll if xll is not None else 0.0
        self._yll = yll if yll is not None else 0.0
        self._xul = xul if xul is not None else 0.0
        self._yul = yul if yul is not None else 0.0
        return

    def __repr__(self):
        s = "xul:{0:<.10G}; yul:{1:<.10G}; rotation:{2:<G}; ".format(
            self.xul, self.yul, self.rotation
        )
        s += "proj4_str:{0}; ".format(self.proj4_str)
        s += "units:{0}; ".format(self.units)
        s += "lenuni:{0}; ".format(self.lenuni)
        s += "length_multiplier:{}".format(self.length_multiplier)
        return s

    @property
    def theta(self):
        return -self.rotation * np.pi / 180.0

    @property
    def xedge(self):
        return self.get_xedge_array()

    @property
    def yedge(self):
        return self.get_yedge_array()

    @property
    def xgrid(self):
        if self._xgrid is None:
            self._set_xygrid()
        return self._xgrid

    @property
    def ygrid(self):
        if self._ygrid is None:
            self._set_xygrid()
        return self._ygrid

    @property
    def xcenter(self):
        return self.get_xcenter_array()

    @property
    def ycenter(self):
        return self.get_ycenter_array()

    @property
    def ycentergrid(self):
        if self._ycentergrid is None:
            self._set_xycentergrid()
        return self._ycentergrid

    @property
    def xcentergrid(self):
        if self._xcentergrid is None:
            self._set_xycentergrid()
        return self._xcentergrid

    def _set_xycentergrid(self):
        self._xcentergrid, self._ycentergrid = np.meshgrid(self.xcenter, self.ycenter)
        self._xcentergrid, self._ycentergrid = self.transform(
            self._xcentergrid, self._ycentergrid
        )

    def _set_xygrid(self):
        self._xgrid, self._ygrid = np.meshgrid(self.xedge, self.yedge)
        self._xgrid, self._ygrid = self.transform(self._xgrid, self._ygrid)

    @staticmethod
    def rotate(x, y, theta, xorigin=0.0, yorigin=0.0):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        # jwhite changed on Oct 11 2016 - rotation is now positive CCW
        # theta = -theta * np.pi / 180.
        theta = theta * np.pi / 180.0

        xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * (y - yorigin)
        yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * (y - yorigin)
        return xrot, yrot

    def transform(self, x, y, inverse=False):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        if not inverse:
            x *= self.length_multiplier
            y *= self.length_multiplier
            x += self.xll
            y += self.yll
            x, y = SpatialReference.rotate(
                x, y, theta=self.rotation, xorigin=self.xll, yorigin=self.yll
            )
        else:
            x, y = SpatialReference.rotate(x, y, -self.rotation, self.xll, self.yll)
            x -= self.xll
            y -= self.yll
            x /= self.length_multiplier
            y /= self.length_multiplier
        return x, y

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        """
        x0 = self.xedge[0]
        x1 = self.xedge[-1]
        y0 = self.yedge[0]
        y1 = self.yedge[-1]

        # upper left point
        x0r, y0r = self.transform(x0, y0)

        # upper right point
        x1r, y1r = self.transform(x1, y0)

        # lower right point
        x2r, y2r = self.transform(x1, y1)

        # lower left point
        x3r, y3r = self.transform(x0, y1)

        xmin = min(x0r, x1r, x2r, x3r)
        xmax = max(x0r, x1r, x2r, x3r)
        ymin = min(y0r, y1r, y2r, y3r)
        ymax = max(y0r, y1r, y2r, y3r)

        return (xmin, xmax, ymin, ymax)

    def get_grid_lines(self):
        """
        Get the grid lines as a list

        """
        xmin = self.xedge[0]
        xmax = self.xedge[-1]
        ymin = self.yedge[-1]
        ymax = self.yedge[0]
        lines = []
        # Vertical lines
        for j in range(self.ncol + 1):
            x0 = self.xedge[j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            x0r, y0r = self.transform(x0, y0)
            x1r, y1r = self.transform(x1, y1)
            lines.append([(x0r, y0r), (x1r, y1r)])

        # horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.yedge[i]
            y1 = y0
            x0r, y0r = self.transform(x0, y0)
            x1r, y1r = self.transform(x1, y1)
            lines.append([(x0r, y0r), (x1r, y1r)])
        return lines

    # def get_grid_line_collection(self, **kwargs):
    #     """
    #     Get a LineCollection of the grid
    #
    #     """
    #     from flopy.plot import ModelMap
    #
    #     map = ModelMap(sr=self)
    #     lc = map.plot_grid(**kwargs)
    #     return lc

    def get_xcenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every column in the grid in model space - not offset or rotated.

        """
        assert self.delr is not None and len(self.delr) > 0, (
            "delr not passed to " "spatial reference object"
        )
        x = np.add.accumulate(self.delr) - 0.5 * self.delr
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid in model space - not offset of rotated.

        """
        assert self.delc is not None and len(self.delc) > 0, (
            "delc not passed to " "spatial reference object"
        )
        Ly = np.add.reduce(self.delc)
        y = Ly - (np.add.accumulate(self.delc) - 0.5 * self.delc)
        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid in model space - not offset
        or rotated.  Array is of size (ncol + 1)

        """
        assert self.delr is not None and len(self.delr) > 0, (
            "delr not passed to " "spatial reference object"
        )
        xedge = np.concatenate(([0.0], np.add.accumulate(self.delr)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid in model space - not offset or
        rotated. Array is of size (nrow + 1)

        """
        assert self.delc is not None and len(self.delc) > 0, (
            "delc not passed to " "spatial reference object"
        )
        length_y = np.add.reduce(self.delc)
        yedge = np.concatenate(([length_y], length_y - np.add.accumulate(self.delc)))
        return yedge

    def write_gridspec(self, filename):
        """write a PEST-style grid specification file"""
        f = open(filename, "w")
        f.write("{0:10d} {1:10d}\n".format(self.delc.shape[0], self.delr.shape[0]))
        f.write(
            "{0:15.6E} {1:15.6E} {2:15.6E}\n".format(
                self.xul * self.length_multiplier,
                self.yul * self.length_multiplier,
                self.rotation,
            )
        )

        for r in self.delr:
            f.write("{0:15.6E} ".format(r))
        f.write("\n")
        for c in self.delc:
            f.write("{0:15.6E} ".format(c))
        f.write("\n")
        return

    def get_vertices(self, i, j):
        """Get vertices for a single cell or sequence if i, j locations."""
        pts = []
        xgrid, ygrid = self.xgrid, self.ygrid
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i + 1, j], ygrid[i + 1, j]])
        pts.append([xgrid[i + 1, j + 1], ygrid[i + 1, j + 1]])
        pts.append([xgrid[i, j + 1], ygrid[i, j + 1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        if np.isscalar(i):
            return pts
        else:
            vrts = np.array(pts).transpose([2, 0, 1])
            return [v.tolist() for v in vrts]

    def get_rc(self, x, y):
        return self.get_ij(x, y)

    def get_ij(self, x, y):
        """Return the row and column of a point or sequence of points
        in real-world coordinates.

        Args:
            x (`float`): scalar or sequence of x coordinates
            y (`float`): scalar or sequence of y coordinates

        Returns:
            tuple of

            - **int** : row or sequence of rows (zero-based)
            - **int** : column or sequence of columns (zero-based)
        """
        if np.isscalar(x):
            c = (np.abs(self.xcentergrid[0] - x)).argmin()
            r = (np.abs(self.ycentergrid[:, 0] - y)).argmin()
        else:
            xcp = np.array([self.xcentergrid[0]] * (len(x)))
            ycp = np.array([self.ycentergrid[:, 0]] * (len(x)))
            c = (np.abs(xcp.transpose() - x)).argmin(axis=0)
            r = (np.abs(ycp.transpose() - y)).argmin(axis=0)
        return r, c

    @property
    def vertices(self):
        """
        Returns a list of vertices for
        """
        if self._vertices is None:
            self._set_vertices()
        return self._vertices

    def _set_vertices(self):
        """
        Populate vertices for the whole grid
        """
        jj, ii = np.meshgrid(range(self.ncol), range(self.nrow))
        jj, ii = jj.ravel(), ii.ravel()
        self._vertices = self.get_vertices(ii, jj)


def maha_based_pdc(sim_en):
    """prototype for detecting prior-data conflict following Alfonso and Oliver 2019

    Args:
        sim_en (`pyemu.ObservationEnsemble`): a simulated outputs ensemble

    Returns:

        tuple containing

        - **pandas.DataFrame**: 1-D subspace squared mahalanobis distances
            that exceed the `l1_crit_val` threshold
        - **pandas.DataFrame**: 2-D subspace squared mahalanobis distances
            that exceed the `l2_crit_val` threshold

    Note:
        Noise realizations are added to `sim_en` to account for measurement
            noise.



    """
    groups = sim_en.pst.nnz_obs_groups
    obs = sim_en.pst.observation_data
    z_scores = {}
    dm_xs = {}
    p_vals = {}
    for group in groups:
        nzobs = obs.loc[obs.obgnme==group,:]
        nzobs = nzobs.loc[nzobs.weight > 0,:].copy()
        nzsim_en = sim_en._df.loc[:,nzobs.obsnme].copy()
        ne,nx = nzsim_en.shape
        ns = ne - 1
        delta = 2.0/(float(ns) + 2.)
        v = nzsim_en.var(axis=0).mean()
        x = nzsim_en.values.copy()
        z = nzobs.obsval.values.copy()
        dm_x,dm_z = [],[]
        for ireal in range(ne):
            x_s = x.copy()
            x_s = np.delete(x_s,(ireal),axis=0)
            first = delta * v * (((ns -1)/(1-delta))*np.eye(ns))
            a_s = first + np.dot(x_s, x_s.T)
            lower = np.linalg.cholesky(a_s)
            lower = np.linalg.inv(lower)
            mu_hat = x_s.mean(axis=0)
            dm_x.append(_maha(delta,v,x_s,x[ireal,:] - mu_hat,lower))
            dm_z.append(_maha(delta,v,x_s,z - mu_hat,lower))
        dm_x = np.array(dm_x)
        dm_z = np.array(dm_z)
        mu_x = np.median(dm_x)
        mu_z = np.median(dm_z)
        mad = np.median(np.abs(dm_x - mu_x))
        sigma_x = 1.4826 * mad
        z_score = (mu_z - mu_x) / sigma_x
        z_scores[group] = z_score
        dm_xs[group] = dm_x
        dm_x.sort()

        p = np.argmin(np.abs(dm_x - mu_z))/dm_x.shape[0]
        p_vals[group] = 1 - p

    z_scores, p_vals, dm_xs = pd.Series(z_scores), pd.Series(p_vals), pd.DataFrame(dm_xs)
    # dm_xs.loc[:,"z_scores"] = z_scores.loc[dm_xs.index]
    # dm_xs.loc[:,"p_vals"] = p_vals.loc[dm_xs.index]
    df = pd.DataFrame({"z_scores":z_scores,"p_vals":p_vals})
    df.index.name = "obgnme"
    return df,pd.DataFrame(dm_xs)

def _maha(delta,v,x,z,lower_inv):

    d_m = np.dot(z.transpose(),z)
    first = np.dot(np.dot(lower_inv,x),z)
    first = np.dot(first.transpose(),first)
    d_m = (1.0/(delta * v)) * (d_m - first)
    return d_m


def get_maha_obs_summary(sim_en, l1_crit_val=6.34, l2_crit_val=9.2):
    """calculate the 1-D and 2-D mahalanobis distance between simulated
    ensemble and observed values.  Used for detecting prior-data conflict

    Args:
        sim_en (`pyemu.ObservationEnsemble`): a simulated outputs ensemble
        l1_crit_val (`float`): the chi squared critical value for the 1-D
            mahalanobis distance.  Default is 6.4 (p=0.01,df=1)
        l2_crit_val (`float`): the chi squared critical value for the 2-D
            mahalanobis distance.  Default is 9.2 (p=0.01,df=2)

    Returns:

        tuple containing

        - **pandas.DataFrame**: 1-D subspace squared mahalanobis distances
            that exceed the `l1_crit_val` threshold
        - **pandas.DataFrame**: 2-D subspace squared mahalanobis distances
            that exceed the `l2_crit_val` threshold

    Note:
        Noise realizations are added to `sim_en` to account for measurement
            noise.

    """

    if not isinstance(sim_en, pyemu.ObservationEnsemble):
        raise Exception("'sim_en' must be a " + " pyemu.ObservationEnsemble instance")
    if sim_en.pst.nnz_obs < 1:
        raise Exception(" at least one non-zero weighted obs is needed")

    # process the simulated ensemblet to only have non-zero weighted obs
    obs = sim_en.pst.observation_data
    nz_names = sim_en.pst.nnz_obs_names
    # get the full cov matrix
    nz_cov_df = sim_en.covariance_matrix().to_dataframe()
    nnz_en = sim_en.loc[:, nz_names].copy()
    nz_cov_df = nz_cov_df.loc[nz_names, nz_names]
    # get some noise realizations
    nnz_en.reseed()
    obsmean = obs.loc[nnz_en.columns.values, "obsval"]
    noise_en = pyemu.ObservationEnsemble.from_gaussian_draw(
        sim_en.pst, num_reals=sim_en.shape[0]
    )
    noise_en -= obsmean  # subtract off the obs val bc we just want the noise
    noise_en.index = nnz_en.index
    nnz_en += noise_en

    # obsval_dict = obs.loc[nnz_en.columns.values,"obsval"].to_dict()

    # first calculate the 1-D subspace maha distances
    print("calculating L-1 maha distances")
    sim_mean = nnz_en.mean()
    obs_mean = obs.loc[nnz_en.columns.values, "obsval"]
    simvar_inv = 1.0 / (nnz_en.std() ** 2)
    res_mean = sim_mean - obs_mean
    l1_maha_sq_df = res_mean ** 2 * simvar_inv
    l1_maha_sq_df = l1_maha_sq_df.loc[l1_maha_sq_df > l1_crit_val]
    # now calculate the 2-D subspace maha distances
    print("preparing L-2 maha distance containers")
    manager = mp.Manager()
    ns = manager.Namespace()
    results = manager.dict()
    mean = manager.dict(res_mean.to_dict())
    var = manager.dict()
    cov = manager.dict()
    var_arr = np.diag(nz_cov_df.values)
    for i1, o1 in enumerate(nz_names):
        var[o1] = var_arr[i1]

        cov_vals = nz_cov_df.loc[o1, :].values[i1 + 1 :]
        ostr_vals = ["{0}_{1}".format(o1, o2) for o2 in nz_names[i1 + 1 :]]
        cd = {o: c for o, c in zip(ostr_vals, cov_vals)}
        cov.update(cd)
    print("starting L-2 maha distance parallel calcs")
    # pool = mp.Pool(processes=5)
    with mp.get_context("spawn").Pool(
            processes=min(mp.cpu_count(), 60)) as pool:
        for i1, o1 in enumerate(nz_names):
            o2names = [o2 for o2 in nz_names[i1 + 1 :]]
            rresults = [
                pool.apply_async(
                    _l2_maha_worker,
                    args=(o1, o2names, mean, var, cov, results, l2_crit_val),
                )
            ]
        [r.get() for r in rresults]

        print("closing pool")
        pool.close()

        print("joining pool")
        pool.join()

    # print(results)
    # print(len(results),len(ostr_vals))

    keys = list(results.keys())
    onames1 = [k.split("|")[0] for k in keys]
    onames2 = [k.split("|")[1] for k in keys]
    l2_maha_sq_vals = [results[k] for k in keys]
    l2_maha_sq_df = pd.DataFrame(
        {"obsnme_1": onames1, "obsnme_2": onames2, "sq_distance": l2_maha_sq_vals}
    )

    return l1_maha_sq_df, l2_maha_sq_df


def _l2_maha_worker(o1, o2names, mean, var, cov, results, l2_crit_val):

    rresults = {}
    v1 = var[o1]
    c = np.zeros((2, 2))
    c[0, 0] = v1
    r1 = mean[o1]
    for o2 in o2names:
        ostr = "{0}_{1}".format(o1, o2)
        cv = cov[ostr]
        v2 = var[o2]
        c[1, 1] = v2
        c[0, 1] = cv
        c[1, 0] = cv
        c_inv = np.linalg.inv(c)

        r2 = mean[o2]
        r_vec = np.array([r1, r2])
        l2_maha_sq_val = np.dot(np.dot(r_vec, c_inv), r_vec.transpose())
        if l2_maha_sq_val > l2_crit_val:
            rresults[ostr] = l2_maha_sq_val
    results.update(rresults)
    print(o1, "done")


def parse_rmr_file(rmr_file):
    """parse a run management record file into a data frame of tokens

    Args:
        rmr_file (`str`):  an rmr file name

    Returns:
        pd.DataFrame: a dataframe of timestamped information

    Note:
        only supports rmr files generated by pest++ version >= 5.1.21

    """

    if not os.path.exists(rmr_file):
        raise FileExistsError("rmr file not found")
    data = {}
    dts = []
    with open(rmr_file,'r') as f:
        for line in f:
            if "->" in line:
                raw = line.split("->")
                dt = datetime.strptime(raw[0],"%m/%d/%y %H:%M:%S")
                tokens = raw[1].split()
                colon_token = [ t for t in tokens if ":" in t]
                if len(colon_token) > 0:
                    idx = tokens.index(colon_token[0])
                    action = "_".join(tokens[:idx+1])
                else:
                    action = '_'.join(tokens)
                if "action" not in data:
                    data["action"] = []
                data["action"].append(action)

                tokens = colon_token
                tokens = {t.split(":")[0]:t.split(':')[1] for t in tokens}
                for t in tokens:
                    if t not in data:
                        data[t] = [np.nan] * len(dts)
                for k in data:
                    if k == "action":
                        continue
                    if k not in tokens:
                        data[k].append(np.nan)
                    else:
                        data[k].append(tokens[k])
                dts.append(dt)

    df = pd.DataFrame(data,index=dts)
    return df






