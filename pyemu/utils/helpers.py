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
from ast import literal_eval
import traceback
import re
import numpy as np
import pandas as pd
import math

pd.options.display.max_colwidth = 100
from ..pyemu_warnings import PyemuWarning


try:
    import flopy
except:
    pass

import pyemu
from pyemu.utils.os_utils import run, start_workers,PyPestWorker


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


def _try_pdcol_numeric(x, first=True, intadj=0, **kwargs):
    try:
        x = pd.to_numeric(x, errors="raise", **kwargs)
        if intadj != 0:
            if first:
                x = pd.Series([xx + intadj if isinstance(xx, (int, np.integer)) else xx for xx in x])
            else:
                x = x + intadj if isinstance(x, (int, np.integer)) else x
    except ValueError as e:
        if first:
            x = x.apply(_try_pdcol_numeric, first=False, intadj=intadj, **kwargs)
        else:
            pass
    return x


def autocorrelated_draw(pst,struct_dict,time_distance_col="distance",num_reals=100,verbose=True,
                        enforce_bounds=False, draw_ineq=False):
    """construct an autocorrelated observation noise ensemble from covariance matrices
        implied by geostatistical structure(s).

        Args:
            pst (`pyemu.Pst`): a control file (or the name of control file).  The
                information in the `* observation data` dataframe is used extensively,
                including weight, standard_deviation (if present), upper_bound/lower_bound (if present).
            time_distance_col (str): the column in `* observation_data` that represents the distance in time
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
    keys = list(struct_dict.keys())
    keys.sort()
    #for gs,onames in struct_dict.items():
    for gs in keys:
        onames = struct_dict[gs]
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
        oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst,gcov,num_reals=num_reals,fill=False,by_groups=False)
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
            lb_dict.update(pst.observation_data.lower_bound.fillna(-1.0e300).to_dict())
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
        #lt_tags = pst.get_constraint_tags("lt")
        #lt_onames = [oname for oname,ogrp in zip(obs.obsnme,obs.obgnme) if True in [True if str(ogrp).startswith(tag) else False for tag in lt_tags]  ]
        lt_onames = pst.less_than_obs_constraints.to_list()
        if verbose:
            print("--> less than ineq obs:",lt_onames)
        lt_dict = obs.loc[lt_onames,"obsval"].to_dict()
        for n,v in lt_dict.items():
            full_oe.loc[:,n] = v
        obs = pst.observation_data
        #gt_tags = pst.get_constraint_tags("gt")
        #gt_onames = [oname for oname, ogrp in zip(obs.obsnme, obs.obgnme) if
        #             True in [True if str(ogrp).startswith(tag) else False for tag in gt_tags]]
        gt_onames = pst.greater_than_obs_constraints.to_list()
        if verbose:
            print("--> greater than ineq obs:", gt_onames)
        gt_dict = obs.loc[gt_onames, "obsval"].to_dict()
        for n, v in gt_dict.items():
            full_oe.loc[:, n] = v
    return full_oe


def draw_by_group(pst, num_reals=100, sigma_range=6, use_specsim=False,
                  struct_dict=None, delr=None, delc=None, scale_offset=True,
                  echo=True, logger=False):
    """Draw a parameter ensemble from the distribution implied by the initial parameter values in the
    control file and a prior parameter covariance matrix derived from grouped geostructures.
    Previously in pst_from.

    Args:
        pst (`pyemu.Pst`): a control file instance
        num_reals (`int`): the number of realizations to draw
        sigma_range (`int`): number of standard deviations represented by parameter bounds.  Default is 6 (99%
            confidence).  4 would be approximately 95% confidence bounds
        use_specsim (`bool`): flag to use spectral simulation for grid-scale pars (highly recommended).
            Default is False
        struct_dict (`dict`): a dict with keys of GeoStruct (or structure file).
            Dictionary values can depend on the values of `use_specsim`.
            If `use_specsim` is True, values are expected to be `list[pd.DataFrame]`
            with dataframes indexed by parameter names and containing columns
            [`parval1`, `pargp`, `i`, `j`, `partype`], where `i` and `j` are
            grid index locations of grid parameters, and `partype` is used to
            indicate grid type parameters. The draw will be independent for each
            unique `pargp`.
            If `use_specsim` is False, dictionary values are expected to be
            `list[pd.DataFrame]` with dataframes indexed by parameter names and
            containing columns ["x", "y", "parnme"], the optional `zone` column
            can be used to draw realizations independently for different zones.
            Alternatively, if use_specsim` is False dictionary keys and values
            can be paths to external structure files and pilotpoint or tpl files,
            respectively.
        delr (`list`, optional): required for specsim (`use_specsim` is True),
            dimension of cells along a row (i.e., column widths), specsim only
            works with regular grids
        delc (`list`, optional):  required for specsim (`use_specsim` is True),
            dimension of cells along a column (i.e., row heights)
        scale_offset (`bool`): flag to apply scale and offset to parameter bounds before calculating prior variance.
            Dfault is True.  If you are using non-default scale and/or offset and you get an exception during
            draw, try changing this value to False.
        echo (`bool`): Verbosity flag passed to new Logger instance if
            `logger`is None
        logger (`pyemu.Logger`, optional): Object for logging process

    Returns:
        `pyemu.ParameterEnsemble`: a prior parameter ensemble

    Note:
        This method draws by parameter group

        If you are using grid-style parameters, please use spectral simulation (`use_specsim=True`)

    """
    if delc is None:
        delc = []
    if delr is None:
        delr = []
    if struct_dict is None:
        struct_dict = {}
    if not logger:
        logger = pyemu.Logger("draw_by_groups.log", echo=echo)
    if pst.npar_adj == 0:
        logger.warn("no adjustable parameters, nothing to draw...")
        return
    # list for holding grid style groups
    gr_pe_l = []
    subset = pst.parameter_data.index
    gr_par_pe = None
    if use_specsim:
        if len(delr)>0 and len(delc)>0 and not pyemu.geostats.SpecSim2d.grid_is_regular(
            delr, delc
        ):
            logger.lraise(
                "draw() error: can't use spectral simulation with irregular grid"
            )
        logger.log("spectral simulation for grid-scale pars")
        # loop over geostructures defined in PestFrom object
        # (setup through add_parameters)
        for geostruct, par_df_l in struct_dict.items():
            par_df = pd.concat(par_df_l)  # force to single df
            if not 'partype' in par_df.columns:
                logger.warn(
                    f"draw() error: use_specsim is {use_specsim} but no column named"
                    f"'partype' to indicate grid based pars in geostruct {geostruct}"
                    f"not using specsim"
                )
            else:
                par_df = par_df.loc[par_df.partype == "grid", :]
                if "i" in par_df.columns:  # need 'i' and 'j' for specsim
                    grd_p = pd.notna(par_df.i)
                else:
                    grd_p = np.array([0])
                # if there are grid pars (also grid pars with i,j info)
                if grd_p.sum() > 0:
                    # select pars to use specsim for
                    gr_df = par_df.loc[grd_p]
                    gr_df = gr_df.astype({"i": int, "j": int})  # make sure int
                    # (won't be if there were nans in concatenated df)
                    if len(gr_df) > 0:
                        # get specsim object for this geostruct
                        ss = pyemu.geostats.SpecSim2d(
                            delx=delr,
                            dely=delc,
                            geostruct=geostruct,
                        )
                        # specsim draw (returns df)
                        gr_pe1 = ss.grid_par_ensemble_helper(
                            pst=pst,
                            gr_df=gr_df,
                            num_reals=num_reals,
                            sigma_range=sigma_range,
                            logger=logger,
                        )
                        # append to list of specsim drawn pars
                        gr_pe_l.append(gr_pe1)
                        # rebuild struct_dict entry for this geostruct
                        # to not include specsim pars
                        struct_dict[geostruct] = []
                        # loop over all in list associated with geostruct
                        for p_df in par_df_l:
                            # if pars are not in the specsim pars just created
                            # assign them to this struct_dict entry
                            # needed if none specsim pars are linked to same geostruct
                            if not p_df.index.isin(gr_df.index).all():
                                struct_dict[geostruct].append(p_df)
                            else:
                                subset = subset.difference(p_df.index)
        if len(gr_pe_l) > 0:
            gr_par_pe = pd.concat(gr_pe_l, axis=1)
        logger.log("spectral simulation for grid-scale pars")
    # draw remaining pars based on their geostruct
    if not subset.empty:
        logger.log(f"Drawing {len(subset)} non-specsim pars")
        pe = pyemu.helpers.geostatistical_draws(
            pst,
            struct_dict=struct_dict,
            num_reals=num_reals,
            sigma_range=sigma_range,
            scale_offset=scale_offset,
            subset=subset
        )
        logger.log(f"Drawing {len(subset)} non-specsim pars")
        if gr_par_pe is not None:
            logger.log(f"Joining specsim and non-specsim pars")
            exist = gr_par_pe.columns.intersection(pe.columns)
            pe = pe._df.drop(exist, axis=1)  # specsim par take precedence
            pe = pd.concat([pe, gr_par_pe], axis=1)
            pe = pyemu.ParameterEnsemble(pst=pst, df=pe)
            logger.log(f"Joining specsim and non-specsim pars")
    else:
        pe = pyemu.ParameterEnsemble(pst=pst, df=gr_par_pe)
    logger.log("drawing realizations")
    return pe.copy()


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
        subset (`array-like`, optional): list, array, set or pandas index defining subset of parameters
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
    full_cov_dict = {n: float(v) for n, v in
                     zip(full_cov.col_names, full_cov.x.ravel())}
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
    diff.sort()
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

    full_cov_dict = {n: float(v)
                     for n, v in zip(full_cov.col_names, full_cov.x.ravel())}
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
                                appended as new rows labelled with 'q_#' where '#' is the selected quantile
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
        ens = ens.set_index("real_name")
    # if 'base' real was lost, then the index is of type int. needs to be string later so set here
    ens.index = [str(i) for i in ens.index]
    if not isinstance(pst, pyemu.Pst):
        raise Exception("pst object must be of type pyemu.Pst")

    # get the observation data
    obs = pst.observation_data.copy()

    # confirm that the indices and weights line up
    if not all(ens.columns==obs.index):
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
    ens_eval = ens.loc[:,trimnames].copy()
    weights = obs.loc[trimnames,:].weight.values.astype(float)

    data = {}
    for cq in quantiles:
        # calculate the point-wise quantile values
        qfit = np.quantile(ens_eval, cq, axis=0)
        # calculate the weighted distance between all reals and the desired quantile
        qreal = np.argmin(
            np.linalg.norm([(i - qfit) * weights for i in ens_eval.values], axis=1)
        )
        quantile_idx["q{}".format(cq)] = qreal
        #ens = ens.append(ens.iloc[qreal])
        #idx = ens.index.values
        #idx[-1] = "q{}".format(cq)
        #ens.set_index(idx, inplace=True)
        data["q{}".format(cq)] = ens.iloc[qreal]

    ens = pd.DataFrame(data=data).transpose()

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
        know_cov.x[i, i] += np.squeeze(var_knowledge_dict[name])

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

        pst.prior_information = pd.concat([pst.prior_information, pi])
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
        pst.prior_information = pd.concat([pst.prior_information, df])

    if pst.control_data.pestmode == "estimation":
        pst.control_data.pestmode = "regularization"


def simple_tpl_from_pars(parnames, tplfilename="model.input.tpl", out_dir="."):
    """Make a simple template file from a list of parameter names.

    Args:
        parnames ([`str`]): list of parameter names to put in the
            new template file
        tplfilename (`str`): Name of the template file to create.  Default
            is "model.input.tpl"
        out_dir (`str`): Directory where the template file should be saved.
            Default is the current working directory (".")

    Note:
        Writes a file `tplfilename` with each parameter name in `parnames` on a line

    """
    tpl_path = os.path.join(out_dir, tplfilename)

    with open(tpl_path, "w") as ofp:
        ofp.write("ptf ~\n")
        [ofp.write("~{0:^12}~\n".format(cname)) for cname in parnames]


def simple_ins_from_obs(obsnames, insfilename="model.output.ins", out_dir="."):
    """write a simple instruction file that reads the values named
     in obsnames in order, one per line from a model output file

    Args:
        obsnames (`str`): list of observation names to put in the
            new instruction file
        insfilename (`str`): the name of the instruction file to
            create. Default is "model.output.ins"
        out_dir (`str`): Directory where the instruction file should be saved.
            Default is the current working directory (".")

    Note:
        writes a file `insfilename` with each observation read off
        of a single line

    """
    ins_path = os.path.join(out_dir, insfilename)

    with open(ins_path, "w") as ofp:
        ofp.write("pif ~\n")
        [ofp.write("l1 !{0}!\n".format(cob)) for cob in obsnames]


def pst_from_parnames_obsnames(
    parnames, obsnames, tplfilename="model.input.tpl", insfilename="model.output.ins", out_dir="."
):
    """Creates a Pst object from a list of parameter names and a list of observation names.

    Args:
        parnames (`str`): list of parameter names
        obsnames (`str`): list of observation names
        tplfilename (`str`): template filename. Default is  "model.input.tpl"
        insfilename (`str`): instruction filename. Default is "model.output.ins"
        out_dir (str): Directory where template and instruction files should be saved.
            Default is the current working directory (".")

    Returns:
        `pyemu.Pst`: the generic control file

    Example::

        parnames = ["p1","p2"]
        obsnames = ["o1","o2"]
        pst = pyemu.helpers.pst_from_parnames_obsnames(parname,obsnames)


    """
    tpl_path = os.path.join(out_dir, tplfilename)
    ins_path = os.path.join(out_dir, insfilename)

    simple_tpl_from_pars(parnames, tplfilename, out_dir)
    simple_ins_from_obs(obsnames, insfilename, out_dir)

    modelinputfilename = tplfilename.replace(".tpl", "")
    modeloutputfilename = insfilename.replace(".ins", "")

    return pyemu.Pst.from_io_files(
        tpl_path, modelinputfilename, ins_path, modeloutputfilename
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
        info_txt = struct.unpack("1001s", f.read(1001))[0].strip().lower().decode()
        info_txt = info_txt.replace("\x00","")
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
    for irun in range(1, int(np.squeeze(header["n_runs"]))):
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
        found in `tpl_files` and `ins_files`, respectively.

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


class PstFromFlopyModel(object):
    """
    Deprecated Class. Try `pyemu.utils.PstFrom()` instead.
    A legacy version can be accessed from `pyemu.legacy`, if desperate.
    """

    def __init__(*args, **kwargs):
        dep_warn = (
            "\n`PstFromFlopyModel()` is deprecated."
            "Checkout `pyemu.utils.PstFrom()` instead.\n"
            "If you really want `PstFromFlopyModel()` a legacy version "
            "sits in `pyemu.legacy`"
        )
        raise DeprecationWarning(dep_warn)


def apply_list_and_array_pars(arr_par_file="mult2model_info.csv", chunk_len=50):
    """Apply multiplier parameters to list and array style model files

    Args:
        arr_par_file (str):
        chunk_len (`int`): the number of files to process per multiprocessing
            chunk in apply_array_pars().  default is 50.

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
    if "apply_order" in df.columns:
        df["apply_order"] = df.apply_order.astype(float)
        uapply_values = df.apply_order.unique()
        uapply_values.sort()
    else:
        df["apply_order"] = 999
        uapply_values = [999]
    for apply_value in uapply_values:
        ddf = df.loc[df.apply_order==apply_value,:].copy()
        assert ddf.shape[0] > 0
        arr_pars = ddf.loc[ddf.index_cols.isna()].copy()
        list_pars = ddf.loc[ddf.index_cols.notna()].copy()
        # extract lists from string in input df
        list_pars["index_cols"] = list_pars.index_cols.apply(literal_eval)
        list_pars["use_cols"] = list_pars.use_cols.apply(literal_eval)
        list_pars["lower_bound"] = list_pars.lower_bound.apply(literal_eval)
        list_pars["upper_bound"] = list_pars.upper_bound.apply(literal_eval)

        if "pre_apply_function" in ddf.columns:
            calls = ddf.pre_apply_function.dropna()
            for call in calls:
                print("...evaluating pre-apply function '{0}'".format(call))
                eval(call)

        # TODO check use_cols is always present
        apply_genericlist_pars(list_pars, chunk_len=chunk_len)
        apply_array_pars(arr_pars, chunk_len=chunk_len)

        if "post_apply_function" in ddf.columns:
            calls = ddf.post_apply_function.dropna()
            for call in calls:
                print("...evaluating post-apply function '{0}'".format(call))
                eval(call)


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
        df["operator"] = "m"
    # find all mults that need to be applied to this array
    df_mf = df.loc[df.model_file == model_file, :]
    results = []
    org_file = df_mf.org_file.unique()
    if org_file.shape[0] != 1:
        raise Exception("wrong number of org_files for {0}".format(model_file))
    if "head_rows" not in df.columns:
        skip = 0
    else:
        skip = df_mf.head_rows.values[0]
    with open(org_file[0], 'r') as fp:
         header = [fp.readline() for _ in range(skip)]
    org_arr = np.loadtxt(org_file[0], ndmin=2, skiprows=skip)


    if "mlt_file" in df_mf.columns:
        for mlt, operator in zip(df_mf.mlt_file, df_mf.operator):
            if pd.isna(mlt):
                continue
            mlt_data = np.loadtxt(mlt, ndmin=2)
            if 1 in list(mlt_data.shape): # if 1d arrays
                org_arr = org_arr.reshape(mlt_data.shape)
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

    try:
        fmt = df_mf.fmt.iloc[0]
    except AttributeError:
        fmt = "%15.6E"
    try:
        # default to space (if fixed format file this should be taken care of
        # in fmt string)
        sep = df_mf.sep.fillna(' ').iloc[0]
    except AttributeError:
        sep = ' '
    with open(model_file, 'w') as fp:
        fp.writelines(header)
        np.savetxt(fp, np.atleast_2d(org_arr), fmt=fmt, delimiter=sep)


def apply_array_pars(arr_par="arr_pars.csv", arr_par_file=None, chunk_len=50):
    """a function to apply array-based multiplier parameters.

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
            with mp.get_context("spawn").Pool(
                    processes=min(mp.cpu_count(), 60)) as pool:
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
    else:
        with mp.get_context("spawn").Pool(
                processes=min(mp.cpu_count(), 60)) as pool:
            x = [
                pool.apply_async(_process_chunk_array_files, args=(chunk, i, df))
                for i, chunk in enumerate(chunks)
            ]
            [xx.get() for xx in x]
            pool.close()
            pool.join()
    print("finished arr mlt", datetime.now())


def setup_temporal_diff_obs(*args, **kwargs):
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
        long_names (`bool`, optional): flag to use long, descriptive names by concatenating the two observation names
            that are being differenced.  This will produce names that are too long for traditional PEST(_HP).
            Default is True.
        prefix (`str`, optional): prefix to prepend to observation names and group names.  Default is "dif".

    Returns:
        tuple containing

        - **str**: the forward run command to execute the binary file process during model runs.

        - **pandas.DataFrame**: a dataframe of observation information for use in the pest control file

    Note:
        This is the companion function of `helpers.apply_temporal_diff_obs()`.



    """
    warnings.warn("This method (and companion method `apply_temporal_diff_obs()`"
                  "are associated with the deprecated `PstFromFlopyModel` class. "
                  "They are now untested.", DeprecationWarning)
    from pyemu.legacy import setup_temporal_diff_obs
    return setup_temporal_diff_obs(*args, **kwargs)


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
                arr[zone_arr == 0] = np.nan
                org_arr[zone_arr == 0] = np.nan

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
        with mp.get_context("spawn").Pool(
                processes=min(mp.cpu_count(), 60)) as pool:
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
    elif isinstance(index_col_eg, (int, np.integer)):
        # index_cols are column numbers in input file
        header = None
        hheader = None
        # actually do need index cols to be list of strings
        # to be compatible when the saved original file is read in.
        df_mf.loc[:, "index_cols"] = df_mf.index_cols.apply(
            lambda x: [str(i) for i in x]
        )

    # if written by PstFrom this should always be comma delim - tidy
    org_data = pd.read_csv(org_file, skiprows=datastrtrow,
                           header=header, dtype='object')
    # mult columns will be string type, so to make sure they align
    org_data.columns = org_data.columns.astype(str)
    # print("org_data columns:", org_data.columns)
    # print("org_data shape:", org_data.shape)
    new_df = org_data.copy()

    for mlt in df_mf.itertuples():
        new_df.loc[:, mlt.index_cols] = new_df.loc[:, mlt.index_cols].apply(
            _try_pdcol_numeric, downcast='integer')
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

            mlts[mlt.index_cols] = mlts[mlt.index_cols].apply(
                _try_pdcol_numeric, intadj=add1, downcast='integer')
            mlts = mlts.set_index(mlt.index_cols)
            if mlts.index.nlevels < 2:  # just in case only one index col is used
                mlts.index = mlts.index.get_level_values(0)
            common_idx = (
                new_df.index.intersection(mlts.index).drop_duplicates()
            )
            mlt_cols = [str(col) for col in mlt.use_cols]
            assert len(common_idx) == mlt.chkpar, (
                "Probable miss-alignment in tpl indices and original file:\n"
                f"mult idx[:10] : {mlts.index.sort_values().tolist()[:10]}\n"
                f"org file idx[:10]: {new_df.index.sort_values().to_list()[:10]}\n"
                f"n common: {len(common_idx)}, n cols: {len(mlt_cols)}, "
                f"expected: {mlt.chkpar}."
            )
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
            kwargs = {"lineterminator": "\n"}
        if len(storehead) != 0:
            fo.write("\n".join(storehead))
            fo.flush()
        if fmt.lower() == "free":
            new_df.to_csv(fo, index=False, mode="a", sep=sep, header=hheader, **kwargs)
        else:
            np.savetxt(
                fo,
                np.atleast_2d(new_df.apply(_try_pdcol_numeric).values),
                fmt=fmt
            )


def build_jac_test_csv(pst, num_steps, par_names=None, forward=True):
    """build a dataframe of jactest inputs for use with pestpp-swp

    Args:
        pst (`pyemu.Pst`): existing control file
        num_steps (`int`): number of perturbation steps for each parameter
        par_names [`str`]: list of parameter names of pars to test.
            If None, all adjustable pars are used. Default is None
        forward (`bool`): flag to start with forward perturbations.
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
    if "lineterminator" not in kwargs:
        if "win" in platform.platform().lower():
            kwargs["lineterminator"] = "\n"
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


# web address of spatial reference dot org
srefhttp = "https://spatialreference.org"


class SpatialReference(object):
    """
    a class to locate a structured model grid in x-y space.
    Lifted wholesale from Flopy, and preserved here...
    ...maybe slightly over-engineered for here

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
            if isinstance(delrc, (float, int, np.integer)):
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
        self.grid_type = "structured"

    @property
    def ncpl(self):
        raise Exception("unstructured grids not supported by SpatialReference")

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


def setup_threshold_pars(orgarr_file,cat_dict,testing_workspace=".",inact_arr=None):
    """setup a thresholding 2-category binary array process.

    Parameters:
        orgarr_file (`str`): the input array that will ultimately be created at runtime
        cat_dict (`str`): dict of info for the two categories.  Keys are (unused) category names.
            values are a len 2 iterable of requested proportion and fill value.
        testing_workspace (`str`): directory where the apply process can be tested.
        inact_arr (`np.ndarray`): an array that indicates inactive nodes (inact_arr=0)

    Returns:
        thresharr_file (`str`): thresholding array file (to be parameterized)
        csv_file (`str`): the csv file that has the inputs needed for the apply process

    Note:
        all required files are based on the `orgarr_file` with suffixes appended to them
        This process was inspired by Todaro and others, 2023, "Experimental sandbox tracer
        tests to characterize a two-facies aquifer via an ensemble smoother"

    """
    assert os.path.exists(orgarr_file)
    #at least 2d for xsections
    org_arr = np.atleast_2d(np.loadtxt(orgarr_file))

    if len(cat_dict) != 2:
        raise Exception("only two categories currently supported, {0} found in target_proportions_dict".\
                        format(len(cat_dict)))

    prop_tags,prop_vals,fill_vals = [],[],[]
    #print(cat_dict[1])
    #for key,(proportion,fill_val) in cat_dict.items():
    keys = list(cat_dict.keys())
    keys.sort()
    for key in keys:
        proportion = cat_dict[key][0]
        fill_val = cat_dict[key][1]

        if int(key) not in cat_dict:
            raise Exception("integer type of key '{0}' not found in target_proportions_dict".format(key))
        prop_tags.append(int(key))
        prop_vals.append(float(proportion))
        # use the key as the fill value for testing purposes
        fill_vals.append(float(fill_val))

    thresharr = org_arr.copy()
    thresharr = thresharr / thresharr.max()
    thresharr_file = orgarr_file+".thresharr.dat"
    np.savetxt(thresharr_file,thresharr,fmt="%15.6E")

    if inact_arr is not None:
        assert inact_arr.shape == org_arr.shape
        inactarr_file = orgarr_file+".threshinact.dat"
        np.savetxt(inactarr_file,inact_arr,fmt="%4.0f")

    df = pd.DataFrame({"threshcat":prop_tags,"threshproportion":prop_vals,"threshfill":fill_vals})
    csv_file = orgarr_file+".threshprops.csv"
    df.to_csv(csv_file,index=False)


    # test that it seems to be working
    rel_csv_file = csv_file
    
    rel_csv_file = os.path.relpath(csv_file,start=testing_workspace)
    bd = os.getcwd()
    os.chdir(testing_workspace)

    #apply_threshold_pars(os.path.split(csv_file)[1])
    apply_threshold_pars(rel_csv_file)
    os.chdir(bd)
    return thresharr_file,csv_file


def apply_threshold_pars(csv_file):
    """apply the thresholding process.  everything keys off of csv_file name...

    Note: if the standard deviation of the continuous thresholding array is too low,
    the line search will fail.  Currently, if this stdev is less than 1.e-10,
    then a homogeneous array of the first category fill value will be created.  User
    beware!

    """
    df = pd.read_csv(csv_file,index_col=0)
    thresarr_file = csv_file.replace("props.csv","arr.dat")
    tarr = np.loadtxt(thresarr_file)
    if np.any(tarr < 0):
        tmin = tarr.min()
        tarr += tmin + 1
        #print(tarr)
        #raise Exception("negatives in thresholding array {0}".format(thresarr_file))
    #norm tarr
    tarr = (tarr - tarr.min()) / tarr.max()
    orgarr_file = csv_file.replace(".threshprops.csv","")
    inactarr_file = csv_file.replace(".threshprops.csv",".threshinact.dat")
    tarr_file = csv_file.replace(".threshprops.csv",".threshcat.dat")
    tcat = df.index.values.astype(int).copy()
    tvals = df.threshproportion.astype(float).values.copy()
    #ttags = df.threshcat.astype(int).values.copy()
    tfill = df.threshfill.astype(float).values.copy()

    iarr = None
    if os.path.exists(inactarr_file):
        iarr = np.loadtxt(inactarr_file, dtype=int)

    # for now...
    assert len(tvals) == 2
    if np.any(tvals <= 0.0):
        print(tvals)
        raise Exception("threshold values much be strictly positive")

    #since we only have two categories, we can just focus on the first proportion
    target_prop = tvals[0]

    tol = 1.0e-10
    if tarr.std() < 1e-5:

        print("WARNING: thresholding array {0} has very low standard deviation".format(thresarr_file))
        print("         using a homogeneous array with first category fill value {0}".format(tfill[0]))

        farr = np.zeros_like(tarr) + tfill[0]
        if iarr is not None:
            farr[iarr == 0] = np.nan
            tarr[iarr == 0] = np.nan
        df.loc[tcat[0], "threshold"] = np.nanmean(tarr)
        df.loc[tcat[1], "threshold"] = np.nanmean(tarr)
        df.loc[tcat[0], "proportion"] = 1
        df.loc[tcat[1], "proportion"] = 0

        if iarr is not None:
            farr[iarr == 0] = -1e30
            tarr[iarr == 0] = -1e30

        df.to_csv(csv_file.replace(".csv", "_results.csv"))
        np.savetxt(orgarr_file, farr, fmt="%15.6E")
        np.savetxt(tarr_file, tarr, fmt="%15.6E")
        return tarr.mean(), 1.0

        #    print("WARNING: thresholding array {0} has very low standard deviation, adding noise".format(thresarr_file))
        #    tarr += np.random.normal(0, tol*2.0, tarr.shape)

    # a classic:
    gr = (np.sqrt(5.) + 1.) / 2.
    a = np.nanmin(tarr)
    b = np.nanmax(tarr)
    c = b - ((b - a) / gr)
    d = a + ((b - a) / gr)


    tot_shape = tarr.shape[0] * tarr.shape[1]
    if iarr is not None:

        # this keeps inact from interfering with calcs later...
        tarr[iarr == 0] = np.nan
        tiarr = iarr.copy()
        tiarr[tiarr <= 0] = 0
        tiarr[tiarr > 0] = 1
        tot_shape = tiarr.sum()

    def get_current_prop(_cur_thresh):
        itarr = np.zeros_like(tarr)
        itarr[tarr <= _cur_thresh] = 1
        current_prop = itarr.sum() / tot_shape
        return current_prop

    numiter = 0
    while True:
        if (b - a) <= tol:
            break
        if numiter > 10000:
            raise Exception("golden section failed to converge")

        cprop = get_current_prop(c)
        dprop = get_current_prop(d)
        #print(a,b,c,d,cprop,dprop,target_prop)
        cphi = (cprop - target_prop)**2
        dphi = (dprop - target_prop)**2
        if cphi < dphi:
            b = d
        else:
            a = c
        c = b - ((b - a) / gr)
        d = a + ((b - a) / gr)
        numiter += 1

    thresh = (a+b) / 2.0
    prop = get_current_prop(thresh)
    farr = np.zeros_like(tarr) - 1
    farr[tarr>thresh] = tfill[1]
    farr[tarr<=thresh] = tfill[0]
    tarr[tarr>thresh] = tcat[0]
    tarr[tarr <= thresh] = tcat[1]

    if iarr is not None:
        farr[iarr==0] = -1e+30
        tarr[iarr==0] = -1e+30
    df.loc[tcat[0],"threshold"] = thresh
    df.loc[tcat[1], "threshold"] = 1.0 - thresh
    df.loc[tcat[0], "proportion"] = prop
    df.loc[tcat[1], "proportion"] = 1.0 - prop
    df.to_csv(csv_file.replace(".csv","_results.csv"))
    np.savetxt(orgarr_file,farr,fmt="%15.6E")
    np.savetxt(tarr_file, tarr, fmt="%15.6E")

    return thresh, prop


def prep_for_gpr(pst_fname,input_fnames,output_fnames,gpr_t_d="gpr_template",gp_kernel=None,nverf=0,
                 plot_fits=False,apply_standard_scalar=False, include_emulated_std_obs=False):
    """helper function to setup a gaussian-process-regression (GPR) emulator for outputs of interest.  This
    is primarily targeted at low-dimensional settings like those encountered in PESTPP-MOU

    Parameters:
        pst_fname (str): existing pest control filename
        input_fnames (str | list[str]): usually a list of decision variable population files
        output_fnames (str | list[str]): usually a list of observation population files that
            corresponds to the simulation results associated with `input_fnames`
        gpr_t_d (str): the template file dir to create that will hold the GPR emulators
        gp_kernel (sklearn GaussianProcess kernel): the kernel to use.  if None, a standard RBF kernel
            is created and used
        nverf (int): the number of input-output pairs to hold back for a simple verification test
        plot_fits (bool): flag to plot the fit GPRs
        apply_standard_scalar (bool): flag to apply sklearn.preprocessing.StandardScaler transform before 
            training/executing the emulator.  Default is False
        include_emulated_std_obs (bool): flag to include the estimated standard deviation in the predicted
            response of each GPR emulator.  If True, additional obserations are added to the GPR pest interface
            , one for each nominated observation quantity.  Can be very useful for designing in-filling strategies

    Returns:
        None

    Note:
        requires scikit-learn

    """

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    import pickle
    import inspect
    pst = pyemu.Pst(pst_fname)

    # work out input variable names
    input_groups = pst.pestpp_options.get("opt_dec_var_groups",None)
    par = pst.parameter_data
    if input_groups is None:
        print("using all adjustable parameters as inputs")
        input_names = pst.adj_par_names
    else:
        input_groups = set([i.strip() for i in input_groups.lower().strip().split(",")])
        print("input groups:",input_groups)
        adj_par = par.loc[pst.adj_par_names,:].copy()
        adj_par = adj_par.loc[adj_par.pargp.apply(lambda x: x in input_groups),:]
        input_names = adj_par.parnme.tolist()
    print("input names:",input_names)

    #work out constraints and objectives
    ineq_names = pst.less_than_obs_constraints.tolist()
    ineq_names.extend(pst.greater_than_obs_constraints.tolist())
    obs = pst.observation_data
    objs = pst.pestpp_options.get("mou_objectives",None)
    constraints = []

    if objs is None:
        print("'mou_objectives' not found in ++ options, using all ineq tagged non-zero weighted obs as objectives")
        objs = ineq_names
    else:
        objs = objs.lower().strip().split(',')
        constraints = [n for n in ineq_names if n not in objs]

    print("objectives:",objs)
    print("constraints:",constraints)
    output_names = objs
    output_names.extend(constraints)

    print("loading input and output files")
    if isinstance(input_fnames,str):
        input_fnames = [input_fnames]
    if isinstance(output_fnames,str):
        output_fnames = [output_fnames]
    if len(output_fnames) != len(input_fnames):
        raise Exception("len(input_fnames) != len(output_fnames)")

    if os.path.exists(gpr_t_d):
        shutil.rmtree(gpr_t_d)
    os.makedirs(gpr_t_d)

    dfs = []
    for input_fname,output_fname in zip(input_fnames,output_fnames):
        if input_fname.lower().endswith(".csv"):
            input_df = pd.read_csv(os.path.join(input_fname),index_col=0)
        elif input_fname.lower().endswith(".jcb"):
            input_df = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=input_fname)._df
        else:
            raise Exception("unrecognized input_fname extension:'{0}', looking for csv or jcb".\
                           format(input_fname.lower()))

        if output_fname.lower().endswith(".csv"):
            output_df = pd.read_csv(os.path.join(output_fname),index_col=0)
        elif output_fname.lower().endswith(".jcb"):
            output_df = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=output_fname)._df
        else:
            raise Exception("unrecognized output_fname extension:'{0}', looking for csv or jcb".\
                            format(output_fname.lower()))

        if input_df.shape[0] != output_df.shape[0]:
            raise Exception("input rows != output rows for {0} and {1}".\
                            format(input_fname,output_fname))
        input_df = input_df.loc[:,input_names]
        assert input_df.shape == input_df.dropna().shape

        output_df = output_df.loc[:, output_names]
        assert output_df.shape == output_df.dropna().shape

        input_df.loc[:,output_names] = output_df.values
        dfs.append(input_df)
        print("...loaded",input_fname,output_fname)

    df = pd.concat(dfs)
    assert df.shape == df.dropna().shape
    df.to_csv(os.path.join(gpr_t_d,"gpr_aggregate_training_data.csv"))
    print("aggregated training dataset shape",df.shape,"saved to",pst_fname + ".aggresults.csv")


    if gp_kernel is None:
        #gp_kernel = ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-8,1e8)) *\
        #                     RBF(length_scale=1000.0, length_scale_bounds=(1e-8, 1e8))
        gp_kernel = Matern(length_scale=100.0, length_scale_bounds=(1e-4, 1e4), nu=0.5)

    for hp in gp_kernel.hyperparameters:
        print(hp)

    cut = df.shape[0] - nverf
    X_train = df.loc[:, input_names].values.copy()[:cut, :]
    X_verf = df.loc[:, input_names].values.copy()[cut:, :]

    model_fnames = []
    if plot_fits:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(os.path.join(gpr_t_d,"gpr_fits.pdf"))
    for output_name in output_names:

        y_verf = df.loc[:,output_name].values.copy()[cut:]
        y_train = df.loc[:, output_name].values.copy()[:cut]
        print("training GPR for {0} with {1} data points".format(output_name,y_train.shape[0]))
        gaussian_process = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=20)
        if not apply_standard_scalar:
            print("WARNING: not applying StandardScalar transformation - user beware!")
            pipeline = Pipeline([("gpr",gaussian_process)])
        else:
            pipeline = Pipeline([("std_scalar", StandardScaler()), ("gpr", gaussian_process)])
        pipeline.fit(X_train, y_train)
        print(output_name,"optimized kernel:",pipeline["gpr"].kernel_)
        if plot_fits:
            print("...plotting fits for",output_name)
            predmean,predstd = pipeline.predict(df.loc[:, input_names].values.copy(), return_std=True)
            df.loc[:,"predmean"] = predmean
            df.loc[:,"predstd"] = predstd
            isverf = np.zeros_like(predmean)
            isverf[cut:] = 1
            df.loc[:,"isverf"] = isverf
            for input_name in input_names:
                fig,ax = plt.subplots(1,1,figsize=(6,6))
                #print(X_train[:,ii])
                #print(y_train)
                ddf = df[[input_name,output_name,"predmean","predstd","isverf"]].copy()
                ddf.sort_values(by=input_name,inplace=True)
                ax.scatter(ddf.loc[ddf.isverf==0,input_name],ddf.loc[ddf.isverf==0,output_name],marker=".",c="r",label="training")
                if nverf > 0:
                    ax.scatter(ddf.loc[ddf.isverf==1,input_name],ddf.loc[ddf.isverf==1,output_name],marker="^",c="c",label="verf")
                ax.plot(ddf[input_name],ddf["predmean"],"k--",label="GPR mean")
                ax.fill_between(ddf[input_name],ddf["predmean"] - (2*ddf["predstd"]),ddf["predmean"]+(2*ddf["predstd"]),alpha=0.5,fc='0.5',label="+/- 95%")
                ax.set_title("input:{0}, output:{1}".format(input_name,output_name),loc="left")
                ax.legend()
                plt.tight_layout()
                pdf.savefig()

                plt.close(fig)



        model_fname = os.path.split(pst_fname)[1]+"."+output_name+".pkl"
        if os.path.exists(os.path.join(gpr_t_d,model_fname)):
            print("WARNING: model_fname '{0}' exists, overwriting...".format(model_fname))
        with open(os.path.join(gpr_t_d,model_fname),'wb') as f:
            pickle.dump(pipeline,f)

        model_fnames.append(model_fname)
        if nverf > 0:
            pred_mean,pred_std = pipeline.predict(X_verf,return_std=True)
            vdf = pd.DataFrame({"y_verf":y_verf,"y_pred":pred_mean,"y_pred_std":pred_std})
            verf_fname = os.path.join(gpr_t_d,"{0}_gpr_verf.csv".format(output_name))
            vdf.to_csv(verf_fname)
            print("saved ",output_fname,"verification csv to",verf_fname)
            mabs = np.abs(vdf.y_verf - vdf.y_pred).mean()
            print("...mean abs error",mabs)
    if plot_fits:
        pdf.close()
    mdf = pd.DataFrame({"output_name":output_names,"model_fname":model_fnames},index=output_names)
    minfo_fname = os.path.join(gpr_t_d,"gprmodel_info.csv")
    mdf.to_csv(minfo_fname)
    print("GPR model info saved to",minfo_fname)

    #write a template file
    tpl_fname = os.path.join(gpr_t_d,"gpr_input.csv.tpl")
    with open(tpl_fname,'w') as f:
        f.write("ptf ~\nparnme,parval1\n")
        for input_name in input_names:
            f.write("{0},~  {0}   ~\n".format(input_name))
    other_pars = list(set(pst.par_names)-set(input_names))
    aux_tpl_fname = None

    if len(other_pars) > 0:

        aux_tpl_fname = os.path.join(gpr_t_d,"aux_par.csv.tpl")
        print("writing aux par tpl file: ",aux_tpl_fname)
        with open(aux_tpl_fname,'w') as f:
            f.write("ptf ~\n")
            for input_name in other_pars:
                f.write("{0},~  {0}   ~\n".format(input_name))
    #write an ins file
    ins_fname = os.path.join(gpr_t_d,"gpr_output.csv.ins")
    with open(ins_fname,'w') as f:
        f.write("pif ~\nl1\n")
        for output_name in output_names:
            if include_emulated_std_obs:
                f.write("l1 ~,~ !{0}! ~,~ !{0}_gprstd!\n".format(output_name))
            else:
                f.write("l1 ~,~ !{0}!\n".format(output_name))
    tpl_list = [tpl_fname]
    if aux_tpl_fname is not None:
        tpl_list.append(aux_tpl_fname)
    input_list = [f.replace(".tpl","") for f in tpl_list]
    gpst = pyemu.Pst.from_io_files(tpl_list,input_list,
                                   [ins_fname],[ins_fname.replace(".ins","")],pst_path=".")
    par_names = pst.par_names
    assert len(set(par_names).symmetric_difference(set(gpst.par_names))) == 0
    for col in pst.parameter_data.columns:
        # this gross thing is to avoid a future error warning in pandas - 
        # why is it getting so strict?!  isn't python duck-typed?
        if col in gpst.parameter_data.columns and\
           gpst.parameter_data.dtypes[col] != pst.parameter_data.dtypes[col]:
            gpst.parameter_data[col] = gpst.parameter_data[col].astype(pst.parameter_data.dtypes[col])
        gpst.parameter_data.loc[par_names,col] = pst.parameter_data.loc[par_names,col].values

    for col in pst.observation_data.columns:
        # this gross thing is to avoid a future error warning in pandas -
        # why is it getting so strict?!  isn't python duck-typed?
        if col in gpst.observation_data.columns and \
                gpst.observation_data.dtypes[col] != pst.observation_data.dtypes[col]:
            gpst.observation_data[col] = gpst.obsveration_data[col].astype(pst.observation_data.dtypes[col])
        gpst.observation_data.loc[output_names,col] = pst.observation_data.loc[output_names,col].values
    if include_emulated_std_obs:
        stdobs = [o for o in gpst.obs_names if o.endswith("_gprstd")]
        assert len(stdobs) > 0
        gpst.observation_data.loc[stdobs,"weight"] = 0.0
    gpst.pestpp_options = pst.pestpp_options
    gpst.prior_information = pst.prior_information.copy()
    #lines = [line[4:] for line in inspect.getsource(gpr_forward_run).split("\n")][1:]
    frun_lines = inspect.getsource(gpr_forward_run)
    getfxn_lines = inspect.getsource(get_gpr_model_dict)
    emulfxn_lines = inspect.getsource(emulate_with_gpr)
    with open(os.path.join(gpr_t_d, "forward_run.py"), 'w') as f:
        f.write("\n")
        for import_name in ["pandas as pd","os","pickle","numpy as np"]:
            f.write("import {0}\n".format(import_name))
        for line in getfxn_lines:
            f.write(line)
        f.write("\n")
        for line in emulfxn_lines:
            f.write(line)
        f.write("\n")
        for line in frun_lines:
            f.write(line)
        f.write("if __name__ == '__main__':\n")
        f.write("    gpr_forward_run()\n")
        

    gpst.control_data.noptmax = 0
    gpst.model_command = "python forward_run.py"
    gpst_fname = os.path.split(pst_fname)[1]
    gpst.write(os.path.join(gpr_t_d,gpst_fname),version=2)
    print("saved gpr pst:",gpst_fname,"in gpr_t_d",gpr_t_d)
    try:
        pyemu.os_utils.run("pestpp-mou {0}".format(gpst_fname),cwd=gpr_t_d)
    except Exception as e:
        print("WARNING: pestpp-mou test run failed: {0}".format(str(e)))
    gpst.control_data.noptmax = pst.control_data.noptmax
    gpst.write(os.path.join(gpr_t_d, gpst_fname), version=2)


def get_gpr_model_dict(mdf):
    import pickle
    gpr_model_dict = {}
    for output_name,model_fname in zip(mdf.output_name,mdf.model_fname):
        gaussian_process = pickle.load(open(model_fname,'rb'))
        gpr_model_dict[output_name] = gaussian_process
    return gpr_model_dict


def emulate_with_gpr(input_df,mdf,gpr_model_dict):
    mdf.loc[:,"sim"] = np.nan
    mdf.loc[:,"sim_std"] = np.nan
    for output_name,gaussian_process in gpr_model_dict.items():
        sim = gaussian_process.predict(np.atleast_2d(input_df.parval1.values),return_std=True)
        mdf.loc[output_name,"sim"] = sim[0]
        mdf.loc[output_name,"sim_std"] = sim[1]
    return mdf


def gpr_pyworker(pst,host,port,input_df=None,mdf=None):
    import os
    import pandas as pd
    import numpy as np
    import pickle

    # if explicit args weren't passed, get the default ones...
    if input_df is None:
        input_df = pd.read_csv("gpr_input.csv",index_col=0)
    if mdf is None:
        mdf = pd.read_csv("gprmodel_info.csv",index_col=0)
    gpr_model_dict = get_gpr_model_dict(mdf)
    ppw = PyPestWorker(pst,host,port,verbose=False)

    # we can only get parameters once the worker has initialize and 
    # is ready to run, so getting the first of pars here
    # essentially blocks until the worker is ready
    parameters = ppw.get_parameters()
    # if its  None, the master already quit...
    if parameters is None:
        return

    obs = ppw._pst.observation_data.copy()
    # align the obsval series with the order sent from the master
    obs = obs.loc[ppw.obs_names,"obsval"]
    
    # work out which par values sent from the master we need to run the emulator
    par = ppw._pst.parameter_data.copy()
    usepar_idx = []
    ppw_par_names = list(ppw.par_names)
    for i,pname in enumerate(input_df.index.values):
        usepar_idx.append(ppw_par_names.index(pname))
    

    while True:
        # map the current dv values in parameters into the 
        # df needed to run the emulator
        input_df["parval1"] = parameters.values[usepar_idx]
        # do the emulation
        simdf = emulate_with_gpr(input_df,mdf,gpr_model_dict)

        # replace the emulated quantities in the obs series
        obs.loc[simdf.index] = simdf.sim.values
        obs.loc[simdf.index.map(lambda x: x+"_gprstd")] = simdf.sim_std.values
        #send the obs series to the master
        ppw.send_observations(obs.values)

        #try to get more pars
        parameters = ppw.get_parameters()
        # if None, we are done
        if parameters is None:
            break
        


def gpr_forward_run():
    """the function to evaluate a set of inputs thru the GPR emulators.\
    This function gets added programmatically to the forward run process"""
    import pandas as pd
    input_df = pd.read_csv("gpr_input.csv",index_col=0)
    mdf = pd.read_csv("gprmodel_info.csv",index_col=0)
    gpr_model_dict = get_gpr_model_dict(mdf)
    mdf = emulate_with_gpr(input_df,mdf,gpr_model_dict)    
    mdf.loc[:,["output_name","sim","sim_std"]].to_csv("gpr_output.csv",index=False)
    return mdf


def dsi_forward_run(pmat=None,ovals=None,pvals=None,
                    write_csv=True
                    
                    ):

    if pvals is None:
        pvals = pd.read_csv("dsi_pars.csv",index_col=0)
    if pmat is None:
        pmat = np.load("dsi_proj_mat.npy")
    if ovals is None:
        ovals = pd.read_csv("dsi_pr_mean.csv",index_col=0)

    try:
        offset = np.load("dsi_obs_offset.npy")
    except:
        #print("no offset file found, assuming no offset")
        offset = np.zeros(ovals.shape[0])
    try:
        log_trans = np.load("dsi_obs_log.npy")
    except:
        #print("no log-tansform file found, assuming no log-transform")
        log_trans = np.zeros(ovals.shape[0])

    try:
        backtransformvals = np.load("dsi_obs_backtransformvals.npy")
        backtransformobsnmes = np.load("dsi_obs_backtransformobsnmes.npy",allow_pickle=True)
        backtransform=True
    except:
        #print("no back-transform file found, assuming no back-transform")
        backtransform=False


    sim_vals = ovals + np.dot(pmat,pvals.values)

    if backtransform:
        #print("applying back-transform")
        obsnmes = np.unique(backtransformobsnmes)
        back_vals = [
                    inverse_normal_score_transform(
                                        backtransformvals[np.where(backtransformobsnmes==o)][:,1],
                                        backtransformvals[np.where(backtransformobsnmes==o)][:,0],
                                        sim_vals.loc[o].mn,
                                        extrap=None
                                        )[0] 
                    for o in obsnmes
                    ]     
        sim_vals.loc[obsnmes,'mn'] = back_vals

    #print("reversing offset and log-transform")
    assert log_trans.shape[0] == sim_vals.mn.values.shape[0], f"log transform shape mismatch: {log_trans.shape[0]},{sim_vals.mn.values.shape[0]}"
    assert offset.shape[0] == sim_vals.mn.values.shape[0], f"offset transform shape mismatch: {offset.shape[0]},{sim_vals.mn.values.shape[0]}"
    vals = sim_vals.mn.values
    vals[np.where(log_trans==1)] = 10**vals[np.where(log_trans==1)]
    vals-= offset
    sim_vals.loc[:,'mn'] = vals
    #print(sim_vals)
    if write_csv:
        sim_vals.to_csv("dsi_sim_vals.csv")
    return sim_vals


def dsi_pyworker(pst,host,port,pmat=None,ovals=None,pvals=None):
    
    import os
    import pandas as pd
    import numpy as np


    # if explicit args weren't passed, get the default ones...
    if pvals is None:
        pvals = pd.read_csv("dsi_pars.csv",index_col=0)
    if pmat is None:
        pmat = np.load("dsi_proj_mat.npy")
    if ovals is None:
        ovals = pd.read_csv("dsi_pr_mean.csv",index_col=0)


    ppw = PyPestWorker(pst,host,port,verbose=False)

    # we can only get parameters once the worker has initialize and 
    # is ready to run, so getting the first of pars here
    # essentially blocks until the worker is ready
    parameters = ppw.get_parameters()
    # if its  None, the master already quit...
    if parameters is None:
        return

    obs = ppw._pst.observation_data.copy()
    # align the obsval series with the order sent from the master
    obs = obs.loc[ppw.obs_names,"obsval"]

    while True:
        # map the current par values in parameters into the 
        # df needed to run the emulator
        pvals.parval1 = parameters.loc[pvals.index]
        # do the emulation
        simdf = dsi_forward_run(pmat=pmat,ovals=ovals,pvals=pvals,write_csv=False)

        # replace the emulated quantities in the obs series
        obs.loc[simdf.index] = simdf.mn.values

        #send the obs series to the master
        ppw.send_observations(obs.values)

        #try to get more pars
        parameters = ppw.get_parameters()
        # if None, we are done
        if parameters is None:
            break


def randrealgen_optimized(nreal, tol=1e-7, max_samples=1000000):
    """
    Generate a set of random realizations with a normal distribution.
    
    Parameters:
    nreal : int
        The number of realizations to generate.
    tol : float
        Tolerance for the stopping criterion.
    max_samples : int
        Maximum number of samples to use.
        
    Returns:
    numpy.ndarray
        An array of nreal random realizations.
    """
    rval = np.zeros(nreal)
    nsamp = 0
    # if nreal is even add 1
    if nreal % 2 == 0:
        numsort = (nreal + 1) // 2
    else:
        numsort = nreal // 2
    while nsamp < max_samples:
        nsamp += 1
        work1 = np.random.normal(size=nreal)
        work1.sort()
        
        if nsamp > 1:
            previous_mean = rval[:numsort] / (nsamp - 1)
            rval[:numsort] += work1[:numsort]
            current_mean = rval[:numsort] / nsamp
            max_diff = np.max(np.abs(current_mean - previous_mean))
            
            if max_diff <= tol:
                break
        else:
            rval[:numsort] = work1[:numsort]
    
    rval[:numsort] /= nsamp
    if nreal % 2 == 0:
        rval[numsort:] = -rval[:numsort][::-1]
    else:
        rval[numsort+1:] = -rval[:numsort][::-1]
    
    return rval


def normal_score_transform(nstval, val, value):
    """
    Transform a value to its normal score using a normal score transform table.
    
    Parameters:
    nstval : array-like
        Normal score transform table values.
    val : array-like
        Original values corresponding to the normal score transform table.
    value : float
        The value to transform.
        
    Returns:
    float
        The normal score of the value.
    int
        The index of the value in the normal score transform table."""
    
    # make sure the input is numpy arrays
    val = np.asarray(val)
    nstval = np.asarray(nstval)
    
    # if the value is outside the range of the table, return the first or last value
    assert value >= val[0], "Value is below the minimum value in the table."
    assert value <= val[-1], "Value is greater than the maximum value in the table."
    # ensure that val is sorted
    assert np.all(np.diff(val) > 0), f"Values in the table must be sorted in ascending order:{list(zip(np.diff(val)>0,val))}"

    # find the rank of the value in the table
    rank = np.searchsorted(val, value, side='right') - 1
    if rank == len(val) - 1:
        return nstval[-1], len(val)
    # if the value coincides with a value in the table, return the corresponding normal score
    nstdiff = nstval[rank + 1] - nstval[rank]
    diff = val[rank + 1] - val[rank]
    if nstdiff <= 0.0 or diff <= 0.0:
        return nstval[rank], rank
    
    # otherwise, interpolate to get the normal score
    dist = value - val[rank]
    interpolated_value = nstval[rank] + (dist / diff) * nstdiff
    return interpolated_value, rank


def inverse_normal_score_transform(nstval, val, value, extrap='quadratic'):
    nreal = len(val)
    # check that nstval is sorted
    assert np.all(np.diff(nstval) > 0), "Values in the table must be sorted in ascending order"
    # check that val is sorted
    assert np.all(np.diff(val) > 0), "Values in the table must be sorted in ascending order"
    
    def linear_extrapolate(x0, y0, x1, y1, x):
        if x1 != x0:
            return y0 + (y1 - y0) / (x1 - x0) * (x - x0)
        return y0

    def quadratic_extrapolate(x1, y1, x2, y2, x3, y3, x4):
        y12=y1-y2
        x23=x2-x3
        y23=y2-y3
        x12=x1-x2
        x13=x1-x3
        if x12==0 or x23==0 or x13==0:
            raise ValueError("Input x values must be distinct")
        a = (y12*x23-y23*x12)
        den = x12*x23*x13
        a = a/den
        b = y23/x23 - a*(x2+x3)
        c=y1-x1*(a*x1+b)
        y4 = a*x4**2 + b*x4 + c
        return y4

    ilim = 0
    if value in nstval:
        rank = np.searchsorted(nstval, value)
        value = val[rank]

    elif value < nstval[0]:
        ilim = -1
        if extrap is None:
            value = val[0]
        elif extrap == 'linear':
            value = linear_extrapolate(nstval[0], val[0], nstval[1], val[1], value)
            #value = min(value, val[0])
        elif extrap == 'quadratic' and nreal >= 3:
            y_vals = np.unique(val)[:3]
            idxs = np.searchsorted(val,y_vals)
            x_vals = nstval[idxs]
            value = quadratic_extrapolate(x_vals[-3], y_vals[-3], x_vals[-2], y_vals[-2], x_vals[-1], y_vals[-1], value)
            #value = min(value, val[0])
        else:
            value = val[0]

    elif value > nstval[-1]:
        ilim = 1
        if extrap is None:
            value = val[-1]
        elif extrap == 'linear':
            value = linear_extrapolate(nstval[-2], val[-2], nstval[-1], val[-1], value)
            #value = max(value, val[-1])
        elif extrap == 'quadratic' and nreal >= 3:
            y_vals = np.unique(val)[-3:]
            idxs = np.searchsorted(val,y_vals)
            x_vals = nstval[idxs]
            value = quadratic_extrapolate(x_vals[-3], y_vals[-3], x_vals[-2], y_vals[-2], x_vals[-1], y_vals[-1], value)
            #value = max(value, val[-1])
        else:
            value = val[-1]

    else:
        rank = np.searchsorted(nstval, value) - 1
        # Get the bounding x and y values
        x0, x1 = nstval[rank], nstval[rank + 1]
        y0, y1 = val[rank], val[rank + 1]
        # Perform linear interpolation
        value = y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    
    return value, ilim

