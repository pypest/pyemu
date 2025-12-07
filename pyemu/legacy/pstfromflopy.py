import copy
import os
import shutil
import warnings

import flopy
import numpy as np
import pandas as pd

import pyemu
from pyemu.pyemu_warnings import PyemuWarning
from pyemu.utils import (
    SpatialReference,
    kl_setup,
    apply_array_pars,
    geostatistical_draws,
)
from pyemu.utils.helpers import _write_df_tpl


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


# TODO: deprecate? Only used by PstFromFlopyModel
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


# TODO: deprecate? Only used by PstFromFlopyModel
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


# TODO: deprecate? Only used by PstFromFlopyModel
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
        # need to get the PAK name to handle stupid horrible exceptions for HFB...
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
    # (can't diff single entry groups)
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
    frun_line = "pyemu.legacy.apply_temporal_diff_obs('{0}')\n".format(config_file)
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
            flag to include stress-period level spatially-global multiplier parameters in addition to
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
        temporal_list_geostruct (`pyemu.geostats.GeoStruct`): the geostastical structure to
            build the prior parameter covariance matrix
            for time-varying list-type multiplier parameters.  This GeoStruct
            express the time correlation so that the 'a' parameter is the length of
            time that boundary condition multiplier parameters are correlated across.
            If None, then a generic GeoStruct is created that uses an 'a' parameter
            of 3 stress periods.  Default is None
        spatial_list_geostruct (`pyemu.geostats.GeoStruct`): the geostastical structure to
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
        """setup the flopy.mbase instance for use with multiplier parameters.
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
        except AttributeError:  # if sr doesn't exist anymore!
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
        """prepare multiplier arrays.  Copies existing model input arrays and
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
        pp_df.loc[:, "fac_file"] = np.nan
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
        # mlt_df.loc[:,"fac_file"] = np.nan
        # mlt_df.loc[:,"pp_file"] = np.nan
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

        mlt_df.loc[mlt_df.suffix == self.pp_suffix, "tpl_file"] = np.nan

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
        mlt_df.loc[mlt_df.suffix == self.kl_suffix, "tpl_file"] = np.nan
        self.par_dfs[self.kl_suffix] = kl_df
        # calc factors for each layer

    def _setup_array_pars(self):
        """main entry point for setting up array multiplier parameters"""
        mlt_df = self._prep_mlt_arrays()
        if mlt_df is None:
            return
        mlt_df.loc[:, "tpl_file"] = mlt_df.mlt_file.apply(
            lambda x: os.path.split(x)[-1] + ".tpl"
        )
        # mlt_df.loc[mlt_df.tpl_file.apply(lambda x:pd.notnull(x.pp_file)),"tpl_file"] = np.nan
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
            filename (`str`): the filename to save the control file to.  If None, the
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
        line = "pyemu.legacy.apply_list_pars()\n"
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
        info_df.loc[:, "itmp"] = np.nan
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
        from pyemu.utils import setup_hds_obs

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
            skip = lambda x: np.nan if x == self.m.bas6.hnoflo else x
        else:
            skip = lambda x: np.nan if x == self.m.bas6.hnoflo or x == inact else x
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
                "could not find hob out file: {0}...skipping".format(org_hob_out_fname)
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

