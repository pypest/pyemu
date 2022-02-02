"""MODFLOW support utilities"""
import os
from datetime import datetime
import shutil
import warnings
import numpy as np
import pandas as pd
import re

pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import (
    SFMT,
    IFMT,
    FFMT,
    pst_config,
    parse_tpl_file,
    try_process_output_file,
)
from pyemu.utils.os_utils import run
from pyemu.utils.helpers import _write_df_tpl
from ..pyemu_warnings import PyemuWarning

PP_FMT = {
    "name": SFMT,
    "x": FFMT,
    "y": FFMT,
    "zone": IFMT,
    "tpl": SFMT,
    "parval1": FFMT,
}
PP_NAMES = ["name", "x", "y", "zone", "parval1"]


def modflow_pval_to_template_file(pval_file, tpl_file=None):
    """write a template file for a modflow parameter value file.

    Args:
        pval_file (`str`): the path and name of the existing modflow pval file
        tpl_file (`str`, optional):  template file to write. If None, use
            `pval_file` +".tpl". Default is None
    Note:
        Uses names in the first column in the pval file as par names.

    Returns:
        **pandas.DataFrame**: a dataFrame with control file parameter information
    """

    if tpl_file is None:
        tpl_file = pval_file + ".tpl"
    pval_df = pd.read_csv(
        pval_file,
        delim_whitespace=True,
        header=None,
        skiprows=2,
        names=["parnme", "parval1"],
    )
    pval_df.index = pval_df.parnme
    pval_df.loc[:, "tpl"] = pval_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x))
    with open(tpl_file, "w") as f:
        f.write("ptf ~\n#pval template file from pyemu\n")
        f.write("{0:10d} #NP\n".format(pval_df.shape[0]))
        f.write(
            pval_df.loc[:, ["parnme", "tpl"]].to_string(
                col_space=0,
                formatters=[SFMT, SFMT],
                index=False,
                header=False,
                justify="left",
            )
        )
    return pval_df


def modflow_hob_to_instruction_file(hob_file, ins_file=None):
    """write an instruction file for a modflow head observation file

    Args:
        hob_file (`str`): the path and name of the existing modflow hob file
        ins_file (`str`, optional): the name of the instruction file to write.
            If `None`, `hob_file` +".ins" is used.  Default is `None`.

    Returns:
        **pandas.DataFrame**: a dataFrame with control file observation information

    """

    hob_df = pd.read_csv(
        hob_file,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        names=["simval", "obsval", "obsnme"],
    )

    hob_df.loc[:, "obsnme"] = hob_df.obsnme.apply(str.lower)
    hob_df.loc[:, "ins_line"] = hob_df.obsnme.apply(lambda x: "l1 !{0:s}!".format(x))
    hob_df.loc[0, "ins_line"] = hob_df.loc[0, "ins_line"].replace("l1", "l2")

    if ins_file is None:
        ins_file = hob_file + ".ins"
    f_ins = open(ins_file, "w")
    f_ins.write("pif ~\n")
    f_ins.write(
        hob_df.loc[:, ["ins_line"]].to_string(
            col_space=0,
            columns=["ins_line"],
            header=False,
            index=False,
            formatters=[SFMT],
        )
        + "\n"
    )
    hob_df.loc[:, "weight"] = 1.0
    hob_df.loc[:, "obgnme"] = "obgnme"
    f_ins.close()
    return hob_df


def modflow_hydmod_to_instruction_file(hydmod_file, ins_file=None):
    """write an instruction file for a modflow hydmod file

    Args:
        hydmod_file (`str`): the path and name of the existing modflow hob file
        ins_file (`str`, optional): the name of the instruction file to write.
            If `None`, `hydmod_file` +".ins" is used.  Default is `None`.

    Returns:
        **pandas.DataFrame**: a dataFrame with control file observation information

    Note:
        calls `pyemu.gw_utils.modflow_read_hydmod_file()`
    """

    hydmod_df, hydmod_outfile = modflow_read_hydmod_file(hydmod_file)
    hydmod_df.loc[:, "ins_line"] = hydmod_df.obsnme.apply(
        lambda x: "l1 w !{0:s}!".format(x)
    )

    if ins_file is None:
        ins_file = hydmod_outfile + ".ins"

    with open(ins_file, "w") as f_ins:
        f_ins.write("pif ~\nl1\n")
        f_ins.write(
            hydmod_df.loc[:, ["ins_line"]].to_string(
                col_space=0,
                columns=["ins_line"],
                header=False,
                index=False,
                formatters=[SFMT],
            )
            + "\n"
        )
    hydmod_df.loc[:, "weight"] = 1.0
    hydmod_df.loc[:, "obgnme"] = "obgnme"

    df = try_process_output_file(hydmod_outfile + ".ins")
    if df is not None:
        df.loc[:, "obsnme"] = df.index.values
        df.loc[:, "obgnme"] = df.obsnme.apply(lambda x: x[:-9])
        df.to_csv("_setup_" + os.path.split(hydmod_outfile)[-1] + ".csv", index=False)
        return df

    return hydmod_df


def modflow_read_hydmod_file(hydmod_file, hydmod_outfile=None):
    """read a binary hydmod file and return a dataframe of the results

    Args:
        hydmod_file (`str`): The path and name of the existing modflow hydmod binary file
        hydmod_outfile (`str`, optional): output file to write.  If `None`, use `hydmod_file` +".dat".
            Default is `None`.

    Returns:
        **pandas.DataFrame**: a dataFrame with hymod_file values
    """
    try:
        import flopy.utils as fu
    except Exception as e:
        print("flopy is not installed - cannot read {0}\n{1}".format(hydmod_file, e))
        return

    obs = fu.HydmodObs(hydmod_file)
    hyd_df = obs.get_dataframe()

    hyd_df.columns = [i[2:] if i.lower() != "totim" else i for i in hyd_df.columns]
    # hyd_df.loc[:,"datetime"] = hyd_df.index
    hyd_df["totim"] = hyd_df.index.map(lambda x: x.strftime("%Y%m%d"))

    hyd_df.rename(columns={"totim": "datestamp"}, inplace=True)

    # reshape into a single column
    hyd_df = pd.melt(hyd_df, id_vars="datestamp")

    hyd_df.rename(columns={"value": "obsval"}, inplace=True)

    hyd_df["obsnme"] = [
        i.lower() + "_" + j.lower() for i, j in zip(hyd_df.variable, hyd_df.datestamp)
    ]

    vc = hyd_df.obsnme.value_counts().sort_values()
    vc = list(vc.loc[vc > 1].index.values)
    if len(vc) > 0:
        hyd_df.to_csv("hyd_df.duplciates.csv")
        obs.get_dataframe().to_csv("hyd_org.duplicates.csv")
        raise Exception("duplicates in obsnme:{0}".format(vc))
    # assert hyd_df.obsnme.value_counts().max() == 1,"duplicates in obsnme"

    if not hydmod_outfile:
        hydmod_outfile = hydmod_file + ".dat"
    hyd_df.to_csv(hydmod_outfile, columns=["obsnme", "obsval"], sep=" ", index=False)
    # hyd_df = hyd_df[['obsnme','obsval']]
    return hyd_df[["obsnme", "obsval"]], hydmod_outfile


def setup_mtlist_budget_obs(
    list_filename,
    gw_filename="mtlist_gw.dat",
    sw_filename="mtlist_sw.dat",
    start_datetime="1-1-1970",
    gw_prefix="gw",
    sw_prefix="sw",
    save_setup_file=False,
):
    """setup observations of gw (and optionally sw) mass budgets from mt3dusgs list file.

    Args:
        list_filename (`str`): path and name of existing modflow list file
        gw_filename (`str`, optional): output filename that will contain the gw budget
            observations. Default is "mtlist_gw.dat"
        sw_filename (`str`, optional): output filename that will contain the sw budget
            observations. Default is "mtlist_sw.dat"
        start_datetime (`str`, optional):  an str that can be parsed into a `pandas.TimeStamp`.
            used to give budget observations meaningful names.  Default is "1-1-1970".
        gw_prefix (`str`, optional): a prefix to add to the GW budget observations.
            Useful if processing more than one list file as part of the forward run process.
            Default is 'gw'.
        sw_prefix (`str`, optional): a prefix to add to the SW budget observations.  Useful
            if processing more than one list file as part of the forward run process.
            Default is 'sw'.
        save_setup_file (`bool`, optional): a flag to save "_setup_"+ `list_filename` +".csv" file
            that contains useful control file information.  Default is `False`.

    Returns:
        tuple containing

        - **str**:  the command to add to the forward run script
        - **str**: the names of the instruction files that were created
        - **pandas.DataFrame**: a dataframe with information for constructing a control file


    Note:
        writes an instruction file and also a _setup_.csv to use when constructing a pest
        control file

        The instruction files are named `out_filename` +".ins"

        It is recommended to use the default value for `gw_filename` or `sw_filename`.

        This is the companion function of `gw_utils.apply_mtlist_budget_obs()`.

    """
    gw, sw = apply_mtlist_budget_obs(
        list_filename, gw_filename, sw_filename, start_datetime
    )
    gw_ins = gw_filename + ".ins"
    _write_mtlist_ins(gw_ins, gw, gw_prefix)
    ins_files = [gw_ins]

    df_gw = try_process_output_file(gw_ins, gw_filename)
    if df_gw is None:
        raise Exception("error processing groundwater instruction file")
    if sw is not None:
        sw_ins = sw_filename + ".ins"
        _write_mtlist_ins(sw_ins, sw, sw_prefix)
        ins_files.append(sw_ins)

        df_sw = try_process_output_file(sw_ins, sw_filename)
        if df_sw is None:
            raise Exception("error processing surface water instruction file")
        df_gw = pd.concat([df_gw, df_sw])
        df_gw.loc[:, "obsnme"] = df_gw.index.values
    if save_setup_file:
        df_gw.to_csv("_setup_" + os.path.split(list_filename)[-1] + ".csv", index=False)

    frun_line = "pyemu.gw_utils.apply_mtlist_budget_obs('{0}')".format(list_filename)
    return frun_line, ins_files, df_gw


def _write_mtlist_ins(ins_filename, df, prefix):
    """write an instruction file for a MT3D-USGS list file"""
    try:
        dt_str = df.index.map(lambda x: x.strftime("%Y%m%d"))
    except:
        dt_str = df.index.map(lambda x: "{0:08.1f}".format(x).strip())
    with open(ins_filename, "w") as f:
        f.write("pif ~\nl1\n")
        for dt in dt_str:
            f.write("l1 ")
            for col in df.columns.str.translate(
                {ord(s): None for s in ["(", ")", "/", "="]}
            ):
                if prefix == "":
                    obsnme = "{0}_{1}".format(col, dt)
                else:
                    obsnme = "{0}_{1}_{2}".format(prefix, col, dt)
                f.write(" w !{0}!".format(obsnme))
            f.write("\n")


def apply_mtlist_budget_obs(
    list_filename,
    gw_filename="mtlist_gw.dat",
    sw_filename="mtlist_sw.dat",
    start_datetime="1-1-1970",
):
    """process an MT3D-USGS list file to extract mass budget entries.

    Args:
        list_filename (`str`): the path and name of an existing MT3D-USGS list file
        gw_filename (`str`, optional): the name of the output file with gw mass
            budget information. Default is "mtlist_gw.dat"
        sw_filename (`str`): the name of the output file with sw mass budget information.
            Default is "mtlist_sw.dat"
        start_datatime (`str`): an str that can be cast to a pandas.TimeStamp.  Used to give
            observations a meaningful name

    Returns:
        2-element tuple containing

        - **pandas.DataFrame**: the gw mass budget dataframe
        - **pandas.DataFrame**: (optional) the sw mass budget dataframe.
          If the SFT process is not active, this returned value is `None`.

    Note:
        This is the companion function of `gw_utils.setup_mtlist_budget_obs()`.
    """
    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    mt = flopy.utils.MtListBudget(list_filename)
    gw, sw = mt.parse(start_datetime=start_datetime, diff=True)
    gw = gw.drop(
        [
            col
            for col in gw.columns
            for drop_col in ["kper", "kstp", "tkstp"]
            if (col.lower().startswith(drop_col))
        ],
        axis=1,
    )
    gw.to_csv(gw_filename, sep=" ", index_label="datetime", date_format="%Y%m%d")
    if sw is not None:
        sw = sw.drop(
            [
                col
                for col in sw.columns
                for drop_col in ["kper", "kstp", "tkstp"]
                if (col.lower().startswith(drop_col))
            ],
            axis=1,
        )
        sw.to_csv(sw_filename, sep=" ", index_label="datetime", date_format="%Y%m%d")
    return gw, sw


def setup_mflist_budget_obs(
    list_filename,
    flx_filename="flux.dat",
    vol_filename="vol.dat",
    start_datetime="1-1'1970",
    prefix="",
    save_setup_file=False,
    specify_times=None,
):
    """setup observations of budget volume and flux from modflow list file.

    Args:
        list_filename (`str`): path and name of the existing modflow list file
        flx_filename (`str`, optional): output filename that will contain the budget flux
            observations. Default is "flux.dat"
        vol_filename (`str`, optional): output filename that will contain the budget volume
            observations.  Default is "vol.dat"
        start_datetime (`str`, optional): a string that can be parsed into a pandas.TimeStamp.
            This is used to give budget observations meaningful names.  Default is "1-1-1970".
        prefix (`str`, optional): a prefix to add to the water budget observations.  Useful if
            processing more than one list file as part of the forward run process. Default is ''.
        save_setup_file (`bool`): a flag to save "_setup_"+ `list_filename` +".csv" file that contains useful
            control file information
        specify_times (`np.ndarray`-like, optional): An array of times to
             extract from the budget dataframes returned by the flopy
             MfListBudget(list_filename).get_dataframe() method. This can be
             useful to ensure consistent observation times for PEST.
             Array needs to be alignable with index of dataframe
             return by flopy method, care should be take to ensure that
             this is the case. If passed will be written to
             "budget_times.config" file as strings to be read by the companion
             `apply_mflist_budget_obs()` method at run time.

    Returns:
        **pandas.DataFrame**: a dataframe with information for constructing a control file.

    Note:
        This method writes instruction files and also a _setup_.csv to use when constructing a pest
        control file.  The instruction files are named <flux_file>.ins and <vol_file>.ins, respectively

        It is recommended to use the default values for flux_file and vol_file.

        This is the companion function of `gw_utils.apply_mflist_budget_obs()`.


    """
    flx, vol = apply_mflist_budget_obs(
        list_filename, flx_filename, vol_filename, start_datetime, times=specify_times
    )
    _write_mflist_ins(flx_filename + ".ins", flx, prefix + "flx")
    _write_mflist_ins(vol_filename + ".ins", vol, prefix + "vol")

    df = try_process_output_file(flx_filename + ".ins")
    if df is None:
        raise Exception("error processing flux instruction file")

    df2 = try_process_output_file(vol_filename + ".ins")
    if df2 is None:
        raise Exception("error processing volume instruction file")

    df = pd.concat([df, df2])
    df.loc[:, "obsnme"] = df.index.values
    if save_setup_file:
        df.to_csv("_setup_" + os.path.split(list_filename)[-1] + ".csv", index=False)
    if specify_times is not None:
        np.savetxt(
            os.path.join(os.path.dirname(flx_filename), "budget_times.config"),
            specify_times,
            fmt="%s",
        )
    return df


def apply_mflist_budget_obs(
    list_filename,
    flx_filename="flux.dat",
    vol_filename="vol.dat",
    start_datetime="1-1-1970",
    times=None,
):
    """process a MODFLOW list file to extract flux and volume water budget
       entries.

    Args:
         list_filename (`str`): path and name of the existing modflow list file
         flx_filename (`str`, optional): output filename that will contain the
             budget flux observations. Default is "flux.dat"
         vol_filename (`str`, optional): output filename that will contain the
             budget volume observations.  Default is "vol.dat"
         start_datetime (`str`, optional): a string that can be parsed into a
             pandas.TimeStamp. This is used to give budget observations
             meaningful names.  Default is "1-1-1970".
         times (`np.ndarray`-like or `str`, optional): An array of times to
             extract from the budget dataframes returned by the flopy
             MfListBudget(list_filename).get_dataframe() method. This can be
             useful to ensure consistent observation times for PEST.
             If type `str`, will assume `times=filename` and attempt to read
             single vector (no header or index) from file, parsing datetime
             using pandas. Array needs to be alignable with index of dataframe
             return by flopy method, care should be take to ensure that
             this is the case. If setup with `setup_mflist_budget_obs()`
             specifying `specify_times` argument `times` should be set to
             "budget_times.config".

     Note:
         This is the companion function of `gw_utils.setup_mflist_budget_obs()`.

     Returns:
         tuple containing

         - **pandas.DataFrame**: a dataframe with flux budget information
         - **pandas.DataFrame**: a dataframe with cumulative budget information

    """
    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    mlf = flopy.utils.MfListBudget(list_filename)
    flx, vol = mlf.get_dataframes(start_datetime=start_datetime, diff=True)
    if times is not None:
        if isinstance(times, str):
            if vol.index.tzinfo:
                parse_date = {"t": [0]}
                names = [None]
            else:
                parse_date = False
                names = ["t"]
            times = pd.read_csv(
                times, header=None, names=names, parse_dates=parse_date
            )["t"].values
        flx = flx.loc[times]
        vol = vol.loc[times]
    flx.to_csv(flx_filename, sep=" ", index_label="datetime", date_format="%Y%m%d")
    vol.to_csv(vol_filename, sep=" ", index_label="datetime", date_format="%Y%m%d")
    return flx, vol


def _write_mflist_ins(ins_filename, df, prefix):
    """write an instruction file for a MODFLOW list file"""

    dt_str = df.index.map(lambda x: x.strftime("%Y%m%d"))
    with open(ins_filename, "w") as f:
        f.write("pif ~\nl1\n")
        for dt in dt_str:
            f.write("l1 ")
            for col in df.columns:
                obsnme = "{0}_{1}_{2}".format(prefix, col, dt)
                f.write(" w !{0}!".format(obsnme))
            f.write("\n")


def setup_hds_timeseries(
    bin_file,
    kij_dict,
    prefix=None,
    include_path=False,
    model=None,
    postprocess_inact=None,
    text=None,
    fill=None,
    precision="single",
):
    """a function to setup a forward process to extract time-series style values
    from a binary modflow binary file (or equivalent format - hds, ucn, sub, cbb, etc).

    Args:
        bin_file (`str`): path and name of existing modflow binary file - headsave, cell budget and MT3D UCN supported.
        kij_dict (`dict`): dictionary of site_name: [k,i,j] pairs. For example: `{"wel1":[0,1,1]}`.
        prefix (`str`, optional): string to prepend to site_name when forming observation names.  Default is None
        include_path (`bool`, optional): flag to setup the binary file processing in directory where the hds_file
            is located (if different from where python is running).  This is useful for setting up
            the process in separate directory for where python is running.
        model (`flopy.mbase`, optional): a `flopy.basemodel` instance.  If passed, the observation names will
            have the datetime of the observation appended to them (using the flopy `start_datetime` attribute.
            If None, the observation names will have the zero-based stress period appended to them. Default is None.
        postprocess_inact (`float`, optional): Inactive value in heads/ucn file e.g. mt.btn.cinit.  If `None`, no
            inactive value processing happens.  Default is `None`.
        text (`str`): the text record entry in the binary file (e.g. "constant_head").
            Used to indicate that the binary file is a MODFLOW cell-by-cell budget file.
            If None, headsave or MT3D unformatted concentration file
            is assummed.  Default is None
        fill (`float`): fill value for NaNs in the extracted timeseries dataframe.  If
            `None`, no filling is done, which may yield model run failures as the resulting
            processed timeseries CSV file (produced at runtime) may have missing values and
            can't be processed with the cooresponding instruction file.  Default is `None`.
        precision (`str`): the precision of the binary file.  Can be "single" or "double".
            Default is "single".

    Returns:
        tuple containing

        - **str**: the forward run command to execute the binary file process during model runs.

        - **pandas.DataFrame**: a dataframe of observation information for use in the pest control file

    Note:
        This function writes hds_timeseries.config that must be in the same
        dir where `apply_hds_timeseries()` is called during the forward run

        Assumes model time units are days

        This is the companion function of `gw_utils.apply_hds_timeseries()`.

    """

    try:
        import flopy
    except Exception as e:
        print("error importing flopy, returning {0}".format(str(e)))
        return

    assert os.path.exists(bin_file), "binary file not found"
    iscbc = False
    if text is not None:
        text = text.upper()

        try:
            # hack: if model is passed and its None, it trips up CellBudgetFile...
            if model is not None:
                bf = flopy.utils.CellBudgetFile(
                    bin_file, precision=precision, model=model
                )
                iscbc = True
            else:
                bf = flopy.utils.CellBudgetFile(bin_file, precision=precision)
                iscbc = True
        except Exception as e:
            try:
                if model is not None:
                    bf = flopy.utils.HeadFile(
                        bin_file, precision=precision, model=model, text=text
                    )
                else:
                    bf = flopy.utils.HeadFile(bin_file, precision=precision, text=text)
            except Exception as e1:
                raise Exception(
                    "error instantiating binary file as either CellBudgetFile:{0} or as HeadFile with text arg: {1}".format(
                        str(e), str(e1)
                    )
                )
        if iscbc:
            tl = [t.decode().strip() for t in bf.textlist]
            if text not in tl:
                raise Exception(
                    "'text' {0} not found in CellBudgetFile.textlist:{1}".format(
                        text, tl
                    )
                )
    elif bin_file.lower().endswith(".ucn"):
        try:
            bf = flopy.utils.UcnFile(bin_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            bf = flopy.utils.HeadFile(bin_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    if text is None:
        text = "none"

    nlay, nrow, ncol = bf.nlay, bf.nrow, bf.ncol

    # if include_path:
    #    pth = os.path.join(*[p for p in os.path.split(hds_file)[:-1]])
    #    config_file = os.path.join(pth,"{0}_timeseries.config".format(hds_file))
    # else:
    config_file = "{0}_timeseries.config".format(bin_file)
    print("writing config file to {0}".format(config_file))
    if fill is None:
        fill = "none"
    f_config = open(config_file, "w")
    if model is not None:
        if model.dis.itmuni != 4:
            warnings.warn(
                "setup_hds_timeseries only supports 'days' time units...", PyemuWarning
            )
        f_config.write(
            "{0},{1},d,{2},{3},{4},{5}\n".format(
                os.path.split(bin_file)[-1],
                model.start_datetime,
                text,
                fill,
                precision,
                iscbc,
            )
        )
        start = pd.to_datetime(model.start_datetime)
    else:
        f_config.write(
            "{0},none,none,{1},{2},{3},{4}\n".format(
                os.path.split(bin_file)[-1], text, fill, precision, iscbc
            )
        )
    f_config.write("site,k,i,j\n")
    dfs = []

    for site, (k, i, j) in kij_dict.items():
        assert k >= 0 and k < nlay, k
        assert i >= 0 and i < nrow, i
        assert j >= 0 and j < ncol, j
        site = site.lower().replace(" ", "")
        if iscbc:
            ts = bf.get_ts((k, i, j), text=text)
            # print(ts)
            df = pd.DataFrame(data=ts, columns=["totim", site])
        else:
            df = pd.DataFrame(data=bf.get_ts((k, i, j)), columns=["totim", site])

        if model is not None:
            dts = start + pd.to_timedelta(df.totim, unit="d")
            df.loc[:, "totim"] = dts
        # print(df)
        f_config.write("{0},{1},{2},{3}\n".format(site, k, i, j))
        df.index = df.pop("totim")
        dfs.append(df)

    f_config.close()
    df = pd.concat(dfs, axis=1).T
    df.to_csv(bin_file + "_timeseries.processed", sep=" ")
    if model is not None:
        t_str = df.columns.map(lambda x: x.strftime("%Y%m%d"))
    else:
        t_str = df.columns.map(lambda x: "{0:08.2f}".format(x))

    ins_file = bin_file + "_timeseries.processed.ins"
    print("writing instruction file to {0}".format(ins_file))
    with open(ins_file, "w") as f:
        f.write("pif ~\n")
        f.write("l1 \n")
        for site in df.index:
            # for t in t_str:
            f.write("l1 w ")
            # for site in df.columns:
            for t in t_str:
                if prefix is not None:
                    obsnme = "{0}_{1}_{2}".format(prefix, site, t)
                else:
                    obsnme = "{0}_{1}".format(site, t)
                f.write(" !{0}!".format(obsnme))
            f.write("\n")
    if postprocess_inact is not None:
        _setup_postprocess_hds_timeseries(
            bin_file, df, config_file, prefix=prefix, model=model
        )
    bd = "."
    if include_path:
        bd = os.getcwd()
        pth = os.path.join(*[p for p in os.path.split(bin_file)[:-1]])
        os.chdir(pth)
    config_file = os.path.split(config_file)[-1]
    try:
        df = apply_hds_timeseries(config_file, postprocess_inact=postprocess_inact)

    except Exception as e:
        os.chdir(bd)
        raise Exception("error in apply_hds_timeseries(): {0}".format(str(e)))
    os.chdir(bd)

    df = try_process_output_file(ins_file)
    if df is None:
        raise Exception("error processing {0} instruction file".format(ins_file))

    df.loc[:, "weight"] = 0.0
    if prefix is not None:
        df.loc[:, "obgnme"] = df.index.map(lambda x: "_".join(x.split("_")[:2]))
    else:
        df.loc[:, "obgnme"] = df.index.map(lambda x: x.split("_")[0])
    frun_line = "pyemu.gw_utils.apply_hds_timeseries('{0}',{1})\n".format(
        config_file, postprocess_inact
    )
    return frun_line, df


def apply_hds_timeseries(config_file=None, postprocess_inact=None):
    """process a modflow binary file using a previously written
    configuration file

    Args:
        config_file (`str`, optional): configuration file written by `pyemu.gw_utils.setup_hds_timeseries`.
            If `None`, looks for `hds_timeseries.config`
        postprocess_inact (`float`, optional): Inactive value in heads/ucn file e.g. mt.btn.cinit.  If `None`, no
            inactive value processing happens.  Default is `None`.

    Note:
        This is the companion function of `gw_utils.setup_hds_timeseries()`.

    """
    import flopy

    if config_file is None:
        config_file = "hds_timeseries.config"

    assert os.path.exists(config_file), config_file
    with open(config_file, "r") as f:
        line = f.readline()
        (
            bf_file,
            start_datetime,
            time_units,
            text,
            fill,
            precision,
            _iscbc,
        ) = line.strip().split(",")
        if len(line.strip().split(",")) == 6:
            (
                bf_file,
                start_datetime,
                time_units,
                text,
                fill,
                precision,
            ) = line.strip().split(",")
            _iscbc = "false"
        else:
            (
                bf_file,
                start_datetime,
                time_units,
                text,
                fill,
                precision,
                _iscbc,
            ) = line.strip().split(",")
        site_df = pd.read_csv(f)
    text = text.upper()
    if _iscbc.lower().strip() == "false":
        iscbc = False
    elif _iscbc.lower().strip() == "true":
        iscbc = True
    else:
        raise Exception(
            "apply_hds_timeseries() error: unrecognized 'iscbc' string in config file: {0}".format(
                _iscbc
            )
        )
    assert os.path.exists(bf_file), "head save file not found"
    if iscbc:
        try:
            bf = flopy.utils.CellBudgetFile(bf_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating CellBudgetFile:{0}".format(str(e)))
    elif bf_file.lower().endswith(".ucn"):
        try:
            bf = flopy.utils.UcnFile(bf_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            if text != "NONE":
                bf = flopy.utils.HeadFile(bf_file, text=text, precision=precision)
            else:
                bf = flopy.utils.HeadFile(bf_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    nlay, nrow, ncol = bf.nlay, bf.nrow, bf.ncol

    dfs = []
    for site, k, i, j in zip(site_df.site, site_df.k, site_df.i, site_df.j):
        assert k >= 0 and k < nlay
        assert i >= 0 and i < nrow
        assert j >= 0 and j < ncol
        if iscbc:
            df = pd.DataFrame(
                data=bf.get_ts((k, i, j), text=text), columns=["totim", site]
            )
        else:
            df = pd.DataFrame(data=bf.get_ts((k, i, j)), columns=["totim", site])
        df.index = df.pop("totim")
        dfs.append(df)
    df = pd.concat(dfs, axis=1).T
    if df.shape != df.dropna().shape:
        warnings.warn("NANs in processed timeseries file", PyemuWarning)
        if fill.upper() != "NONE":
            fill = float(fill)
            df.fillna(fill, inplace=True)
    # print(df)
    df.to_csv(bf_file + "_timeseries.processed", sep=" ")
    if postprocess_inact is not None:
        _apply_postprocess_hds_timeseries(config_file, postprocess_inact)
    return df


def _setup_postprocess_hds_timeseries(
    hds_file, df, config_file, prefix=None, model=None
):
    """Dirty function to setup post processing concentrations in inactive/dry cells"""

    warnings.warn(
        "Setting up post processing of hds or ucn timeseries obs. "
        "Prepending 'pp' to obs name may cause length to exceed 20 chars",
        PyemuWarning,
    )
    if model is not None:
        t_str = df.columns.map(lambda x: x.strftime("%Y%m%d"))
    else:
        t_str = df.columns.map(lambda x: "{0:08.2f}".format(x))
    if prefix is not None:
        prefix = "pp{0}".format(prefix)
    else:
        prefix = "pp"
    ins_file = hds_file + "_timeseries.post_processed.ins"
    print("writing instruction file to {0}".format(ins_file))
    with open(ins_file, "w") as f:
        f.write("pif ~\n")
        f.write("l1 \n")
        for site in df.index:
            f.write("l1 w ")
            # for site in df.columns:
            for t in t_str:
                obsnme = "{0}{1}_{2}".format(prefix, site, t)
                f.write(" !{0}!".format(obsnme))
            f.write("\n")
    frun_line = "pyemu.gw_utils._apply_postprocess_hds_timeseries('{0}')\n".format(
        config_file
    )
    return frun_line


def _apply_postprocess_hds_timeseries(config_file=None, cinact=1e30):
    """private function to post processing binary files"""
    import flopy

    if config_file is None:
        config_file = "hds_timeseries.config"

    assert os.path.exists(config_file), config_file
    with open(config_file, "r") as f:
        line = f.readline()
        (
            hds_file,
            start_datetime,
            time_units,
            text,
            fill,
            precision,
            _iscbc,
        ) = line.strip().split(",")
        if len(line.strip().split(",")) == 6:
            (
                hds_file,
                start_datetime,
                time_units,
                text,
                fill,
                precision,
            ) = line.strip().split(",")
            _iscbc = "false"
        else:
            (
                hds_file,
                start_datetime,
                time_units,
                text,
                fill,
                precision,
                _iscbc,
            ) = line.strip().split(",")
        site_df = pd.read_csv(f)

    # print(site_df)
    text = text.upper()
    assert os.path.exists(hds_file), "head save file not found"
    if hds_file.lower().endswith(".ucn"):
        try:
            hds = flopy.utils.UcnFile(hds_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            if text != "NONE":
                hds = flopy.utils.HeadFile(hds_file, text=text, precision=precision)
            else:
                hds = flopy.utils.HeadFile(hds_file, precision=precision)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    nlay, nrow, ncol = hds.nlay, hds.nrow, hds.ncol

    dfs = []
    for site, k, i, j in zip(site_df.site, site_df.k, site_df.i, site_df.j):
        assert k >= 0 and k < nlay
        assert i >= 0 and i < nrow
        assert j >= 0 and j < ncol
        if text.upper() != "NONE":
            df = pd.DataFrame(data=hds.get_ts((k, i, j)), columns=["totim", site])
        else:
            df = pd.DataFrame(data=hds.get_ts((k, i, j)), columns=["totim", site])
        df.index = df.pop("totim")
        inact_obs = df[site].apply(lambda x: np.isclose(x, cinact))
        if inact_obs.sum() > 0:
            assert k + 1 < nlay, "Inactive observation in lowest layer"
            df_lower = pd.DataFrame(
                data=hds.get_ts((k + 1, i, j)), columns=["totim", site]
            )
            df_lower.index = df_lower.pop("totim")
            df.loc[inact_obs] = df_lower.loc[inact_obs]
            print(
                "{0} observation(s) post-processed for site {1} at kij ({2},{3},{4})".format(
                    inact_obs.sum(), site, k, i, j
                )
            )
        dfs.append(df)
    df = pd.concat(dfs, axis=1).T
    # print(df)
    df.to_csv(hds_file + "_timeseries.post_processed", sep=" ")
    return df


def setup_hds_obs(
    hds_file,
    kperk_pairs=None,
    skip=None,
    prefix="hds",
    text="head",
    precision="single",
    include_path=False,
):
    """a function to setup using all values from a layer-stress period
    pair for observations.

    Args:
        hds_file (`str`): path and name of an existing MODFLOW head-save file.
            If the hds_file endswith 'ucn', then the file is treated as a UcnFile type.
        kperk_pairs ([(int,int)]): a list of len two tuples which are pairs of kper
            (zero-based stress period index) and k (zero-based layer index) to
            setup observations for.  If None, then all layers and stress period records
            found in the file will be used.  Caution: a shit-ton of observations may be produced!
        skip (variable, optional): a value or function used to determine which values
            to skip when setting up observations.  If np.scalar(skip)
            is True, then values equal to skip will not be used.
            If skip can also be a np.ndarry with dimensions equal to the model.
            Observations are set up only for cells with Non-zero values in the array.
            If not np.ndarray or np.scalar(skip), then skip will be treated as a lambda function that
            returns np.NaN if the value should be skipped.
        prefix (`str`): the prefix to use for the observation names. default is "hds".
        text (`str`): the text tag the flopy HeadFile instance.  Default is "head"
        precison (`str`): the precision string for the flopy HeadFile instance.  Default is "single"
        include_path (`bool`, optional): flag to setup the binary file processing in directory where the hds_file
        is located (if different from where python is running).  This is useful for setting up
            the process in separate directory for where python is running.

    Returns:
        tuple containing

        - **str**: the forward run script line needed to execute the headsave file observation
          operation
        - **pandas.DataFrame**: a dataframe of pest control file information

    Note:
        Writes an instruction file and a _setup_ csv used construct a control file.

        This is the companion function to `gw_utils.apply_hds_obs()`.

    """
    try:
        import flopy
    except Exception as e:
        print("error importing flopy, returning {0}".format(str(e)))
        return

    assert os.path.exists(hds_file), "head save file not found"
    if hds_file.lower().endswith(".ucn"):
        try:
            hds = flopy.utils.UcnFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    elif text.lower() == "headu":
        try:
            hds = flopy.utils.HeadUFile(hds_file, text=text, precision=precision)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))
    else:
        try:
            hds = flopy.utils.HeadFile(hds_file, text=text, precision=precision)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    if kperk_pairs is None:
        kperk_pairs = []
        for kstp, kper in hds.kstpkper:
            kperk_pairs.extend([(kper - 1, k) for k in range(hds.nlay)])
    if len(kperk_pairs) == 2:
        try:
            if len(kperk_pairs[0]) == 2:
                pass
        except:
            kperk_pairs = [kperk_pairs]

    # if start_datetime is not None:
    #    start_datetime = pd.to_datetime(start_datetime)
    #    dts = start_datetime + pd.to_timedelta(hds.times,unit='d')
    data = {}
    kpers = [kper - 1 for kstp, kper in hds.kstpkper]
    for kperk_pair in kperk_pairs:
        kper, k = kperk_pair
        assert kper in kpers, "kper not in hds:{0}".format(kper)
        assert k in range(hds.nlay), "k not in hds:{0}".format(k)
        kstp = last_kstp_from_kper(hds, kper)
        d = hds.get_data(kstpkper=(kstp, kper))[k]

        data["{0}_{1}".format(kper, k)] = d.flatten()
        # data[(kper,k)] = d.flatten()
    idx, iidx, jidx = [], [], []
    for _ in range(len(data)):
        for i in range(hds.nrow):
            iidx.extend([i for _ in range(hds.ncol)])
            jidx.extend([j for j in range(hds.ncol)])
            idx.extend(["i{0:04d}_j{1:04d}".format(i, j) for j in range(hds.ncol)])
    idx = idx[: hds.nrow * hds.ncol]

    df = pd.DataFrame(data, index=idx)
    data_cols = list(df.columns)
    data_cols.sort()
    # df.loc[:,"iidx"] = iidx
    # df.loc[:,"jidx"] = jidx
    if skip is not None:
        for col in data_cols:
            if np.isscalar(skip):
                df.loc[df.loc[:, col] == skip, col] = np.NaN
            elif isinstance(skip, np.ndarray):
                assert (
                    skip.ndim >= 2
                ), "skip passed as {}D array, At least 2D (<= 4D) array required".format(
                    skip.ndim
                )
                assert skip.shape[-2:] == (
                    hds.nrow,
                    hds.ncol,
                ), "Array dimensions of arg. skip needs to match model dimensions ({0},{1}). ({2},{3}) passed".format(
                    hds.nrow, hds.ncol, skip.shape[-2], skip.shape[-1]
                )
                if skip.ndim == 2:
                    print(
                        "2D array passed for skip, assuming constant for all layers and kper"
                    )
                    skip = np.tile(skip, (len(kpers), hds.nlay, 1, 1))
                if skip.ndim == 3:
                    print("3D array passed for skip, assuming constant for all kper")
                    skip = np.tile(skip, (len(kpers), 1, 1, 1))
                kper, k = [int(c) for c in col.split("_")]
                df.loc[
                    df.index.map(
                        lambda x: skip[
                            kper,
                            k,
                            int(x.split("_")[0].strip("i")),
                            int(x.split("_")[1].strip("j")),
                        ]
                        == 0
                    ),
                    col,
                ] = np.NaN
            else:
                df.loc[:, col] = df.loc[:, col].apply(skip)

    # melt to long form
    df = df.melt(var_name="kperk", value_name="obsval")
    # set row and col identifies
    df.loc[:, "iidx"] = iidx
    df.loc[:, "jidx"] = jidx
    # drop nans from skip
    df = df.dropna()
    # set some additional identifiers
    df.loc[:, "kper"] = df.kperk.apply(lambda x: int(x.split("_")[0]))
    df.loc[:, "kidx"] = df.pop("kperk").apply(lambda x: int(x.split("_")[1]))

    # form obs names
    # def get_kper_str(kper):
    #    if start_datetime is not None:
    #        return  dts[int(kper)].strftime("%Y%m%d")
    #    else:
    #        return "kper{0:04.0f}".format(kper)
    fmt = prefix + "_{0:02.0f}_{1:03.0f}_{2:03.0f}_{3:03.0f}"
    # df.loc[:,"obsnme"] = df.apply(lambda x: fmt.format(x.kidx,x.iidx,x.jidx,
    #                                                   get_kper_str(x.kper)),axis=1)
    df.loc[:, "obsnme"] = df.apply(
        lambda x: fmt.format(x.kidx, x.iidx, x.jidx, x.kper), axis=1
    )

    df.loc[:, "ins_str"] = df.obsnme.apply(lambda x: "l1 w !{0}!".format(x))
    df.loc[:, "obgnme"] = prefix
    # write the instruction file
    with open(hds_file + ".dat.ins", "w") as f:
        f.write("pif ~\nl1\n")
        df.ins_str.to_string(f, index=False, header=False)

    # write the corresponding output file
    df.loc[:, ["obsnme", "obsval"]].to_csv(hds_file + ".dat", sep=" ", index=False)

    hds_path = os.path.dirname(hds_file)
    setup_file = os.path.join(
        hds_path, "_setup_{0}.csv".format(os.path.split(hds_file)[-1])
    )
    df.to_csv(setup_file)
    if not include_path:
        hds_file = os.path.split(hds_file)[-1]
    fwd_run_line = (
        "pyemu.gw_utils.apply_hds_obs('{0}',precision='{1}',text='{2}')\n".format(
            hds_file, precision, text
        )
    )
    df.index = df.obsnme
    return fwd_run_line, df


def last_kstp_from_kper(hds, kper):
    """function to find the last time step (kstp) for a
    give stress period (kper) in a modflow head save file.

    Args:
        hds (`flopy.utils.HeadFile`): head save file
        kper (`int`): the zero-index stress period number

    Returns:
        **int**: the zero-based last time step during stress period
        kper in the head save file

    """
    # find the last kstp with this kper
    kstp = -1
    for kkstp, kkper in hds.kstpkper:
        if kkper == kper + 1 and kkstp > kstp:
            kstp = kkstp
    if kstp == -1:
        raise Exception("kstp not found for kper {0}".format(kper))
    kstp -= 1
    return kstp


def apply_hds_obs(hds_file, inact_abs_val=1.0e20, precision="single", text="head"):
    """process a modflow head save file.  A companion function to
    `gw_utils.setup_hds_obs()` that is called during the forward run process

    Args:
        hds_file (`str`): a modflow head save filename. if hds_file ends with 'ucn',
            then the file is treated as a UcnFile type.
        inact_abs_val (`float`, optional): the value that marks the mininum and maximum
            active value.  values in the headsave file greater than `inact_abs_val` or less
            than -`inact_abs_val` are reset to `inact_abs_val`
    Returns:
        **pandas.DataFrame**: a dataframe with extracted simulated values.
    Note:
        This is the companion function to `gw_utils.setup_hds_obs()`.

    """

    try:
        import flopy
    except Exception as e:
        raise Exception("apply_hds_obs(): error importing flopy: {0}".format(str(e)))
    from .. import pst_utils

    assert os.path.exists(hds_file)
    out_file = hds_file + ".dat"
    ins_file = out_file + ".ins"
    assert os.path.exists(ins_file)
    df = pd.DataFrame({"obsnme": pst_utils.parse_ins_file(ins_file)})
    df.index = df.obsnme

    # populate metdata
    items = ["k", "i", "j", "kper"]
    for i, item in enumerate(items):
        df.loc[:, item] = df.obsnme.apply(lambda x: int(x.split("_")[i + 1]))

    if hds_file.lower().endswith("ucn"):
        hds = flopy.utils.UcnFile(hds_file)
    elif text.lower() == "headu":
        hds = flopy.utils.HeadUFile(hds_file)
    else:
        hds = flopy.utils.HeadFile(hds_file, precision=precision, text=text)
    kpers = df.kper.unique()
    df.loc[:, "obsval"] = np.NaN
    for kper in kpers:
        kstp = last_kstp_from_kper(hds, kper)
        data = hds.get_data(kstpkper=(kstp, kper))
        # jwhite 15jan2018 fix for really large values that are getting some
        # trash added to them...
        if text.lower() != "headu":
            data[np.isnan(data)] = 0.0
            data[data > np.abs(inact_abs_val)] = np.abs(inact_abs_val)
            data[data < -np.abs(inact_abs_val)] = -np.abs(inact_abs_val)
            df_kper = df.loc[df.kper == kper, :]
            df.loc[df_kper.index, "obsval"] = data[df_kper.k, df_kper.i, df_kper.j]
        else:

            df_kper = df.loc[df.kper == kper, :]
            for k, d in enumerate(data):
                d[np.isnan(d)] = 0.0
                d[d > np.abs(inact_abs_val)] = np.abs(inact_abs_val)
                d[d < -np.abs(inact_abs_val)] = -np.abs(inact_abs_val)
                df_kperk = df_kper.loc[df_kper.k == k, :]
                df.loc[df_kperk.index, "obsval"] = d[df_kperk.i]

    assert df.dropna().shape[0] == df.shape[0]
    df.loc[:, ["obsnme", "obsval"]].to_csv(out_file, index=False, sep=" ")
    return df


def setup_sft_obs(sft_file, ins_file=None, start_datetime=None, times=None, ncomp=1):
    """writes a post-processor and instruction file for a mt3d-usgs sft output file

    Args:
        sft_file (`str`): path and name of an existing sft output file (ASCII)
        ins_file (`str`, optional): the name of the instruction file to create.
            If None, the name is `sft_file`+".ins".  Default is `None`.
        start_datetime (`str`): a pandas.to_datetime() compatible str.  If not None,
            then the resulting observation names have the datetime
            suffix.  If None, the suffix is the output totim.  Default
            is `None`.
        times ([`float`]): a list of times to make observations for.  If None, all times
            found in the file are used. Default is None.
        ncomp (`int`): number of components in transport model. Default is 1.

    Returns:
        **pandas.DataFrame**: a dataframe with observation names and values for the sft simulated
        concentrations.

    Note:
        This is the companion function to `gw_utils.apply_sft_obs()`.

    """

    df = pd.read_csv(sft_file, skiprows=1, delim_whitespace=True)
    df.columns = [c.lower().replace("-", "_") for c in df.columns]
    if times is None:
        times = df.time.unique()
    missing = []
    utimes = df.time.unique()
    for t in times:
        if t not in utimes:
            missing.append(str(t))
    if len(missing) > 0:
        print(df.time)
        raise Exception("the following times are missing:{0}".format(",".join(missing)))
    with open("sft_obs.config", "w") as f:
        f.write(sft_file + "\n")
        [f.write("{0:15.6E}\n".format(t)) for t in times]
    df = apply_sft_obs()
    utimes = df.time.unique()
    for t in times:
        assert t in utimes, "time {0} missing in processed dataframe".format(t)
    idx = df.time.apply(lambda x: x in times)
    if start_datetime is not None:
        start_datetime = pd.to_datetime(start_datetime)
        df.loc[:, "time_str"] = pd.to_timedelta(df.time, unit="d") + start_datetime
        df.loc[:, "time_str"] = df.time_str.apply(
            lambda x: datetime.strftime(x, "%Y%m%d")
        )
    else:
        df.loc[:, "time_str"] = df.time.apply(lambda x: "{0:08.2f}".format(x))
    df.loc[:, "ins_str"] = "l1\n"
    # check for multiple components
    df_times = df.loc[idx, :]
    df.loc[:, "icomp"] = 1
    icomp_idx = list(df.columns).index("icomp")
    for t in times:
        df_time = df.loc[df.time == t, :].copy()
        vc = df_time.sfr_node.value_counts()
        ncomp = vc.max()
        assert np.all(vc.values == ncomp)
        nstrm = df_time.shape[0] / ncomp
        for icomp in range(ncomp):
            s = int(nstrm * (icomp))
            e = int(nstrm * (icomp + 1))
            idxs = df_time.iloc[s:e, :].index
            # df_time.iloc[nstrm*(icomp):nstrm*(icomp+1),icomp_idx.loc["icomp"] = int(icomp+1)
            df_time.loc[idxs, "icomp"] = int(icomp + 1)

        # df.loc[df_time.index,"ins_str"] = df_time.apply(lambda x: "l1 w w !sfrc{0}_{1}_{2}! !swgw{0}_{1}_{2}! !gwcn{0}_{1}_{2}!\n".\
        #                                 format(x.sfr_node,x.icomp,x.time_str),axis=1)
        df.loc[df_time.index, "ins_str"] = df_time.apply(
            lambda x: "l1 w w !sfrc{0}_{1}_{2}!\n".format(
                x.sfr_node, x.icomp, x.time_str
            ),
            axis=1,
        )
    df.index = np.arange(df.shape[0])
    if ins_file is None:
        ins_file = sft_file + ".processed.ins"

    with open(ins_file, "w") as f:
        f.write("pif ~\nl1\n")
        [f.write(i) for i in df.ins_str]
    # df = try_process_ins_file(ins_file,sft_file+".processed")
    df = try_process_output_file(ins_file, sft_file + ".processed")
    return df


def apply_sft_obs():
    """process an mt3d-usgs sft ASCII output file using a previous-written
    config file

    Returns:
        **pandas.DataFrame**: a dataframe of extracted simulated outputs

    Note:
        This is the companion function to `gw_utils.setup_sft_obs()`.

    """
    # this is for dealing with the missing 'e' problem
    def try_cast(x):
        try:
            return float(x)
        except:
            return 0.0

    times = []
    with open("sft_obs.config") as f:
        sft_file = f.readline().strip()
        for line in f:
            times.append(float(line.strip()))
    df = pd.read_csv(sft_file, skiprows=1, delim_whitespace=True)  # ,nrows=10000000)
    df.columns = [c.lower().replace("-", "_") for c in df.columns]
    df = df.loc[df.time.apply(lambda x: x in times), :]
    # print(df.dtypes)
    # normalize
    for c in df.columns:
        # print(c)
        if not "node" in c:
            df.loc[:, c] = df.loc[:, c].apply(try_cast)
        # print(df.loc[df.loc[:,c].apply(lambda x : type(x) == str),:])
        if df.dtypes[c] == float:
            df.loc[df.loc[:, c] < 1e-30, c] = 0.0
            df.loc[df.loc[:, c] > 1e30, c] = 1.0e30
    df.loc[:, "sfr_node"] = df.sfr_node.apply(np.int64)

    df.to_csv(sft_file + ".processed", sep=" ", index=False)
    return df


def setup_sfr_seg_parameters(
    nam_file, model_ws=".", par_cols=None, tie_hcond=True, include_temporal_pars=None
):
    """Setup multiplier parameters for SFR segment data.

    Args:
        nam_file (`str`): MODFLOw name file.  DIS, BAS, and SFR must be
            available as pathed in the nam_file.  Optionally, `nam_file` can be
            an existing `flopy.modflow.Modflow`.
        model_ws (`str`): model workspace for flopy to load the MODFLOW model from
        par_cols ([`str`]): a list of segment data entires to parameterize
        tie_hcond (`bool`):  flag to use same mult par for hcond1 and hcond2 for a
            given segment.  Default is `True`.
        include_temporal_pars ([`str`]):  list of spatially-global multipliers to set up for
            each stress period.  Default is None

    Returns:
        **pandas.DataFrame**: a dataframe with useful parameter setup information

    Note:
        This function handles the standard input case, not all the cryptic SFR options.  Loads the
        dis, bas, and sfr files with flopy using model_ws.

        This is the companion function to `gw_utils.apply_sfr_seg_parameters()` .
        The number (and numbering) of segment data entries must consistent across
        all stress periods.

        Writes `nam_file` +"_backup_.sfr" as the backup of the original sfr file
        Skips values = 0.0 since multipliers don't work for these

    """

    try:
        import flopy
    except Exception as e:
        return
    if par_cols is None:
        par_cols = ["flow", "runoff", "hcond1", "pptsw"]
    if tie_hcond:
        if "hcond1" not in par_cols or "hcond2" not in par_cols:
            tie_hcond = False

    if isinstance(nam_file, flopy.modflow.mf.Modflow) and nam_file.sfr is not None:
        m = nam_file
        nam_file = m.namefile
        model_ws = m.model_ws
    else:
        # load MODFLOW model # is this needed? could we just pass the model if it has already been read in?
        m = flopy.modflow.Modflow.load(
            nam_file, load_only=["sfr"], model_ws=model_ws, check=False, forgive=False
        )
    if include_temporal_pars:
        if include_temporal_pars is True:
            tmp_par_cols = {col: range(m.dis.nper) for col in par_cols}
        elif isinstance(include_temporal_pars, str):
            tmp_par_cols = {include_temporal_pars: range(m.dis.nper)}
        elif isinstance(include_temporal_pars, list):
            tmp_par_cols = {col: range(m.dis.nper) for col in include_temporal_pars}
        elif isinstance(include_temporal_pars, dict):
            tmp_par_cols = include_temporal_pars
        include_temporal_pars = True
    else:
        tmp_par_cols = {}
        include_temporal_pars = False

    # make backup copy of sfr file
    shutil.copy(
        os.path.join(model_ws, m.sfr.file_name[0]),
        os.path.join(model_ws, nam_file + "_backup_.sfr"),
    )

    # get the segment data (dict)
    segment_data = m.sfr.segment_data
    shape = segment_data[list(segment_data.keys())[0]].shape
    # check
    for kper, seg_data in m.sfr.segment_data.items():
        assert (
            seg_data.shape == shape
        ), "cannot use: seg data must have the same number of entires for all kpers"
    seg_data_col_order = list(seg_data.dtype.names)
    # convert segment_data dictionary to multi index df - this could get ugly
    reform = {
        (k, c): segment_data[k][c]
        for k in segment_data.keys()
        for c in segment_data[k].dtype.names
    }
    seg_data_all_kper = pd.DataFrame.from_dict(reform)
    seg_data_all_kper.columns.names = ["kper", "col"]

    # extract the first seg data kper to a dataframe
    seg_data = seg_data_all_kper[0].copy()  # pd.DataFrame.from_records(seg_data)

    # make sure all par cols are found and search of any data in kpers
    missing = []
    cols = par_cols.copy()
    for par_col in set(par_cols + list(tmp_par_cols.keys())):
        if par_col not in seg_data.columns:
            if par_col in cols:
                missing.append(cols.pop(cols.index(par_col)))
            if par_col in tmp_par_cols.keys():
                _ = tmp_par_cols.pop(par_col)
        # look across all kper in multiindex df to check for values entry - fill with absmax should capture entries
        else:
            seg_data.loc[:, par_col] = (
                seg_data_all_kper.loc[:, (slice(None), par_col)]
                    .abs().groupby(level=1, axis=1).max()
            )
    if len(missing) > 0:
        warnings.warn(
            "the following par_cols were not found in segment data: {0}".format(
                ",".join(missing)
            ),
            PyemuWarning,
        )
        if len(missing) >= len(par_cols):
            warnings.warn(
                "None of the passed par_cols ({0}) were found in segment data.".format(
                    ",".join(par_cols)
                ),
                PyemuWarning,
            )
    seg_data = seg_data[seg_data_col_order]  # reset column orders to inital
    seg_data_org = seg_data.copy()
    seg_data.to_csv(os.path.join(model_ws, "sfr_seg_pars.dat"), sep=",")

    # the data cols not to parameterize
    # better than a column indexer as pandas can change column orders
    idx_cols = ["nseg", "icalc", "outseg", "iupseg", "iprior", "nstrpts"]
    notpar_cols = [c for c in seg_data.columns if c not in cols + idx_cols]

    # process par cols
    tpl_str, pvals = [], []
    if include_temporal_pars:
        tmp_pnames, tmp_tpl_str = [], []
        tmp_df = pd.DataFrame(
            data={c: 1.0 for c in tmp_par_cols.keys()},
            index=list(m.sfr.segment_data.keys()),
        )
        tmp_df.sort_index(inplace=True)
        tmp_df.to_csv(os.path.join(model_ws, "sfr_seg_temporal_pars.dat"))
    for par_col in set(cols + list(tmp_par_cols.keys())):
        print(par_col)
        prefix = par_col
        if tie_hcond and par_col == "hcond2":
            prefix = "hcond1"
        if seg_data.loc[:, par_col].sum() == 0.0:
            print("all zeros for {0}...skipping...".format(par_col))
            # seg_data.loc[:,par_col] = 1
            # all zero so no need to set up
            if par_col in cols:
                # - add to notpar
                notpar_cols.append(cols.pop(cols.index(par_col)))
            if par_col in tmp_par_cols.keys():
                _ = tmp_par_cols.pop(par_col)
        if par_col in cols:
            seg_data.loc[:, par_col] = seg_data.apply(
                lambda x: "~    {0}_{1:04d}   ~".format(prefix, int(x.nseg))
                if float(x[par_col]) != 0.0
                else "1.0",
                axis=1,
            )
            org_vals = seg_data_org.loc[seg_data_org.loc[:, par_col] != 0.0, par_col]
            pnames = seg_data.loc[org_vals.index, par_col]
            pvals.extend(list(org_vals.values))
            tpl_str.extend(list(pnames.values))
        if par_col in tmp_par_cols.keys():
            parnme = tmp_df.index.map(
                lambda x: "{0}_{1:04d}_tmp".format(par_col, int(x))
                if x in tmp_par_cols[par_col]
                else 1.0
            )
            sel = parnme != 1.0
            tmp_df.loc[sel, par_col] = parnme[sel].map(lambda x: "~   {0}  ~".format(x))
            tmp_tpl_str.extend(list(tmp_df.loc[sel, par_col].values))
            tmp_pnames.extend(list(parnme[sel].values))
    pnames = [t.replace("~", "").strip() for t in tpl_str]
    df = pd.DataFrame(
        {"parnme": pnames, "org_value": pvals, "tpl_str": tpl_str}, index=pnames
    )
    df.drop_duplicates(inplace=True)
    if df.empty:
        warnings.warn(
            "No spatial sfr segment parameters have been set up, "
            "either none of {0} were found or all were zero.".format(
                ",".join(par_cols)
            ),
            PyemuWarning,
        )
        # return df
    # set not par cols to 1.0
    seg_data.loc[:, notpar_cols] = "1.0"

    # write the template file
    _write_df_tpl(os.path.join(model_ws, "sfr_seg_pars.dat.tpl"), seg_data, sep=",")

    # make sure the tpl file exists and has the same num of pars
    parnme = parse_tpl_file(os.path.join(model_ws, "sfr_seg_pars.dat.tpl"))
    assert len(parnme) == df.shape[0]

    # set some useful par info
    df["pargp"] = df.parnme.apply(lambda x: x.split("_")[0])

    if include_temporal_pars:
        _write_df_tpl(
            filename=os.path.join(model_ws, "sfr_seg_temporal_pars.dat.tpl"), df=tmp_df
        )
        pargp = [pname.split("_")[0] + "_tmp" for pname in tmp_pnames]
        tmp_df = pd.DataFrame(
            data={"parnme": tmp_pnames, "pargp": pargp}, index=tmp_pnames
        )
        if not tmp_df.empty:
            tmp_df.loc[:, "org_value"] = 1.0
            tmp_df.loc[:, "tpl_str"] = tmp_tpl_str
            df = pd.concat([df, tmp_df[df.columns]])
    if df.empty:
        warnings.warn(
            "No sfr segment parameters have been set up, "
            "either none of {0} were found or all were zero.".format(
                ",".join(set(par_cols + list(tmp_par_cols.keys())))
            ),
            PyemuWarning,
        )
        return df

    # write the config file used by apply_sfr_pars()
    with open(os.path.join(model_ws, "sfr_seg_pars.config"), "w") as f:
        f.write("nam_file {0}\n".format(nam_file))
        f.write("model_ws {0}\n".format(model_ws))
        f.write("mult_file sfr_seg_pars.dat\n")
        f.write("sfr_filename {0}\n".format(m.sfr.file_name[0]))
        if include_temporal_pars:
            f.write("time_mult_file sfr_seg_temporal_pars.dat\n")

    # set some useful par info
    df.loc[:, "parubnd"] = 1.25
    df.loc[:, "parlbnd"] = 0.75
    hpars = df.loc[df.pargp.apply(lambda x: x.startswith("hcond")), "parnme"]
    df.loc[hpars, "parubnd"] = 100.0
    df.loc[hpars, "parlbnd"] = 0.01

    return df


def setup_sfr_reach_parameters(nam_file, model_ws=".", par_cols=["strhc1"]):
    """Setup multiplier paramters for reach data, when reachinput option is specififed in sfr.


    Args:
        nam_file (`str`): MODFLOw name file.  DIS, BAS, and SFR must be
            available as pathed in the nam_file.  Optionally, `nam_file` can be
            an existing `flopy.modflow.Modflow`.
        model_ws (`str`): model workspace for flopy to load the MODFLOW model from
        par_cols ([`str`]): a list of segment data entires to parameterize
        tie_hcond (`bool`):  flag to use same mult par for hcond1 and hcond2 for a
            given segment.  Default is `True`.
        include_temporal_pars ([`str`]):  list of spatially-global multipliers to set up for
            each stress period.  Default is None

    Returns:
        **pandas.DataFrame**: a dataframe with useful parameter setup information

    Note:
        Similar to `gw_utils.setup_sfr_seg_parameters()`, method will apply params to sfr reachdata

        Can load the dis, bas, and sfr files with flopy using model_ws. Or can pass a model object
        (SFR loading can be slow)

        This is the companion function of `gw_utils.apply_sfr_reach_parameters()`
        Skips values = 0.0 since multipliers don't work for these

    """
    try:
        import flopy
    except Exception as e:
        return
    if par_cols is None:
        par_cols = ["strhc1"]
    if isinstance(nam_file, flopy.modflow.mf.Modflow) and nam_file.sfr is not None:
        # flopy MODFLOW model has been passed and has SFR loaded
        m = nam_file
        nam_file = m.namefile
        model_ws = m.model_ws
    else:
        # if model has not been passed or SFR not loaded # load MODFLOW model
        m = flopy.modflow.Modflow.load(
            nam_file, load_only=["sfr"], model_ws=model_ws, check=False, forgive=False
        )
    # get reachdata as dataframe
    reach_data = pd.DataFrame.from_records(m.sfr.reach_data)
    # write inital reach_data as csv
    reach_data_orig = reach_data.copy()
    reach_data.to_csv(os.path.join(m.model_ws, "sfr_reach_pars.dat"), sep=",")

    # generate template file with pars in par_cols
    # process par cols
    tpl_str, pvals = [], []
    # par_cols=["strhc1"]
    idx_cols = ["node", "k", "i", "j", "iseg", "ireach", "reachID", "outreach"]
    # the data cols not to parameterize
    notpar_cols = [c for c in reach_data.columns if c not in par_cols + idx_cols]
    # make sure all par cols are found and search of any data in kpers
    missing = []
    cols = par_cols.copy()
    for par_col in par_cols:
        if par_col not in reach_data.columns:
            missing.append(par_col)
            cols.remove(par_col)
    if len(missing) > 0:
        warnings.warn(
            "the following par_cols were not found in reach data: {0}".format(
                ",".join(missing)
            ),
            PyemuWarning,
        )
        if len(missing) >= len(par_cols):
            warnings.warn(
                "None of the passed par_cols ({0}) were found in reach data.".format(
                    ",".join(par_cols)
                ),
                PyemuWarning,
            )
    for par_col in cols:
        if par_col == "strhc1":
            prefix = "strk"  # shorten par
        else:
            prefix = par_col
        reach_data.loc[:, par_col] = reach_data.apply(
            lambda x: "~    {0}_{1:04d}   ~".format(prefix, int(x.reachID))
            if float(x[par_col]) != 0.0
            else "1.0",
            axis=1,
        )
        org_vals = reach_data_orig.loc[reach_data_orig.loc[:, par_col] != 0.0, par_col]
        pnames = reach_data.loc[org_vals.index, par_col]
        pvals.extend(list(org_vals.values))
        tpl_str.extend(list(pnames.values))
    pnames = [t.replace("~", "").strip() for t in tpl_str]
    df = pd.DataFrame(
        {"parnme": pnames, "org_value": pvals, "tpl_str": tpl_str}, index=pnames
    )
    df.drop_duplicates(inplace=True)
    if df.empty:
        warnings.warn(
            "No sfr reach parameters have been set up, either none of {0} were found or all were zero.".format(
                ",".join(par_cols)
            ),
            PyemuWarning,
        )
    else:
        # set not par cols to 1.0
        reach_data.loc[:, notpar_cols] = "1.0"

        # write the template file
        _write_df_tpl(
            os.path.join(model_ws, "sfr_reach_pars.dat.tpl"), reach_data, sep=","
        )

        # write the config file used by apply_sfr_pars()
        with open(os.path.join(model_ws, "sfr_reach_pars.config"), "w") as f:
            f.write("nam_file {0}\n".format(nam_file))
            f.write("model_ws {0}\n".format(model_ws))
            f.write("mult_file sfr_reach_pars.dat\n")
            f.write("sfr_filename {0}".format(m.sfr.file_name[0]))

        # make sure the tpl file exists and has the same num of pars
        parnme = parse_tpl_file(os.path.join(model_ws, "sfr_reach_pars.dat.tpl"))
        assert len(parnme) == df.shape[0]

        # set some useful par info
        df.loc[:, "pargp"] = df.parnme.apply(lambda x: x.split("_")[0])
        df.loc[:, "parubnd"] = 1.25
        df.loc[:, "parlbnd"] = 0.75
        hpars = df.loc[df.pargp.apply(lambda x: x.startswith("strk")), "parnme"]
        df.loc[hpars, "parubnd"] = 100.0
        df.loc[hpars, "parlbnd"] = 0.01
    return df


def apply_sfr_seg_parameters(seg_pars=True, reach_pars=False):
    """apply the SFR segement multiplier parameters.

    Args:
        seg_pars (`bool`, optional): flag to apply segment-based parameters.
            Default is True
        reach_pars (`bool`, optional): flag to apply reach-based parameters.
            Default is False

    Returns:
        **flopy.modflow.ModflowSfr**: the modified SFR package instance

    Note:
        Expects "sfr_seg_pars.config" to exist

        Expects `nam_file` +"_backup_.sfr" to exist



    """

    if not seg_pars and not reach_pars:
        raise Exception(
            "gw_utils.apply_sfr_pars() error: both seg_pars and reach_pars are False"
        )
    # if seg_pars and reach_pars:
    #    raise Exception("gw_utils.apply_sfr_pars() error: both seg_pars and reach_pars are True")

    import flopy

    bak_sfr_file, pars = None, None

    if seg_pars:
        assert os.path.exists("sfr_seg_pars.config")

        with open("sfr_seg_pars.config", "r") as f:
            pars = {}
            for line in f:
                line = line.strip().split()
                pars[line[0]] = line[1]
        bak_sfr_file = pars["nam_file"] + "_backup_.sfr"
        # m = flopy.modflow.Modflow.load(pars["nam_file"],model_ws=pars["model_ws"],load_only=["sfr"],check=False)
        m = flopy.modflow.Modflow.load(pars["nam_file"], load_only=[], check=False)
        sfr = flopy.modflow.ModflowSfr2.load(os.path.join(bak_sfr_file), m)
        sfrfile = pars["sfr_filename"]
        mlt_df = pd.read_csv(pars["mult_file"], delim_whitespace=False, index_col=0)
        # time_mlt_df = None
        # if "time_mult_file" in pars:
        #     time_mult_file = pars["time_mult_file"]
        #     time_mlt_df = pd.read_csv(pars["time_mult_file"], delim_whitespace=False,index_col=0)

        idx_cols = ["nseg", "icalc", "outseg", "iupseg", "iprior", "nstrpts"]
        present_cols = [c for c in idx_cols if c in mlt_df.columns]
        mlt_cols = mlt_df.columns.drop(present_cols)
        for key, val in m.sfr.segment_data.items():
            df = pd.DataFrame.from_records(val)
            df.loc[:, mlt_cols] *= mlt_df.loc[:, mlt_cols]
            val = df.to_records(index=False)
            sfr.segment_data[key] = val
    if reach_pars:
        assert os.path.exists("sfr_reach_pars.config")
        with open("sfr_reach_pars.config", "r") as f:
            r_pars = {}
            for line in f:
                line = line.strip().split()
                r_pars[line[0]] = line[1]
        if bak_sfr_file is None:  # will be the case is seg_pars is false
            bak_sfr_file = r_pars["nam_file"] + "_backup_.sfr"
            # m = flopy.modflow.Modflow.load(pars["nam_file"],model_ws=pars["model_ws"],load_only=["sfr"],check=False)
            m = flopy.modflow.Modflow.load(
                r_pars["nam_file"], load_only=[], check=False
            )
            sfr = flopy.modflow.ModflowSfr2.load(os.path.join(bak_sfr_file), m)
            sfrfile = r_pars["sfr_filename"]
        r_mlt_df = pd.read_csv(r_pars["mult_file"], sep=",", index_col=0)
        r_idx_cols = ["node", "k", "i", "j", "iseg", "ireach", "reachID", "outreach"]
        r_mlt_cols = r_mlt_df.columns.drop(r_idx_cols)
        r_df = pd.DataFrame.from_records(m.sfr.reach_data)
        r_df.loc[:, r_mlt_cols] *= r_mlt_df.loc[:, r_mlt_cols]
        sfr.reach_data = r_df.to_records(index=False)

    # m.remove_package("sfr")
    if pars is not None and "time_mult_file" in pars:
        time_mult_file = pars["time_mult_file"]
        time_mlt_df = pd.read_csv(time_mult_file, delim_whitespace=False, index_col=0)
        for kper, sdata in m.sfr.segment_data.items():
            assert kper in time_mlt_df.index, (
                "gw_utils.apply_sfr_seg_parameters() error: kper "
                + "{0} not in time_mlt_df index".format(kper)
            )
            for col in time_mlt_df.columns:
                sdata[col] *= time_mlt_df.loc[kper, col]

    sfr.write_file(filename=sfrfile)
    return sfr


def apply_sfr_parameters(seg_pars=True, reach_pars=False):
    """thin wrapper around `gw_utils.apply_sfr_seg_parameters()`

    Args:
        seg_pars (`bool`, optional): flag to apply segment-based parameters.
            Default is True
        reach_pars (`bool`, optional): flag to apply reach-based parameters.
            Default is False

    Returns:
        **flopy.modflow.ModflowSfr**: the modified SFR package instance

    Note:
        Expects "sfr_seg_pars.config" to exist

        Expects `nam_file` +"_backup_.sfr" to exist


    """
    sfr = apply_sfr_seg_parameters(seg_pars=seg_pars, reach_pars=reach_pars)
    return sfr


def setup_sfr_obs(
    sfr_out_file, seg_group_dict=None, ins_file=None, model=None, include_path=False
):
    """setup observations using the sfr ASCII output file.  Setups
    the ability to aggregate flows for groups of segments.  Applies
    only flow to aquier and flow out.

    Args:
        sft_out_file (`str`): the name and path to an existing SFR output file
        seg_group_dict (`dict`): a dictionary of SFR segements to aggregate together for a single obs.
            the key value in the dict is the base observation name. If None, all segments
            are used as individual observations. Default is None
        model (`flopy.mbase`): a flopy model.  If passed, the observation names will have
            the datetime of the observation appended to them.  If None, the observation names
            will have the stress period appended to them. Default is None.
        include_path (`bool`): flag to prepend sfr_out_file path to sfr_obs.config.  Useful for setting up
            process in separate directory for where python is running.


    Returns:
        **pandas.DataFrame**: dataframe of observation name, simulated value and group.

    Note:
        This is the companion function of `gw_utils.apply_sfr_obs()`.

        This function writes "sfr_obs.config" which must be kept in the dir where
        "gw_utils.apply_sfr_obs()" is being called during the forward run

    """

    sfr_dict = load_sfr_out(sfr_out_file)
    kpers = list(sfr_dict.keys())
    kpers.sort()

    if seg_group_dict is None:
        seg_group_dict = {"seg{0:04d}".format(s): s for s in sfr_dict[kpers[0]].segment}
    else:
        warnings.warn(
            "Flow out (flout) of grouped segments will be aggregated... ", PyemuWarning
        )
    sfr_segs = set(sfr_dict[list(sfr_dict.keys())[0]].segment)
    keys = ["sfr_out_file"]
    if include_path:
        values = [os.path.split(sfr_out_file)[-1]]
    else:
        values = [sfr_out_file]
    for oname, segs in seg_group_dict.items():
        if np.isscalar(segs):
            segs_set = {segs}
            segs = [segs]
        else:
            segs_set = set(segs)
        diff = segs_set.difference(sfr_segs)
        if len(diff) > 0:
            raise Exception(
                "the following segs listed with oname {0} where not found: {1}".format(
                    oname, ",".join([str(s) for s in diff])
                )
            )
        for seg in segs:
            keys.append(oname)
            values.append(seg)

    df_key = pd.DataFrame({"obs_base": keys, "segment": values})
    if include_path:
        pth = os.path.join(*[p for p in os.path.split(sfr_out_file)[:-1]])
        config_file = os.path.join(pth, "sfr_obs.config")
    else:
        config_file = "sfr_obs.config"
    print("writing 'sfr_obs.config' to {0}".format(config_file))
    df_key.to_csv(config_file)

    bd = "."
    if include_path:
        bd = os.getcwd()
        os.chdir(pth)
    try:
        df = apply_sfr_obs()
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in apply_sfr_obs(): {0}".format(str(e)))
    os.chdir(bd)
    if model is not None:
        dts = (
            pd.to_datetime(model.start_datetime)
            + pd.to_timedelta(np.cumsum(model.dis.perlen.array), unit="d")
        ).date
        df.loc[:, "datetime"] = df.kper.apply(lambda x: dts[x])
        df.loc[:, "time_str"] = df.datetime.apply(lambda x: x.strftime("%Y%m%d"))
    else:
        df.loc[:, "time_str"] = df.kper.apply(lambda x: "{0:04d}".format(x))
    df.loc[:, "flaqx_obsnme"] = df.apply(
        lambda x: "{0}_{1}_{2}".format("fa", x.obs_base, x.time_str), axis=1
    )
    df.loc[:, "flout_obsnme"] = df.apply(
        lambda x: "{0}_{1}_{2}".format("fo", x.obs_base, x.time_str), axis=1
    )

    if ins_file is None:
        ins_file = sfr_out_file + ".processed.ins"

    with open(ins_file, "w") as f:
        f.write("pif ~\nl1\n")
        for fla, flo in zip(df.flaqx_obsnme, df.flout_obsnme):
            f.write("l1 w w !{0}! !{1}!\n".format(fla, flo))

    df = None
    pth = os.path.split(ins_file)[:-1]
    pth = os.path.join(*pth)
    if pth == "":
        pth = "."
    bd = os.getcwd()
    os.chdir(pth)
    df = try_process_output_file(
        os.path.split(ins_file)[-1], os.path.split(sfr_out_file + ".processed")[-1]
    )
    os.chdir(bd)
    if df is not None:
        df.loc[:, "obsnme"] = df.index.values
        df.loc[:, "obgnme"] = df.obsnme.apply(
            lambda x: "flaqx" if x.startswith("fa") else "flout"
        )
        return df


def apply_sfr_obs():
    """apply the sfr observation process

    Args:
        None

    Returns:
        **pandas.DataFrame**: a dataframe of aggregrated sfr segment aquifer and outflow

    Note:
        This is the companion function of `gw_utils.setup_sfr_obs()`.

        Requires `sfr_obs.config`.

        Writes `sfr_out_file`+".processed", where `sfr_out_file` is defined in "sfr_obs.config"
    """
    assert os.path.exists("sfr_obs.config")
    df_key = pd.read_csv("sfr_obs.config", index_col=0)

    assert df_key.iloc[0, 0] == "sfr_out_file", df_key.iloc[0, :]
    sfr_out_file = df_key.iloc[0, 1]
    df_key = df_key.iloc[1:, :]
    df_key.loc[:, "segment"] = df_key.segment.apply(np.int64)
    df_key.index = df_key.segment
    seg_group_dict = df_key.groupby(df_key.obs_base).groups

    sfr_kper = load_sfr_out(sfr_out_file)
    kpers = list(sfr_kper.keys())
    kpers.sort()
    # results = {o:[] for o in seg_group_dict.keys()}
    results = []
    for kper in kpers:
        df = sfr_kper[kper]
        for obs_base, segs in seg_group_dict.items():
            agg = df.loc[
                segs.values, :
            ].sum()  # still agg flout where seg groups are passed!
            # print(obs_base,agg)
            results.append([kper, obs_base, agg["flaqx"], agg["flout"]])
    df = pd.DataFrame(data=results, columns=["kper", "obs_base", "flaqx", "flout"])
    df.sort_values(by=["kper", "obs_base"], inplace=True)
    df.to_csv(sfr_out_file + ".processed", sep=" ", index=False)
    return df


def load_sfr_out(sfr_out_file, selection=None):
    """load an ASCII SFR output file into a dictionary of kper: dataframes.

    Args:
        sfr_out_file (`str`): SFR ASCII output file
        selection (`pandas.DataFrame`): a dataframe of `reach` and `segment` pairs to
            load.  If `None`, all reach-segment pairs are loaded.  Default is `None`.

    Returns:
        **dict**: dictionary of {kper:`pandas.DataFrame`} of SFR output.

    Note:
        Aggregates flow to aquifer for segments and returns and flow out at
        downstream end of segment.

    """
    assert os.path.exists(sfr_out_file), "couldn't find sfr out file {0}".format(
        sfr_out_file
    )
    tag = " stream listing"
    lcount = 0
    sfr_dict = {}
    if selection is None:
        pass
    elif isinstance(selection, str):
        assert (
            selection == "all"
        ), "If string passed as selection only 'all' allowed: " "{}".format(selection)
    else:
        assert isinstance(
            selection, pd.DataFrame
        ), "'selection needs to be pandas Dataframe. " "Type {} passed.".format(
            type(selection)
        )
        assert np.all(
            [sr in selection.columns for sr in ["segment", "reach"]]
        ), "Either 'segment' or 'reach' not in selection columns"
    with open(sfr_out_file) as f:
        while True:
            line = f.readline().lower()
            lcount += 1
            if line == "":
                break
            if line.startswith(tag):
                raw = line.strip().split()
                kper = int(raw[3]) - 1
                kstp = int(raw[5]) - 1
                [f.readline() for _ in range(4)]  # skip to where the data starts
                lcount += 4
                dlines = []
                while True:
                    dline = f.readline()
                    lcount += 1
                    if dline.strip() == "":
                        break
                    draw = dline.strip().split()
                    dlines.append(draw)
                df = pd.DataFrame(data=np.array(dlines)).iloc[:, [3, 4, 6, 7]]
                df.columns = ["segment", "reach", "flaqx", "flout"]
                df["segment"] = df.segment.astype(np.int64)
                df["reach"] = df.reach.astype(np.int64)
                df["flaqx"] = df.flaqx.astype(np.float64)
                df["flout"] = df.flout.astype(np.float64)
                df.index = [
                    "{0:03d}_{1:03d}".format(s, r)
                    for s, r in np.array([df.segment.values, df.reach.values]).T
                ]
                # df.index = df.apply(
                # lambda x: "{0:03d}_{1:03d}".format(
                # int(x.segment), int(x.reach)), axis=1)
                if selection is None:  # setup for all segs, aggregate
                    gp = df.groupby(df.segment)
                    bot_reaches = (
                        gp[["reach"]]
                        .max()
                        .apply(
                            lambda x: "{0:03d}_{1:03d}".format(
                                int(x.name), int(x.reach)
                            ),
                            axis=1,
                        )
                    )
                    # only sum distributed output # take flow out of seg
                    df2 = pd.DataFrame(
                        {
                            "flaqx": gp.flaqx.sum(),
                            "flout": df.loc[bot_reaches, "flout"].values,
                        },
                        index=gp.groups.keys(),
                    )
                    # df = df.groupby(df.segment).sum()
                    df2["segment"] = df2.index
                elif isinstance(selection, str) and selection == "all":
                    df2 = df
                else:
                    seg_reach_id = selection.apply(
                        lambda x: "{0:03d}_{1:03d}".format(
                            int(x.segment), int(x.reach)
                        ),
                        axis=1,
                    ).values
                    for sr in seg_reach_id:
                        if sr not in df.index:
                            s, r = [x.lstrip("0") for x in sr.split("_")]
                            warnings.warn(
                                "Requested segment reach pair ({0},{1}) "
                                "is not in sfr output. Dropping...".format(
                                    int(r), int(s)
                                ),
                                PyemuWarning,
                            )
                            seg_reach_id = np.delete(
                                seg_reach_id, np.where(seg_reach_id == sr), axis=0
                            )
                    df2 = df.loc[seg_reach_id].copy()
                if kper in sfr_dict.keys():
                    print(
                        "multiple entries found for kper {0}, "
                        "replacing...".format(kper)
                    )
                sfr_dict[kper] = df2
    return sfr_dict


def setup_sfr_reach_obs(
    sfr_out_file, seg_reach=None, ins_file=None, model=None, include_path=False
):
    """setup observations using the sfr ASCII output file.  Setups
    sfr point observations using segment and reach numbers.

    Args:
        sft_out_file (`str`): the path and name of an existing SFR output file
        seg_reach (varies): a dict, or list of SFR [segment,reach] pairs identifying
            locations of interest.  If `dict`, the key value in the dict is the base
            observation name. If None, all reaches are used as individual observations.
            Default is None - THIS MAY SET UP A LOT OF OBS!
        model (`flopy.mbase`): a flopy model.  If passed, the observation names will
            have the datetime of the observation appended to them.  If None, the
            observation names will have the stress period appended to them. Default is None.
        include_path (`bool`): a flag to prepend sfr_out_file path to sfr_obs.config.  Useful
            for setting up process in separate directory for where python is running.


    Returns:
        `pd.DataFrame`: a dataframe of observation names, values, and groups

    Note:
        This is the companion function of `gw_utils.apply_sfr_reach_obs()`.

        This function writes "sfr_reach_obs.config" which must be kept in the dir where
        "apply_sfr_reach_obs()" is being called during the forward run

    """
    if seg_reach is None:
        warnings.warn("Obs will be set up for every reach", PyemuWarning)
        seg_reach = "all"
    elif isinstance(seg_reach, list) or isinstance(seg_reach, np.ndarray):
        if np.ndim(seg_reach) == 1:
            seg_reach = [seg_reach]
        assert (
            np.shape(seg_reach)[1] == 2
        ), "varible seg_reach expected shape (n,2), received {0}".format(
            np.shape(seg_reach)
        )
        seg_reach = pd.DataFrame(seg_reach, columns=["segment", "reach"])
        seg_reach.index = seg_reach.apply(
            lambda x: "s{0:03d}r{1:03d}".format(int(x.segment), int(x.reach)), axis=1
        )
    elif isinstance(seg_reach, dict):
        seg_reach = pd.DataFrame.from_dict(
            seg_reach, orient="index", columns=["segment", "reach"]
        )
    else:
        assert isinstance(
            seg_reach, pd.DataFrame
        ), "'selection needs to be pandas Dataframe. Type {} passed.".format(
            type(seg_reach)
        )
        assert np.all(
            [sr in seg_reach.columns for sr in ["segment", "reach"]]
        ), "Either 'segment' or 'reach' not in selection columns"

    sfr_dict = load_sfr_out(sfr_out_file, selection=seg_reach)
    kpers = list(sfr_dict.keys())
    kpers.sort()

    if isinstance(seg_reach, str) and seg_reach == "all":
        seg_reach = sfr_dict[kpers[0]][["segment", "reach"]]
        seg_reach.index = seg_reach.apply(
            lambda x: "s{0:03d}r{1:03d}".format(int(x.segment), int(x.reach)), axis=1
        )
    keys = ["sfr_out_file"]
    if include_path:
        values = [os.path.split(sfr_out_file)[-1]]
    else:
        values = [sfr_out_file]
    diff = seg_reach.loc[
        seg_reach.apply(
            lambda x: "{0:03d}_{1:03d}".format(int(x.segment), int(x.reach))
            not in sfr_dict[list(sfr_dict.keys())[0]].index,
            axis=1,
        )
    ]

    if len(diff) > 0:
        for ob in diff.itertuples():
            warnings.warn(
                "segs,reach pair listed with onames {0} was not found: {1}".format(
                    ob.Index, "({},{})".format(ob.segment, ob.reach)
                ),
                PyemuWarning,
            )
    seg_reach = seg_reach.drop(diff.index)
    seg_reach["obs_base"] = seg_reach.index
    df_key = pd.DataFrame({"obs_base": keys, "segment": 0, "reach": values})
    df_key = pd.concat([df_key, seg_reach], sort=True).reset_index(drop=True)
    if include_path:
        pth = os.path.join(*[p for p in os.path.split(sfr_out_file)[:-1]])
        config_file = os.path.join(pth, "sfr_reach_obs.config")
    else:
        config_file = "sfr_reach_obs.config"
    print("writing 'sfr_reach_obs.config' to {0}".format(config_file))
    df_key.to_csv(config_file)

    bd = "."
    if include_path:
        bd = os.getcwd()
        os.chdir(pth)
    try:
        df = apply_sfr_reach_obs()
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in apply_sfr_reach_obs(): {0}".format(str(e)))
    os.chdir(bd)
    if model is not None:
        dts = (
            pd.to_datetime(model.start_datetime)
            + pd.to_timedelta(np.cumsum(model.dis.perlen.array), unit="d")
        ).date
        df.loc[:, "datetime"] = df.kper.apply(lambda x: dts[x])
        df.loc[:, "time_str"] = df.datetime.apply(lambda x: x.strftime("%Y%m%d"))
    else:
        df.loc[:, "time_str"] = df.kper.apply(lambda x: "{0:04d}".format(x))
    df.loc[:, "flaqx_obsnme"] = df.apply(
        lambda x: "{0}_{1}_{2}".format("fa", x.obs_base, x.time_str), axis=1
    )
    df.loc[:, "flout_obsnme"] = df.apply(
        lambda x: "{0}_{1}_{2}".format("fo", x.obs_base, x.time_str), axis=1
    )

    if ins_file is None:
        ins_file = sfr_out_file + ".reach_processed.ins"

    with open(ins_file, "w") as f:
        f.write("pif ~\nl1\n")
        for fla, flo in zip(df.flaqx_obsnme, df.flout_obsnme):
            f.write("l1 w w !{0}! !{1}!\n".format(fla, flo))

    df = None
    pth = os.path.split(ins_file)[:-1]
    pth = os.path.join(*pth)
    if pth == "":
        pth = "."
    bd = os.getcwd()
    os.chdir(pth)
    try:
        df = try_process_output_file(
            os.path.split(ins_file)[-1], os.path.split(sfr_out_file + ".processed")[-1]
        )

    except Exception as e:
        pass
    os.chdir(bd)
    if df is not None:
        df.loc[:, "obsnme"] = df.index.values
        df.loc[:, "obgnme"] = df.obsnme.apply(
            lambda x: "flaqx" if x.startswith("fa") else "flout"
        )
        return df


def apply_sfr_reach_obs():
    """apply the sfr reach observation process.

    Returns:
        `pd.DataFrame`: a dataframe of sfr aquifer and outflow ad segment,reach locations

    Note:
        This is the companion function of `gw_utils.setup_sfr_reach_obs()`.

        Requires sfr_reach_obs.config.

        Writes <sfr_out_file>.processed, where <sfr_out_file> is defined in
        "sfr_reach_obs.config"

    """
    assert os.path.exists("sfr_reach_obs.config")
    df_key = pd.read_csv("sfr_reach_obs.config", index_col=0)

    assert df_key.iloc[0, 0] == "sfr_out_file", df_key.iloc[0, :]
    sfr_out_file = df_key.iloc[0].reach
    df_key = df_key.iloc[1:, :].copy()
    df_key.loc[:, "segment"] = df_key.segment.apply(np.int64)
    df_key.loc[:, "reach"] = df_key.reach.apply(np.int64)
    df_key = df_key.set_index("obs_base")

    sfr_kper = load_sfr_out(sfr_out_file, df_key)
    kpers = list(sfr_kper.keys())
    kpers.sort()

    results = []
    for kper in kpers:
        df = sfr_kper[kper]
        for sr in df_key.itertuples():
            ob = df.loc["{0:03d}_{1:03d}".format(sr.segment, sr.reach), :]
            results.append([kper, sr.Index, ob["flaqx"], ob["flout"]])
    df = pd.DataFrame(data=results, columns=["kper", "obs_base", "flaqx", "flout"])
    df.sort_values(by=["kper", "obs_base"], inplace=True)
    df.to_csv(sfr_out_file + ".reach_processed", sep=" ", index=False)
    return df


def modflow_sfr_gag_to_instruction_file(
    gage_output_file, ins_file=None, parse_filename=False
):
    """writes an instruction file for an SFR gage output file to read Flow only at all times

    Args:
        gage_output_file (`str`): the gage output filename (ASCII).

        ins_file (`str`, optional): the name of the instruction file to
            create.  If None, the name is `gage_output_file` +".ins".
            Default is None

        parse_filename (`bool`): if True, get the gage_num parameter by
            parsing the gage output file filename if False, get the gage
            number from the file itself

    Returns:
        tuple containing

        - **pandas.DataFrame**: a dataframe with obsnme and obsval for the sfr simulated flows.
        - **str**: file name of instructions file relating to gage output.
        - **str**: file name of processed gage output for all times

    Note:
        Sets up observations for gage outputs only for the Flow column.

        If `parse_namefile` is true, only text up to first '.' is used as the gage_num


    """

    if ins_file is None:
        ins_file = gage_output_file + ".ins"

    # navigate the file to be sure the header makes sense
    indat = [line.strip() for line in open(gage_output_file, "r").readlines()]
    header = [i for i in indat if i.startswith('"')]
    # yank out the gage number to identify the observation names
    if parse_filename:
        gage_num = os.path.basename(gage_output_file).split(".")[0]
    else:
        gage_num = re.sub(
            "[^0-9]", "", indat[0].lower().split("gage no.")[-1].strip().split()[0]
        )

    # get the column names
    cols = (
        [i.lower() for i in header if "data" in i.lower()][0]
        .lower()
        .replace('"', "")
        .replace("data:", "")
        .split()
    )

    # make sure "Flow" is included in the columns
    if "flow" not in cols:
        raise Exception('Requested field "Flow" not in gage output columns')

    # find which column is for  "Flow"
    flowidx = np.where(np.array(cols) == "flow")[0][0]

    # write out the instruction file lines
    inslines = [
        "l1 " + (flowidx + 1) * "w " + "!g{0}_{1:d}!".format(gage_num, j)
        for j in range(len(indat) - len(header))
    ]
    inslines[0] = inslines[0].replace("l1", "l{0:d}".format(len(header) + 1))

    # write the instruction file
    with open(ins_file, "w") as ofp:
        ofp.write("pif ~\n")
        [ofp.write("{0}\n".format(line)) for line in inslines]

    df = try_process_output_file(ins_file, gage_output_file)
    return df, ins_file, gage_output_file


def setup_gage_obs(gage_file, ins_file=None, start_datetime=None, times=None):
    """setup a forward run post processor routine for the modflow gage file

    Args:
        gage_file (`str`): the gage output file (ASCII)
        ins_file (`str`, optional): the name of the instruction file to create.  If None, the name
            is `gage_file`+".processed.ins".  Default is `None`
        start_datetime (`str`): a `pandas.to_datetime()` compatible `str`.  If not `None`,
            then the resulting observation names have the datetime suffix.  If `None`,
            the suffix is the output totim.  Default is `None`.
        times ([`float`]):  a container of times to make observations for.  If None,
            all times are used. Default is None.

    Returns:
        tuple containing

        - **pandas.DataFrame**: a dataframe with observation name and simulated values for the
          values in the gage file.
        - **str**: file name of instructions file that was created relating to gage output.
        - **str**: file name of processed gage output (processed according to times passed above.)


    Note:
         Setups up observations for gage outputs (all columns).

         This is the companion function of `gw_utils.apply_gage_obs()`
    """

    with open(gage_file, "r") as f:
        line1 = f.readline()
        gage_num = int(
            re.sub("[^0-9]", "", line1.split("GAGE No.")[-1].strip().split()[0])
        )
        gage_type = line1.split("GAGE No.")[-1].strip().split()[1].lower()
        obj_num = int(line1.replace('"', "").strip().split()[-1])
        line2 = f.readline()
        df = pd.read_csv(
            f, delim_whitespace=True, names=line2.replace('"', "").split()[1:]
        )

    df.columns = [
        c.lower().replace("-", "_").replace(".", "_").strip("_") for c in df.columns
    ]
    # get unique observation ids
    obs_ids = {
        col: "" for col in df.columns[1:]
    }  # empty dictionary for observation ids
    for col in df.columns[1:]:  # exclude column 1 (TIME)
        colspl = col.split("_")
        if len(colspl) > 1:
            # obs name built out of "g"(for gage) "s" or "l"(for gage type) 2 chars from column name - date added later
            obs_ids[col] = "g{0}{1}{2}".format(
                gage_type[0], colspl[0][0], colspl[-1][0]
            )
        else:
            obs_ids[col] = "g{0}{1}".format(gage_type[0], col[0:2])
    with open(
        "_gage_obs_ids.csv", "w"
    ) as f:  # write file relating obs names to meaningfull keys!
        [f.write("{0},{1}\n".format(key, obs)) for key, obs in obs_ids.items()]
    # find passed times in df
    if times is None:
        times = df.time.unique()
    missing = []
    utimes = df.time.unique()
    for t in times:
        if not np.isclose(t, utimes).any():
            missing.append(str(t))
    if len(missing) > 0:
        print(df.time)
        raise Exception("the following times are missing:{0}".format(",".join(missing)))
    # write output times to config file
    with open("gage_obs.config", "w") as f:
        f.write(gage_file + "\n")
        [f.write("{0:15.10E}\n".format(t)) for t in times]
    # extract data for times: returns dataframe and saves a processed df - read by pest
    df, obs_file = apply_gage_obs(return_obs_file=True)
    utimes = df.time.unique()
    for t in times:
        assert np.isclose(
            t, utimes
        ).any(), "time {0} missing in processed dataframe".format(t)
    idx = df.time.apply(
        lambda x: np.isclose(x, times).any()
    )  # boolean selector of desired times in df
    if start_datetime is not None:
        # convert times to usable observation times
        start_datetime = pd.to_datetime(start_datetime)
        df.loc[:, "time_str"] = pd.to_timedelta(df.time, unit="d") + start_datetime
        df.loc[:, "time_str"] = df.time_str.apply(
            lambda x: datetime.strftime(x, "%Y%m%d")
        )
    else:
        df.loc[:, "time_str"] = df.time.apply(lambda x: "{0:08.2f}".format(x))
    # set up instructions (line feed for lines without obs (not in time)
    df.loc[:, "ins_str"] = "l1\n"
    df_times = df.loc[idx, :]  # Slice by desired times
    # TODO include GAGE No. in obs name (if permissible)
    df.loc[df_times.index, "ins_str"] = df_times.apply(
        lambda x: "l1 w {}\n".format(
            " w ".join(
                ["!{0}{1}!".format(obs, x.time_str) for key, obs in obs_ids.items()]
            )
        ),
        axis=1,
    )
    df.index = np.arange(df.shape[0])
    if ins_file is None:
        ins_file = gage_file + ".processed.ins"

    with open(ins_file, "w") as f:
        f.write("pif ~\nl1\n")
        [f.write(i) for i in df.ins_str]
    df = try_process_output_file(ins_file, gage_file + ".processed")
    return df, ins_file, obs_file


def apply_gage_obs(return_obs_file=False):
    """apply the modflow gage obs post-processor

    Args:
        return_obs_file (`bool`): flag to return the processed
            observation file.  Default is `False`.

    Note:
        This is the companion function of `gw_utils.setup_gage_obs()`



    """
    times = []
    with open("gage_obs.config") as f:
        gage_file = f.readline().strip()
        for line in f:
            times.append(float(line.strip()))
    obs_file = gage_file + ".processed"
    with open(gage_file, "r") as f:
        line1 = f.readline()
        gage_num = int(
            re.sub("[^0-9]", "", line1.split("GAGE No.")[-1].strip().split()[0])
        )
        gage_type = line1.split("GAGE No.")[-1].strip().split()[1].lower()
        obj_num = int(line1.replace('"', "").strip().split()[-1])
        line2 = f.readline()
        df = pd.read_csv(
            f, delim_whitespace=True, names=line2.replace('"', "").split()[1:]
        )
    df.columns = [c.lower().replace("-", "_").replace(".", "_") for c in df.columns]
    df = df.loc[df.time.apply(lambda x: np.isclose(x, times).any()), :]
    df.to_csv(obs_file, sep=" ", index=False)
    if return_obs_file:
        return df, obs_file
    else:
        return df


def apply_hfb_pars(par_file="hfb6_pars.csv"):
    """a function to apply HFB multiplier parameters.

    Args:
        par_file (`str`): the HFB parameter info file.
            Default is `hfb_pars.csv`

    Note:
        This is the companion function to
        `gw_utils.write_hfb_zone_multipliers_template()`

        This is to account for the horrible HFB6 format that differs from other
        BCs making this a special case

        Requires "hfb_pars.csv"

        Should be added to the forward_run.py script
    """
    hfb_pars = pd.read_csv(par_file)

    hfb_mults_contents = open(hfb_pars.mlt_file.values[0], "r").readlines()
    skiprows = (
        sum([1 if i.strip().startswith("#") else 0 for i in hfb_mults_contents]) + 1
    )
    header = hfb_mults_contents[:skiprows]

    # read in the multipliers
    names = ["lay", "irow1", "icol1", "irow2", "icol2", "hydchr"]
    hfb_mults = pd.read_csv(
        hfb_pars.mlt_file.values[0],
        skiprows=skiprows,
        delim_whitespace=True,
        names=names,
    ).dropna()

    # read in the original file
    hfb_org = pd.read_csv(
        hfb_pars.org_file.values[0],
        skiprows=skiprows,
        delim_whitespace=True,
        names=names,
    ).dropna()

    # multiply it out
    hfb_org.hydchr *= hfb_mults.hydchr

    for cn in names[:-1]:
        hfb_mults[cn] = hfb_mults[cn].astype(np.int64)
        hfb_org[cn] = hfb_org[cn].astype(np.int64)
    # write the results
    with open(hfb_pars.model_file.values[0], "w", newline="") as ofp:
        [ofp.write("{0}\n".format(line.strip())) for line in header]
        ofp.flush()
        hfb_org[["lay", "irow1", "icol1", "irow2", "icol2", "hydchr"]].to_csv(
            ofp, sep=" ", header=None, index=None
        )


def write_hfb_zone_multipliers_template(m):
    """write a template file for an hfb using multipliers per zone (double yuck!)

    Args:
        m (`flopy.modflow.Modflow`): a model instance with an HFB package

    Returns:
        tuple containing

        - **dict**: a dictionary with original unique HFB conductivity values and their
          corresponding parameter names
        - **str**: the template filename that was created

    """
    if m.hfb6 is None:
        raise Exception("no HFB package found")
    # find the model file
    hfb_file = os.path.join(m.model_ws, m.hfb6.file_name[0])

    # this will use multipliers, so need to copy down the original
    if not os.path.exists(os.path.join(m.model_ws, "hfb6_org")):
        os.mkdir(os.path.join(m.model_ws, "hfb6_org"))
    # copy down the original file
    shutil.copy2(
        os.path.join(m.model_ws, m.hfb6.file_name[0]),
        os.path.join(m.model_ws, "hfb6_org", m.hfb6.file_name[0]),
    )

    if not os.path.exists(os.path.join(m.model_ws, "hfb6_mlt")):
        os.mkdir(os.path.join(m.model_ws, "hfb6_mlt"))

    # read in the model file
    hfb_file_contents = open(hfb_file, "r").readlines()

    # navigate the header
    skiprows = (
        sum([1 if i.strip().startswith("#") else 0 for i in hfb_file_contents]) + 1
    )
    header = hfb_file_contents[:skiprows]

    # read in the data
    names = ["lay", "irow1", "icol1", "irow2", "icol2", "hydchr"]
    hfb_in = pd.read_csv(
        hfb_file, skiprows=skiprows, delim_whitespace=True, names=names
    ).dropna()
    for cn in names[:-1]:
        hfb_in[cn] = hfb_in[cn].astype(np.int64)

    # set up a multiplier for each unique conductivity value
    unique_cond = hfb_in.hydchr.unique()
    hfb_mults = dict(
        zip(unique_cond, ["hbz_{0:04d}".format(i) for i in range(len(unique_cond))])
    )
    # set up the TPL line for each parameter and assign
    hfb_in["tpl"] = "blank"
    for cn, cg in hfb_in.groupby("hydchr"):
        hfb_in.loc[hfb_in.hydchr == cn, "tpl"] = "~{0:^10s}~".format(hfb_mults[cn])

    assert "blank" not in hfb_in.tpl

    # write out the TPL file
    tpl_file = os.path.join(m.model_ws, "hfb6.mlt.tpl")
    with open(tpl_file, "w", newline="") as ofp:
        ofp.write("ptf ~\n")
        [ofp.write("{0}\n".format(line.strip())) for line in header]
        ofp.flush()
        hfb_in[["lay", "irow1", "icol1", "irow2", "icol2", "tpl"]].to_csv(
            ofp, sep=" ", quotechar=" ", header=None, index=None, mode="a"
        )

    # make a lookup for lining up the necessary files to
    # perform multiplication with the helpers.apply_hfb_pars() function
    # which must be added to the forward run script
    with open(os.path.join(m.model_ws, "hfb6_pars.csv"), "w") as ofp:
        ofp.write("org_file,mlt_file,model_file\n")
        ofp.write(
            "{0},{1},{2}\n".format(
                os.path.join(m.model_ws, "hfb6_org", m.hfb6.file_name[0]),
                os.path.join(
                    m.model_ws,
                    "hfb6_mlt",
                    os.path.basename(tpl_file).replace(".tpl", ""),
                ),
                hfb_file,
            )
        )

    return hfb_mults, tpl_file


def write_hfb_template(m):
    """write a template file for an hfb (yuck!)

    Args:
         m (`flopy.modflow.Modflow`): a model instance with an HFB package

     Returns:
         tuple containing

         - **str**: name of the template file that was created

         - **pandas.DataFrame**: a dataframe with use control file info for the
           HFB parameters

    """

    assert m.hfb6 is not None
    hfb_file = os.path.join(m.model_ws, m.hfb6.file_name[0])
    assert os.path.exists(hfb_file), "couldn't find hfb_file {0}".format(hfb_file)
    f_in = open(hfb_file, "r")
    tpl_file = hfb_file + ".tpl"
    f_tpl = open(tpl_file, "w")
    f_tpl.write("ptf ~\n")
    parnme, parval1, xs, ys = [], [], [], []
    iis, jjs, kks = [], [], []
    try:
        xc = m.sr.xcentergrid
        yc = m.sr.ycentergrid
    except AttributeError:
        xc = m.modelgrid.xcellcenters
        yc = m.modelgrid.ycellcenters

    while True:
        line = f_in.readline()
        if line == "":
            break
        f_tpl.write(line)
        if not line.startswith("#"):
            raw = line.strip().split()
            nphfb = int(raw[0])
            mxfb = int(raw[1])
            nhfbnp = int(raw[2])
            if nphfb > 0 or mxfb > 0:
                raise Exception("not supporting terrible HFB pars")
            for i in range(nhfbnp):
                line = f_in.readline()
                if line == "":
                    raise Exception("EOF")
                raw = line.strip().split()
                k = int(raw[0]) - 1
                i = int(raw[1]) - 1
                j = int(raw[2]) - 1
                pn = "hb{0:02}{1:04d}{2:04}".format(k, i, j)
                pv = float(raw[5])
                raw[5] = "~ {0}  ~".format(pn)
                line = " ".join(raw) + "\n"
                f_tpl.write(line)
                parnme.append(pn)
                parval1.append(pv)
                xs.append(xc[i, j])
                ys.append(yc[i, j])
                iis.append(i)
                jjs.append(j)
                kks.append(k)

            break

    f_tpl.close()
    f_in.close()
    df = pd.DataFrame(
        {
            "parnme": parnme,
            "parval1": parval1,
            "x": xs,
            "y": ys,
            "i": iis,
            "j": jjs,
            "k": kks,
        },
        index=parnme,
    )
    df.loc[:, "pargp"] = "hfb_hydfac"
    df.loc[:, "parubnd"] = df.parval1.max() * 10.0
    df.loc[:, "parlbnd"] = df.parval1.min() * 0.1
    return tpl_file, df


class GsfReader:
    """
    a helper class to read a standard modflow-usg gsf file

    Args:
        gsffilename (`str`): filename



    """

    def __init__(self, gsffilename):

        with open(gsffilename, "r") as f:
            self.read_data = f.readlines()

        self.nnode, self.nlay, self.iz, self.ic = [
            int(n) for n in self.read_data[1].split()
        ]

        self.nvertex = int(self.read_data[2])

    def get_vertex_coordinates(self):
        """


        Returns:
            Dictionary containing list of x, y and z coordinates for each vertex
        """
        # vdata = self.read_data[3:self.nvertex+3]
        vertex_coords = {}
        for vert in range(self.nvertex):
            x, y, z = self.read_data[3 + vert].split()
            vertex_coords[vert + 1] = [float(x), float(y), float(z)]
        return vertex_coords

    def get_node_data(self):
        """

        Returns:
            nodedf: a pd.DataFrame containing Node information; Node, X, Y, Z, layer, numverts, vertidx

        """

        node_data = []
        for node in range(self.nnode):
            nid, x, y, z, lay, numverts = self.read_data[
                self.nvertex + 3 + node
            ].split()[:6]

            # vertidx = {'ivertex': [int(n) for n in self.read_data[self.nvertex+3 + node].split()[6:]]}
            vertidx = [
                int(n) for n in self.read_data[self.nvertex + 3 + node].split()[6:]
            ]

            node_data.append(
                [
                    int(nid),
                    float(x),
                    float(y),
                    float(z),
                    int(lay),
                    int(numverts),
                    vertidx,
                ]
            )

        nodedf = pd.DataFrame(
            node_data, columns=["node", "x", "y", "z", "layer", "numverts", "vertidx"]
        )
        return nodedf

    def get_node_coordinates(self, zcoord=False, zero_based=False):
        """
        Args:
            zcoord (`bool`): flag to add z coord to coordinates.  Default is False
            zero_based (`bool`): flag to subtract one from the node numbers in the returned
                node_coords dict.  This is needed to support PstFrom.  Default is False


        Returns:
            node_coords: Dictionary containing x and y coordinates for each node
        """
        node_coords = {}
        for node in range(self.nnode):
            nid, x, y, z, lay, numverts = self.read_data[
                self.nvertex + 3 + node
            ].split()[:6]
            nid = int(nid)
            if zero_based:
                nid -= 1
            node_coords[nid] = [float(x), float(y)]
            if zcoord:
                node_coords[nid] += [float(z)]

        return node_coords
