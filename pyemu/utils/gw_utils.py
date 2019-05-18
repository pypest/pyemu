""" module of utilities for groundwater modeling
"""

import os
import copy
import csv
from datetime import datetime
import shutil
import warnings
import numpy as np
import pandas as pd
import re
pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT,pst_config,_try_run_inschek,\
    parse_tpl_file,try_process_ins_file
from pyemu.utils.os_utils import run
from pyemu.utils.helpers import write_df_tpl
from ..pyemu_warnings import PyemuWarning
PP_FMT = {"name": SFMT, "x": FFMT, "y": FFMT, "zone": IFMT, "tpl": SFMT,
          "parval1": FFMT}
PP_NAMES = ["name","x","y","zone","parval1"]


def modflow_pval_to_template_file(pval_file,tpl_file=None):
    """write a template file for a modflow parameter value file.
    Uses names in the first column in the pval file as par names.

    Parameters
    ----------
    pval_file : str
        parameter value file
    tpl_file : str, optional
        template file to write.  If None, use <pval_file>.tpl.
        Default is None

    Returns
    -------
    df : pandas.DataFrame
        pandas DataFrame with control file parameter information
    """

    if tpl_file is None:
        tpl_file = pval_file + ".tpl"
    pval_df = pd.read_csv(pval_file,delim_whitespace=True,
                          header=None,skiprows=2,
                          names=["parnme","parval1"])
    pval_df.index = pval_df.parnme
    pval_df.loc[:,"tpl"] = pval_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x))
    with open(tpl_file,'w') as f:
        f.write("ptf ~\n#pval template file from pyemu\n")
        f.write("{0:10d} #NP\n".format(pval_df.shape[0]))
        f.write(pval_df.loc[:,["parnme","tpl"]].to_string(col_space=0,
                                                          formatters=[SFMT,SFMT],
                                                          index=False,
                                                          header=False,
                                                          justify="left"))
    return pval_df

def modflow_hob_to_instruction_file(hob_file):
    """write an instruction file for a modflow head observation file

    Parameters
    ----------
    hob_file : str
        modflow hob file

    Returns
    -------
    df : pandas.DataFrame
        pandas DataFrame with control file observation information
    """

    hob_df = pd.read_csv(hob_file,delim_whitespace=True,skiprows=1,
                         header=None,names=["simval","obsval","obsnme"])

    hob_df.loc[:,"obsnme"] = hob_df.obsnme.apply(str.lower)
    hob_df.loc[:,"ins_line"] = hob_df.obsnme.apply(lambda x:"l1 !{0:s}!".format(x))
    hob_df.loc[0,"ins_line"] = hob_df.loc[0,"ins_line"].replace('l1','l2')

    ins_file = hob_file + ".ins"
    f_ins = open(ins_file, 'w')
    f_ins.write("pif ~\n")
    f_ins.write(hob_df.loc[:,["ins_line"]].to_string(col_space=0,
                                                     columns=["ins_line"],
                                                     header=False,
                                                     index=False,
                                                     formatters=[SFMT]) + '\n')
    hob_df.loc[:,"weight"] = 1.0
    hob_df.loc[:,"obgnme"] = "obgnme"
    f_ins.close()
    return hob_df

def modflow_hydmod_to_instruction_file(hydmod_file):
    """write an instruction file for a modflow hydmod file

    Parameters
    ----------
    hydmod_file : str
        modflow hydmod file


    Returns
    -------
    df : pandas.DataFrame
        pandas DataFrame with control file observation information

    Note
    ----
    calls modflow_read_hydmod_file()
    """

    hydmod_df, hydmod_outfile = modflow_read_hydmod_file(hydmod_file)


    hydmod_df.loc[:,"ins_line"] = hydmod_df.obsnme.apply(lambda x:"l1 w !{0:s}!".format(x))

    ins_file = hydmod_outfile + ".ins"

    with open(ins_file, 'w') as f_ins:
        f_ins.write("pif ~\nl1\n")
        f_ins.write(hydmod_df.loc[:,["ins_line"]].to_string(col_space=0,
                                                     columns=["ins_line"],
                                                     header=False,
                                                     index=False,
                                                     formatters=[SFMT]) + '\n')
    hydmod_df.loc[:,"weight"] = 1.0
    hydmod_df.loc[:,"obgnme"] = "obgnme"

    try:
        os.system("inschek {0}.ins {0}".format(hydmod_outfile))
    except:
        print("error running inschek")

    obs_obf = hydmod_outfile + ".obf"
    if os.path.exists(obs_obf):
        df = pd.read_csv(obs_obf,delim_whitespace=True,header=None,names=["obsnme","obsval"])
        df.loc[:,"obgnme"] = df.obsnme.apply(lambda x: x[:-9])
        df.to_csv("_setup_"+os.path.split(hydmod_outfile)[-1]+'.csv',index=False)
        df.index = df.obsnme
        return df


    return hydmod_df

def modflow_read_hydmod_file(hydmod_file, hydmod_outfile=None):
    """ read in a binary hydmod file and return a dataframe of the results

    Parameters
    ----------
    hydmod_file : str
        modflow hydmod binary file
    hydmod_outfile : str
        output file to write.  If None, use <hydmod_file>.dat.
        Default is None

    Returns
    -------
    df : pandas.DataFrame
        pandas DataFrame with hymod_file values

    Note
    ----
    requires flopy
    """
    try:
        import flopy.utils as fu
    except Exception as e:
        print('flopy is not installed - cannot read {0}\n{1}'.format(hydmod_file, e))
        return
    #print('Starting to read HYDMOD data from {0}'.format(hydmod_file))
    obs = fu.HydmodObs(hydmod_file)
    hyd_df = obs.get_dataframe()

    hyd_df.columns = [i[2:] if i.lower() != 'totim' else i for i in hyd_df.columns]
    #hyd_df.loc[:,"datetime"] = hyd_df.index
    hyd_df['totim'] = hyd_df.index.map(lambda x: x.strftime("%Y%m%d"))

    hyd_df.rename(columns={'totim': 'datestamp'}, inplace=True)


    # reshape into a single column
    hyd_df = pd.melt(hyd_df, id_vars='datestamp')

    hyd_df.rename(columns={'value': 'obsval'}, inplace=True)

    hyd_df['obsnme'] = [i.lower() + '_' + j.lower() for i, j in zip(hyd_df.variable, hyd_df.datestamp)]

    vc = hyd_df.obsnme.value_counts().sort_values()
    vc = list(vc.loc[vc>1].index.values)
    if len(vc) > 0:
        hyd_df.to_csv("hyd_df.duplciates.csv")
        obs.get_dataframe().to_csv("hyd_org.duplicates.csv")
        raise Exception("duplicates in obsnme:{0}".format(vc))
    #assert hyd_df.obsnme.value_counts().max() == 1,"duplicates in obsnme"

    if not hydmod_outfile:
        hydmod_outfile = hydmod_file + '.dat'
    hyd_df.to_csv(hydmod_outfile, columns=['obsnme','obsval'], sep=' ',index=False)
    #hyd_df = hyd_df[['obsnme','obsval']]
    return hyd_df[['obsnme','obsval']], hydmod_outfile


def setup_pilotpoints_grid(ml=None,sr=None,ibound=None,prefix_dict=None,
                           every_n_cell=4,
                           use_ibound_zones=False,
                           pp_dir='.',tpl_dir='.',
                           shapename="pp.shp"):
    """ setup regularly-spaced (gridded) pilot point parameterization

    Parameters
    ----------
    ml : flopy.mbase
        a flopy mbase dervied type.  If None, sr must not be None.
    sr : flopy.utils.reference.SpatialReference
        a spatial reference use to locate the model grid in space.  If None,
        ml must not be None.  Default is None
    ibound : numpy.ndarray
        the modflow ibound integer array.  Used to set pilot points only in active areas.
        If None and ml is None, then pilot points are set in all rows and columns according to
        every_n_cell.  Default is None.
    prefix_dict : dict
        a dictionary of pilot point parameter prefix, layer pairs.  example : {"hk":[0,1,2,3]} would
        setup pilot points with the prefix "hk" for model layers 1 - 4 (zero based). If None, a generic set
        of pilot points with the "pp" prefix are setup for a generic nrowXncol grid. Default is None
    use_ibound_zones : bool
        a flag to use the greater-than-zero values in the ibound as pilot point zones.  If False,ibound
        values greater than zero are treated as a single zone.  Default is False.
    pp_dir : str
        directory to write pilot point files to.  Default is '.'
    tpl_dir : str
        directory to write pilot point template file to.  Default is '.'
    shapename : str
        name of shapefile to write that containts pilot point information. Default is "pp.shp"

    Returns
    -------
        pp_df : pandas.DataFrame
            a dataframe summarizing pilot point information (same information
            written to shapename

    """
    from . import pp_utils
    warnings.warn("setup_pilotpoint_grid has moved to pp_utils...",PyemuWarning)
    return pp_utils.setup_pilotpoints_grid(ml=ml,sr=sr,ibound=ibound,
                                           prefix_dict=prefix_dict,
                                           every_n_cell=every_n_cell,
                                           use_ibound_zones=use_ibound_zones,
                                           pp_dir=pp_dir,tpl_dir=tpl_dir,
                                           shapename=shapename)


def pp_file_to_dataframe(pp_filename):


    from . import pp_utils
    warnings.warn("pp_file_to_dataframe has moved to pp_utils",PyemuWarning)
    return pp_utils.pp_file_to_dataframe(pp_filename)

def pp_tpl_to_dataframe(tpl_filename):

    from . import pp_utils
    warnings.warn("pp_tpl_to_dataframe has moved to pp_utils",PyemuWarning)
    return pp_utils.pp_tpl_to_dataframe(tpl_filename)

def write_pp_shapfile(pp_df,shapename=None):
    from . import pp_utils
    warnings.warn("write_pp_shapefile has moved to pp_utils",PyemuWarning)
    pp_utils.write_pp_shapfile(pp_df,shapename=shapename)


def write_pp_file(filename,pp_df):
    from . import pp_utils
    warnings.warn("write_pp_file has moved to pp_utils",PyemuWarning)
    return pp_utils.write_pp_file(filename,pp_df)

def pilot_points_to_tpl(pp_file,tpl_file=None,name_prefix=None):
    from . import pp_utils
    warnings.warn("pilot_points_to_tpl has moved to pp_utils",PyemuWarning)
    return pp_utils.pilot_points_to_tpl(pp_file,tpl_file=tpl_file,
                                        name_prefix=name_prefix)

def fac2real(pp_file=None,factors_file="factors.dat",out_file="test.ref",
             upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30):
    from . import geostats as gs
    warnings.warn("fac2real has moved to geostats",PyemuWarning)
    return gs.fac2real(pp_file=pp_file,factors_file=factors_file,
                       out_file=out_file,upper_lim=upper_lim,
                       lower_lim=lower_lim,fill_value=fill_value)


def setup_mtlist_budget_obs(list_filename,gw_filename="mtlist_gw.dat",sw_filename="mtlist_sw.dat",
                            start_datetime="1-1-1970",gw_prefix='gw',sw_prefix="sw",
                            save_setup_file=False):
    """ setup observations of gw (and optionally sw) mass budgets from mt3dusgs list file.  writes
        an instruction file and also a _setup_.csv to use when constructing a pest
        control file

        Parameters
        ----------
        list_filename : str
                modflow list file
        gw_filename : str
            output filename that will contain the gw budget observations. Default is
            "mtlist_gw.dat"
        sw_filename : str
            output filename that will contain the sw budget observations. Default is
            "mtlist_sw.dat"
        start_datetime : str
            an str that can be parsed into a pandas.TimeStamp.  used to give budget
            observations meaningful names
        gw_prefix : str
            a prefix to add to the GW budget observations.  Useful if processing
            more than one list file as part of the forward run process. Default is 'gw'.
        sw_prefix : str
            a prefix to add to the SW budget observations.  Useful if processing
            more than one list file as part of the forward run process. Default is 'sw'.
        save_setup_file : (boolean)
            a flag to save _setup_<list_filename>.csv file that contains useful
            control file information

        Returns
        -------
        frun_line, ins_filenames, df :str, list(str), pandas.DataFrame
            the command to add to the forward run script, the names of the instruction
            files and a dataframe with information for constructing a control file.  If INSCHEK fails
            to run, df = None

        Note
        ----
        This function uses INSCHEK to get observation values; the observation values are
        the values of the list file list_filename.  If INSCHEK fails to run, the obseravtion
        values are set to 1.0E+10

        the instruction files are named <out_filename>.ins

        It is recommended to use the default value for gw_filename or sw_filename.

        """
    gw,sw = apply_mtlist_budget_obs(list_filename, gw_filename, sw_filename, start_datetime)
    gw_ins = gw_filename + ".ins"
    _write_mtlist_ins(gw_ins, gw, gw_prefix)
    ins_files = [gw_ins]
    try:
        run("inschek {0}.ins {0}".format(gw_filename))
    except:
        print("error running inschek")
    if sw is not None:
        sw_ins = sw_filename + ".ins"
        _write_mtlist_ins(sw_ins, sw, sw_prefix)
        ins_files.append(sw_ins)
        try:
            run("inschek {0}.ins {0}".format(sw_filename))
        except:
            print("error running inschek")
    frun_line = "pyemu.gw_utils.apply_mtlist_budget_obs('{0}')".format(list_filename)
    gw_obf = gw_filename + ".obf"
    df_gw = None
    if os.path.exists(gw_obf):
        df_gw = pd.read_csv(gw_obf, delim_whitespace=True, header=None, names=["obsnme", "obsval"])
        df_gw.loc[:, "obgnme"] = df_gw.obsnme.apply(lambda x: x[:-9])
        sw_obf = sw_filename + ".obf"
        if os.path.exists(sw_obf):
            df_sw = pd.read_csv(sw_obf, delim_whitespace=True, header=None, names=["obsnme", "obsval"])
            df_sw.loc[:, "obgnme"] = df_sw.obsnme.apply(lambda x: x[:-9])
            df_gw = df_gw.append(df_sw)

        if save_setup_file:
            df_gw.to_csv("_setup_" + os.path.split(list_filename)[-1] + '.csv', index=False)
        df_gw.index = df_gw.obsnme
    return frun_line,ins_files,df_gw

def _write_mtlist_ins(ins_filename,df,prefix):
    """ write an instruction file for a MODFLOW list file

    Parameters
    ----------
    ins_filename : str
        name of the instruction file to write
    df : pandas.DataFrame
        the dataframe of list file entries
    prefix : str
        the prefix to add to the column names to form
        obseravtions names

    """
    try:
        dt_str = df.index.map(lambda x: x.strftime("%Y%m%d"))
    except:
        dt_str = df.index.map(lambda x: "{0:08.1f}".format(x).strip())
    if prefix == '':
        name_len = 11
    else:
        name_len = 11 - (len(prefix)+1)
    with open(ins_filename,'w') as f:
        f.write('pif ~\nl1\n')

        for dt in dt_str:
            f.write("l1 ")
            for col in df.columns:
                col = col.replace("(",'').replace(")",'')
                raw = col.split('_')
                name = ''.join([r[:2] for r in raw[:-2]])[:6] + raw[-2] + raw[-1][0]
                #raw[0] = raw[0][:6]
                #name = ''.join(raw)
                if prefix == '':
                    obsnme = "{1}_{2}".format(prefix,name[:name_len],dt)
                else:
                    obsnme = "{0}_{1}_{2}".format(prefix, name[:name_len], dt)
                f.write(" w !{0}!".format(obsnme))
            f.write("\n")

def apply_mtlist_budget_obs(list_filename,gw_filename="mtlist_gw.dat",
                            sw_filename="mtlist_sw.dat",
                            start_datetime="1-1-1970"):
    """ process an MT3D list file to extract mass budget entries.

    Parameters
    ----------
    list_filename : str
        the mt3d list file
    gw_filename : str
        the name of the output file with gw mass budget information.
        Default is "mtlist_gw.dat"
    sw_filename : str
        the name of the output file with sw mass budget information.
        Default is "mtlist_sw.dat"
    start_datatime : str
        an str that can be cast to a pandas.TimeStamp.  Used to give
        observations a meaningful name

    Returns
    -------
    gw : pandas.DataFrame
        the gw mass dataframe
    sw : pandas.DataFrame (optional)
        the sw mass dataframe

    Note
    ----
    requires flopy

    if SFT is not active, no SW mass budget will be returned

    """
    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    mt = flopy.utils.MtListBudget(list_filename)
    gw, sw = mt.parse(start_datetime=start_datetime, diff=True)
    gw = gw.drop([col for col in gw.columns
                  for drop_col in ["kper", "kstp", "tkstp"]
                  if (col.lower().startswith(drop_col))], axis=1)
    gw.to_csv(gw_filename, sep=' ', index_label="datetime", date_format="%Y%m%d")
    if sw is not None:
        sw = sw.drop([col for col in sw.columns
                      for drop_col in ["kper", "kstp", "tkstp"]
                      if (col.lower().startswith(drop_col))], axis=1)
        sw.to_csv(sw_filename, sep=' ', index_label="datetime", date_format="%Y%m%d")
    return gw, sw

def setup_mflist_budget_obs(list_filename,flx_filename="flux.dat",
                            vol_filename="vol.dat",start_datetime="1-1'1970",prefix='',
                            save_setup_file=False):
    """ setup observations of budget volume and flux from modflow list file.  writes
    an instruction file and also a _setup_.csv to use when constructing a pest
    control file

    Parameters
    ----------
    list_filename : str
            modflow list file
    flx_filename : str
        output filename that will contain the budget flux observations. Default is
        "flux.dat"
    vol_filename : str)
        output filename that will contain the budget volume observations.  Default
        is "vol.dat"
    start_datetime : str
        an str that can be parsed into a pandas.TimeStamp.  used to give budget
        observations meaningful names
    prefix : str
        a prefix to add to the water budget observations.  Useful if processing
        more than one list file as part of the forward run process. Default is ''.
    save_setup_file : (boolean)
        a flag to save _setup_<list_filename>.csv file that contains useful
        control file information

    Returns
    -------
    df : pandas.DataFrame
        a dataframe with information for constructing a control file.  If INSCHEK fails
        to run, reutrns None

    Note
    ----
    This function uses INSCHEK to get observation values; the observation values are
    the values of the list file list_filename.  If INSCHEK fails to run, the obseravtion
    values are set to 1.0E+10

    the instruction files are named <flux_file>.ins and <vol_file>.ins, respectively

    It is recommended to use the default values for flux_file and vol_file.


    """



    flx,vol = apply_mflist_budget_obs(list_filename,flx_filename,vol_filename,
                                      start_datetime)
    _write_mflist_ins(flx_filename+".ins",flx,prefix+"flx")
    _write_mflist_ins(vol_filename+".ins",vol, prefix+"vol")

    #run("inschek {0}.ins {0}".format(flx_filename))
    #run("inschek {0}.ins {0}".format(vol_filename))

    try:
        #os.system("inschek {0}.ins {0}".format(flx_filename))
        #os.system("inschek {0}.ins {0}".format(vol_filename))
        run("inschek {0}.ins {0}".format(flx_filename))
        run("inschek {0}.ins {0}".format(vol_filename))

    except:
        print("error running inschek")
        return None
    flx_obf = flx_filename+".obf"
    vol_obf = vol_filename + ".obf"
    if os.path.exists(flx_obf) and os.path.exists(vol_obf):
        df = pd.read_csv(flx_obf,delim_whitespace=True,header=None,names=["obsnme","obsval"])
        df.loc[:,"obgnme"] = df.obsnme.apply(lambda x: x[:-9])
        df2 = pd.read_csv(vol_obf, delim_whitespace=True, header=None, names=["obsnme", "obsval"])
        df2.loc[:, "obgnme"] = df2.obsnme.apply(lambda x: x[:-9])
        df = df.append(df2)
        if save_setup_file:
            df.to_csv("_setup_"+os.path.split(list_filename)[-1]+'.csv',index=False)
        df.index = df.obsnme
        return df

def apply_mflist_budget_obs(list_filename,flx_filename="flux.dat",
                            vol_filename="vol.dat",
                            start_datetime="1-1-1970"):
    """ process a MODFLOW list file to extract flux and volume water budget entries.

    Parameters
    ----------
    list_filename : str
        the modflow list file
    flx_filename : str
        the name of the output file with water budget flux information.
        Default is "flux.dat"
    vol_filename : str
        the name of the output file with water budget volume information.
        Default is "vol.dat"
    start_datatime : str
        an str that can be cast to a pandas.TimeStamp.  Used to give
        observations a meaningful name

    Returns
    -------
    flx : pandas.DataFrame
        the flux dataframe
    vol : pandas.DataFrame
        the volume dataframe

    Note
    ----
    requires flopy

    """
    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    mlf = flopy.utils.MfListBudget(list_filename)
    flx,vol = mlf.get_dataframes(start_datetime=start_datetime,diff=True)
    flx.to_csv(flx_filename,sep=' ',index_label="datetime",date_format="%Y%m%d")
    vol.to_csv(vol_filename,sep=' ',index_label="datetime",date_format="%Y%m%d")
    return flx,vol


def _write_mflist_ins(ins_filename,df,prefix):
    """ write an instruction file for a MODFLOW list file

    Parameters
    ----------
    ins_filename : str
        name of the instruction file to write
    df : pandas.DataFrame
        the dataframe of list file entries
    prefix : str
        the prefix to add to the column names to form
        obseravtions names

    """

    dt_str = df.index.map(lambda x: x.strftime("%Y%m%d"))
    name_len = 11 - (len(prefix)+1)
    with open(ins_filename,'w') as f:
        f.write('pif ~\nl1\n')

        for dt in dt_str:
            f.write("l1 ")
            for col in df.columns:
                obsnme = "{0}_{1}_{2}".format(prefix,col[:name_len],dt)
                f.write(" w !{0}!".format(obsnme))
            f.write("\n")


def setup_hds_timeseries(hds_file,kij_dict,prefix=None,include_path=False,
                         model=None, postprocess_inact=None):
    """a function to setup extracting time-series from a binary modflow
    head save (or equivalent format - ucn, sub, etc).  Writes
    an instruction file and a _set_ csv

    Parameters
    ----------
    hds_file : str
        binary filename
    kij_dict : dict
        dictionary of site_name: [k,i,j] pairs
    prefix : str
        string to prepend to site_name when forming obsnme's.  Default is None
    include_path : bool
        flag to prepend hds_file path. Useful for setting up
        process in separate directory for where python is running.
    model : flopy.mbase
        a flopy model.  If passed, the observation names will have the datetime of the
        observation appended to them.  If None, the observation names will have the
        stress period appended to them. Default is None.
    postprocess_inact : float
        Inactive flag in heads/ucn file e.g. mt.btn.cinit

    Returns
    -------



    Note
    ----
    This function writes hds_timeseries.config that must be in the same
    dir where apply_hds_timeseries() is called during the forward run

    assumes model time units are days!!!

    """

    try:
        import flopy
    except Exception as e:
        print("error importing flopy, returning {0}".format(str(e)))
        return

    assert os.path.exists(hds_file),"head save file not found"
    if hds_file.lower().endswith(".ucn"):
        try:
            hds = flopy.utils.UcnFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            hds = flopy.utils.HeadFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    nlay,nrow,ncol = hds.nlay,hds.nrow,hds.ncol

    #if include_path:
    #    pth = os.path.join(*[p for p in os.path.split(hds_file)[:-1]])
    #    config_file = os.path.join(pth,"{0}_timeseries.config".format(hds_file))
    #else:
    config_file = "{0}_timeseries.config".format(hds_file)
    print("writing config file to {0}".format(config_file))

    f_config = open(config_file,'w')
    if model is not None:
        if model.dis.itmuni != 4:
            warnings.warn("setup_hds_timeseries only supports 'days' time units...",PyemuWarning)
        f_config.write("{0},{1},d\n".format(os.path.split(hds_file)[-1],model.start_datetime))
        start = pd.to_datetime(model.start_datetime)
    else:
        f_config.write("{0},none,none\n".format(os.path.split(hds_file)[-1]))
    f_config.write("site,k,i,j\n")
    dfs = []

    for site,(k,i,j) in kij_dict.items():
        assert k >= 0 and k < nlay, k
        assert i >= 0 and i < nrow, i
        assert j >= 0 and j < ncol, j
        site = site.lower().replace(" ",'')
        df = pd.DataFrame(data=hds.get_ts((k,i,j)),columns=["totim",site])

        if model is not None:
            dts = start + pd.to_timedelta(df.totim,unit='d')
            df.loc[:,"totim"] = dts
        #print(df)
        f_config.write("{0},{1},{2},{3}\n".format(site,k,i,j))
        df.index = df.pop("totim")
        dfs.append(df)

    f_config.close()
    df = pd.concat(dfs,axis=1)
    df.to_csv(hds_file+"_timeseries.processed",sep=' ')
    if model is not None:
        t_str = df.index.map(lambda x: x.strftime("%Y%m%d"))
    else:
        t_str = df.index.map(lambda x: "{0:08.2f}".format(x))

    ins_file = hds_file+"_timeseries.processed.ins"
    print("writing instruction file to {0}".format(ins_file))
    with open(ins_file,'w') as f:
        f.write('pif ~\n')
        f.write("l1 \n")
        for t in t_str:
            f.write("l1 w ")
            for site in df.columns:
                if prefix is not None:
                    obsnme = "{0}_{1}_{2}".format(prefix,site,t)
                else:
                    obsnme = "{0}_{1}".format(site, t)
                f.write(" !{0}!".format(obsnme))
            f.write('\n')
    if postprocess_inact is not None:
        _setup_postprocess_hds_timeseries(hds_file, df, config_file, prefix=prefix, model=model)
    bd = '.'
    if include_path:
        bd = os.getcwd()
        pth = os.path.join(*[p for p in os.path.split(hds_file)[:-1]])
        os.chdir(pth)
    config_file = os.path.split(config_file)[-1]
    try:
        df = apply_hds_timeseries(config_file, postprocess_inact=postprocess_inact)
    except Exception as e:
        os.chdir(bd)
        raise Exception("error in apply_hds_timeseries(): {0}".format(str(e)))
    os.chdir(bd)

    #df = _try_run_inschek(ins_file,ins_file.replace(".ins",""))
    df = try_process_ins_file(ins_file,ins_file.replace(".ins",""))
    if df is not None:
        df.loc[:,"weight"] = 0.0
        if prefix is not None:
            df.loc[:,"obgnme"] = df.index.map(lambda x: '_'.join(x.split('_')[:2]))
        else:
            df.loc[:, "obgnme"] = df.index.map(lambda x: x.split('_')[0])
    frun_line = "pyemu.gw_utils.apply_hds_timeseries('{0}',{1})\n".format(config_file, postprocess_inact)
    return frun_line,df


def apply_hds_timeseries(config_file=None, postprocess_inact=None):

    import flopy

    if config_file is None:
        config_file = "hds_timeseries.config"

    assert os.path.exists(config_file), config_file
    with open(config_file,'r') as f:
        line = f.readline()
        hds_file,start_datetime,time_units = line.strip().split(',')
        site_df = pd.read_csv(f)

    #print(site_df)

    assert os.path.exists(hds_file), "head save file not found"
    if hds_file.lower().endswith(".ucn"):
        try:
            hds = flopy.utils.UcnFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            hds = flopy.utils.HeadFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    nlay, nrow, ncol = hds.nlay, hds.nrow, hds.ncol

    dfs = []
    for site,k,i,j in zip(site_df.site,site_df.k,site_df.i,site_df.j):
        assert k >= 0 and k < nlay
        assert i >= 0 and i < nrow
        assert j >= 0 and j < ncol
        df = pd.DataFrame(data=hds.get_ts((k,i,j)),columns=["totim",site])
        df.index = df.pop("totim")
        dfs.append(df)
    df = pd.concat(dfs,axis=1)
    #print(df)
    df.to_csv(hds_file+"_timeseries.processed",sep=' ')
    if postprocess_inact is not None:
        _apply_postprocess_hds_timeseries(config_file, postprocess_inact)
    return df


def _setup_postprocess_hds_timeseries(hds_file, df, config_file, prefix=None, model=None):
    """Dirty function to post process concentrations in inactive/dry cells"""
    warnings.warn(
        "Setting up post processing of hds or ucn timeseries obs. "
        "Prepending 'pp' to obs name may cause length to exceed 20 chars", PyemuWarning)
    if model is not None:
        t_str = df.index.map(lambda x: x.strftime("%Y%m%d"))
    else:
        t_str = df.index.map(lambda x: "{0:08.2f}".format(x))
    if prefix is not None:
        prefix = "pp{0}".format(prefix)
    else:
        prefix = "pp"
    ins_file = hds_file+"_timeseries.post_processed.ins"
    print("writing instruction file to {0}".format(ins_file))
    with open(ins_file,'w') as f:
        f.write('pif ~\n')
        f.write("l1 \n")
        for t in t_str:
            f.write("l1 w ")
            for site in df.columns:
                obsnme = "{0}{1}_{2}".format(prefix, site, t)
                f.write(" !{0}!".format(obsnme))
            f.write('\n')
    frun_line = "pyemu.gw_utils._apply_postprocess_hds_timeseries('{0}')\n".format(config_file)
    return frun_line


def _apply_postprocess_hds_timeseries(config_file=None, cinact=1e30):

    import flopy

    if config_file is None:
        config_file = "hds_timeseries.config"

    assert os.path.exists(config_file), config_file
    with open(config_file,'r') as f:
        line = f.readline()
        hds_file,start_datetime,time_units = line.strip().split(',')
        site_df = pd.read_csv(f)

    #print(site_df)

    assert os.path.exists(hds_file), "head save file not found"
    if hds_file.lower().endswith(".ucn"):
        try:
            hds = flopy.utils.UcnFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            hds = flopy.utils.HeadFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    nlay, nrow, ncol = hds.nlay, hds.nrow, hds.ncol

    dfs = []
    for site, k, i, j in zip(site_df.site, site_df.k, site_df.i, site_df.j):
        assert k >= 0 and k < nlay
        assert i >= 0 and i < nrow
        assert j >= 0 and j < ncol
        df = pd.DataFrame(data=hds.get_ts((k, i, j)), columns=["totim", site])
        df.index = df.pop("totim")
        inact_obs = df[site].apply(lambda x: np.isclose(x, cinact))
        if inact_obs.sum() > 0:
            assert k+1 < nlay, "Inactive observation in lowest layer"
            df_lower = pd.DataFrame(data=hds.get_ts((k+1, i, j)), columns=["totim", site])
            df_lower.index = df_lower.pop("totim")
            df.loc[inact_obs] = df_lower.loc[inact_obs]
            print("{0} observation(s) post-processed for site {1} at kij ({2},{3},{4})".
                  format(inact_obs.sum(), site, k, i, j))
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    #print(df)
    df.to_csv(hds_file+"_timeseries.post_processed", sep=' ')
    return df

def setup_hds_obs(hds_file,kperk_pairs=None,skip=None,prefix="hds"):
    """a function to setup using all values from a
    layer-stress period pair for observations.  Writes
    an instruction file and a _setup_ csv used
    construct a control file.

    Parameters
    ----------
    hds_file : str
        a MODFLOW head-save file.  If the hds_file endswith 'ucn',
        then the file is treated as a UcnFile type.
    kperk_pairs : iterable
        an iterable of pairs of kper (zero-based stress
        period index) and k (zero-based layer index) to
        setup observations for.  If None, then a shit-ton
        of observations may be produced!
    skip : variable
        a value or function used to determine which values
        to skip when setting up observations.  If np.scalar(skip)
        is True, then values equal to skip will not be used.
        If skip can also be a np.ndarry with dimensions equal to the model.
        Obscervations are set up only for cells with Non-zero values in the array.
        If not np.ndarray or np.scalar(skip), then skip will be treated as a lambda function that
        returns np.NaN if the value should be skipped.
    prefix : str
        the prefix to use for the observation names. default is "hds".

    Returns
    -------
    (forward_run_line, df) : str, pd.DataFrame
        a python code str to add to the forward run script and the setup info for the observations

    Note
    ----
    requires flopy

    writes <hds_file>.dat.ins instruction file

    writes _setup_<hds_file>.csv which contains much
    useful information for construction a control file


    """
    try:
        import flopy
    except Exception as e:
        print("error importing flopy, returning {0}".format(str(e)))
        return

    assert os.path.exists(hds_file),"head save file not found"
    if hds_file.lower().endswith(".ucn"):
        try:
            hds = flopy.utils.UcnFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating UcnFile:{0}".format(str(e)))
    else:
        try:
            hds = flopy.utils.HeadFile(hds_file)
        except Exception as e:
            raise Exception("error instantiating HeadFile:{0}".format(str(e)))

    if kperk_pairs is None:
        kperk_pairs = []
        for kstp,kper in hds.kstpkper:
            kperk_pairs.extend([(kper-1,k) for k in range(hds.nlay)])
    if len(kperk_pairs) == 2:
        try:
            if len(kperk_pairs[0]) == 2:
                pass
        except:
            kperk_pairs = [kperk_pairs]

    #if start_datetime is not None:
    #    start_datetime = pd.to_datetime(start_datetime)
    #    dts = start_datetime + pd.to_timedelta(hds.times,unit='d')
    data = {}
    kpers = [kper-1 for kstp,kper in hds.kstpkper]
    for kperk_pair in kperk_pairs:
        kper,k = kperk_pair
        assert kper in kpers, "kper not in hds:{0}".format(kper)
        assert k in range(hds.nlay), "k not in hds:{0}".format(k)
        kstp = last_kstp_from_kper(hds,kper)
        d = hds.get_data(kstpkper=(kstp,kper))[k,:,:]

        data["{0}_{1}".format(kper,k)] = d.flatten()
        #data[(kper,k)] = d.flatten()
    idx,iidx,jidx = [],[],[]
    for _ in range(len(data)):
        for i in range(hds.nrow):
            iidx.extend([i for _ in range(hds.ncol)])
            jidx.extend([j for j in range(hds.ncol)])
            idx.extend(["i{0:04d}_j{1:04d}".format(i,j) for j in range(hds.ncol)])
    idx = idx[:hds.nrow*hds.ncol]

    df = pd.DataFrame(data,index=idx)
    data_cols = list(df.columns)
    data_cols.sort()
    #df.loc[:,"iidx"] = iidx
    #df.loc[:,"jidx"] = jidx
    if skip is not None:
        for col in data_cols:
            if np.isscalar(skip):
                df.loc[df.loc[:,col]==skip,col] = np.NaN
            elif isinstance(skip, np.ndarray):
                assert skip.ndim >= 2, "skip passed as {}D array, At least 2D (<= 4D) array required".format(skip.ndim)
                assert skip.shape[-2:] == (hds.nrow, hds.ncol), \
                    "Array dimensions of arg. skip needs to match model dimensions ({0},{1}). ({2},{3}) passed".\
                        format(hds.nrow, hds.ncol, skip.shape[-2], skip.shape[-1])
                if skip.ndim == 2:
                    print("2D array passed for skip, assuming constant for all layers and kper")
                    skip = np.tile(skip, (len(kpers), hds.nlay, 1, 1))
                if skip.ndim == 3:
                    print("3D array passed for skip, assuming constant for all kper")
                    skip = np.tile(skip, (len(kpers), 1, 1, 1))
                kper, k = [int(c) for c in col.split('_')]
                df.loc[df.index.map(
                    lambda x: skip[kper, k, int(x.split('_')[0].strip('i')), int(x.split('_')[1].strip('j'))] == 0),
                    col] = np.NaN
            else:
                df.loc[:,col] = df.loc[:,col].apply(skip)

    # melt to long form
    df = df.melt(var_name="kperk",value_name="obsval")
    # set row and col identifies
    df.loc[:,"iidx"] = iidx
    df.loc[:,"jidx"] = jidx
    #drop nans from skip
    df = df.dropna()
    #set some additional identifiers
    df.loc[:,"kper"] = df.kperk.apply(lambda x: int(x.split('_')[0]))
    df.loc[:,"kidx"] = df.pop("kperk").apply(lambda x: int(x.split('_')[1]))

    # form obs names
    #def get_kper_str(kper):
    #    if start_datetime is not None:
    #        return  dts[int(kper)].strftime("%Y%m%d")
    #    else:
    #        return "kper{0:04.0f}".format(kper)
    fmt = prefix + "_{0:02.0f}_{1:03.0f}_{2:03.0f}_{3:03.0f}"
    # df.loc[:,"obsnme"] = df.apply(lambda x: fmt.format(x.kidx,x.iidx,x.jidx,
    #                                                   get_kper_str(x.kper)),axis=1)
    df.loc[:,"obsnme"] = df.apply(lambda x: fmt.format(x.kidx,x.iidx,x.jidx,
                                                      x.kper),axis=1)

    df.loc[:,"ins_str"] = df.obsnme.apply(lambda x: "l1 w !{0}!".format(x))
    df.loc[:,"obgnme"] = prefix
    #write the instruction file
    with open(hds_file+".dat.ins","w") as f:
        f.write("pif ~\nl1\n")
        df.ins_str.to_string(f,index=False,header=False)

    #write the corresponding output file
    df.loc[:,["obsnme","obsval"]].to_csv(hds_file+".dat",sep=' ',index=False)

    hds_path = os.path.dirname(hds_file)
    setup_file = os.path.join(hds_path,"_setup_{0}.csv".format(os.path.split(hds_file)[-1]))
    df.to_csv(setup_file)
    fwd_run_line = "pyemu.gw_utils.apply_hds_obs('{0}')\n".format(hds_file)
    df.index = df.obsnme
    return fwd_run_line, df


def last_kstp_from_kper(hds,kper):
    """ function to find the last time step (kstp) for a
    give stress period (kper) in a modflow head save file.


    Parameters
    ----------
    hds : flopy.utils.HeadFile

    kper : int
        the zero-index stress period number

    Returns
    -------
    kstp : int
        the zero-based last time step during stress period
        kper in the head save file


    """
    #find the last kstp with this kper
    kstp = -1
    for kkstp,kkper in hds.kstpkper:
        if kkper == kper+1 and kkstp > kstp:
            kstp = kkstp
    if kstp == -1:
        raise Exception("kstp not found for kper {0}".format(kper))
    kstp -= 1
    return kstp


def apply_hds_obs(hds_file):
    """ process a modflow head save file.  A companion function to
    setup_hds_obs that is called during the forward run process

    Parameters
    ----------
    hds_file : str
        a modflow head save filename. if hds_file ends with 'ucn',
        then the file is treated as a UcnFile type.

    Note
    ----
    requires flopy

    writes <hds_file>.dat

    expects <hds_file>.dat.ins to exist

    uses pyemu.pst_utils.parse_ins_file to get observation names

    """

    try:
        import flopy
    except Exception as e:
        raise Exception("apply_hds_obs(): error importing flopy: {0}".\
                        format(str(e)))
    from .. import pst_utils
    assert os.path.exists(hds_file)
    out_file = hds_file+".dat"
    ins_file = out_file + ".ins"
    assert os.path.exists(ins_file)
    df = pd.DataFrame({"obsnme":pst_utils.parse_ins_file(ins_file)})
    df.index = df.obsnme

    # populate metdata
    items = ["k","i","j","kper"]
    for i,item in enumerate(items):
        df.loc[:,item] = df.obsnme.apply(lambda x: int(x.split('_')[i+1]))

    if hds_file.lower().endswith('ucn'):
        hds = flopy.utils.UcnFile(hds_file)
    else:
        hds = flopy.utils.HeadFile(hds_file)
    kpers = df.kper.unique()
    df.loc[:,"obsval"] = np.NaN
    for kper in kpers:
        kstp = last_kstp_from_kper(hds,kper)
        data = hds.get_data(kstpkper=(kstp,kper))
        #jwhite 15jan2018 fix for really large values that are getting some
        #trash added to them...
        data[np.isnan(data)] = 0.0
        data[data>1.0e+20] = 1.0e+20
        data[data<-1.0e+20] = -1.0e+20
        df_kper = df.loc[df.kper==kper,:]
        df.loc[df_kper.index,"obsval"] = data[df_kper.k,df_kper.i,df_kper.j]
    assert df.dropna().shape[0] == df.shape[0]
    df.loc[:,["obsnme","obsval"]].to_csv(out_file,index=False,sep=" ")


def setup_sft_obs(sft_file,ins_file=None,start_datetime=None,times=None,ncomp=1):
    """writes an instruction file for a mt3d-usgs sft output file

    Parameters
    ----------
        sft_file : str
            the sft output file (ASCII)
        ins_file : str
            the name of the instruction file to create.  If None, the name
            is <sft_file>.ins.  Default is None
        start_datetime : str
            a pandas.to_datetime() compatible str.  If not None,
            then the resulting observation names have the datetime
            suffix.  If None, the suffix is the output totim.  Default
            is None
        times : iterable
            a container of times to make observations for.  If None, all times are used.
            Default is None.
        ncomp : int
            number of components in transport model. Default is 1.


    Returns
    -------
        df : pandas.DataFrame
            a dataframe with obsnme and obsval for the sft simulated concentrations and flows.
            If inschek was not successfully run, then returns None


    Note
    ----
        setups up observations for SW conc, GW conc and flowgw for all times and reaches.
    """

    df = pd.read_csv(sft_file,skiprows=1,delim_whitespace=True)
    df.columns = [c.lower().replace("-","_") for c in df.columns]
    if times is None:
        times = df.time.unique()
    missing = []
    utimes = df.time.unique()
    for t in times:
        if t not in utimes:
            missing.append(str(t))
    if len(missing) > 0:
        print(df.time)
        raise Exception("the following times are missing:{0}".format(','.join(missing)))
    with open("sft_obs.config",'w') as f:
        f.write(sft_file+'\n')
        [f.write("{0:15.6E}\n".format(t)) for t in times]
    df = apply_sft_obs()
    utimes = df.time.unique()
    for t in times:
        assert t in utimes,"time {0} missing in processed dataframe".format(t)
    idx = df.time.apply(lambda x: x in times)
    if start_datetime is not None:
        start_datetime = pd.to_datetime(start_datetime)
        df.loc[:,"time_str"] = pd.to_timedelta(df.time,unit='d') + start_datetime
        df.loc[:,"time_str"] = df.time_str.apply(lambda x: datetime.strftime(x,"%Y%m%d"))
    else:
        df.loc[:,"time_str"] = df.time.apply(lambda x: "{0:08.2f}".format(x))
    df.loc[:,"ins_str"] = "l1\n"
    # check for multiple components
    df_times = df.loc[idx,:]
    df.loc[:,"icomp"] = 1
    icomp_idx = list(df.columns).index("icomp")
    for t in times:
        df_time = df.loc[df.time==t,:]
        vc = df_time.sfr_node.value_counts()
        ncomp = vc.max()
        assert np.all(vc.values==ncomp)
        nstrm = df_time.shape[0] / ncomp
        for icomp in range(ncomp):
            s = int(nstrm*(icomp))
            e = int(nstrm*(icomp+1))
            idxs = df_time.iloc[s:e,:].index
            #df_time.iloc[nstrm*(icomp):nstrm*(icomp+1),icomp_idx.loc["icomp"] = int(icomp+1)
            df_time.loc[idxs,"icomp"] = int(icomp+1)

        df.loc[df_time.index,"ins_str"] = df_time.apply(lambda x: "l1 w w !sfrc{0}_{1}_{2}! !swgw{0}_{1}_{2}! !gwcn{0}_{1}_{2}!\n".\
                                         format(x.sfr_node,x.icomp,x.time_str),axis=1)
    df.index = np.arange(df.shape[0])
    if ins_file is None:
        ins_file = sft_file+".processed.ins"

    with open(ins_file,'w') as f:
        f.write("pif ~\nl1\n")
        [f.write(i) for i in df.ins_str]
    #df = _try_run_inschek(ins_file,sft_file+".processed")
    df = try_process_ins_file(ins_file,sft_file+".processed")
    if df is not None:
        return df
    else:
        return None


def apply_sft_obs():
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
    df = pd.read_csv(sft_file,skiprows=1,delim_whitespace=True)#,nrows=10000000)
    df.columns = [c.lower().replace("-", "_") for c in df.columns]
    df = df.loc[df.time.apply(lambda x: x in times), :]
    #print(df.dtypes)
    #normalize
    for c in df.columns:
        #print(c)
        if not "node" in c:
            df.loc[:,c] = df.loc[:,c].apply(try_cast)
        #print(df.loc[df.loc[:,c].apply(lambda x : type(x) == str),:])
        df.loc[df.loc[:,c].apply(lambda x: x<1e-30),c] = 0.0
        df.loc[df.loc[:, c] > 1e+30, c] = 1.0e+30
    df.loc[:,"sfr_node"] = df.sfr_node.apply(np.int)

    df.to_csv(sft_file+".processed",sep=' ',index=False)
    return df


def setup_sfr_seg_parameters(nam_file, model_ws='.', par_cols=["flow", "runoff", "hcond1", "pptsw"],
                             tie_hcond=True, include_temporal_pars=False):
    """Setup multiplier parameters for SFR segment data.  Just handles the
    standard input case, not all the cryptic SFR options.  Loads the dis, bas, and sfr files
    with flopy using model_ws.  However, expects that apply_sfr_seg_parameters() will be called
    from within model_ws at runtime.

    Parameters
    ----------
        nam_file : str
            MODFLOw name file.  DIS, BAS, and SFR must be available as pathed in the nam_file
        model_ws : str
            model workspace for flopy to load the MODFLOW model from
        OR
        nam_file : flopy.modflow.mf.Modflow
            flopy modflow model object
        par_cols : list(str)
            segment data entires to parameterize
        tie_hcond : bool
            flag to use same mult par for hcond1 and hcond2 for a given segment.  Default is True
        include_temporal_pars : bool
            flag to include spatially-global multipliers for each par_col for each stress period.  Default is False

    Returns
    -------
        df : pandas.DataFrame
            a dataframe with useful parameter setup information

    Note
    ----
        the number (and numbering) of segment data entries must consistent across
        all stress periods.
        writes <nam_file>+"_backup_.sfr" as the backup of the original sfr file
        skips values = 0.0 since multipliers don't work for these

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

    if isinstance(nam_file,flopy.modflow.mf.Modflow) and nam_file.sfr is not None:
        m = nam_file
        nam_file = m.namefile
        model_ws = m.model_ws
    else:
        # load MODFLOW model # is this needed? could we just pass the model if it has already been read in?
        m = flopy.modflow.Modflow.load(nam_file,load_only=["sfr"],model_ws=model_ws,check=False,forgive=False)
    #make backup copy of sfr file
    shutil.copy(os.path.join(model_ws,m.sfr.file_name[0]),os.path.join(model_ws,nam_file+"_backup_.sfr"))

    #get the segment data (dict)
    segment_data = m.sfr.segment_data
    shape = segment_data[list(segment_data.keys())[0]].shape
    # check
    for kper,seg_data in m.sfr.segment_data.items():
        assert seg_data.shape == shape,"cannot use: seg data must have the same number of entires for all kpers"
    seg_data_col_order = list(seg_data.dtype.names)
    # convert segment_data dictionary to multi index df - this could get ugly
    reform = {(k, c): segment_data[k][c] for k in segment_data.keys() for c in segment_data[k].dtype.names}
    seg_data_all_kper = pd.DataFrame.from_dict(reform)
    seg_data_all_kper.columns.names = ['kper', 'col']

    # extract the first seg data kper to a dataframe
    seg_data = seg_data_all_kper[0].copy() # pd.DataFrame.from_records(seg_data)

    #make sure all par cols are found and search of any data in kpers
    missing = []
    cols = par_cols.copy()
    for par_col in par_cols:
        if par_col not in seg_data.columns:
            missing.append(par_col)
            cols.remove(par_col)
        # look across all kper in multiindex df to check for values entry - fill with absmax should capture entries
        else:
            seg_data.loc[:, par_col] = seg_data_all_kper.loc[:, (slice(None), par_col)].abs().max(level=1, axis=1)
    if len(missing) > 0:
        warnings.warn("the following par_cols were not found in segment data: {0}".format(','.join(missing)), PyemuWarning)
        if len(missing) >= len(par_cols):
            warnings.warn("None of the passed par_cols ({0}) were found in segment data.".
                          format(','.join(par_cols)), PyemuWarning)
    seg_data = seg_data[seg_data_col_order]  # reset column orders to inital
    seg_data_org = seg_data.copy()
    seg_data.to_csv(os.path.join(model_ws, "sfr_seg_pars.dat"), sep=',')

    #the data cols not to parameterize
    # better than a column indexer as pandas can change column orders
    idx_cols=['nseg', 'icalc', 'outseg', 'iupseg', 'iprior', 'nstrpts']
    notpar_cols = [c for c in seg_data.columns if c not in cols+idx_cols]

    #process par cols
    tpl_str, pvals = [], []
    for par_col in cols:
        prefix = par_col
        if tie_hcond and par_col == 'hcond2':
            prefix = 'hcond1'
        if seg_data.loc[:, par_col].sum() == 0.0:
            print("all zeros for {0}...skipping...".format(par_col))
            #seg_data.loc[:,par_col] = 1
        else:
            seg_data.loc[:, par_col] = seg_data.apply(lambda x: "~    {0}_{1:04d}   ~".
                                                      format(prefix, int(x.nseg)) if float(x[par_col]) != 0.0
                                                      else "1.0", axis=1)

            org_vals = seg_data_org.loc[seg_data_org.loc[:, par_col] != 0.0, par_col]
            pnames = seg_data.loc[org_vals.index, par_col]
            pvals.extend(list(org_vals.values))
            tpl_str.extend(list(pnames.values))

    pnames = [t.replace('~','').strip() for t in tpl_str]
    df = pd.DataFrame({"parnme":pnames,"org_value":pvals,"tpl_str":tpl_str},index=pnames)
    df.drop_duplicates(inplace=True)
    if df.empty:
        warnings.warn("No sfr segment parameters have been set up, either none of {0} were found or all were zero.".
                      format(','.join(par_cols)), PyemuWarning)
        return df

    # set not par cols to 1.0
    seg_data.loc[:, notpar_cols] = "1.0"

    #write the template file
    write_df_tpl(os.path.join(model_ws,"sfr_seg_pars.dat.tpl"), seg_data, sep=',')



    #make sure the tpl file exists and has the same num of pars
    parnme = parse_tpl_file(os.path.join(model_ws,"sfr_seg_pars.dat.tpl"))
    assert len(parnme) == df.shape[0]

    #set some useful par info
    df.loc[:,"pargp"] = df.parnme.apply(lambda x: x.split('_')[0])
    df.loc[:,"parubnd"] = 1.25
    df.loc[:,"parlbnd"] = 0.75
    hpars = df.loc[df.pargp.apply(lambda x: x.startswith("hcond")),"parnme"]
    df.loc[hpars,"parubnd"] = 100.0
    df.loc[hpars, "parlbnd"] = 0.01

    if include_temporal_pars:
        # include only stress periods that are explicitly listed in segment data
        pnames,tpl_str = [],[]
        tmp_df = pd.DataFrame(data={c:1.0 for c in cols},index=list(m.sfr.segment_data.keys()))
        tmp_df.sort_index(inplace=True)
        tmp_df.to_csv(os.path.join(model_ws,"sfr_seg_temporal_pars.dat"))
        for par_col in cols:
            print(par_col)
            parnme = tmp_df.index.map(lambda x: "{0}_{1:04d}_tmp".format(par_col,int(x)))
            tmp_df.loc[:,par_col] = parnme.map(lambda x: "~   {0}  ~".format(x))
            tpl_str.extend(list(tmp_df.loc[:,par_col].values))
            pnames.extend(list(parnme.values))
        write_df_tpl(filename=os.path.join(model_ws,"sfr_seg_temporal_pars.dat.tpl"),df=tmp_df)
        pargp = [pname.split('_')[0]+"_tmp" for pname in pnames]
        tmp_df = pd.DataFrame(data={"parnme":pnames,"pargp":pargp},index=pnames)
        tmp_df.loc[:,"org_value"] = 1.0
        tmp_df.loc[:,"tpl_str"] = tpl_str
        df = df.append(tmp_df)

    # write the config file used by apply_sfr_pars()
    with open(os.path.join(model_ws, "sfr_seg_pars.config"), 'w') as f:
        f.write("nam_file {0}\n".format(nam_file))
        f.write("model_ws {0}\n".format(model_ws))
        f.write("mult_file sfr_seg_pars.dat\n")
        f.write("sfr_filename {0}\n".format(m.sfr.file_name[0]))
        if include_temporal_pars:
            f.write("time_mult_file sfr_seg_temporal_pars.dat\n")


    # set some useful par info
    df.loc[:, "pargp"] = df.parnme.apply(lambda x: x.split('_')[0])
    df.loc[:, "parubnd"] = 1.25
    df.loc[:, "parlbnd"] = 0.75
    hpars = df.loc[df.pargp.apply(lambda x: x.startswith("hcond")), "parnme"]
    df.loc[hpars, "parubnd"] = 100.0
    df.loc[hpars, "parlbnd"] = 0.01

    return df


def setup_sfr_reach_parameters(nam_file,model_ws='.', par_cols=['strhc1']):
    """Setup multiplier paramters for reach data, when reachinput option is specififed in sfr.
    Similare to setup_sfr_seg_parameters() method will apply params to sfr reachdata
    Can load the dis, bas, and sfr files with flopy using model_ws. Or can pass a model object (SFR loading can be slow)
    However, expects that apply_sfr_reach_parameters() will be called
    from within model_ws at runtime.

    Parameters
    ----------
        nam_file : str
            MODFLOw name file.  DIS, BAS, and SFR must be available as pathed in the nam_file
        model_ws : str
            model workspace for flopy to load the MODFLOW model from
        OR
        nam_file : flopy.modflow.mf.Modflow
            flopy modflow model object

        par_cols : list(str)
            segment data entires to parameterize

    Returns
    -------
        df : pandas.DataFrame
            a dataframe with useful parameter setup information

    Note
    ----
        skips values = 0.0 since multipliers don't work for these

    """
    try:
        import flopy
    except Exception as e:
        return
    if par_cols is None:
        par_cols = ['strhc1']
    if isinstance(nam_file,flopy.modflow.mf.Modflow) and nam_file.sfr is not None:
        # flopy MODFLOW model has been passed and has SFR loaded
        m = nam_file
        nam_file = m.namefile
        model_ws = m.model_ws
    else:
        # if model has not been passed or SFR not loaded # load MODFLOW model
        m = flopy.modflow.Modflow.load(nam_file, load_only=["sfr"], model_ws=model_ws, check=False, forgive=False)
    # get reachdata as dataframe
    reach_data = pd.DataFrame.from_records(m.sfr.reach_data)
    # write inital reach_data as csv
    reach_data_orig = reach_data.copy()
    reach_data.to_csv(os.path.join(m.model_ws, "sfr_reach_pars.dat"), sep=',')

    # generate template file with pars in par_cols
    #process par cols
    tpl_str,pvals = [],[]
    # par_cols=["strhc1"]
    idx_cols=["node", "k", "i", "j", "iseg", "ireach", "reachID", "outreach"]
    #the data cols not to parameterize
    notpar_cols = [c for c in reach_data.columns if c not in par_cols+idx_cols]
    # make sure all par cols are found and search of any data in kpers
    missing = []
    cols = par_cols.copy()
    for par_col in par_cols:
        if par_col not in reach_data.columns:
            missing.append(par_col)
            cols.remove(par_col)
    if len(missing) > 0:
        warnings.warn("the following par_cols were not found in reach data: {0}".format(','.join(missing)),
                      PyemuWarning)
        if len(missing) >= len(par_cols):
            warnings.warn("None of the passed par_cols ({0}) were found in reach data.".
                          format(','.join(par_cols)), PyemuWarning)
    for par_col in cols:
        if par_col == "strhc1":
            prefix = 'strk'  # shorten par
        else:
            prefix = par_col
        reach_data.loc[:, par_col] = reach_data.apply(
            lambda x: "~    {0}_{1:04d}   ~".format(prefix, int(x.reachID))
            if float(x[par_col]) != 0.0 else "1.0", axis=1)
        org_vals = reach_data_orig.loc[reach_data_orig.loc[:, par_col] != 0.0, par_col]
        pnames = reach_data.loc[org_vals.index, par_col]
        pvals.extend(list(org_vals.values))
        tpl_str.extend(list(pnames.values))
    pnames = [t.replace('~','').strip() for t in tpl_str]
    df = pd.DataFrame({"parnme":pnames,"org_value":pvals,"tpl_str":tpl_str},index=pnames)
    df.drop_duplicates(inplace=True)
    if df.empty:
        warnings.warn("No sfr reach parameters have been set up, either none of {0} were found or all were zero.".
                      format(','.join(par_cols)), PyemuWarning)
    else:
        # set not par cols to 1.0
        reach_data.loc[:, notpar_cols] = "1.0"

        # write the template file
        write_df_tpl(os.path.join(model_ws, "sfr_reach_pars.dat.tpl"),reach_data,sep=',')

        # write the config file used by apply_sfr_pars()
        with open(os.path.join(model_ws, "sfr_reach_pars.config"), 'w') as f:
            f.write("nam_file {0}\n".format(nam_file))
            f.write("model_ws {0}\n".format(model_ws))
            f.write("mult_file sfr_reach_pars.dat\n")
            f.write("sfr_filename {0}".format(m.sfr.file_name[0]))

        # make sure the tpl file exists and has the same num of pars
        parnme = parse_tpl_file(os.path.join(model_ws, "sfr_reach_pars.dat.tpl"))
        assert len(parnme) == df.shape[0]

        # set some useful par info
        df.loc[:, "pargp"] = df.parnme.apply(lambda x: x.split('_')[0])
        df.loc[:, "parubnd"] = 1.25
        df.loc[:, "parlbnd"] = 0.75
        hpars = df.loc[df.pargp.apply(lambda x: x.startswith("strk")), "parnme"]
        df.loc[hpars, "parubnd"] = 100.0
        df.loc[hpars, "parlbnd"] = 0.01
    return df


def apply_sfr_seg_parameters(seg_pars=True, reach_pars=False):
    """apply the SFR segement multiplier parameters.  Expected to be run in the same dir
    as the model exists

    Parameters
    ----------
        reach_pars : bool
            if reach paramters need to be applied

    Returns
    -------
        sfr : flopy.modflow.ModflowSfr instance

    Note
    ----
        expects "sfr_seg_pars.config" to exist
        expects <nam_file>+"_backup_.sfr" to exist


    """

    if not seg_pars and not reach_pars:
        raise Exception("gw_utils.apply_sfr_pars() error: both seg_pars and reach_pars are False")
    #if seg_pars and reach_pars:
    #    raise Exception("gw_utils.apply_sfr_pars() error: both seg_pars and reach_pars are True")

    import flopy
    bak_sfr_file,pars = None,None

    # if seg_pars:
    #     config_file = "sfr_seg_pars.config"
    #     idx_cols = ['nseg', 'icalc', 'outseg', 'iupseg', 'iprior', 'nstrpts']
    # else:
    #     config_file = "sfr_reach_pars.config"
    #     idx_cols = ["node", "k", "i", "j", "iseg", "ireach", "reachID", "outreach"]
    #
    # assert os.path.exists(config_file),"gw_utils.apply_sfr_pars() error: config file {0} missing".format(config_file)
    # with open(config_file, 'r') as f:
    #     pars = {}
    #     for line in f:
    #         line = line.strip().split()
    #         pars[line[0]] = line[1]
    #
    # m = flopy.modflow.Modflow.load(pars["nam_file"], load_only=[], check=False)
    # bak_sfr_file = pars["nam_file"] + "_backup_.sfr"
    # sfr = flopy.modflow.ModflowSfr2.load(os.path.join(bak_sfr_file), m)
    # sfrfile = pars["sfr_filename"]
    #
    #
    # mlt_df = pd.read_csv(pars["mult_file"], delim_whitespace=False, index_col=0)
    # present_cols = [c for c in idx_cols if c in mlt_df.columns]
    # mlt_cols = mlt_df.columns.drop(present_cols)
    #
    # if seg_pars:
    #     for key, val in m.sfr.segment_data.items():
    #         df = pd.DataFrame.from_records(val)
    #         df.loc[:, mlt_cols] *= mlt_df.loc[:, mlt_cols]
    #         val = df.to_records(index=False)
    #         sfr.segment_data[key] = val
    # else:
    #     df = pd.DataFrame.from_records(m.sfr.reach_data)
    #     df.loc[:, mlt_cols] *= mlt_df.loc[:, mlt_cols]
    #     sfr.reach_data = df.to_records(index=False)
    #
    #
    # if "time_mult_file" in pars:
    #     time_mult_file = pars["time_mult_file"]
    #     time_mlt_df = pd.read_csv(pars["time_mult_file"], delim_whitespace=False, index_col=0)
    #     for kper,sdata in m.sfr.segment_data.items():
    #         assert kper in time_mlt_df.index,"gw_utils.apply_sfr_seg_parameters() error: kper "+\
    #                                          "{0} not in time_mlt_df index".format(kper)
    #         for col in time_mlt_df.columns:
    #             sdata[col] *= time_mlt_df.loc[kper,col]
    #
    #
    # sfr.write_file(filename=sfrfile)
    # return sfr

    if seg_pars:
        assert os.path.exists("sfr_seg_pars.config")

        with open("sfr_seg_pars.config",'r') as f:
            pars = {}
            for line in f:
                line = line.strip().split()
                pars[line[0]] = line[1]
        bak_sfr_file = pars["nam_file"]+"_backup_.sfr"
        #m = flopy.modflow.Modflow.load(pars["nam_file"],model_ws=pars["model_ws"],load_only=["sfr"],check=False)
        m = flopy.modflow.Modflow.load(pars["nam_file"], load_only=[], check=False)
        sfr = flopy.modflow.ModflowSfr2.load(os.path.join(bak_sfr_file), m)
        sfrfile = pars["sfr_filename"]
        mlt_df = pd.read_csv(pars["mult_file"], delim_whitespace=False, index_col=0)
        time_mlt_df = None
        if "time_mult_file" in pars:
            time_mult_file = pars["time_mult_file"]
            time_mlt_df = pd.read_csv(pars["time_mult_file"], delim_whitespace=False,index_col=0)

        idx_cols = ['nseg', 'icalc', 'outseg', 'iupseg', 'iprior', 'nstrpts']
        present_cols = [c for c in idx_cols if c in mlt_df.columns]
        mlt_cols = mlt_df.columns.drop(present_cols)
        for key, val in m.sfr.segment_data.items():
            df = pd.DataFrame.from_records(val)
            df.loc[:, mlt_cols] *= mlt_df.loc[:, mlt_cols]
            val = df.to_records(index=False)
            sfr.segment_data[key] = val
    if reach_pars:
        assert os.path.exists("sfr_reach_pars.config")
        with open("sfr_reach_pars.config", 'r') as f:
            r_pars = {}
            for line in f:
                line = line.strip().split()
                r_pars[line[0]] = line[1]
        if bak_sfr_file is None:  # will be the case is seg_pars is false
            bak_sfr_file = r_pars["nam_file"]+"_backup_.sfr"
            #m = flopy.modflow.Modflow.load(pars["nam_file"],model_ws=pars["model_ws"],load_only=["sfr"],check=False)
            m = flopy.modflow.Modflow.load(r_pars["nam_file"], load_only=[], check=False)
            sfr = flopy.modflow.ModflowSfr2.load(os.path.join(bak_sfr_file), m)
            sfrfile = r_pars["sfr_filename"]
        r_mlt_df = pd.read_csv(r_pars["mult_file"],sep=',',index_col=0)
        r_idx_cols = ["node", "k", "i", "j", "iseg", "ireach", "reachID", "outreach"]
        r_mlt_cols = r_mlt_df.columns.drop(r_idx_cols)
        r_df = pd.DataFrame.from_records(m.sfr.reach_data)
        r_df.loc[:, r_mlt_cols] *= r_mlt_df.loc[:, r_mlt_cols]
        sfr.reach_data = r_df.to_records(index=False)


    #m.remove_package("sfr")
    if pars is not None and "time_mult_file" in pars:
        time_mult_file = pars["time_mult_file"]
        time_mlt_df = pd.read_csv(time_mult_file, delim_whitespace=False, index_col=0)
        for kper, sdata in m.sfr.segment_data.items():
            assert kper in time_mlt_df.index, "gw_utils.apply_sfr_seg_parameters() error: kper " + \
                                              "{0} not in time_mlt_df index".format(kper)
            for col in time_mlt_df.columns:
                sdata[col] *= time_mlt_df.loc[kper, col]

    sfr.write_file(filename=sfrfile)
    return sfr

def apply_sfr_parameters(seg_pars=True, reach_pars=False):
    sfr = apply_sfr_seg_parameters(seg_pars=seg_pars, reach_pars=reach_pars)
    return sfr


def setup_sfr_obs(sfr_out_file,seg_group_dict=None,ins_file=None,model=None,
                  include_path=False):
    """setup observations using the sfr ASCII output file.  Setups
    the ability to aggregate flows for groups of segments.  Applies
    only flow to aquier and flow out.

    Parameters
    ----------
    sft_out_file : str
        the existing SFR output file
    seg_group_dict : dict
        a dictionary of SFR segements to aggregate together for a single obs.
        the key value in the dict is the base observation name. If None, all segments
        are used as individual observations. Default is None
    model : flopy.mbase
        a flopy model.  If passed, the observation names will have the datetime of the
        observation appended to them.  If None, the observation names will have the
        stress period appended to them. Default is None.
    include_path : bool
        flag to prepend sfr_out_file path to sfr_obs.config.  Useful for setting up
        process in separate directory for where python is running.


    Returns
    -------
    df : pd.DataFrame
        dataframe of obsnme, obsval and obgnme if inschek run was successful.  Else None

    Note
    ----
    This function writes "sfr_obs.config" which must be kept in the dir where
    "apply_sfr_obs()" is being called during the forward run

    """

    sfr_dict = load_sfr_out(sfr_out_file)
    kpers = list(sfr_dict.keys())
    kpers.sort()

    if seg_group_dict is None:
        seg_group_dict = {"seg{0:04d}".format(s):s for s in sfr_dict[kpers[0]].segment}
    else:
        warnings.warn("Flow out (flout) of grouped segments will be aggregated... ", PyemuWarning)
    sfr_segs = set(sfr_dict[list(sfr_dict.keys())[0]].segment)
    keys = ["sfr_out_file"]
    if include_path:
        values = [os.path.split(sfr_out_file)[-1]]
    else:
        values = [sfr_out_file]
    for oname,segs in seg_group_dict.items():
        if np.isscalar(segs):
            segs_set = {segs}
            segs = [segs]
        else:
            segs_set = set(segs)
        diff = segs_set.difference(sfr_segs)
        if len(diff) > 0:
            raise Exception("the following segs listed with oname {0} where not found: {1}".
                            format(oname,','.join([str(s) for s in diff])))
        for seg in segs:
            keys.append(oname)
            values.append(seg)

    df_key = pd.DataFrame({"obs_base":keys,"segment":values})
    if include_path:
        pth = os.path.join(*[p for p in os.path.split(sfr_out_file)[:-1]])
        config_file = os.path.join(pth,"sfr_obs.config")
    else:
        config_file = "sfr_obs.config"
    print("writing 'sfr_obs.config' to {0}".format(config_file))
    df_key.to_csv(config_file)

    bd = '.'
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
        dts = (pd.to_datetime(model.start_datetime) + pd.to_timedelta(np.cumsum(model.dis.perlen.array),unit='d')).date
        df.loc[:,"datetime"] = df.kper.apply(lambda x: dts[x])
        df.loc[:,"time_str"] = df.datetime.apply(lambda x: x.strftime("%Y%m%d"))
    else:
        df.loc[:,"time_str"] = df.kper.apply(lambda x: "{0:04d}".format(x))
    df.loc[:,"flaqx_obsnme"] = df.apply(lambda x: "{0}_{1}_{2}".format("fa",x.obs_base,x.time_str),axis=1)
    df.loc[:,"flout_obsnme"] = df.apply(lambda x: "{0}_{1}_{2}".format("fo",x.obs_base,x.time_str),axis=1)

    if ins_file is None:
        ins_file = sfr_out_file + ".processed.ins"

    with open(ins_file,'w') as f:
        f.write("pif ~\nl1\n")
        for fla,flo in zip(df.flaqx_obsnme,df.flout_obsnme):
            f.write("l1 w w !{0}! !{1}!\n".format(fla,flo))

    df = None
    pth = os.path.split(ins_file)[:-1]
    pth = os.path.join(*pth)
    if pth == '':
        pth = '.'
    bd = os.getcwd()
    os.chdir(pth)
    try:
        #df = _try_run_inschek(os.path.split(ins_file)[-1],os.path.split(sfr_out_file+".processed")[-1])
        df = try_process_ins_file(os.path.split(ins_file)[-1],os.path.split(sfr_out_file+".processed")[-1])
    except Exception as e:
        pass
    os.chdir(bd)
    if df is not None:
        df.loc[:,"obsnme"] = df.index.values
        df.loc[:,"obgnme"] = df.obsnme.apply(lambda x: "flaqx" if x.startswith("fa") else "flout")
        return df


def apply_sfr_obs():
    """apply the sfr observation process - pairs with setup_sfr_obs().
    requires sfr_obs.config.  Writes <sfr_out_file>.processed, where
    <sfr_out_file> is defined in "sfr_obs.config"


    Parameters
    ----------
    None

    Returns
    -------
    df : pd.DataFrame
        a dataframe of aggregrated sfr segment aquifer and outflow
    """
    assert os.path.exists("sfr_obs.config")
    df_key = pd.read_csv("sfr_obs.config",index_col=0)

    assert df_key.iloc[0,0] == "sfr_out_file",df_key.iloc[0,:]
    sfr_out_file = df_key.iloc[0,1]
    df_key = df_key.iloc[1:,:]
    df_key.loc[:, "segment"] = df_key.segment.apply(np.int)
    df_key.index = df_key.segment
    seg_group_dict = df_key.groupby(df_key.obs_base).groups

    sfr_kper = load_sfr_out(sfr_out_file)
    kpers = list(sfr_kper.keys())
    kpers.sort()
    #results = {o:[] for o in seg_group_dict.keys()}
    results = []
    for kper in kpers:
        df = sfr_kper[kper]
        for obs_base,segs in seg_group_dict.items():
            agg = df.loc[segs.values,:].sum()  # still agg flout where seg groups are passed!
            #print(obs_base,agg)
            results.append([kper,obs_base,agg["flaqx"],agg["flout"]])
    df = pd.DataFrame(data=results,columns=["kper","obs_base","flaqx","flout"])
    df.sort_values(by=["kper","obs_base"],inplace=True)
    df.to_csv(sfr_out_file+".processed",sep=' ',index=False)
    return df


def load_sfr_out(sfr_out_file, selection=None):
    """load an ASCII SFR output file into a dictionary of kper: dataframes.  aggregates
    flow to aquifer for segments and returns and flow out at downstream end of segment.

    Parameters
    ----------
    sfr_out_file : str
        SFR ASCII output file

    Returns
    -------
        sfr_dict : dict
            dictionary of {kper:dataframe}

    """
    assert os.path.exists(sfr_out_file),"couldn't find sfr out file {0}".\
        format(sfr_out_file)
    tag = " stream listing"
    lcount = 0
    sfr_dict = {}
    if selection is None:
        pass
    elif isinstance(selection, str):
        assert selection == 'all', "If string passed as selection only 'all' allowed: {}".format(selection)
    else:
        assert isinstance(
            selection, pd.DataFrame), "'selection needs to be pandas Dataframe. Type {} passed.".format(type(selection))
        assert np.all([sr in selection.columns for sr in ['segment', 'reach']]
                      ), "Either 'segment' or 'reach' not in selection columns"
    with open(sfr_out_file) as f:
        while True:
            line = f.readline().lower()
            lcount += 1
            if line == '':
                break
            if line.startswith(tag):
                raw = line.strip().split()
                kper = int(raw[3]) - 1
                kstp = int(raw[5]) - 1
                [f.readline() for _ in range(4)] #skip to where the data starts
                lcount += 4
                dlines = []
                while True:
                    dline = f.readline()
                    lcount += 1
                    if dline.strip() == '':
                        break
                    draw = dline.strip().split()
                    dlines.append(draw)
                df = pd.DataFrame(data=np.array(dlines)).iloc[:, [3, 4, 6, 7]]
                df.columns = ["segment", "reach", "flaqx", "flout"]
                df.loc[:, "segment"] = df.segment.apply(np.int)
                df.loc[:, "reach"] = df.reach.apply(np.int)
                df.loc[:, "flaqx"] = df.flaqx.apply(np.float)
                df.loc[:, "flout"] = df.flout.apply(np.float)
                df.index = df.apply(lambda x: "{0:03d}_{1:03d}".format(int(x.segment), int(x.reach)), axis=1)
                if selection is None:  # setup for all segs, aggregate
                    gp = df.groupby(df.segment)
                    bot_reaches = gp[['reach']].max().apply(
                        lambda x: "{0:03d}_{1:03d}".format(int(x.name), int(x.reach)), axis=1)
                    df2 = pd.DataFrame(index=gp.groups.keys(), columns=['flaqx', 'flout'])
                    df2['flaqx'] = gp.flaqx.sum()  # only sum distributed output
                    df2['flout'] = df.loc[bot_reaches, 'flout'].values  # take flow out of seg
                    # df = df.groupby(df.segment).sum()
                    df2.loc[:,"segment"] = df2.index
                elif isinstance(selection, str) and selection == 'all':
                    df2 = df
                else:
                    seg_reach_id = selection.apply(lambda x: "{0:03d}_{1:03d}".
                                                   format(int(x.segment), int(x.reach)), axis=1).values
                    for sr in seg_reach_id:
                        if sr not in df.index:
                            s, r = [x.lstrip('0') for x in sr.split('_')]
                            warnings.warn("Requested segment reach pair ({0},{1}) is not in sfr output. Dropping...".
                                          format(int(r), int(s)), PyemuWarning)
                            seg_reach_id = np.delete(seg_reach_id, np.where(seg_reach_id == sr), axis=0)
                    df2 = df.loc[seg_reach_id].copy()
                if kper in sfr_dict.keys():
                    print("multiple entries found for kper {0}, replacing...".format(kper))
                sfr_dict[kper] = df2
    return sfr_dict


def setup_sfr_reach_obs(sfr_out_file,seg_reach=None,ins_file=None,model=None,
                        include_path=False):
    """setup observations using the sfr ASCII output file.  Setups
    sfr point observations using segment and reach numbers.

    Parameters
    ----------
    sft_out_file : str
        the existing SFR output file
    seg_reach : dict, list or pandas.DataFrame
        a dict, or list of SFR [segment,reach] pairs identifying observation locations.
        If dict the key value in the dict is the base observation name.
        If None, all reaches are used as individual observations. Default is None - THIS MAY SET UP A LOT OF OBS!
    model : flopy.mbase
        a flopy model.  If passed, the observation names will have the datetime of the
        observation appended to them.  If None, the observation names will have the
        stress period appended to them. Default is None.
    include_path : bool
        flag to prepend sfr_out_file path to sfr_obs.config.  Useful for setting up
        process in separate directory for where python is running.


    Returns
    -------
    df : pd.DataFrame
        dataframe of obsnme, obsval and obgnme if inschek run was successful.  Else None

    Note
    ----
    This function writes "sfr_reach_obs.config" which must be kept in the dir where
    "apply_sfr_reach_obs()" is being called during the forward run

    """
    if seg_reach is None:
        warnings.warn("Obs will be set up for every reach", PyemuWarning)
        seg_reach = 'all'
    elif isinstance(seg_reach, list) or isinstance(seg_reach, np.ndarray):
        if np.ndim(seg_reach) == 1:
            seg_reach = [seg_reach]
        assert np.shape(
            seg_reach)[1] == 2, "varible seg_reach expected shape (n,2), received {0}".format(np.shape(seg_reach))
        seg_reach = pd.DataFrame(seg_reach, columns=['segment', 'reach'])
        seg_reach.index = seg_reach.apply(lambda x: "s{0:03d}r{1:03d}".format(int(x.segment), int(x.reach)), axis=1)
    elif isinstance(seg_reach, dict):
        seg_reach = pd.DataFrame.from_dict(seg_reach, orient='index', columns=['segment', 'reach'])
    else:
        assert isinstance(
            seg_reach, pd.DataFrame), "'selection needs to be pandas Dataframe. Type {} passed.".format(type(seg_reach))
        assert np.all([sr in seg_reach.columns for sr in ['segment', 'reach']]
                      ), "Either 'segment' or 'reach' not in selection columns"

    sfr_dict = load_sfr_out(sfr_out_file, selection=seg_reach)
    kpers = list(sfr_dict.keys())
    kpers.sort()

    if isinstance(seg_reach, str) and seg_reach == 'all':
        seg_reach = sfr_dict[kpers[0]][['segment', 'reach']]
        seg_reach.index = seg_reach.apply(lambda x: "s{0:03d}r{1:03d}".format(int(x.segment), int(x.reach)), axis=1)
    keys = ["sfr_out_file"]
    if include_path:
        values = [os.path.split(sfr_out_file)[-1]]
    else:
        values = [sfr_out_file]
    diff = seg_reach.loc[seg_reach.apply(lambda x: "{0:03d}_{1:03d}".format(int(x.segment), int(x.reach))
                                         not in sfr_dict[list(sfr_dict.keys())[0]].index, axis=1)]

    if len(diff) > 0:
        for ob in diff.itertuples():
            warnings.warn("segs,reach pair listed with onames {0} was not found: {1}".
                          format(ob.Index, "({},{})".format(ob.segment, ob.reach)), PyemuWarning)
    seg_reach = seg_reach.drop(diff.index)
    seg_reach['obs_base'] = seg_reach.index
    df_key = pd.DataFrame({"obs_base": keys, "segment": 0, 'reach': values})
    df_key = pd.concat([df_key, seg_reach], sort=True).reset_index(drop=True)
    if include_path:
        pth = os.path.join(*[p for p in os.path.split(sfr_out_file)[:-1]])
        config_file = os.path.join(pth,"sfr_reach_obs.config")
    else:
        config_file = "sfr_reach_obs.config"
    print("writing 'sfr_reach_obs.config' to {0}".format(config_file))
    df_key.to_csv(config_file)

    bd = '.'
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
        dts = (pd.to_datetime(model.start_datetime) + pd.to_timedelta(np.cumsum(model.dis.perlen.array), unit='d')).date
        df.loc[:, "datetime"] = df.kper.apply(lambda x: dts[x])
        df.loc[:, "time_str"] = df.datetime.apply(lambda x: x.strftime("%Y%m%d"))
    else:
        df.loc[:, "time_str"] = df.kper.apply(lambda x: "{0:04d}".format(x))
    df.loc[:, "flaqx_obsnme"] = df.apply(lambda x: "{0}_{1}_{2}".format("fa", x.obs_base, x.time_str), axis=1)
    df.loc[:, "flout_obsnme"] = df.apply(lambda x: "{0}_{1}_{2}".format("fo", x.obs_base, x.time_str), axis=1)

    if ins_file is None:
        ins_file = sfr_out_file + ".reach_processed.ins"

    with open(ins_file, 'w') as f:
        f.write("pif ~\nl1\n")
        for fla, flo in zip(df.flaqx_obsnme, df.flout_obsnme):
            f.write("l1 w w !{0}! !{1}!\n".format(fla, flo))

    df = None
    pth = os.path.split(ins_file)[:-1]
    pth = os.path.join(*pth)
    if pth == '':
        pth = '.'
    bd = os.getcwd()
    os.chdir(pth)
    try:
        #df = _try_run_inschek(os.path.split(ins_file)[-1],os.path.split(sfr_out_file+".processed")[-1])
        df = try_process_ins_file(os.path.split(ins_file)[-1], os.path.split(sfr_out_file+".reach_processed")[-1])
    except Exception as e:
        pass
    os.chdir(bd)
    if df is not None:
        df.loc[:, "obsnme"] = df.index.values
        df.loc[:, "obgnme"] = df.obsnme.apply(lambda x: "flaqx" if x.startswith("fa") else "flout")
        return df


def apply_sfr_reach_obs():
    """apply the sfr reach observation process - pairs with setup_sfr_reach_obs().
    requires sfr_reach_obs.config.  Writes <sfr_out_file>.processed, where
    <sfr_out_file> is defined in "sfr_reach_obs.config"


    Parameters
    ----------
    None

    Returns
    -------
    df : pd.DataFrame
        a dataframe of sfr aquifer and outflow ad segment,reach locations
    """
    assert os.path.exists("sfr_reach_obs.config")
    df_key = pd.read_csv("sfr_reach_obs.config", index_col=0)

    assert df_key.iloc[0, 0] == "sfr_out_file", df_key.iloc[0, :]
    sfr_out_file = df_key.iloc[0].reach
    df_key = df_key.iloc[1:, :].copy()
    df_key.loc[:, "segment"] = df_key.segment.apply(np.int)
    df_key.loc[:, "reach"] = df_key.reach.apply(np.int)
    df_key = df_key.set_index('obs_base')

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
    df.to_csv(sfr_out_file+".reach_processed", sep=' ', index=False)
    return df


def modflow_sfr_gag_to_instruction_file(gage_output_file, ins_file=None, parse_filename=False):
    """writes an instruction file for an SFR gage output file to read Flow only at all times

        Parameters
        ----------
            gage_output_file : str
                the gage output filename (ASCII).

            ins_file : str
                the name of the instruction file to create.  If None, the name
                is <gage_output_file>.ins.  Default is None

            parse_filename : bool
                if True, get the gage_num parameter by parsing the gage output file filename
                if False, get the gage number from the file itself

        Returns
        -------
            df : pandas.DataFrame
                a dataframe with obsnme and obsval for the sfr simulated flows.
                If inschek was not successfully run, then returns None
            ins_file : str
                file name of instructions file relating to gage output.
            obs_file : str
                file name of processed gage output for all times


        Note
        ----
            sets up observations for gage outputs only for the Flow column.

            if parse_namefile is true, only text up to first '.' is used as the gage_num

        TODO : allow other observation types and align explicitly with times - now returns all values
        """

    if ins_file is None:
        ins_file = gage_output_file + '.ins'

    # navigate the file to be sure the header makes sense
    indat = [line.strip() for line in open(gage_output_file, 'r').readlines()]
    header = [i for i in indat if i.startswith('"')]
    # yank out the gage number to identify the observation names
    if parse_filename:
        gage_num = os.path.basename(gage_output_file).split('.')[0]
    else:
        gage_num = re.sub("[^0-9]", "", indat[0].lower().split("gage no.")[-1].strip().split()[0])

    # get the column names
    cols = [i.lower() for i in header if 'data' in i.lower()][0].lower().replace('"', '').replace('data:', '').split()

    # make sure "Flow" is included in the columns
    if 'flow' not in cols:
        raise Exception('Requested field "Flow" not in gage output columns')

    # find which column is for  "Flow"
    flowidx = np.where(np.array(cols) == 'flow')[0][0]

    # write out the instruction file lines
    inslines = ['l1 ' + (flowidx + 1) * 'w ' + '!g{0}_{1:d}!'.format(gage_num, j)
                for j in range(len(indat) - len(header))]
    inslines[0] = inslines[0].replace('l1', 'l{0:d}'.format(len(header) + 1))

    # write the instruction file
    with open(ins_file, 'w') as ofp:
        ofp.write('pif ~\n')
        [ofp.write('{0}\n'.format(line)) for line in inslines]

    df = _try_run_inschek(ins_file, gage_output_file)
    if df is not None:
        return df, ins_file, gage_output_file
    else:
        print("Inschek didn't run so nothing returned")
        return None

def setup_gage_obs(gage_file,ins_file=None,start_datetime=None,times=None):
    """writes an instruction file for a mt3d-usgs sft output file

    Parameters
    ----------
        gage_file : str
            the gage output file (ASCII)
        ins_file : str
            the name of the instruction file to create.  If None, the name
            is <gage_file>.processed.ins.  Default is None
        start_datetime : str
            a pandas.to_datetime() compatible str.  If not None,
            then the resulting observation names have the datetime
            suffix.  If None, the suffix is the output totim.  Default
            is None
        times : iterable
            a container of times to make observations for.  If None, all times are used.
            Default is None.


    Returns
    -------
        df : pandas.DataFrame
            a dataframe with obsnme and obsval for the sft simulated concentrations and flows.
            If inschek was not successfully run, then returns None
        ins_file : str
            file name of instructions file relating to gage output.
        obs_file : str
            file name of processed gage output (processed according to times passed above.)


    Note
    ----
        setups up observations for gage outputs (all columns).
    """

    with open(gage_file, 'r') as f:
        line1 = f.readline()
        gage_num = int(re.sub("[^0-9]", "", line1.split("GAGE No.")[-1].strip().split()[0]))
        gage_type = line1.split("GAGE No.")[-1].strip().split()[1].lower()
        obj_num = int(line1.replace('"', '').strip().split()[-1])
        line2 = f.readline()
        df = pd.read_csv(f, delim_whitespace=True, names=line2.replace('"', '').split()[1:])

    df.columns = [c.lower().replace("-", "_").replace('.', '_').strip('_') for c in df.columns]
    # get unique observation ids
    obs_ids = {col:"" for col in df.columns[1:]} # empty dictionary for observation ids
    for col in df.columns[1:]: # exclude column 1 (TIME)
        colspl = col.split('_')
        if len(colspl) > 1:
            # obs name built out of "g"(for gage) "s" or "l"(for gage type) 2 chars from column name - date added later
            obs_ids[col] = "g{0}{1}{2}".format(gage_type[0], colspl[0][0],colspl[-1][0])
        else:
            obs_ids[col] = "g{0}{1}".format(gage_type[0], col[0:2])
    with open("_gage_obs_ids.csv", "w") as f: # write file relating obs names to meaningfull keys!
        [f.write('{0},{1}\n'.format(key, obs)) for key, obs in obs_ids.items()]
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
        raise Exception("the following times are missing:{0}".format(','.join(missing)))
    # write output times to config file
    with open("gage_obs.config", 'w') as f:
        f.write(gage_file+'\n')
        [f.write("{0:15.10E}\n".format(t)) for t in times]
    # extract data for times: returns dataframe and saves a processed df - read by pest
    df, obs_file = apply_gage_obs(return_obs_file=True)
    utimes = df.time.unique()
    for t in times:
        assert np.isclose(t, utimes).any(), "time {0} missing in processed dataframe".format(t)
    idx = df.time.apply(lambda x: np.isclose(x, times).any())  # boolean selector of desired times in df
    if start_datetime is not None:
        # convert times to usable observation times
        start_datetime = pd.to_datetime(start_datetime)
        df.loc[:, "time_str"] = pd.to_timedelta(df.time, unit='d') + start_datetime
        df.loc[:, "time_str"] = df.time_str.apply(lambda x: datetime.strftime(x, "%Y%m%d"))
    else:
        df.loc[:, "time_str"] = df.time.apply(lambda x: "{0:08.2f}".format(x))
    # set up instructions (line feed for lines without obs (not in time)
    df.loc[:, "ins_str"] = "l1\n"
    df_times = df.loc[idx, :]  # Slice by desired times
    # TODO include GAGE No. in obs name (if permissible)
    df.loc[df_times.index, "ins_str"] = df_times.apply(lambda x: "l1 w {}\n".format(
        ' w '.join(["!{0}{1}!".format(obs, x.time_str) for key,obs in obs_ids.items()])), axis=1)
    df.index = np.arange(df.shape[0])
    if ins_file is None:
        ins_file = gage_file+".processed.ins"

    with open(ins_file, 'w') as f:
        f.write("pif ~\nl1\n")
        [f.write(i) for i in df.ins_str]
    df = _try_run_inschek(ins_file, gage_file+".processed")
    if df is not None:
        return df, ins_file, obs_file
    else:
        return None


def apply_gage_obs(return_obs_file=False):
    times = []
    with open("gage_obs.config") as f:
        gage_file = f.readline().strip()
        for line in f:
            times.append(float(line.strip()))
    obs_file = gage_file+".processed"
    with open(gage_file, 'r') as f:
        line1 = f.readline()
        gage_num = int(re.sub("[^0-9]", "", line1.split("GAGE No.")[-1].strip().split()[0]))
        gage_type = line1.split("GAGE No.")[-1].strip().split()[1].lower()
        obj_num = int(line1.replace('"', '').strip().split()[-1])
        line2 = f.readline()
        df = pd.read_csv(f, delim_whitespace=True, names=line2.replace('"', '').split()[1:])
    df.columns = [c.lower().replace("-", "_").replace('.', '_') for c in df.columns]
    df = df.loc[df.time.apply(lambda x: np.isclose(x, times).any()), :]
    df.to_csv(obs_file, sep=' ', index=False)
    if return_obs_file:
        return df, obs_file
    else:
        return df


def write_hfb_zone_multipliers_template(m):
    """write a template file for an hfb using multipliers per zone (double yuck!)

    Parameters
    ----------
        m : flopy.modflow.Modflow instance with an HFB file

    Returns
    -------
        (hfb_mults, tpl_filename) : (dict, str)
            a dictionary with original unique HFB conductivity values and their corresponding parameter names
            and the name of the template file

    """
    assert m.hfb6 is not None
    # find the model file
    hfb_file = os.path.join(m.model_ws, m.hfb6.file_name[0])

    # this will use multipliers, so need to copy down the original
    if not os.path.exists(os.path.join(m.model_ws, 'hfb6_org')):
        os.mkdir(os.path.join(m.model_ws, 'hfb6_org'))
    # copy down the original file
    shutil.copy2(os.path.join(m.model_ws, m.hfb6.file_name[0]), os.path.join(m.model_ws,'hfb6_org', m.hfb6.file_name[0]))

    if not os.path.exists(os.path.join(m.model_ws, 'hfb6_mlt')):
        os.mkdir(os.path.join(m.model_ws, 'hfb6_mlt'))

    # read in the model file
    hfb_file_contents = open(hfb_file, 'r').readlines()

    # navigate the header
    skiprows = sum([1 if i.strip().startswith('#') else 0 for i in hfb_file_contents]) + 1
    header = hfb_file_contents[:skiprows]

    # read in the data
    names = ['lay', 'irow1','icol1','irow2','icol2', 'hydchr']
    hfb_in = pd.read_csv(hfb_file, skiprows=skiprows, delim_whitespace=True, names=names).dropna()
    for cn in names[:-1]:
        hfb_in[cn] = hfb_in[cn].astype(np.int)

    # set up a multiplier for each unique conductivity value
    unique_cond = hfb_in.hydchr.unique()
    hfb_mults = dict(zip(unique_cond, ['hbz_{0:04d}'.format(i) for i in range(len(unique_cond))]))
    # set up the TPL line for each parameter and assign
    hfb_in['tpl'] = 'blank'
    for cn, cg in hfb_in.groupby('hydchr'):
        hfb_in.loc[hfb_in.hydchr == cn, 'tpl'] = '~{0:^10s}~'.format(hfb_mults[cn])

    assert 'blank' not in hfb_in.tpl

    # write out the TPL file
    tpl_file = os.path.join(m.model_ws, "hfb6.mlt.tpl")
    with open(tpl_file, 'w') as ofp:
        ofp.write('ptf ~\n')
        [ofp.write('{0}\n'.format(line.strip())) for line in header]
        ofp.flush()
        hfb_in[['lay', 'irow1','icol1','irow2','icol2', 'tpl']].to_csv(ofp, sep=' ', quotechar=' ',
                header=None, index=None, mode='a')

    # make a lookup for lining up the necessary files to perform multiplication with the
    # helpers.apply_hfb_pars() function which must be added to the forward run script
    with open(os.path.join(m.model_ws, 'hfb6_pars.csv'), 'w') as ofp:
        ofp.write('org_file,mlt_file,model_file\n')
        ofp.write('{0},{1},{2}\n'.format(os.path.join(m.model_ws, 'hfb6_org', m.hfb6.file_name[0]),
                                         os.path.join(m.model_ws, 'hfb6_mlt', os.path.basename(tpl_file).replace('.tpl','')),
                                         hfb_file))

    return hfb_mults, tpl_file


def write_hfb_template(m):
    """write a template file for an hfb (yuck!)

    Parameters
    ----------
        m : flopy.modflow.Modflow instance with an HFB file

    Returns
    -------
        (tpl_filename, df) : (str, pandas.DataFrame)
            the name of the template file and a dataframe with useful info.

    """

    assert m.hfb6 is not None
    hfb_file = os.path.join(m.model_ws,m.hfb6.file_name[0])
    assert os.path.exists(hfb_file),"couldn't find hfb_file {0}".format(hfb_file)
    f_in = open(hfb_file,'r')
    tpl_file = hfb_file+".tpl"
    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    parnme,parval1,xs,ys = [],[],[],[]
    iis,jjs,kks = [],[],[]
    xc = m.sr.xcentergrid
    yc = m.sr.ycentergrid

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
                pn = "hb{0:02}{1:04d}{2:04}".format(k,i,j)
                pv = float(raw[5])
                raw[5] = "~ {0}  ~".format(pn)
                line = ' '.join(raw)+'\n'
                f_tpl.write(line)
                parnme.append(pn)
                parval1.append(pv)
                xs.append(xc[i,j])
                ys.append(yc[i,j])
                iis.append(i)
                jjs.append(j)
                kks.append(k)

            break

    f_tpl.close()
    f_in.close()
    df = pd.DataFrame({"parnme":parnme,"parval1":parval1,"x":xs,"y":ys,
                       "i":iis,"j":jjs,"k":kks},index=parnme)
    df.loc[:,"pargp"] = "hfb_hydfac"
    df.loc[:,"parubnd"] = df.parval1.max() * 10.0
    df.loc[:,"parlbnd"] = df.parval1.min() * 0.1
    return tpl_file,df


