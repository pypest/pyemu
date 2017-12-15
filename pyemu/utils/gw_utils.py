""" module of utilities for groundwater modeling
"""

import os
import copy
from datetime import datetime
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT,pst_config,_try_run_inschek
from pyemu.utils.helpers import run
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

    hob_df.loc[:,"ins_line"] = hob_df.obsnme.apply(lambda x:"l1 w w !{0:s}!".format(x))
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
    print('Starting to read HYDMOD data from {0}'.format(hydmod_file))
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
    import warnings
    warnings.warn("setup_pilotpoint_grid has moved to pp_utils...")
    return pp_utils.setup_pilotpoints_grid(ml=ml,sr=sr,ibound=ibound,
                                           prefix_dict=prefix_dict,
                                           every_n_cell=every_n_cell,
                                           use_ibound_zones=use_ibound_zones,
                                           pp_dir=pp_dir,tpl_dir=tpl_dir,
                                           shapename=shapename)


def pp_file_to_dataframe(pp_filename):

    import warnings
    from . import pp_utils
    warnings.warn("pp_file_to_dataframe has moved to pp_utils")
    return pp_utils.pp_file_to_dataframe(pp_filename)

def pp_tpl_to_dataframe(tpl_filename):
    import warnings
    from . import pp_utils
    warnings.warn("pp_tpl_to_dataframe has moved to pp_utils")
    return pp_utils.pp_tpl_to_dataframe(tpl_filename)

def write_pp_shapfile(pp_df,shapename=None):
    from . import pp_utils
    import warnings
    warnings.warn("write_pp_shapefile has moved to pp_utils")
    pp_utils.write_pp_shapfile(pp_df,shapename=shapename)


def write_pp_file(filename,pp_df):
    from . import pp_utils
    import warnings
    warnings.warn("write_pp_file has moved to pp_utils")
    return pp_utils.write_pp_file(filename,pp_df)

def pilot_points_to_tpl(pp_file,tpl_file=None,name_prefix=None):
    from . import pp_utils
    import warnings
    warnings.warn("pilot_points_to_tpl has moved to pp_utils")
    return pp_utils.pilot_points_to_tpl(pp_file,tpl_file=tpl_file,
                                        name_prefix=name_prefix)

def fac2real(pp_file=None,factors_file="factors.dat",out_file="test.ref",
             upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30):
    from . import geostats as gs
    import warnings
    warnings.warn("fac2real has moved to geostats")
    return gs.fac2real(pp_file=pp_file,factors_file=factors_file,
                       out_file=out_file,upper_lim=upper_lim,
                       lower_lim=lower_lim,fill_value=fill_value)


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

    run("inschek {0}.ins {0}".format(flx_filename))
    run("inschek {0}.ins {0}".format(vol_filename))

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
    flx.to_csv(flx_filename,sep=' ',index_label="datetime",date_format="%Y%M%d")
    vol.to_csv(vol_filename,sep=' ',index_label="datetime",date_format="%Y%M%d")
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
        is True, then values equal to skip will not be used.  If not
        np.scalar(skip), then skip will be treated as a lambda function that
        returns np.NaN if the value should be skipped.
    prefix : str
        the prefix to use for the observation names. default is "hds".

    Returns
    -------
    forward_run_line : str
        a python code str to add to the forward run script

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
    return fwd_run_line

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
        df_kper = df.loc[df.kper==kper,:]
        df.loc[df_kper.index,"obsval"] = data[df_kper.k,df_kper.i,df_kper.j]
    assert df.dropna().shape[0] == df.shape[0]
    df.loc[:,["obsnme","obsval"]].to_csv(out_file,index=False,sep=" ")


def setup_sft_obs(sft_file,ins_file=None,start_datetime=None,times=None):
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
    idx = df.time.apply(lambda x: x in times)
    if start_datetime is not None:
        start_datetime = pd.to_datetime(start_datetime)
        df.loc[:,"time_str"] = pd.to_timedelta(df.time,unit='d') + start_datetime
        df.loc[:,"time_str"] = df.time_str.apply(lambda x: datetime.strftime(x,"%d%m%Y"))
    else:
        df.loc[:,"time_str"] = df.time.apply(lambda x: "{0:08.2f}".format(x))
    df.loc[:,"ins_str"] = "l1\n"
    df.loc[idx,"ins_str"] = df.apply(lambda x: "l1 w w w !sfrconc{0}_{1}! !flowgw{0}_{1}! !gwconc{0}_{1}!\n".\
                                     format(x.sfr_node,x.time_str),axis=1)
    df.index = np.arange(df.shape[0])
    if ins_file is None:
        ins_file = sft_file+".ins"

    with open(ins_file,'w') as f:
        f.write("pif ~\nl2\n")
        [f.write(i) for i in df.ins_str]
    df = _try_run_inschek(ins_file,sft_file)
    if df is not None:
        return df
    else:
        return None



# def setup_ssm_parameters(mt):
#     """Set up ssm file multiplier parameters for the point sources and sinks
#
#     Parameters
#     ----------
#         ssm_file : str
#             the ssm file to parameterize
#
#     """
#     try:
#         import flopy
#     except Exception as e:
#         raise Exception("error importing flopy: {0}".format(str(e)))
#
#     # first load the stress period list data in the ssm file
#
#     ssm = flopy.mt3d.Mt3dSsm.load(ssm_file,
