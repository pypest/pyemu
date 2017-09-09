import os
import copy
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT,pst_config
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
        tpl_file : str (optional)
            template file to write.  If None, use <pval_file>.tpl
    Returns
    -------
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
        pandas DataFrame with control file observation information
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
        hydmod_outfile :str (optional)
            output file to write.  If None, use <hydmod_file>.dat
    Returns
    -------
        pandas DataFrame with control file observation information
    """
    try:
        import flopy.utils as fu
    except Exception as e:
        print('flopy is not installed - cannot read {0}\n{1}'.format(hydmod_file, e))
        return
    print('Starting to read HYDMOD data from {0}'.format(hydmod_file))
    obs = fu.HydmodObs(hydmod_file)
    hyd_df = obs.get_dataframe()

    hyd_df.columns = [i[6:] if i.lower() != 'totim' else i for i in hyd_df.columns]
    #hyd_df.loc[:,"datetime"] = hyd_df.index
    hyd_df['totim'] = hyd_df.index.map(lambda x: x.strftime("%Y%m%d"))

    hyd_df.rename(columns={'totim': 'datestamp'}, inplace=True)


    # reshape into a single column
    hyd_df = pd.melt(hyd_df, id_vars='datestamp')

    hyd_df.rename(columns={'value': 'obsval'}, inplace=True)

    hyd_df['obsnme'] = [i + '_' + j for i, j in zip(hyd_df.variable, hyd_df.datestamp)]

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
    from . import pp_utils
    return pp_utils.setup_pilotpoints_grid(ml=ml,sr=sr,ibound=ibound,
                                           prefix_dict=prefix_dict,
                                           every_n_cell=every_n_cell,
                                           use_ibound_zones=use_ibound_zones,
                                           pp_dir=pp_dir,tpl_dir=tpl_dir,
                                           shapename=shapename)


def pp_file_to_dataframe(pp_filename):
    from . import pp_utils
    return pp_utils.pp_file_to_dataframe(pp_filename)

def pp_tpl_to_dataframe(tpl_filename):
    from . import pp_utils
    return pp_utils.pp_tpl_to_dataframe(tpl_filename)

def write_pp_shapfile(pp_df,shapename=None):
    from . import pp_utils
    pp_utils.write_pp_shapfile(pp_df,shapename=shapename)


def write_pp_file(filename,pp_df):
    from . import pp_utils
    return pp_utils.write_pp_file(filename,pp_df)

def pilot_points_to_tpl(pp_file,tpl_file=None,name_prefix=None):
    from . import pp_utils
    return pp_utils.pilot_points_to_tpl(pp_file,tpl_file=tpl_file,
                                        name_prefix=name_prefix)

def fac2real(pp_file=None,factors_file="factors.dat",out_file="test.ref",
             upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30):
    from . import geostats as gs
    return gs.fac2real(pp_file=pp_file,factors_file=factors_file,
                       out_file=out_file,upper_lim=upper_lim,
                       lower_lim=lower_lim,fill_value=fill_value)


def setup_mflist_budget_obs(list_filename,flx_filename="flux.dat",
                            vol_filename="vol.dat",start_datetime="1-1'1970",prefix='',
                            save_setup_file=False):
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
    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    mlf = flopy.utils.MfListBudget(list_filename)
    flx,vol = mlf.get_dataframes(start_datetime=start_datetime,diff=True)
    flx.to_csv(flx_filename,sep=' ',index_label="datetime")
    vol.to_csv(vol_filename,sep=' ',index_label="datetime")
    return flx,vol


def _write_mflist_ins(ins_filename,df,prefix):
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

