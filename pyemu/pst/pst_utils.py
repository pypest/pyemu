"""This module contains helpers and default values that support
the pyemu.Pst object.
"""
from __future__ import print_function, division
import os, sys
import stat
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100

import pyemu
from ..pyemu_warnings import PyemuWarning
#formatters
#SFMT = lambda x: "{0:>20s}".format(str(x.decode()))
def SFMT(item):
    try:
        s = "{0:<20s} ".format(item.decode())
    except:
        s = "{0:<20s} ".format(str(item))
    return s

SFMT_LONG = lambda x: "{0:<50s} ".format(str(x))
IFMT = lambda x: "{0:<10d} ".format(int(x))
FFMT = lambda x: "{0:<20.10E} ".format(float(x))

def str_con(item):
    if len(item) == 0:
        return np.NaN
    return item.lower().strip()

pst_config = {}

# parameter stuff
pst_config["tied_dtype"] = np.dtype([("parnme", "U20"), ("partied","U20")])
pst_config["tied_fieldnames"] = ["parnme","partied"]
pst_config["tied_format"] = {"parnme":SFMT,"partied":SFMT}
pst_config["tied_converters"] = {"parnme":str_con,"partied":str_con}
pst_config["tied_defaults"] = {"parnme":"dum","partied":"dum"}

pst_config["par_dtype"] = np.dtype([("parnme", "U20"), ("partrans","U20"),
                                   ("parchglim","U20"),("parval1", np.float64),
                                   ("parlbnd",np.float64),("parubnd",np.float64),
                                   ("pargp","U20"),("scale", np.float64),
                                   ("offset", np.float64),("dercom",np.int)])
pst_config["par_fieldnames"] = "PARNME PARTRANS PARCHGLIM PARVAL1 PARLBND PARUBND " +\
                              "PARGP SCALE OFFSET DERCOM"
pst_config["par_fieldnames"] = pst_config["par_fieldnames"].lower().strip().split()
pst_config["par_format"] = {"parnme": SFMT, "partrans": SFMT,
                           "parchglim": SFMT, "parval1": FFMT,
                           "parlbnd": FFMT, "parubnd": FFMT,
                           "pargp": SFMT, "scale": FFMT,
                           "offset": FFMT, "dercom": IFMT}
pst_config["par_converters"] = {"parnme": str_con, "pargp": str_con,
                                "parval1":np.float,"parubnd":np.float,
                                "parlbnd":np.float,"scale":np.float,
                                "offset":np.float}
pst_config["par_defaults"] = {"parnme":"dum","partrans":"log","parchglim":"factor",
                             "parval1":1.0,"parlbnd":1.1e-10,"parubnd":1.1e+10,
                             "pargp":"pargp","scale":1.0,"offset":0.0,"dercom":1}


# parameter group stuff
pst_config["pargp_dtype"] = np.dtype([("pargpnme", "U20"), ("inctyp","U20"),
                                   ("derinc", np.float64),
                                   ("derinclb",np.float64),("forcen","U20"),
                                   ("derincmul",np.float64),("dermthd", "U20"),
                                   ("splitthresh", np.float64),("splitreldiff",np.float64),
                                      ("splitaction","U20")])
pst_config["pargp_fieldnames"] = "PARGPNME INCTYP DERINC DERINCLB FORCEN DERINCMUL " +\
                        "DERMTHD SPLITTHRESH SPLITRELDIFF SPLITACTION"
pst_config["pargp_fieldnames"] = pst_config["pargp_fieldnames"].lower().strip().split()

pst_config["pargp_format"] = {"pargpnme":SFMT,"inctyp":SFMT,"derinc":FFMT,"forcen":SFMT,
                      "derincmul":FFMT,"dermthd":SFMT,"splitthresh":FFMT,
                      "splitreldiff":FFMT,"splitaction":SFMT}

pst_config["pargp_converters"] = {"pargpnme":str_con,"inctyp":str_con,
                         "dermethd":str_con,"derinc":np.float,"derinclb":np.float,
                         "splitaction":str_con,"forcen":str_con,"derincmul":np.float}
pst_config["pargp_defaults"] = {"pargpnme":"pargp","inctyp":"relative","derinc":0.01,
                       "derinclb":0.0,"forcen":"switch","derincmul":2.0,
                     "dermthd":"parabolic","splitthresh":1.0e-5,
                       "splitreldiff":0.5,"splitaction":"smaller"}


# observation stuff
pst_config["obs_fieldnames"] = "OBSNME OBSVAL WEIGHT OBGNME".lower().split()
pst_config["obs_dtype"] = np.dtype([("obsnme","U20"),("obsval",np.float64),
                           ("weight",np.float64),("obgnme","U20")])
pst_config["obs_format"] = {"obsnme": SFMT, "obsval": FFMT,
                   "weight": FFMT, "obgnme": SFMT}
pst_config["obs_converters"] = {"obsnme": str_con, "obgnme": str_con,
                                "weight":np.float,"obsval":np.float}
pst_config["obs_defaults"] = {"obsnme":"dum","obsval":1.0e+10,
                     "weight":1.0,"obgnme":"obgnme"}


# prior info stuff
pst_config["null_prior"] = pd.DataFrame({"pilbl": None,
                                    "obgnme": None}, index=[])
pst_config["prior_format"] = {"pilbl": SFMT, "equation": SFMT_LONG,
                     "weight": FFMT, "obgnme": SFMT}
pst_config["prior_fieldnames"] = ["pilbl","equation", "weight", "obgnme"]


# other containers
pst_config["model_command"] = []
pst_config["template_files"] = []
pst_config["input_files"] = []
pst_config["instruction_files"] = []
pst_config["output_files"] = []
pst_config["other_lines"] = []
pst_config["tied_lines"] = []
pst_config["regul_lines"] = []
pst_config["pestpp_options"] = {}


def read_resfile(resfile):
        """load a residual file into a pandas.DataFrame

        Parameters
        ----------
        resfile : str
            residual file name

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame

        """
        assert os.path.exists(resfile),"read_resfile() error: resfile " +\
                                       "{0} not found".format(resfile)
        converters = {"name": str_con, "group": str_con}
        f = open(resfile, 'r')
        while True:
            line = f.readline()
            if line == '':
                raise Exception("Pst.get_residuals: EOF before finding "+
                                "header in resfile: " + resfile)
            if "name" in line.lower():
                header = line.lower().strip().split()
                break
        res_df = pd.read_csv(f, header=None, names=header, sep="\s+",
                                 converters=converters)
        res_df.index = res_df.name
        f.close()
        return res_df

def res_from_en(pst,enfile):
    """load ensemble file for residual into a pandas.DataFrame

        Parameters
        ----------
        enfile : str
            ensemble file name

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame

        """
    converters = {"name": str_con, "group": str_con}
    try: #substitute ensemble for res, 'base' if there, otherwise mean
        obs=pst.observation_data
        if isinstance(enfile,str):
            df=pd.read_csv(enfile,converters=converters)
            df.columns=df.columns.str.lower()
            df = df.set_index('real_name').T.rename_axis('name').rename_axis(None, 1)
        else:
            df = enfile.T
        if 'base' in df.columns:
            df['modelled']=df['base']
            df['std']=df.std(axis=1)
        else:
            df['modelled']=df.mean(axis=1)
            df['std']=df.std(axis=1)
        #probably a more pandastic way to do this
        res_df=df[['modelled','std']].copy()
        res_df['group']=obs.loc[:,'obgnme'].copy()
        res_df['measured']=obs['obsval'].copy()
        res_df['weight']=obs['weight'].copy()
        res_df['residual']=res_df['measured']-res_df['modelled']
    except Exception as e:
        raise Exception("Pst.res_from_en:{0}".format(str(e)))
    return res_df

def read_parfile(parfile):
    """load a pest-compatible .par file into a pandas.DataFrame

    Parameters
    ----------
    parfile : str
        pest parameter file name

    Returns
    -------
    pandas.DataFrame : pandas.DataFrame

    """
    assert os.path.exists(parfile), "Pst.parrep(): parfile not found: " +\
                                    str(parfile)
    f = open(parfile, 'r')
    header = f.readline()
    par_df = pd.read_csv(f, header=None,
                             names=["parnme", "parval1", "scale", "offset"],
                             sep="\s+")
    par_df.index = par_df.parnme
    return par_df

def write_parfile(df,parfile):
    """ write a pest parameter file from a dataframe

    Parameters
    ----------
    df : (pandas.DataFrame)
        dataframe with column names that correspond to the entries
        in the parameter data section of a pest control file
    parfile : str
        name of the parameter file to write

    """
    columns = ["parnme","parval1","scale","offset"]
    formatters = {"parnme":lambda x:"{0:20s}".format(x),
                  "parval1":lambda x:"{0:20.7E}".format(x),
                  "scale":lambda x:"{0:20.7E}".format(x),
                  "offset":lambda x:"{0:20.7E}".format(x)}

    for col in columns:
        assert col in df.columns,"write_parfile() error: " +\
                                 "{0} not found in df".format(col)
    with open(parfile,'w') as f:
        f.write("single point\n")
        f.write(df.to_string(col_space=0,
                      columns=columns,
                      formatters=formatters,
                      justify="right",
                      header=False,
                      index=False,
                      index_names=False) + '\n')

def parse_tpl_file(tpl_file):
    """ parse a pest template file to get the parameter names

    Parameters
    ----------
    tpl_file : str
        template file name

    Returns
    -------
    par_names : list
        list of parameter names

    """
    par_names = set()
    with open(tpl_file,'r') as f:
        try:
            header = f.readline().strip().split()
            assert header[0].lower() in ["ptf","jtf"],\
                "template file error: must start with [ptf,jtf], not:" +\
                str(header[0])
            assert len(header) == 2,\
                "template file error: header line must have two entries: " +\
                str(header)

            marker = header[1]
            assert len(marker) == 1,\
                "template file error: marker must be a single character, not:" +\
                str(marker)
            for line in f:
                par_line = set(line.lower().strip().split(marker)[1::2])
                par_names.update(par_line)
                #par_names.extend(par_line)
                #for p in par_line:
                #    if p not in par_names:
                #        par_names.append(p)
        except Exception as e:
            raise Exception("error processing template file " +\
                            tpl_file+" :\n" + str(e))
    #par_names = [pn.strip().lower() for pn in par_names]
    #seen = set()
    #seen_add = seen.add
    #return [x for x in par_names if not (x in seen or seen_add(x))]
    return [p.strip() for p in list(par_names)]


def write_input_files(pst):
    """write parameter values to a model input files using a template files with
    current parameter values (stored in Pst.parameter_data.parval1).
    This is a simple implementation of what PEST does.  It does not
    handle all the special cases, just a basic function...user beware

    Parameters
    ----------
    pst : (pyemu.Pst)
        a Pst instance

    """
    par = pst.parameter_data
    par.loc[:,"parval1_trans"] = (par.parval1 * par.scale) + par.offset
    for tpl_file,in_file in zip(pst.template_files,pst.input_files):
        write_to_template(pst.parameter_data.parval1_trans,tpl_file,in_file)

def write_to_template(parvals,tpl_file,in_file):
    """ write parameter values to model input files using template files

    Parameters
    ----------
    parvals : dict or pandas.Series
        a way to look up parameter values using parameter names
    tpl_file : str
        template file
    in_file : str
        input file

    """
    f_in = open(in_file,'w')
    f_tpl = open(tpl_file,'r')
    header = f_tpl.readline().strip().split()
    assert header[0].lower() in ["ptf", "jtf"], \
        "template file error: must start with [ptf,jtf], not:" + \
        str(header[0])
    assert len(header) == 2, \
        "template file error: header line must have two entries: " + \
        str(header)

    marker = header[1]
    assert len(marker) == 1, \
        "template file error: marker must be a single character, not:" + \
        str(marker)
    for line in f_tpl:
        if marker not in line:
            f_in.write(line)
        else:
            line = line.rstrip()
            par_names = line.lower().split(marker)[1::2]
            par_names = [name.strip() for name in par_names]
            start,end = get_marker_indices(marker,line)
            assert len(par_names) == len(start)
            new_line = line[:start[0]]
            between = [line[e:s] for s,e in zip(start[1:],end[:-1])]
            for i,name in enumerate(par_names):
                s,e = start[i],end[i]
                w = e - s
                if w > 15:
                    d = 6
                else:
                    d = 3
                fmt = "{0:" + str(w)+"."+str(d)+"E}"
                val_str = fmt.format(parvals[name])
                new_line += val_str
                if i != len(par_names) - 1:
                    new_line += between[i]
            new_line += line[end[-1]:]
            f_in.write(new_line+'\n')
    f_tpl.close()
    f_in.close()


def get_marker_indices(marker,line):
    """ method to find the start and end parameter markers
    on a template file line.  Used by write_to_template()

    Parameters
    ----------
    marker : str
        template file marker char
    line : str
        template file line

    Returns
    -------
    indices : list
        list of start and end indices (zero based)

    """
    indices = [i for i, ltr in enumerate(line) if ltr == marker]
    start = indices[0:-1:2]
    end = [i+1 for i in indices[1::2]]
    assert len(start) == len(end)
    return start,end


def parse_ins_file(ins_file):
    """parse a pest instruction file to get observation names

    Parameters
    ----------
    ins_file : str
        instruction file name

    Returns
    -------
    list of observation names

    """

    obs_names = []
    with open(ins_file,'r') as f:
        header = f.readline().strip().split()
        assert header[0].lower() in ["pif","jif"],\
            "instruction file error: must start with [pif,jif], not:" +\
            str(header[0])
        marker = header[1]
        assert len(marker) == 1,\
            "instruction file error: marker must be a single character, not:" +\
            str(marker)
        for line in f:
            line = line.lower()
            if marker in line:
                raw = line.lower().strip().split(marker)
                for item in raw[::2]:
                    obs_names.extend(parse_ins_string(item))
            else:
                obs_names.extend(parse_ins_string(line.strip()))
    #obs_names = [on.strip().lower() for on in obs_names]
    return obs_names


def parse_ins_string(string):
    """ split up an instruction file line to get the observation names

    Parameters
    ----------
    string : str
        instruction file line

    Returns
    -------
    obs_names : list
        list of observation names

    """
    istart_markers = ["[","(","!"]
    iend_markers = ["]",")","!"]

    obs_names = []

    idx = 0
    while True:
        if idx >= len(string) - 1:
            break
        char = string[idx]
        if char in istart_markers:
            em = iend_markers[istart_markers.index(char)]
            # print("\n",idx)
            # print(string)
            # print(string[idx+1:])
            # print(string[idx+1:].index(em))
            # print(string[idx+1:].index(em)+idx+1)
            eidx = min(len(string),string[idx+1:].index(em)+idx+1)
            obs_name = string[idx+1:eidx]
            if obs_name.lower() != "dum":
                obs_names.append(obs_name)
            idx = eidx + 1
        else:
            idx += 1
    return obs_names


def populate_dataframe(index,columns, default_dict, dtype):
    """ helper function to populate a generic Pst dataframe attribute.  This
    function is called as part of constructing a generic Pst instance

    Parameters
    ----------
    index : (varies)
        something to use as the dataframe index
    columns: (varies)
        something to use as the dataframe columns
    default_dict : (dict)
        dictionary of default values for columns
    dtype : numpy.dtype
        dtype used to cast dataframe columns

    Returns
    -------
    new_df : pandas.DataFrame

    """
    new_df = pd.DataFrame(index=index,columns=columns)
    for fieldname,dt in zip(columns,dtype.descr):
        default = default_dict[fieldname]
        new_df.loc[:,fieldname] = default
        new_df.loc[:,fieldname] = new_df.loc[:,fieldname].astype(dt[1])
    return new_df


def generic_pst(par_names=["par1"],obs_names=["obs1"],addreg=False):
    """generate a generic pst instance.  This can used to later fill in
    the Pst parts programatically.

    Parameters
    ----------
    par_names : (list)
        parameter names to setup
    obs_names : (list)
        observation names to setup

    Returns
    -------
    new_pst : pyemu.Pst

    """
    if not isinstance(par_names,list):
        par_names = list(par_names)
    if not isinstance(obs_names,list):
        obs_names = list(obs_names)
    new_pst = pyemu.Pst("pest.pst",load=False)
    pargp_data = populate_dataframe(["pargp"], new_pst.pargp_fieldnames,
                                    new_pst.pargp_defaults, new_pst.pargp_dtype)
    new_pst.parameter_groups = pargp_data

    par_data = populate_dataframe(par_names,new_pst.par_fieldnames,
                                  new_pst.par_defaults,new_pst.par_dtype)
    par_data.loc[:,"parnme"] = par_names
    par_data.index = par_names
    par_data.sort_index(inplace=True)
    new_pst.parameter_data = par_data
    obs_data = populate_dataframe(obs_names,new_pst.obs_fieldnames,
                                  new_pst.obs_defaults,new_pst.obs_dtype)
    obs_data.loc[:,"obsnme"] = obs_names
    obs_data.index = obs_names
    obs_data.sort_index(inplace=True)
    new_pst.observation_data = obs_data

    new_pst.template_files = ["file.tpl"]
    new_pst.input_files = ["file.in"]
    new_pst.instruction_files = ["file.ins"]
    new_pst.output_files = ["file.out"]
    new_pst.model_command = ["model.bat"]

    new_pst.prior_information = new_pst.null_prior

    #new_pst.other_lines = ["* singular value decomposition\n","1\n",
    #                       "{0:d} {1:15.6E}\n".format(new_pst.npar_adj,1.0E-6),
    #                       "1 1 1\n"]
    if addreg:
        new_pst.zero_order_tikhonov()

    return new_pst


def pst_from_io_files(tpl_files,in_files,ins_files,out_files,pst_filename=None):
    """ generate a new pyemu.Pst instance from model interface files.  This
    function is emulated in the Pst.from_io_files() class method.

    Parameters
    ----------
    tpl_files : (list)
        template file names
    in_files : (list)
        model input file names
    ins_files : (list)
        instruction file names
    out_files : (list)
        model output file names
    pst_filename : str
        filename to save new pyemu.Pst.  If None, Pst is not written.
        default is None

    Returns
    -------
    new_pst : pyemu.Pst

    """

    warnings.warn("pst_from_io_files has moved to pyemu.helpers and is also "+\
                  "now avaiable as a Pst class method (Pst.from_io_files())",PyemuWarning)
    from pyemu import helpers
    return helpers.pst_from_io_files(tpl_files=tpl_files,in_files=in_files,
                              ins_files=ins_files,out_files=out_files,
                              pst_filename=pst_filename)


def try_run_inschek(pst):
    """ attempt to run INSCHEK for each instruction file, model output
    file pair in a pyemu.Pst.  If the run is successful, the INSCHEK written
    .obf file is used to populate the pst.observation_data.obsval attribute

    Parameters
    ----------
    pst : (pyemu.Pst)

    """
    for ins_file,out_file in zip(pst.instruction_files,pst.output_files):
        df = _try_run_inschek(ins_file,out_file)
        if df is not None:
            pst.observation_data.loc[df.index, "obsval"] = df.obsval


def try_process_ins_file(ins_file,out_file=None):
    assert os.path.exists(ins_file),"instruction file {0} not found".format(ins_file)
    try:
        obs_names = parse_ins_file(ins_file)
    except Exception as e:
        raise Exception("error parsing ins file {0} for obs names: {1}".format(ins_file,str(e)))

    if out_file is None:
        out_file = ins_file.replace(".ins","")
    if not os.path.exists(out_file):
        print("out file {0} not found".format(out_file))
        return pd.DataFrame({"obsnme":obs_names},index="obsnme")
    try:
        f_ins = open(ins_file,'r')
        f_out = open(out_file,'r')

        header = f_ins.readline().lower().strip().split()
        assert header[0] == "pif"
        marker = header[1]
        icount,ocount = 1,0
        obsnme,obsval = [],[]
        for iline in f_ins:
            iraw = iline.lower().strip().split()
            if iraw[0][0] != 'l':
                raise Exception("not support instruction:{0}".format(iraw[0]))
            num_lines = int(iraw[0][1:])
            for i in range(num_lines):
                oline = f_out.readline().lower().strip()
                ocount += 1
            if '(' in oline:
                raise Exception("semi-fixed obs index instructions not supported")
            elif '[' in oline:
                raise Exception("fixed obs index instructions not supported")
            elif marker in oline:
                raise Exception("marker-based file seeking not supported")
            oraw = oline.strip().split()
            if oline[0] in [' ','   ']:
                oraw.insert(0,'')
            oc = 0
            for ir in iraw[1:]:
                if ir == 'w':
                    oc += 1
                    if oc > len(oraw):
                        raise Exception("out file line {0} too short".format(ocount))
                elif '!' in ir:
                    n = ir.replace('!','')
                    try:
                        v = float(oraw[oc])
                    except Exception as e:
                        raise Exception("error processing ins {0} for obs {1}, string: {2} on line {3} (ins line {4}):{5}".
                                        format(ir,n,oline,ocount,icount,str(e)))
                    obsnme.append(n)
                    obsval.append(v)
                    oc += 1

        f_ins.close()
        f_out.close()
        df = pd.DataFrame({"obsnme":obsnme,"obsval":obsval},index=obsnme)

        return df
    except Exception as e:
        print("error processing ins file {0}: {1}".format(ins_file,str(e)))
        return pd.DataFrame({"obsnme":obs_names},index=obs_names)



def _try_run_inschek(ins_file,out_file,cwd='.'):

    try:
        pyemu.os_utils.run("inschek {0} {1}".format(ins_file, out_file),cwd=cwd)
        obf_file = os.path.join(cwd,ins_file.replace(".ins", ".obf"))
        df = pd.read_csv(obf_file, delim_whitespace=True,
                         skiprows=0, index_col=0, names=["obsval"])
        df.index = df.index.map(str.lower)
        return df
    except Exception as e:
        print("error using inschek for instruction file {0}:{1}".
              format(ins_file, str(e)))
        print("observations in this instruction file will have" +
              "generic values.")
        return None


def get_phi_comps_from_recfile(recfile):
    """read the phi components from a record file by iteration

    Parameters
    ----------
    recfile : str
        pest record file name

    Returns
    -------
    iters : dict
        nested dictionary of iteration number, {group,contribution}

    """
    iiter = 1
    iters = {}
    f = open(recfile,'r')
    while True:
        line = f.readline()
        if line == '':
            break
        if "starting phi for this iteration" in line.lower() or \
            "final phi" in line.lower():
            contributions = {}
            while True:
                line = f.readline()
                if line == '':
                    break
                if "contribution to phi" not in line.lower():
                    iters[iiter] = contributions
                    iiter += 1
                    break
                raw = line.strip().split()
                val = float(raw[-1])
                group = raw[-3].lower().replace('\"', '')
                contributions[group] = val
    return iters


def del_rw(action, name, exc):
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)

def start_slaves(slave_dir,exe_rel_path,pst_rel_path,num_slaves=None,slave_root="..",
                 port=4004,rel_path=None):


    warnings.warn("deprecation warning:start_slaves() has moved to the utils.helpers module",PyemuWarning)
    from pyemu.utils import start_slaves
    start_slaves(slave_dir,exe_rel_path,pst_rel_path,num_slaves=num_slaves,slave_root=slave_root,
                 port=port,rel_path=rel_path)

def res_from_obseravtion_data(observation_data):
    """create a generic residual dataframe filled with np.NaN for
    missing information

    Parameters
    ----------
    observation_data : pandas.DataFrame
        pyemu.Pst.observation_data

    Returns
    -------
    res_df : pandas.DataFrame

    """
    res_df = observation_data.copy()
    res_df.loc[:, "name"] = res_df.pop("obsnme")
    res_df.loc[:, "measured"] = res_df.pop("obsval")
    res_df.loc[:, "group"] = res_df.pop("obgnme")
    res_df.loc[:, "modelled"] = np.NaN
    res_df.loc[:, "residual"] = np.NaN
    return res_df

def clean_missing_exponent(pst_filename,clean_filename="clean.pst"):
    """fixes the issue where some terrible fortran program may have
    written a floating point format without the 'e' - like 1.0-3, really?!

    Parameters
    ----------
    pst_filename : str
        the pest control file
    clean_filename : str
        the new pest control file to write. Default is "clean.pst"

    Returns
    -------
    None


    """
    lines = []
    with open(pst_filename,'r') as f:
        for line in f:
            line = line.lower().strip()
            if '+' in line:
                raw = line.split('+')
                for i,r in enumerate(raw[:-1]):
                    if r[-1] != 'e':
                        r = r + 'e'
                    raw[i] = r
                lines.append('+'.join(raw))
            else:
                lines.append(line)
    with open(clean_filename,'w') as f:
        for line in lines:
            f.write(line+'\n')

def csv_to_ins_file(csv_filename,ins_filename=None,only_cols=None,only_rows=None,
                    marker='~',includes_header=True,includes_index=True,prefix=''):

    # process the csv_filename in case it is a dataframe
    if isinstance(csv_filename,str):
        df = pd.read_csv(csv_filename,index_col=0)
        df.columns = df.columns.map(str.lower)
        df.index = df.index.map(lambda x: str(x).lower())
    else:
        df = csv_filename

    # process only_cols
    if only_cols is None:
        only_cols = set(df.columns.map(str.lower))
    else:
        if isinstance(only_cols,str): # incase it is a single name
            only_cols = [only_cols]
        only_cols = set(only_cols)

    if only_rows is None:
        only_rows = set(df.index.map(str.lower))
    else:
        if isinstance(only_rows,str): # incase it is a single name
            only_rows = [only_rows]
        only_rows = set(only_rows)

    # process the row labels, handling duplicates
    rlabels = []
    row_visit = {}
    only_rlabels = []
    for rname in df.index:
        rname = str(rname).strip().lower()

        if rname in row_visit:
            rsuffix = str(int(row_visit[rname] + 1))
            row_visit[rname] += 1
        else:
            row_visit[rname] = 1
            rsuffix = ''
        rlabel = rname + rsuffix
        rlabels.append(rlabel)
        if rname in only_rows:
            only_rlabels.append(rlabel)
    only_rlabels = set(only_rlabels)

    #process the col labels, handling duplicates
    clabels = []
    col_visit = {}
    only_clabels = []
    for cname in df.columns:
        cname = str(cname).strip().lower()
        if cname in col_visit:
            csuffix = str(int(col_visit[cname]+1))
            col_visit[cname] += 1
        else:
            col_visit[cname] = 1
            csuffix = ''
        clabel = cname + csuffix
        clabels.append(clabel)
        if cname in only_cols:
            only_clabels.append(clabel)

    if ins_filename is None:
        if not isinstance(csv_filename,str):
            raise Exception("ins_filename is None but csv_filename is not string")
        ins_filename = csv_filename + ".ins"
    row_visit, col_visit = {},{}
    onames = []
    ovals = []
    with open(ins_filename,'w') as f:
        f.write("pif ~\nl1\n")
        for i,rlabel in enumerate(rlabels):
            if includes_header:
                f.write("l1 ") #skip the row (index) label
            for j,clabel in enumerate(clabels):
                if rlabel in only_rlabels and clabel in only_clabels:
                    oname = prefix+rlabel+"_"+clabel
                    onames.append(oname)
                    ovals.append(df.iloc[i,j])
                else:
                    oname = "dum"
                if j == 0:
                    if includes_index:
                        f.write(" {0},{0} ".format(marker))
                else:
                    f.write(" {0},{0} ".format(marker))
                f.write(" !{0}! ".format(oname))
            f.write('\n')
    odf = pd.DataFrame({"obsnme":onames,"obsval":ovals},index=onames)
    return odf













