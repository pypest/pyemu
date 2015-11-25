from __future__ import print_function, division
import os
import multiprocessing as mp
import subprocess as sp
import socket
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100

import pyemu

#formatters
#SFMT = lambda x: "{0:>20s}".format(str(x.decode()))
def SFMT(item):
    try:
        s = "{0:>20s}".format(item.decode())
    except:
        s = "{0:>20s}".format(str(item))
    return s

SFMT_LONG = lambda x: "{0:>50s}".format(str(x))
IFMT = lambda x: "{0:>10d}".format(int(x))
FFMT = lambda x: "{0:>15.6E}".format(float(x))


def str_con(item):
    if len(item) == 0:
        return np.NaN
    return item.lower()

pst_config = {}

# parameter stuff
pst_config["par_dtype"] = np.dtype([("parnme", "a20"), ("partrans","a20"),
                                   ("parchglim","a20"),("parval1", np.float64),
                                   ("parlbnd",np.float64),("parubnd",np.float64),
                                   ("pargp","a20"),("scale", np.float64),
                                   ("offset", np.float64),("dercom",np.int)])
pst_config["par_fieldnames"] = "PARNME PARTRANS PARCHGLIM PARVAL1 PARLBND PARUBND " +\
                              "PARGP SCALE OFFSET DERCOM"
pst_config["par_fieldnames"] = pst_config["par_fieldnames"].lower().strip().split()
pst_config["par_format"] = {"parnme": SFMT, "partrans": SFMT,
                           "parchglim": SFMT, "parval1": FFMT,
                           "parlbnd": FFMT, "parubnd": FFMT,
                           "pargp": SFMT, "scale": FFMT,
                           "offset": FFMT, "dercom": IFMT}
pst_config["par_converters"] = {"parnme": str_con, "pargp": str_con}
pst_config["par_defaults"] = {"parnme":"dum","partrans":"log","parchglim":"factor",
                             "parval1":1.0,"parlbnd":1.1e-10,"parubnd":1.1e+10,
                             "pargp":"pargp","scale":1.0,"offset":0.0,"dercom":1}


# parameter group stuff
pst_config["pargp_dtype"] = np.dtype([("pargpnme", "a20"), ("inctyp","a20"),
                                   ("derinc", np.float64),
                                   ("derinclb",np.float64),("forcen","a20"),
                                   ("derincmul",np.float64),("dermthd", "a20"),
                                   ("splitthresh", np.float64),("splitreldiff",np.float64),
                                      ("splitaction","a20")])
pst_config["pargp_fieldnames"] = "PARGPNME INCTYP DERINC DERINCLB FORCEN DERINCMUL " +\
                        "DERMTHD SPLITTHRESH SPLITRELDIFF SPLITACTION"
pst_config["pargp_fieldnames"] = pst_config["pargp_fieldnames"].lower().strip().split()

pst_config["pargp_format"] = {"pargpnme":SFMT,"inctyp":SFMT,"derinc":FFMT,"forcen":SFMT,
                      "derincmul":FFMT,"dermthd":SFMT,"splitthresh":FFMT,
                      "splitreldiff":FFMT,"splitaction":SFMT}

pst_config["pargp_converters"] = {"pargpnme":str_con,"inctype":str_con,
                         "dermethd":str_con,
                         "splitaction":str_con}
pst_config["pargp_defaults"] = {"pargpnme":"pargp","inctyp":"relative","derinc":0.01,
                       "derinclb":0.0,"forcen":"switch","derincmul":2.0,
                     "dermthd":"parabolic","splitthresh":1.0e-5,
                       "splitreldiff":0.5,"splitaction":"smaller"}


# observation stuff
pst_config["obs_fieldnames"] = "OBSNME OBSVAL WEIGHT OBGNME".lower().split()
pst_config["obs_dtype"] = np.dtype([("obsnme","a20"),("obsval",np.float64),
                           ("weight",np.float64),("obgnme","a20")])
pst_config["obs_format"] = {"obsnme": SFMT, "obsval": FFMT,
                   "weight": FFMT, "obgnme": SFMT}
pst_config["obs_converters"] = {"obsnme": str_con, "obgnme": str_con}
pst_config["obs_defaults"] = {"obsnme":"dum","obsval":1.0e+10,
                     "weight":1.0,"obgnme":"obgnme"}


# prior info stuff
pst_config["null_prior"] = pd.DataFrame({"pilbl": None,
                                    "obgnme": None}, index=[])
pst_config["prior_format"] = {"pilbl": SFMT, "equation": SFMT_LONG,
                     "weight": FFMT, "obgnme": SFMT}
pst_config["prior_fieldnames"] = ["equation", "weight", "obgnme"]


# other containers
pst_config["model_command"] = []
pst_config["template_files"] = []
pst_config["input_files"] = []
pst_config["instruction_files"] = []
pst_config["output_files"] = []
pst_config["other_lines"] = []
pst_config["tied_lines"] = []
pst_config["regul_lines"] = []
pst_config["pestpp_lines"] = []


def read_parfile(parfile):
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
    par_names = []
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
                par_line = line.strip().split(marker)[1::2]
                for p in par_line:
                    if p not in par_names:
                        par_names.append(p)
        except Exception as e:
            raise Exception("error processing template file " +\
                            tpl_file+" :\n" + str(e))
    par_names = [pn.strip().lower() for pn in par_names]
    return par_names


def parse_ins_file(ins_file):
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
            if marker in line:
                raw = line.strip().split(marker)
                for item in raw[::2]:
                    obs_names.extend(parse_ins_string(item))
            else:
                obs_names.extend(parse_ins_string(line.strip()))
    obs_names = [on.strip().lower() for on in obs_names]
    return obs_names


def parse_ins_string(string):
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
    new_df = pd.DataFrame(index=index,columns=columns)
    for fieldname,dt in zip(columns,dtype.descr):
        default = default_dict[fieldname]
        new_df.loc[:,fieldname] = default
        new_df.loc[:,fieldname] = new_df.loc[:,fieldname].astype(dt[1])
    return new_df


def pst_from_io_files(tpl_files,in_files,ins_files,out_files,pst_filename=None):
    par_names = []
    if not isinstance(tpl_files,list):
        tpl_files = [tpl_files]
    for tpl_file in tpl_files:
        assert os.path.exists(tpl_file),"template file not found: "+str(tpl_file)
        par_names.extend(parse_tpl_file(tpl_file))

    if not isinstance(ins_files,list):
        ins_files = [ins_files]
    obs_names = []
    for ins_file in ins_files:
        assert os.path.exists(ins_file),"instruction file not found: "+str(ins_file)
        obs_names.extend(parse_ins_file(ins_file))


    new_pst = pyemu.Pst(pst_filename,load=False)

    pargp_data = populate_dataframe(["pargp"], new_pst.pargp_fieldnames,
                                    new_pst.pargp_defaults, new_pst.pargp_dtype)
    new_pst.parameter_groups = pargp_data

    par_data = populate_dataframe(par_names,new_pst.par_fieldnames,
                                  new_pst.par_defaults,new_pst.par_dtype)
    par_data.loc[:,"parnme"] = par_names
    new_pst.parameter_data = par_data
    obs_data = populate_dataframe(obs_names,new_pst.obs_fieldnames,
                                  new_pst.obs_defaults,new_pst.obs_dtype)
    obs_data.loc[:,"obsnme"] = obs_names
    new_pst.observation_data = obs_data
    new_pst.template_files = tpl_files
    new_pst.input_files = in_files
    new_pst.instruction_files = ins_files
    new_pst.output_files = out_files
    new_pst.model_command = ["model.bat"]

    if pst_filename:
        new_pst.zero_order_tikhonov()

        new_pst.write(pst_filename,update_regul=True)
    return new_pst


def get_phi_comps_from_recfile(recfile):
        """read the phi components from a record file
        Args:
            recfile (str) : record file
        Returns:
            dict{iteration number:{group,contribution}}
        Raises:
            None
        """
        iiter = 1
        iters = {}
        f = open(recfile,'r')
        while True:
            line = f.readline()
            if line == '':
                break
            if "starting phi for this iteration" in line.lower():
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

def smp_to_ins(smp_filename,ins_filename=None):
    """ create an instruction file from an smp file
    :param smp_filename: existing smp file
    :param ins_filename: instruction file to create.  If None, create
        an instruction file using the smp filename with the ".ins" suffix
    :return: dataframe instance of the smp file with the observation names and
        instruction lines as additional columns
    """
    if ins_filename is None:
        ins_filename = smp_filename+".ins"
    df = smp_to_dataframe(smp_filename)
    df.loc[:,"ins_strings"] = None
    df.loc[:,"observation_names"] = None
    name_groups = df.groupby("name").groups
    for name,idxs in name_groups.items():
        onames = [name+"_{0:d}".format(i) for i in range(len(idxs))]
        if False in (map(lambda x :len(x) <= 12,onames)):
            long_names = [oname for oname in onames if len(oname) > 12]
            raise Exception("observation names longer than 12 chars:\n{0}".format(str(long_names)))
        ins_strs = ["l1  ({0:s})39:46".format(on) for on in onames]

        df.loc[idxs,"observation_names"] = onames
        df.loc[idxs,"ins_strings"] = ins_strs

    with open(ins_filename,'w') as f:
        f.write("pif ~\n")
        [f.write(ins_str+"\n") for ins_str in df.loc[:,"ins_strings"]]
    return df


def dataframe_to_smp(dataframe,smp_filename,name_col="name",
                     datetime_col="datetime",value_col="value",
                     datetime_format="dd/mm/yyyy",
                     value_format="{0:15.6E}"):
    """ write a dataframe as an smp file

    :param dataframe: a pandas dataframe
    :param smp_filename: smp file to write
    :param name_col: the column in the dataframe the marks the site namne
    :param datetime_col: the column in the dataframe that is a datetime instance
    :param value_col: the column in the dataframe that is the values
    :param datetime_format: either 'dd/mm/yyyy' or 'mm/dd/yyy'
    :param value_format: a python float-compatible format
    :return: None
    """

    formatters = {"name":lambda x:"{0:10s}".format(str(x)[:10]),
                  "value":lambda x:value_format.format(x)}
    if datetime_format.lower().startswith("d"):
        dt_fmt = "%d/%m/%Y    %H:%M:%S"
    elif datetime_format.lower().startswith("m"):
        dt_fmt = "%m/%d/%Y    %H:%M:%S"
    else:
        raise Exception("unrecognized datetime_format: " +\
                        "{0}".format(str(datetime_format)))

    for col in [name_col,datetime_col,value_col]:
        assert col in dataframe.columns

    dataframe.loc[:,"datetime_str"] = dataframe.loc[:,"datetime"].\
        apply(lambda x:x.strftime(dt_fmt))
    if isinstance(smp_filename,str):
        smp_filename = open(smp_filename,'w')
        smp_filename.write(dataframe.loc[:,[name_col,"datetime_str",value_col]].\
                to_string(col_space=0,
                          formatters=formatters,
                          justify="right",
                          header=False,
                          index=False) + '\n')
    dataframe.pop("datetime_str")


def date_parser(items):
    """ datetime parser to help load smp files
    :param items: tuple of date and time strings from smp file
    :return: a datetime instance
    """
    try:
        dt = datetime.strptime(items,"%d/%m/%Y %H:%M:%S")
    except Exception as e:
        try:
            dt = datetime.strptime(items,"%m/%d/%Y %H:%M:%S")
        except Exception as ee:
            raise Exception("error parsing datetime string" +\
                            " {0}: \n{1}\n{2}".format(str(items),str(e),str(ee)))
    return dt


def smp_to_dataframe(smp_filename):
    """ load an smp file into a pandas dataframe
    :param smp_filename: smp filename to load
    :return: a pandas dataframe instance
    """
    df = pd.read_csv(smp_filename, delim_whitespace=True,
                     parse_dates={"datetime":["date","time"]},
                     header=None,names=["name","date","time","value"],
                     dtype={"name":object,"value":np.float64},
                     na_values=["dry"],
                     date_parser=date_parser)
    return df


def start_slaves(slave_dir,exe_rel_path,pst_rel_path,num_slaves=None,slave_root="..",
                 port=4004,rel_path='.'):
    """ start a group of pest(++) slaves on the local machine

    Parameters:
    ----------
        slave_dir : (str) the path to a complete set of input files

        exe_rel_path : (str) the relative path to the pest(++)
                        executable from within the slave_dir
        pst_rel_path : (str) the relative path to the pst file
                        from within the slave_dir

        num_slaves : (int) number of slaves to start. defaults to number of cores

        slave_root : (str) the root to make the new slave directories in

        rel_path: (str) the relative path to where pest(++) should be run
                  from within the slave_dir, defaults to the uppermost level of the slave dir

    """

    assert os.path.isdir(slave_dir)
    assert os.path.isdir(slave_root)
    if num_slaves is None:
        num_slaves = mp.cpu_count()
    else:
        num_slaves = int(num_slaves)
    #assert os.path.exists(os.path.join(slave_dir,exe_rel_path))
    if not os.path.exists(os.path.join(slave_dir,exe_rel_path)):
        print("warning: exe_rel_path not verified...hopefully exe is in the PATH var")
    assert os.path.exists(os.path.join(slave_dir,pst_rel_path))

    hostname = socket.gethostname()
    port = int(port)

    tcp_arg = "{0}:{1}".format(hostname,port)

    procs = []
    for i in range(num_slaves):
        new_slave_dir = os.path.join(slave_root,"slave_{0}".format(i))
        if os.path.exists(new_slave_dir):
            try:
                shutil.rmtree(new_slave_dir)
            except Exception as e:
                raise Exception("unable to remove existing slave dir:" + \
                                "{0}\n{1}".format(new_slave_dir,str(e)))
        try:
            shutil.copytree(slave_dir,new_slave_dir)
        except Exception as e:
            raise Exception("unable to copy files from slave dir: " + \
                            "{0} to new slave dir: {1}\n{2}".format(slave_dir,new_slave_dir,str(e)))
        try:
            args = [exe_rel_path,pst_rel_path,"/h",tcp_arg]
            print("starting slave in {0} with args: {1}".format(new_slave_dir,args))
            p = sp.Popen(args,cwd=os.path.join(new_slave_dir,rel_path))
            procs.append(p)
        except Exception as e:
            raise Exception("error starting slave: {0}".format(str(e)))

    for p in procs:
        p.wait()


def test_smp():
    smp_filename = os.path.join('..','..',"examples","smp","sim_hds_v6.smp")
    df = smp_to_dataframe(smp_filename)
    print(df.dtypes)
    dataframe_to_smp(df,smp_filename+".test")
    smp_to_ins(smp_filename)
    obs_names = parse_ins_file(smp_filename+".ins")
    print(len(obs_names))

if __name__ == "__main__":
    test_smp()



