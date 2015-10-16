from __future__ import print_function, division
import os
import copy
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
    return par_df


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
                par_names.extend(line.strip().split(marker)[1::2])
        except Exception as e:
            raise Exception("error processing template file " +\
                            tpl_file+" :\n" + str(e))

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
            eidx = min(len(string),string[idx+1:].index(em)+idx+1)
            obs_name = string[idx+1:eidx]
            obs_names.append(obs_name)
            idx = eidx
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


def pst_from_io_files(pst_filename,tpl_files,in_files,ins_files,out_files):
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

    new_pst.zero_order_tikhonov()

    new_pst.write(pst_filename,update_regul=True)


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