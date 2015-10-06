from __future__ import print_function, division
import os
import copy
import numpy as np
import pandas
pandas.options.display.max_colwidth = 100

def parse_tpl_file(tpl_file):
    par_names = []
    with open(tpl_file,'r') as f:
        header = f.readline().strip().split()
        assert header[0].lower() in ["ptf","jtf"],\
            "template file error: must start with [ptf,jtf], not:" +\
            str(header[0])
        marker = header[1]
        assert len(marker) == 1,\
            "template file error: marker must be a single character, not:" +\
            str(marker)
        for line in f:
            par_names.extend(line.strip().split(marker)[1::2])
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


def populate_dataframe(index,columns,defaults,dtype):
    new_df = pandas.DataFrame(index=index,columns=columns)
    for fieldname,default,dt in zip(columns,defaults,dtype.descr):
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

    new_pst = pst(pst_filename,load=False)
    par_data = populate_dataframe(par_names,new_pst.par_fieldnames,
                                  new_pst.par_defaults,new_pst.par_dtype)
    par_data.loc[:,"parnme"] = par_names
    new_pst.parameter_data = par_data

    obs_data = populate_dataframe(obs_names,new_pst.obs_fieldnames,
                                  new_pst.obs_defaults,new_pst.obs_dtype)
    obs_data.loc[:,"obsnme"] = obs_names
    new_pst.observation_data = obs_data
    new_pst.mode = "estimation"
    raise NotImplementedError()