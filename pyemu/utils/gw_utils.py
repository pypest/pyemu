import os
import numpy as np
import pandas as pd
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT

def pilot_points_to_tpl(pp_file,tpl_file=None,name_prefix=None):
    assert os.path.exists(pp_file)
    if tpl_file is None:
        tpl_file = pp_file+".tpl"

    pp_df = pd.read_csv(pp_file,delim_whitespace=True,header=None,
                        names=["name","x","y","zone","value"])

    if name_prefix is not None:
        digits = str(len(str(pp_df.shape[0])))
        fmt = "{0:0"+digits+"d}"
        names = [name_prefix+fmt.format(i) for i in range(pp_df.shape[0])]
    else:
        names = pp_df.name.copy()

    too_long = []
    for name in names:
        if len(name) > 12:
            too_long.append(name)
    if len(too_long) > 0:
        raise Exception("the following parameter names are too long:" +\
                        ",".join(too_long))

    tpl_entries = ["~    {0}    ~".format(name) for name in names]
    pp_df.loc[:,"tpl"] = tpl_entries
    fmt = {"name":SFMT,"x":FFMT,"y":FFMT,"zone":IFMT,"tpl":SFMT}

    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(pp_df.to_string(col_space=0,
                              columns=["name","x","y","zone","tpl"],
                              formatters=fmt,
                              justify="left",
                              header=False,
                              index=False) + '\n')

    return pp_df

def fac2real(pp_file,factors_file,out_file="test.ref",
             upper_lim=1.0e+30,lower_lim=-1.0e+30):
    assert os.path.exists(pp_file)
    assert os.path.exists(factors_file)
    pp_data = pd.read_csv(pp_file,delim_whitespace=True,header=None,
                          names=["name","value"],usecols=[0,4])
    pp_data.loc[:,"name"] = pp_data.name.apply(lambda x: x.lower())

    f_fac = open(factors_file,'r')
    fpp_file = f_fac.readline()
    fzone_file = f_fac.readline()
    ncol,nrow = [int(i) for i in f_fac.readline().strip().split()]
    npp = int(f_fac.readline().strip())
    pp_names = [f_fac.readline().strip().lower() for _ in range(npp)]

    # check that pp_names is sync'd with pp_data
    diff = set(list(pp_data.name)).symmetric_difference(set(pp_names))
    if len(diff) > 0:
        raise Exception("the following pilot point names are not common " +\
                        "between the factors file and the pilot points file " +\
                        ','.join(list(diff)))

    arr = np.zeros((nrow,ncol),dtype=np.float32) - 1.0e+30
    for i in range(nrow):
        for j in range(ncol):
            line = f_fac.readline()
            if len(line) == 0:
                raise Exception("unexpected EOF in factors file")
            try:
                fac_data = parse_factor_line(line)
            except Exception as e:
                raise Exception("error parsing factor line {0}:{1}".format(line,str(e)))
            fac_prods = [pp_data.loc[pp,"value"]*fac_data[pp] for pp in fac_data]
            arr[i,j] = np.sum(np.array(fac_prods))
    arr[arr<lower_lim] = lower_lim
    arr[arr>upper_lim] = upper_lim
    np.savetxt(out_file,arr,fmt="%15.6E",delimiter='')

def parse_factor_line(line):
    raw = line.strip().split()
    inode,zone,nfac = [int(i) for i in raw[:3]]
    offset = float(raw[3])
    fac_data = {}
    for ifac in range(4,4+nfac*2,2):
        pnum = int(raw[ifac]) - 1 #zero based to sync with pandas
        fac = float(raw[ifac+1])
        fac_data[pnum] = fac
    return fac_data



