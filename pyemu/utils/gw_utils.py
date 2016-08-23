import os
import copy
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT,pst_config

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

def modflow_hob_to_instruction_file(hob_file,ins_file=None):
    """write an instruction file for a modflow head observation file
    Parameters
    ----------
        hob_file : str
            modflow hob file
        ins_file :str (optional)
            instruction file to write.  If None, use <hob_file>.ins
    Returns
    -------
        pandas DataFrame with control file observation information
    """

    hob_df = pd.read_csv(hob_file,delim_whitespace=True,skiprows=1,
                         header=None,names=["simval","obsval","obsnme"])

    hob_df.loc[:,"ins_line"] = hob_df.obsnme.apply(lambda x:"l1 w w !{0:s}!".format(x))
    hob_df.loc[0,"ins_line"] = hob_df.loc[0,"ins_line"].replace('l1','l2')

    if ins_file is None:
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


def setup_pilotpoints_grid(ml,prefix_dict=None,
                           every_n_cell=4,
                           use_ibound_zones=False,
                           pp_dir='.',tpl_dir='.',
                           shapename="pp.shp"):
    """setup grid-based pilot points.  Uses the ibound to determine
       where to set pilot points. pilot points are given generic "pp_"
       names.  write template files as well...hopefully this is useful
        to someone...
    Parameters
    ----------
        ml : flopy.modflow.Modflow instance
        prefix_dict : (optional)dict{k:list}
            a dictionary of parameter prefixes to use for each model
            layer (e.g. {0:["hk_1","sy_1","rch"],1:["hk_2","ss_2"]}).
            layer indices not list in prefix_dict will not have
            pilot points written for them. If None, then "pp_<k>_" is
            used for each layer. Zero-based layer index!!!
        every_n_cell : int
            the stride in the row and col loops. controls how dense the
            point point network is compared to the model grid.
            every_n_cell = 1 results in a pilot point in every cell
        use_ibound_zones : bool
            flag to use ibound values as zones for the pilot points
        pp_dir : str
            directory for pilot point files
        tpl_dir : str
            directory for template files
        shapename : str
            name of pilot point shapefile.  set to None to disable
    Returns
    -------
        par_info : pd.DataFrame
            a combined dataframe with pilot point, control file and
            file location information

    """
    import flopy
    assert isinstance(ml,flopy.modflow.Modflow)

    #build a generic prefix_dict
    if prefix_dict is None:
        prefix_dict = {k:["pp_{0:02d}_".format(k)] for k in range(ml.nlay)}

    #check prefix_dict
    for k, prefix in prefix_dict.items():
        assert k < ml.nlay,"layer index {0} > nlay {1}".format(k,ml.nlay)
        if not isinstance(prefix,list):
            prefix_dict[k] = [prefix]

    try:
        ibound = ml.bas6.ibound.array
    except Exception as e:
        raise Exception("error getting model.bas6.ibound:{0}".format(str(e)))
    par_info = []
    pp_files,tpl_files = [],[]
    pp_names = copy.copy(PP_NAMES)
    pp_names.extend(["k","i","j"])
    for k in range(ml.nlay):
        pp_df = None
        ib = ibound[k]
        pp_count = 0
        #skip this layer if not in prefix_dict
        if k not in prefix_dict.keys():
            continue
        #cycle through rows and cols
        for i in range(0,ml.nrow,every_n_cell):
            for j in range(0,ml.ncol,every_n_cell):
                # skip if this is an inactive cell
                if ib[i,j] < 1:
                    continue

                # get the attributes we need
                x = ml.sr.xcentergrid[i,j]
                y = ml.sr.ycentergrid[i,j]
                name = "pp_{0:04d}".format(pp_count)
                parval1 = 1.0

                #decide what to use as the zone
                zone = 1
                if use_ibound_zones:
                    zone = ib[i,j]
                #stick this pilot point into a dataframe container
                if pp_df is None:
                    data = {"name": name, "x": x, "y": y, "zone": zone,
                            "parval1": parval1, "k":k, "i":i, "j":j}
                    pp_df = pd.DataFrame(data=data,index=[0],columns=pp_names)
                else:
                    data = [name, x, y, zone, parval1, k, i, j]
                    pp_df.loc[pp_count,:] = data
                pp_count += 1
        #if we found some acceptable locs...
        if pp_df is not None:
            for prefix in prefix_dict[k]:
                base_filename = prefix+"pp.dat"
                pp_filename = os.path.join(pp_dir, base_filename)
                # write the base pilot point file
                write_pp_file(pp_filename, pp_df)

                tpl_filename = os.path.join(tpl_dir, base_filename + ".tpl")
                #write the tpl file
                pilot_points_to_tpl(pp_df, tpl_filename,
                                    name_prefix=prefix)
                pp_df.loc[:,"tpl_filename"] = tpl_filename
                pp_df.loc[:,"pp_filename"] = pp_filename
                pp_df.loc[:,"pargp"] = prefix
                #save the parameter names and parval1s for later
                par_info.append(pp_df.copy())
                #save the pp_filename and tpl_filename for later
                pp_files.append(pp_filename)
                tpl_files.append(tpl_filename)

    par_info = pd.concat(par_info)
    for key,default in pst_config["par_defaults"].items():
        if key in par_info.columns:
            continue
        par_info.loc[:,key] = default

    if shapename is not None:
        try:
            import shapefile
        except:
            print("error importing shapefile, try pip install pyshp...")
            return par_info
        shp = shapefile.Writer(shapeType=shapefile.POINT)
        for name,dtype in par_info.dtypes.iteritems():
            if dtype == object:
                shp.field(name=name,fieldType='C',size=50)
            elif dtype in [int,np.int,np.int64,np.int32]:
                shp.field(name=name, fieldType='N', size=50, decimal=0)
            elif dtype in [float,np.float,np.float32,np.float32]:
                shp.field(name=name, fieldType='N', size=50, decimal=8)
            else:
                raise Exception("unrecognized field type in par_info:{0}:{1}".format(name,dtype))

        #some pandas awesomeness..
        par_info.apply(lambda x:shp.poly([[[x.x,x.y]]]), axis=1)
        par_info.apply(lambda x:shp.record(*x),axis=1)

        shp.save(shapename)
        shp = shapefile.Reader(shapename)
        assert shp.numRecords == par_info.shape[0]
    return par_info


def pp_file_to_dataframe(pp_filename):
    return pd.read_csv(pp_filename, delim_whitespace=True,
                     header=None, names=PP_NAMES)


def write_pp_shapfile(pp_df,shapename=None):
    """write pilot points to a shapefile
    Parameters
    ----------
        pp_df : pandas.DataFrame or str
            pilot dataframe or a pilot point filename
        shapename : (optional) str
            shapefile name.  If None, pp_df must be str
    Returns
    -------
        None

    """
    try:
        import shapefile
    except Exception as e:
        raise Exception("error importing shapefile: {0}, \ntry pip install pyshp...".format(str(e)))

    if not isinstance(pp_df,list):
        pp_df = [pp_df]
    dfs = []
    for pp in pp_df:
        if isinstance(pp,pd.DataFrame):
            dfs.append(pp)
        elif isinstance(pp,str):
            dfs.append(pp_file_to_dataframe(pp))
        else:
            raise Exception("unsupported arg type:{0}".format(type(pp)))

    if shapename is None:
        shapename = "pp_locs.shp"

    shp = shapefile.Writer(shapeType=shapefile.POINT)
    for name, dtype in dfs[0].dtypes.iteritems():
        if dtype == object:
            shp.field(name=name, fieldType='C', size=50)
        elif dtype in [int, np.int, np.int64, np.int32]:
            shp.field(name=name, fieldType='N', size=50, decimal=0)
        elif dtype in [float, np.float, np.float32, np.float32]:
            shp.field(name=name, fieldType='N', size=50, decimal=8)
        else:
            raise Exception("unrecognized field type in par_info:{0}:{1}".format(name, dtype))


    # some pandas awesomeness..
    for df in dfs:
        df.apply(lambda x: shp.poly([[[x.x, x.y]]]), axis=1)
        df.apply(lambda x: shp.record(*x), axis=1)

    shp.save(shapename)



def write_pp_file(filename,pp_df):
    """write a pilot points file from a dataframe
    Parameters
    ----------
        filename : str
            pilot points file to write
        pp_df : pandas DataFrame
            must have columns name, x, y, zone and value
    Returns
    -------
        None
    """
    with open(filename,'w') as f:
       f.write(pp_df.to_string(col_space=0,
                                columns=PP_NAMES,
                                formatters=PP_FMT,
                                justify="right",
                                header=False,
                                index=False) + '\n')


def pilot_points_to_tpl(pp_file,tpl_file=None,name_prefix=None):
    """write a template file from a pilot points file
    Parameters
    ----------
        pp_file : str
            pilot points file
        tpl_file : (optional)str
            template file name to create.  If None, append ".tpl" to
            the pp_file arg
        name_prefix : (optional)str
            name to prepend to parameter names for each pilot point.  for example,
            if name_prefix = "hk_", then each pilot point parameter will be named
            "hk_0001","hk_0002", etc
    Returns
    -------
        pp_df : pandas.DataFrame
            pilot point information (name,x,y,zone,parval1) with the parameter
            information (parnme,tpl),where is the parmaeter marker that went
            into the template file.

    """

    if isinstance(pp_file,pd.DataFrame):
        pp_df = pp_file
        assert tpl_file is not None
    else:
        assert os.path.exists(pp_file)
        pp_df = pd.read_csv(pp_file, delim_whitespace=True,
                            header=None, names=PP_NAMES)

    if tpl_file is None:
        tpl_file = pp_file + ".tpl"

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
    pp_df.loc[:,"parnme"] = names


    f_tpl = open(tpl_file,'w')
    f_tpl.write("ptf ~\n")
    f_tpl.write(pp_df.to_string(col_space=0,
                              columns=["name","x","y","zone","tpl"],
                              formatters=PP_FMT,
                              justify="left",
                              header=False,
                              index=False) + '\n')

    return pp_df


def fac2real(pp_file,factors_file,out_file="test.ref",
             upper_lim=1.0e+30,lower_lim=-1.0e+30):
    """A python replication of the PEST fac2real utility
    Parameters
    ----------
        pp_file : str
            existing pilot points file
        factors_file : str
            existing factors file from ppk2fac, etc
        out_file : str
            filename of array to write
    Returns
    -------
        None
    """
    if isinstance(pp_file,str):
        assert os.path.exists(pp_file)

        pp_data = pd.read_csv(pp_file,delim_whitespace=True,header=None,
                              names=["name","parval1"],usecols=[0,4])
        pp_data.loc[:,"name"] = pp_data.name.apply(lambda x: x.lower())
    elif isinstance(pp_file,pd.DataFrame):
        assert "name" in pp_file.columns
        assert "parval1" in pp_file.columns
        pp_data = pp_file
    else:
        raise Exception("unrecognized pp_file arg: must be str or pandas.DataFrame, not {0}"\
                        .format(type(pp_file)))
    assert os.path.exists(factors_file)
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

    arr = np.zeros((nrow,ncol),dtype=np.float32) + 1.0e+30
    pp_dict = {name:val for name,val in zip(pp_data.index,pp_data.parval1)}
    pp_dict_log = {name:np.log10(val) for name,val in zip(pp_data.index,pp_data.parval1)}
    #for i in range(nrow):
    #    for j in range(ncol):
    while True:
        line = f_fac.readline()
        if len(line) == 0:
            #raise Exception("unexpected EOF in factors file")
            break
        try:
            inode,itrans,fac_data = parse_factor_line(line)
        except Exception as e:
            raise Exception("error parsing factor line {0}:{1}".format(line,str(e)))
        #fac_prods = [pp_data.loc[pp,"value"]*fac_data[pp] for pp in fac_data]
        if itrans == 0:
            fac_sum = sum([pp_dict[pp] * fac_data[pp] for pp in fac_data])
        else:
            fac_sum = sum([pp_dict_log[pp] * fac_data[pp] for pp in fac_data])
        if itrans != 0:
            fac_sum = 10**fac_sum
        #col = ((inode - 1) // nrow) + 1
        #row = inode - ((col - 1) * nrow)
        row = ((inode-1) // ncol) + 1
        col = inode - ((row - 1) * ncol)
        #arr[row-1,col-1] = np.sum(np.array(fac_prods))
        arr[row - 1, col - 1] = fac_sum
    arr[arr<lower_lim] = lower_lim
    arr[arr>upper_lim] = upper_lim
    if out_file is not None:
        np.savetxt(out_file,arr,fmt="%15.6E",delimiter='')
        return out_file
    return arr

def parse_factor_line(line):
    raw = line.strip().split()
    inode,itrans,nfac = [int(i) for i in raw[:3]]
    fac_data = {}
    for ifac in range(4,4+nfac*2,2):
        pnum = int(raw[ifac]) - 1 #zero based to sync with pandas
        fac = float(raw[ifac+1])
        fac_data[pnum] = fac
    return inode,itrans,fac_data



