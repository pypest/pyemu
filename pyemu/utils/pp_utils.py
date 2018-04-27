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


def setup_pilotpoints_grid(ml=None,sr=None,ibound=None,prefix_dict=None,
                           every_n_cell=4,
                           use_ibound_zones=False,
                           pp_dir='.',tpl_dir='.',
                           shapename="pp.shp"):
    """setup grid-based pilot points.  Uses the ibound to determine
    where to set pilot points. pilot points are given generic ``pp_``
    names unless ``prefix_dict`` is passed.  Write template files and a
    shapefile as well...hopefully this is useful to someone...

    Parameters
    ----------
    ml : flopy.mbase
        a flopy mbase dervied type.  If None, sr must not be None.
    sr : flopy.utils.reference.SpatialReference
        a spatial reference use to locate the model grid in space.  If None,
        ml must not be None.  Default is None
    ibound : numpy.ndarray
        the modflow ibound integer array.  Used to set pilot points only in active areas (!=0).
        If None and ml is None, then pilot points are set in all rows and columns according to
        every_n_cell.  Default is None.
    prefix_dict : dict
        a dictionary of layer index, pilot point parameter prefixes.
        Example : ``{0:["hk_"],1:["vka_"]}`` would setup pilot points with
        the prefix ``"hk"`` for model layers 1 and ``"vka"`` for model layer 2 (zero based).
        If None, a generic set of pilot points with the "pp" prefix are setup
        for a generic nrowXncol grid. Default is None.
    use_ibound_zones : bool
        a flag to use the greater-than-zero values in the ibound as pilot point zones.  If False,ibound
        values greater than zero are treated as a single zone.  Default is False.
    pp_dir : str
        directory to write pilot point files to.  Default is '.'
    tpl_dir : str
        directory to write pilot point template file to.  Default is '.'
    shapename : str
        name of shapefile to write that contains pilot point information. Default is "pp.shp"

    Returns
    -------
    pp_df : pandas.DataFrame
        a dataframe summarizing pilot point information (same information
        written to shapename
    Example
    -------
    ``>>>import flopy``

    ``>>>from pyemu.utils import setup_pilotpoints_grid``

    ``>>>m = flopy.modflow.Modfow.load("mymodel.nam")``

    ``>>>setup_pilotpoints_grid(m,prefix_dict={0:['hk_'],1:['vka_']},``

    ``>>>                       every_n_cell=3,shapename='layer1_pp.shp')``
    """

    import flopy

    if ml is not None:
        assert isinstance(ml,flopy.modflow.Modflow)
        sr = ml.sr
        if ibound is None:
            ibound = ml.bas6.ibound.array
    else:
        assert sr is not None,"if 'ml' is not passed, 'sr' must be passed"
        if ibound is None:

            print("ibound not passed, using array of ones")
            ibound = {k:np.ones((sr.nrow,sr.ncol)) for k in prefix_dict.keys()}
        #assert ibound is not None,"if 'ml' is not pass, 'ibound' must be passed"

    try:
        xcentergrid = sr.xcentergrid
        ycentergrid = sr.ycentergrid
    except Exception as e:
        raise Exception("error getting xcentergrid and/or ycentergrid from 'sr':{0}".\
                        format(str(e)))
    start = int(float(every_n_cell) / 2.0)

    #build a generic prefix_dict
    if prefix_dict is None:
        prefix_dict = {k:["pp_{0:02d}_".format(k)] for k in range(ml.nlay)}

    #check prefix_dict
    for k, prefix in prefix_dict.items():
        assert k < len(ibound),"layer index {0} > nlay {1}".format(k,len(ibound))
        if not isinstance(prefix,list):
            prefix_dict[k] = [prefix]

    #try:
        #ibound = ml.bas6.ibound.array
    #except Exception as e:
    #    raise Exception("error getting model.bas6.ibound:{0}".format(str(e)))
    par_info = []
    pp_files,tpl_files = [],[]
    pp_names = copy.copy(PP_NAMES)
    pp_names.extend(["k","i","j"])
    for k in range(len(ibound)):
        pp_df = None
        ib = ibound[k]
        assert ib.shape == xcentergrid.shape,"ib.shape != xcentergrid.shape for k {0}".\
            format(k)
        pp_count = 0
        #skip this layer if not in prefix_dict
        if k not in prefix_dict.keys():
            continue
        #cycle through rows and cols
        for i in range(start,ib.shape[0]-start,every_n_cell):
            for j in range(start,ib.shape[1]-start,every_n_cell):
                # skip if this is an inactive cell
                if ib[i,j] == 0:
                    continue

                # get the attributes we need
                x = xcentergrid[i,j]
                y = ycentergrid[i,j]
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
        except Exception as e:
            print("error importing shapefile, try pip install pyshp...{0}"\
                  .format(str(e)))
            return par_info
        shp = shapefile.Writer(shapeType=shapefile.POINT)
        for name,dtype in par_info.dtypes.iteritems():
            if dtype == object:
                shp.field(name=name,fieldType='C',size=50)
            elif dtype in [int,np.int,np.int64,np.int32]:
                shp.field(name=name, fieldType='N', size=50, decimal=0)
            elif dtype in [float,np.float,np.float32,np.float64]:
                shp.field(name=name, fieldType='N', size=50, decimal=10)
            else:
                raise Exception("unrecognized field type in par_info:{0}:{1}".format(name,dtype))

        #some pandas awesomeness..
        par_info.apply(lambda x:shp.poly([[[x.x,x.y]]],shapeType=shapefile.POINT), axis=1)
        par_info.apply(lambda x:shp.record(*x),axis=1)

        shp.save(shapename)
        shp = shapefile.Reader(shapename)
        assert shp.numRecords == par_info.shape[0]
    return par_info


def pp_file_to_dataframe(pp_filename):

    """ read a pilot point file to a pandas Dataframe

    Parameters
    ----------
    pp_filename : str
        pilot point file

    Returns
    -------
    df : pandas.DataFrame
        a dataframe with pp_utils.PP_NAMES for columns

    """

    df = pd.read_csv(pp_filename, delim_whitespace=True,
                     header=None, names=PP_NAMES,usecols=[0,1,2,3,4])
    df.loc[:,"name"] = df.name.apply(str).apply(str.lower)
    return df

def pp_tpl_to_dataframe(tpl_filename):
    """ read a pilot points template file to a pandas dataframe

    Parameters
    ----------
    tpl_filename : str
        pilot points template file

    Returns
    -------
    df : pandas.DataFrame
        a dataframe with "parnme" included

    """
    inlines = open(tpl_filename, 'r').readlines()
    header = inlines.pop(0)
    marker = header.strip().split()[1]
    assert len(marker) == 1
    usecols = [0,1,2,3]
    df = pd.read_csv(tpl_filename, delim_whitespace=True,skiprows=1,
                     header=None, names=PP_NAMES[:-1],usecols=usecols)
    df.loc[:,"name"] = df.name.apply(str).apply(str.lower)
    df["parnme"] = [i.split(marker)[1].strip() for i in inlines]


    return df

def write_pp_shapfile(pp_df,shapename=None):
    """write pilot points dataframe to a shapefile

    Parameters
    ----------
    pp_df : pandas.DataFrame or str
        pilot point dataframe or a pilot point filename.  Dataframe
        must include "x" and "y"
    shapename : str
        shapefile name.  If None, pp_df must be str and shapefile
        is saved as <pp_df>.shp

    Note
    ----
        requires pyshp

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
            raise Exception("unrecognized field type in pp_df:{0}:{1}".format(name, dtype))


    # some pandas awesomeness..
    for df in dfs:
        df.apply(lambda x: shp.poly([[[x.x, x.y]]]), axis=1)
        df.apply(lambda x: shp.record(*x), axis=1)

    shp.save(shapename)



def write_pp_file(filename,pp_df):
    """write a pilot points dataframe to a pilot points file

    Parameters
    ----------
    filename : str
        pilot points file to write
    pp_df : pandas.DataFrame
        a dataframe that has columns "x","y","zone", and "value"

    """
    with open(filename,'w') as f:
       f.write(pp_df.to_string(col_space=0,
                                columns=PP_NAMES,
                                formatters=PP_FMT,
                                justify="right",
                                header=False,
                                index=False) + '\n')


def pilot_points_to_tpl(pp_file,tpl_file=None,name_prefix=None):
    """write a template file for a pilot points file

    Parameters
    ----------
    pp_file : str
        pilot points file
    tpl_file : str
        template file name to write.  If None, append ".tpl" to
        the pp_file arg. Default is None
    name_prefix : str
        name to prepend to parameter names for each pilot point.  For example,
        if ``name_prefix = "hk_"``, then each pilot point parameter will be named
        "hk_0001","hk_0002", etc.  If None, parameter names from pp_df.name
        are used.  Default is None.

    Returns
    -------
        pp_df : pandas.DataFrame
            a dataframe with pilot point information (name,x,y,zone,parval1)
             with the parameter information (parnme,tpl_str)
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
