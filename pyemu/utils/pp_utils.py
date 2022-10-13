"""Pilot point support utilities
"""
import os
import copy
import numpy as np
import pandas as pd
import warnings

pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import SFMT, IFMT, FFMT, pst_config
from pyemu.utils.helpers import run, _write_df_tpl
from ..pyemu_warnings import PyemuWarning

PP_FMT = {
    "name": SFMT,
    "x": FFMT,
    "y": FFMT,
    "zone": IFMT,
    "tpl": SFMT,
    "parval1": FFMT,
}
PP_NAMES = ["name", "x", "y", "zone", "parval1"]


def setup_pilotpoints_grid(
    ml=None,
    sr=None,
    ibound=None,
    prefix_dict=None,
    every_n_cell=4,
    ninst=1,
    use_ibound_zones=False,
    pp_dir=".",
    tpl_dir=".",
    shapename="pp.shp",
    pp_filename_dict={},
):
    """setup a regularly-spaced (gridded) pilot point parameterization

    Args:
        ml (`flopy.mbase`, optional): a flopy mbase dervied type.  If None, `sr` must not be None.
        sr (`flopy.utils.reference.SpatialReference`, optional):  a spatial reference use to
            locate the model grid in space.  If None, `ml` must not be None.  Default is None
        ibound (`numpy.ndarray`, optional): the modflow ibound integer array.  THis is used to
            set pilot points only in active areas. If None and ml is None, then pilot points
            are set in all rows and columns according to `every_n_cell`.  Default is None.
        prefix_dict (`dict`): a dictionary of layer index, pilot point parameter prefix(es) pairs.
            For example : `{0:["hk,"vk"]}` would setup pilot points with the prefix "hk" and "vk" for
            model layer 1. If None, a generic set of pilot points with
            the "pp" prefix are setup for a generic nrow by ncol grid. Default is None
        ninst (`int`): Number of instances of pilot_points to set up.
            e.g. number of layers. If ml is None and prefix_dict is None,
            this is used to set up default prefix_dict.
        use_ibound_zones (`bool`): a flag to use the greater-than-zero values in the
            ibound as pilot point zones.  If False ,ibound values greater than zero are
            treated as a single zone.  Default is False.
        pp_dir (`str`, optional): directory to write pilot point files to.  Default is '.'
        tpl_dir (`str`, optional): directory to write pilot point template file to.  Default is '.'
        shapename (`str`, optional): name of shapefile to write that contains pilot
            point information. Default is "pp.shp"
        pp_filename_dict (`dict`): optional dict of prefix-pp filename pairs.  prefix values must
            match the values in `prefix_dict`.  If None, then pp filenames are based on the
            key values in `prefix_dict`.  Default is None

    Returns:
        `pandas.DataFrame`: a dataframe summarizing pilot point information (same information
        written to `shapename`

    Example::

        m = flopy.modflow.Modflow.load("my.nam")
        df = pyemu.pp_utils.setup_pilotpoints_grid(ml=m)

    """

    if ml is not None:
        try:
            import flopy
        except Exception as e:
            raise ImportError("error importing flopy: {0}".format(str(e)))
        assert isinstance(ml, flopy.modflow.Modflow)
        try:
            sr = ml.sr
        except AttributeError:
            from pyemu.utils.helpers import SpatialReference

            sr = SpatialReference.from_namfile(
                os.path.join(ml.model_ws, ml.namefile),
                delr=ml.modelgrid.delr,
                delc=ml.modelgrid.delc,
            )
        if ibound is None:
            ibound = ml.bas6.ibound.array
            # build a generic prefix_dict
        if prefix_dict is None:
            prefix_dict = {k: ["pp_{0:02d}_".format(k)] for k in range(ml.nlay)}
    else:
        assert sr is not None, "if 'ml' is not passed, 'sr' must be passed"
        if prefix_dict is None:
            prefix_dict = {k: ["pp_{0:02d}_".format(k)] for k in range(ninst)}
        if ibound is None:
            print("ibound not passed, using array of ones")
            ibound = {k: np.ones((sr.nrow, sr.ncol)) for k in prefix_dict.keys()}
        # assert ibound is not None,"if 'ml' is not pass, 'ibound' must be passed"

    if isinstance(ibound, np.ndarray):
        assert np.ndim(ibound) == 2 or np.ndim(ibound) == 3, (
            "ibound needs to be either 3d np.ndarray or k_dict of 2d arrays. "
            "Array of {0} dimensions passed".format(np.ndim(ibound))
        )
        if np.ndim(ibound) == 2:
            ibound = {0: ibound}
        else:
            ibound = {k: arr for k, arr in enumerate(ibound)}

    try:
        xcentergrid = sr.xcentergrid
        ycentergrid = sr.ycentergrid
    except AttributeError as e0:
        warnings.warn("xcentergrid and/or ycentergrid not in 'sr':{0}", PyemuWarning)
        try:
            xcentergrid = sr.xcellcenters
            ycentergrid = sr.ycellcenters
        except AttributeError as e1:
            raise Exception(
                "error getting xcentergrid and/or ycentergrid "
                "from 'sr':{0}:{1}".format(str(e0), str(e1))
            )
    start = int(float(every_n_cell) / 2.0)

    # fix for x-section models
    if xcentergrid.shape[0] == 1:
        start_row = 0
    else:
        start_row = start

    if xcentergrid.shape[1] == 1:
        start_col = 0
    else:
        start_col = start

    # check prefix_dict
    keys = list(prefix_dict.keys())
    keys.sort()

    # for k, prefix in prefix_dict.items():
    for k in keys:
        prefix = prefix_dict[k]
        if not isinstance(prefix, list):
            prefix_dict[k] = [prefix]
        if np.all([isinstance(v, dict) for v in ibound.values()]):
            for p in prefix_dict[k]:
                if np.any([p.startswith(key) for key in ibound.keys()]):
                    ib_sel = next(key for key in ibound.keys() if p.startswith(key))
                else:
                    ib_sel = "general_zn"
                assert k < len(ibound[ib_sel]), "layer index {0} > nlay {1}".format(
                    k, len(ibound[ib_sel])
                )
        else:
            assert k < len(ibound), "layer index {0} > nlay {1}".format(k, len(ibound))

    # try:
    # ibound = ml.bas6.ibound.array
    # except Exception as e:
    #    raise Exception("error getting model.bas6.ibound:{0}".format(str(e)))
    par_info = []
    pp_files, tpl_files = [], []
    pp_names = copy.copy(PP_NAMES)
    pp_names.extend(["k", "i", "j"])

    if not np.all([isinstance(v, dict) for v in ibound.values()]):
        ibound = {"general_zn": ibound}
    par_keys = list(ibound.keys())
    par_keys.sort()
    for par in par_keys:
        for k in range(len(ibound[par])):
            pp_df = None
            ib = ibound[par][k]
            assert (
                ib.shape == xcentergrid.shape
            ), "ib.shape != xcentergrid.shape for k {0}".format(k)
            pp_count = 0
            # skip this layer if not in prefix_dict
            if k not in prefix_dict.keys():
                continue
            # cycle through rows and cols
            for i in range(start_row, ib.shape[0] - start_row, every_n_cell):
                for j in range(start_col, ib.shape[1] - start_col, every_n_cell):
                    # skip if this is an inactive cell
                    if ib[i, j] <= 0:  # this will account for MF6 style ibound as well
                        continue

                    # get the attributes we need
                    x = xcentergrid[i, j]
                    y = ycentergrid[i, j]
                    name = "pp_{0:04d}".format(pp_count)
                    parval1 = 1.0

                    # decide what to use as the zone
                    zone = 1
                    if use_ibound_zones:
                        zone = ib[i, j]
                    # stick this pilot point into a dataframe container

                    if pp_df is None:
                        data = {
                            "name": name,
                            "x": x,
                            "y": y,
                            "zone": zone,  # if use_ibound_zones is False this will always be 1
                            "parval1": parval1,
                            "k": k,
                            "i": i,
                            "j": j,
                        }
                        pp_df = pd.DataFrame(data=data, index=[0], columns=pp_names)
                    else:
                        data = [name, x, y, zone, parval1, k, i, j]
                        pp_df.loc[pp_count, :] = data
                    pp_count += 1
            # if we found some acceptable locs...
            if pp_df is not None:
                for prefix in prefix_dict[k]:
                    # if parameter prefix relates to current zone definition
                    if prefix.startswith(par) or (
                        ~np.any([prefix.startswith(p) for p in ibound.keys()])
                        and par == "general_zn"
                    ):
                        if prefix in pp_filename_dict.keys():
                            base_filename = pp_filename_dict[prefix].replace(":", "")
                        else:
                            base_filename = "{0}pp.dat".format(prefix.replace(":", ""))
                        pp_filename = os.path.join(pp_dir, base_filename)
                        # write the base pilot point file
                        write_pp_file(pp_filename, pp_df)

                        tpl_filename = os.path.join(tpl_dir, base_filename + ".tpl")
                        # write the tpl file
                        pp_df = pilot_points_to_tpl(
                            pp_df, tpl_filename, name_prefix=prefix,
                        )
                        pp_df.loc[:, "tpl_filename"] = tpl_filename
                        pp_df.loc[:, "pp_filename"] = pp_filename
                        pp_df.loc[:, "pargp"] = prefix
                        # save the parameter names and parval1s for later
                        par_info.append(pp_df.copy())
                        # save the pp_filename and tpl_filename for later
                        pp_files.append(pp_filename)
                        tpl_files.append(tpl_filename)

    par_info = pd.concat(par_info)
    fields = ["k", "i", "j", "zone"]
    par_info = par_info.astype({f: int for f in fields}, errors='ignore')
    defaults = pd.DataFrame(pst_config["par_defaults"], index=par_info.index)
    missingcols = defaults.columns.difference(par_info.columns)
    par_info.loc[:, missingcols] = defaults

    if shapename is not None:
        try:
            import shapefile
        except Exception as e:
            print(
                "error importing shapefile, try pip install pyshp...{0}".format(str(e))
            )
            return par_info
        try:
            shp = shapefile.Writer(target=shapename, shapeType=shapefile.POINT)
        except:
            shp = shapefile.Writer(shapeType=shapefile.POINT)
        for name, dtype in par_info.dtypes.items():
            if dtype == object:
                shp.field(name=name, fieldType="C", size=50)
            elif dtype in [int, np.int64, np.int32]:
                shp.field(name=name, fieldType="N", size=50, decimal=0)
            elif dtype in [float, np.float32, np.float64]:
                shp.field(name=name, fieldType="N", size=50, decimal=10)
            else:
                raise Exception(
                    "unrecognized field type in par_info:{0}:{1}".format(name, dtype)
                )

        # some pandas awesomeness..
        par_info.apply(lambda x: shp.point(x.x, x.y), axis=1)
        par_info.apply(lambda x: shp.record(*x), axis=1)
        try:
            shp.save(shapename)
        except:
            shp.close()
        shp.close()
        shp = shapefile.Reader(shapename)
        assert shp.numRecords == par_info.shape[0]
        shp.close()
    return par_info


def pp_file_to_dataframe(pp_filename):
    """read a pilot point file to a pandas Dataframe

    Args:
        pp_filename (`str`): path and name of an existing pilot point file

    Returns:
        `pandas.DataFrame`: a dataframe with `pp_utils.PP_NAMES` for columns

    Example::

        df = pyemu.pp_utils.pp_file_to_dataframe("my_pp.dat")

    """

    df = pd.read_csv(
        pp_filename,
        delim_whitespace=True,
        header=None,
        names=PP_NAMES,
        usecols=[0, 1, 2, 3, 4],
    )
    df.loc[:, "name"] = df.name.apply(str).apply(str.lower)
    return df


def pp_tpl_to_dataframe(tpl_filename):
    """read a pilot points template file to a pandas dataframe

    Args:
        tpl_filename (`str`): path and name of an existing pilot points
            template file

    Returns:
        `pandas.DataFrame`: a dataframe of pilot point info with "parnme" included

    Notes:
        Use for processing pilot points since the point point file itself may
        have generic "names".

    Example::

        df = pyemu.pp_utils.pp_tpl_file_to_dataframe("my_pp.dat.tpl")

    """
    inlines = open(tpl_filename, "r").readlines()
    header = inlines.pop(0)
    marker = header.strip().split()[1]
    assert len(marker) == 1
    usecols = [0, 1, 2, 3]
    df = pd.read_csv(
        tpl_filename,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        names=PP_NAMES[:-1],
        usecols=usecols,
    )
    df.loc[:, "name"] = df.name.apply(str).apply(str.lower)
    df["parnme"] = [i.split(marker)[1].strip() for i in inlines]

    return df


def pilot_points_from_shapefile(shapename):
    """read pilot points from shapefile into a dataframe

    Args:
        shapename (`str`): the shapefile name to read.

    Notes:
        requires pyshp

    """
    try:
        import shapefile
    except Exception as e:
        raise Exception(
            "error importing shapefile: {0}, \ntry pip install pyshp...".format(str(e))
        )
    shp = shapefile.Reader(shapename)
    if shp.shapeType != shapefile.POINT:
        raise Exception("shapefile '{0}' is not POINT type")
    names = [n[0].lower() for n in shp.fields[1:]]
    if "name" not in names:
        raise Exception("pilot point shapefile missing 'name' attr")

    data = {name: [] for name in names}
    xvals = []
    yvals = []

    for shape, rec in zip(shp.shapes(), shp.records()):
        pt = shape.points[0]
        for name, val in zip(names, rec):
            data[name].append(val)
        xvals.append(pt[0])
        yvals.append(pt[1])

    df = pd.DataFrame(data)
    df.loc[:, "x"] = xvals
    df.loc[:, "y"] = yvals
    if "parval1" not in df.columns:
        print("adding generic parval1 to pp shapefile dataframe")
        df.loc[:, "parval1"] = 1.0

    return df


def write_pp_shapfile(pp_df, shapename=None):
    """write pilot points dataframe to a shapefile

    Args:
        pp_df (`pandas.DataFrame`): pilot point dataframe (must include "x" and "y"
            columns).  If `pp_df` is a string, it is assumed to be a pilot points file
            and is loaded with `pp_utils.pp_file_to_dataframe`. Can also be a list of
            `pandas.DataFrames` and/or filenames.
        shapename (`str`): the shapefile name to write.  If `None` , `pp_df` must be a string
            and shapefile is saved as `pp_df` +".shp"

    Notes:
        requires pyshp

    """
    try:
        import shapefile
    except Exception as e:
        raise Exception(
            "error importing shapefile: {0}, \ntry pip install pyshp...".format(str(e))
        )

    if not isinstance(pp_df, list):
        pp_df = [pp_df]
    dfs = []
    for pp in pp_df:
        if isinstance(pp, pd.DataFrame):
            dfs.append(pp)
        elif isinstance(pp, str):
            dfs.append(pp_file_to_dataframe(pp))
        else:
            raise Exception("unsupported arg type:{0}".format(type(pp)))

    if shapename is None:
        shapename = "pp_locs.shp"
    try:
        shp = shapefile.Writer(shapeType=shapefile.POINT)
    except:
        shp = shapefile.Writer(target=shapename, shapeType=shapefile.POINT)
    for name, dtype in dfs[0].dtypes.iteritems():
        if dtype == object:
            shp.field(name=name, fieldType="C", size=50)
        elif dtype in [int, np.int, np.int64, np.int32]:
            shp.field(name=name, fieldType="N", size=50, decimal=0)
        elif dtype in [float, np.float32, np.float32]:
            shp.field(name=name, fieldType="N", size=50, decimal=8)
        else:
            raise Exception(
                "unrecognized field type in pp_df:{0}:{1}".format(name, dtype)
            )

    # some pandas awesomeness..
    for df in dfs:
        # df.apply(lambda x: shp.poly([[[x.x, x.y]]]), axis=1)
        df.apply(lambda x: shp.point(x.x, x.y), axis=1)
        df.apply(lambda x: shp.record(*x), axis=1)

    try:
        shp.save(shapename)
    except:
        shp.close()


def write_pp_file(filename, pp_df):
    """write a pilot points dataframe to a pilot points file

    Args:
        filename (`str`): pilot points file to write
        pp_df (`pandas.DataFrame`):  a dataframe that has
            at least columns "x","y","zone", and "value"

    """
    with open(filename, "w") as f:
        f.write(
            pp_df.to_string(
                col_space=0,
                columns=PP_NAMES,
                formatters=PP_FMT,
                justify="right",
                header=False,
                index=False,
            )
            + "\n"
        )


def pilot_points_to_tpl(pp_file, tpl_file=None, name_prefix=None):
    """write a template file for a pilot points file

    Args:
        pp_file : (`str`): existing pilot points file
        tpl_file (`str`): template file name to write.  If None,
            `pp_file`+".tpl" is used.  Default is `None`.
        name_prefix (`str`): name to prepend to parameter names for each
            pilot point.  For example, if `name_prefix = "hk_"`, then each
            pilot point parameters will be named "hk_0001","hk_0002", etc.
            If None, parameter names from `pp_df.name` are used.
            Default is None.

    Returns:
        `pandas.DataFrame`: a dataframe with pilot point information
        (name,x,y,zone,parval1) with the parameter information
        (parnme,tpl_str)

    Example::

        pyemu.pp_utils.pilot_points_to_tpl("my_pps.dat",name_prefix="my_pps")


    """

    if isinstance(pp_file, pd.DataFrame):
        pp_df = pp_file
        assert tpl_file is not None
    else:
        assert os.path.exists(pp_file)
        pp_df = pd.read_csv(pp_file, delim_whitespace=True, header=None, names=PP_NAMES)
    pp_df = pp_df.astype({'zone': int}, errors='ignore')
    if tpl_file is None:
        tpl_file = pp_file + ".tpl"

    if name_prefix is not None:
        if "i" in pp_df.columns and "j" in pp_df.columns:
            pp_df.loc[:, "parnme"] = pp_df.apply(
                lambda x: "{0}_i:{1}_j:{2}".format(name_prefix, int(x.i), int(x.j)),
                axis=1,
            )
        elif "x" in pp_df.columns and "y" in pp_df.columns:
            pp_df.loc[:, "parnme"] = pp_df.apply(
                lambda x: "{0}_x:{1:0.2f}_y:{2:0.2f}".format(name_prefix, x.x, x.y),
                axis=1,
            )
        else:
            pp_df.loc[:, "idx"] = np.arange(pp_df.shape[0])
            pp_df.loc[:, "parnme"] = pp_df.apply(
                lambda x: "{0}_ppidx:{1}".format(name_prefix, x.idx),
                axis=1,
            )
        if "zone" in pp_df.columns:
            pp_df.loc[:, "parnme"] = pp_df.apply(
                lambda x: x.parnme + "_zone:{0}".format(x.zone), axis=1
            )
        pp_df.loc[:, "tpl"] = pp_df.parnme.apply(
            lambda x: "~    {0}    ~".format(x)
        )
    else:
        pp_df.loc[:, "parnme"] = pp_df.name
        pp_df.loc[:, "tpl"] = pp_df.parnme.apply(
            lambda x: "~    {0}    ~".format(x)
        )
    _write_df_tpl(
        tpl_file,
        pp_df.loc[:, ["name", "x", "y", "zone", "tpl"]],
        sep=" ",
        index_label="index",
        header=False,
        index=False,
        quotechar=" ",
        quoting=2,
    )

    return pp_df
