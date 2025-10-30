"""Pilot point support utilities
"""
from __future__ import division, print_function

import os
import copy
import numpy as np
import pandas as pd
import warnings

import pyemu

pd.options.display.max_colwidth = 100
from pyemu.pst.pst_utils import SFMT, IFMT, FFMT, pst_config
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
        ml (`flopy.mbase`, optional): a flopy mbase derived type.  If None, `sr` must not be None.
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

    if sr.grid_type=='vertex':
        if len(xcentergrid.shape)==1:
            xcentergrid = np.reshape(xcentergrid, (xcentergrid.shape[0], 1))
            ycentergrid = np.reshape(ycentergrid, (ycentergrid.shape[0], 1))


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
    if sr.grid_type == "vertex":
        pp_names.extend(["k"])
    else:
        pp_names.extend(["k", "i", "j"])

    if not np.all([isinstance(v, dict) for v in ibound.values()]):
        ibound = {"general_zn": ibound}
    par_keys = list(ibound.keys())
    par_keys.sort()
    for par in par_keys:
        for k in range(len(ibound[par])):
            # skip this layer if not in prefix_dict
            if k not in prefix_dict.keys():
                continue

            pp_df = []
            ib = ibound[par][k]
            assert (
                ib.shape == xcentergrid.shape
            ), "ib.shape != xcentergrid.shape for k {0}".format(k)

            pp_count = 0
            if sr.grid_type == "vertex":
                spacing=every_n_cell
                
                for zone in np.unique(ib):
                    # escape <zero idomain values
                    if zone <= 0:
                        continue
                    ppoint_xys = get_zoned_ppoints_for_vertexgrid(spacing, ib, sr, zone_number=zone)

                    parval1 = 1.0

                    for x, y in ppoint_xys:
                        # name from pilot point count
                        name = "pp_{0:04d}".format(pp_count)
                        pp_df.append([name, x, y, zone, parval1, k,])
                        pp_count += 1
            else:
                # cycle through rows and cols
                # allow to run closer to outside edge rather than leaving a gap
                for i in range(start_row, ib.shape[0] - start_row//2, every_n_cell):
                    for j in range(start_col, ib.shape[1] - start_col//2, every_n_cell):
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
                        pp_df.append([name, x, y, zone, parval1, k, i, j])
                        pp_count += 1
            pp_df = pd.DataFrame(pp_df, columns=pp_names)
            # if we found some acceptable locs...
            if not pp_df.empty:
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
    if sr.grid_type == "vertex":
        fields = ["k", "zone"]
    else:
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
                shp.field(name, "C", size=50)
            elif dtype in [int]:#, np.int64, np.int32]:
                shp.field(name, "N", size=50, decimal=0)
            elif dtype in [float, np.float32, np.float64]:
                shp.field(name, "N", size=50, decimal=10)
            else:
                try:
                    if dtype in [np.int64, np.int32]:
                        shp.field(name, "N", size=50, decimal=0)
                    else:
                        raise Exception(
                            "unrecognized field type in par_info:{0}:{1}".format(name, dtype)
                        )

                except Exception as e:
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
    """read a space-delim pilot point file to a pandas Dataframe

    Args:
        pp_filename (`str`): path and name of an existing pilot point file

    Returns:
        `pandas.DataFrame`: a dataframe with `pp_utils.PP_NAMES` for columns

    Example::

        df = pyemu.pp_utils.pp_file_to_dataframe("my_pp.dat")

    """

    df = pd.read_csv(
        pp_filename,
        sep=r'\s+',
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
        sep=r"\s+",
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
    for name, dtype in dfs[0].dtypes.items():
        if dtype == object:
            shp.field(name, "C", size=50)
        elif dtype in [int]:#, np.int, np.int64, np.int32]:
            shp.field(name, "N", size=50, decimal=0)
        elif dtype in [float, np.float32, np.float32]:
            shp.field(name, "N", size=50, decimal=8)
        else:
            try:
                if dtype in [np.int64, np.int32]:
                    shp.field(name, "N", size=50, decimal=0)
                else:
                    raise Exception(
                        "unrecognized field type in par_info:{0}:{1}".format(name, dtype)
                    )

            except Exception as e:
                raise Exception(
                    "unrecognized field type in par_info:{0}:{1}".format(name, dtype)
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
        pp_df = pd.read_csv(pp_file, sep=r"\s+",
                            header=None, names=PP_NAMES)
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
    with open(tpl_file, "w") as f:
        f.write("ptf ~\n")
        pp_df.loc[:, ["name", "x", "y", "zone", "tpl"]].apply(
            lambda x: f.write(' '.join(x.astype(str)) + '\n'), axis=1)
    # _write_df_tpl(
    #     tpl_file,
    #     pp_df.loc[:, ["name", "x", "y", "zone", "tpl"]],
    #     sep=" ",
    #     index_label="index",
    #     header=False,
    #     index=False,
    #     quotechar=" ",
    #     quoting=2,
    # )

    return pp_df

def get_zoned_ppoints_for_vertexgrid(spacing, zone_array, mg, zone_number=None, add_buffer=True):
    """Generate equally spaced pilot points for active area of DISV type MODFLOW6 grid. 
 
    Args:
        spacing (`float`): spacing in model length units between pilot points. 
        zone_array (`numpy.ndarray`): the modflow 6 idomain integer array.  This is used to
            set pilot points only in active areas and to assign zone numbers. 
        mg  (`flopy.discretization.vertexgrid.VertexGrid`): a VertexGrid flopy discretization derived type.
        zone_number (`int`): zone number 
        add_buffer (`boolean`): specifies whether pilot points ar eplaced within a buffer zone of size `distance` around the zone/active domain

    Returns:
        `list`: a list of tuples with pilot point x and y coordinates

    Example::

        get_zoned_ppoints_for_vertexgrid(spacing=100, ib=idomain, mg, zone_number=1, add_buffer=False)

    """

    try:
        from shapely.ops import unary_union
        from shapely.geometry import Polygon #, Point, MultiPoint
        from shapely import points
    except ImportError:
        raise ImportError('The `shapely` library was not found. Please make sure it is installed.')

    
    if mg.grid_type=='vertex' and zone_array is not None and len(zone_array.shape)==1:
            zone_array = np.reshape(zone_array, (zone_array.shape[0], ))


    assert zone_array.shape[0] == mg.xcellcenters.shape[0], "The ib idomain array should be of shape (ncpl,). i.e. For a single layer."

    if zone_number:
        assert zone_array[zone_array==zone_number].shape[0]>0, f"The zone_number: {zone_number} is not in the ib array."

    # get zone cell x,y 
    xc, yc = mg.xcellcenters, mg.ycellcenters
    if zone_number != None:
        xc_zone = xc[np.where(zone_array==zone_number)[0]]
        yc_zone = yc[np.where(zone_array==zone_number)[0]]
    else:
        xc_zone = xc[np.where(zone_array>0)]
        yc_zone = yc[np.where(zone_array>0)]   

    # get outer bounds
    xmin, xmax = min(xc_zone), max(xc_zone)
    ymin, ymax = min(yc_zone), max(yc_zone)
    # n-ppoints
    nx = int(np.ceil((xmax - xmin) / spacing))
    ny = int(np.ceil((ymax - ymin) / spacing))
    # make even spaced points
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xv, yv = np.meshgrid(x, y)
    def _get_ppoints():
        # make grid
        grid_points = points(list(zip(xv.flatten(), yv.flatten())))

        # get vertices for model grid/zone polygon
        verts = [mg.get_cell_vertices(cellid) for cellid in range(mg.ncpl)]
        if zone_number != None:
            # select zone area
            verts = [verts[i] for i in np.where(zone_array==zone_number)[0]]
        # dissolve
        polygon = unary_union([Polygon(v) for v in verts])
        # add buffer
        if add_buffer==True:
            polygon = polygon.buffer(spacing)
        # select ppoint coords within area
        ppoints = [(p.x, p.y) for p in grid_points[polygon.covers(grid_points)]]
        return ppoints
    #
    # def _get_ppoints_new(): # alternative method searching for speedup (not too successful)
    #     # make grid
    #     grid_points = points(list(zip(xv.flatten(), yv.flatten())))
    #
    #     # get vertices for model grid/zone polygon
    #     verts = [mg.get_cell_vertices(cellid) for cellid in range(mg.ncpl)]
    #     if zone_number != None:
    #         # select zone area
    #         verts = [verts[i] for i in np.where(zone_array == zone_number)[0]]
    #     # dissolve
    #     polygon = unary_union([Polygon(v) for v in verts])
    #     # add buffer
    #     if add_buffer == True:
    #         polygon = polygon.buffer(spacing)
    #     # select ppoint coords within area
    #     ppoints = [(p.x, p.y) for p in grid_points[polygon.covers(grid_points)]]
    #     return ppoints

    ppoints = _get_ppoints()
    # ppoints = _get_ppoints_new()
    assert len(ppoints)>0
    return ppoints


def parse_pp_options_with_defaults(pp_kwargs, threads=10, log=True, logger=None, **depr_kwargs):
    default_dict = dict(pp_space=10,  # default pp spacing
                        use_pp_zones=False, # don't setup pp by zone, as default
                        spatial_reference=None,  # if not passed pstfrom will use class attrib
                        # factor calc options
                        try_use_ppu=True,  # default to using ppu
                        num_threads=threads,  # fallback if num_threads not in pp_kwargs, only used if ppu fails
                        # factor calc options, incl at run time.
                        minpts_interp=1,
                        maxpts_interp=20,
                        search_radius=1e10,
                        # ult lims
                        fill_value=1.0 if log else 0.,
                        lower_limit=1.0e-30 if log else -1.0e30,
                        upper_limit=1.0e30
                        )
    # for run time options we need to be strict about dtypes
    default_dtype = dict(# pp_space=10,  # default pp spacing
                         # use_pp_zones=False, # don't setup pp by zone, as default
                         # spatial_reference=None,  # if not passed pstfrom will use class attrib
                         # # factor calc options
                         # try_use_ppu=True,  # default to using ppu
                         # num_threads=threads,  # fallback if num_threads not in pp_kwargs, only used if ppu fails
                         # # factor calc options, incl at run time.
                         minpts_interp=int,
                         maxpts_interp=int,
                         search_radius=float,
                         # ult lims
                         fill_value=float,
                         lower_limit=float,
                         upper_limit=float)

    # parse deprecated kwargs first
    for key, val in depr_kwargs.items():
        if val is not None:
            if key in pp_kwargs:
                if logger is not None:
                    logger.lraise(f"'{key}' passed but its also in 'pp_options")
            if logger is not None:
                logger.warn(f"Directly passing '{key}' has been deprecated and will eventually be removed" +
                            f", please use pp_options['{key}'] instead.")
            pp_kwargs[key] = val

    # fill defaults
    for key, val in default_dict.items():
        if key not in pp_kwargs:
            if logger is not None:
                logger.statement(f"'{key}' not set in pp_options, "
                                 f"Setting to default value: [{val}]")
            pp_kwargs[key] = val

    for key, typ in default_dtype.items():
        pp_kwargs[key] = typ(pp_kwargs[key])

    return pp_kwargs


def prep_pp_hyperpars(file_tag,pp_filename,pp_info,out_filename,grid_dict,
                       geostruct,arr_shape,pp_options,zone_array=None,
                      ws = "."):
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception("prep_pp_hyperpars() error importing pypestutils: '{0}'".format(str(e)))

    illegal_chars = [i for i in r"/:*?<>\|"]
    for i in illegal_chars:
        if i in file_tag:
            print("warning: replacing illegal character '{0}' with '-' in file_tag name '{1}'".format(i,file_tag))
            file_tag = file_tag.replace(i,"-")

    gridinfo_filename = file_tag + ".gridinfo.dat"
    corrlen_filename = file_tag + ".corrlen.dat"
    bearing_filename = file_tag + ".bearing.dat"
    aniso_filename = file_tag + ".aniso.dat"
    zone_filename = file_tag + ".zone.dat"

    if len(arr_shape) == 1 and type(arr_shape) is tuple:
        arr_shape = (1,arr_shape[0])


    nodes = list(grid_dict.keys())
    nodes.sort()
    with open(os.path.join(ws,gridinfo_filename), 'w') as f:
        f.write("node,x,y\n")
        for node in nodes:
            f.write("{0},{1},{2}\n".format(node, grid_dict[node][0], grid_dict[node][1]))

    corrlen = np.zeros(arr_shape) + geostruct.variograms[0].a
    np.savetxt(os.path.join(ws,corrlen_filename), corrlen, fmt="%20.8E")
    bearing = np.zeros(arr_shape) + geostruct.variograms[0].bearing
    np.savetxt(os.path.join(ws,bearing_filename), bearing, fmt="%20.8E")
    aniso = np.zeros(arr_shape) + geostruct.variograms[0].anisotropy
    np.savetxt(os.path.join(ws,aniso_filename), aniso, fmt="%20.8E")

    if zone_array is None:
        zone_array = np.ones(arr_shape,dtype=int)
    np.savetxt(os.path.join(ws,zone_filename),zone_array,fmt="%5d")


    # fnx_call = "pyemu.utils.apply_ppu_hyperpars('{0}','{1}','{2}','{3}','{4}'". \
    #     format(pp_filename, gridinfo_filename, out_filename, corrlen_filename,
    #            bearing_filename)
    # fnx_call += "'{0}',({1},{2}))".format(aniso_filename, arr_shape[0], arr_shape[1])

    # apply_ppu_hyperpars(pp_filename, gridinfo_filename, out_filename, corrlen_filename,
    #                     bearing_filename, aniso_filename, arr_shape)

    config_df = pd.DataFrame(columns=["value"])
    config_df.index.name = "key"

    config_df.loc["pp_filename", "value"] = pp_filename  # this might be in pp_options too in which case does this get stomped on?
    config_df.loc["out_filename","value"] = out_filename
    config_df.loc["corrlen_filename", "value"] = corrlen_filename
    config_df.loc["bearing_filename", "value"] = bearing_filename
    config_df.loc["aniso_filename", "value"] = aniso_filename
    config_df.loc["gridinfo_filename", "value"] = gridinfo_filename
    config_df.loc["zone_filename", "value"] = zone_filename

    config_df.loc["vartransform","value"] = geostruct.transform
    v = geostruct.variograms[0]
    vartype = 2
    if isinstance(v, pyemu.geostats.ExpVario):
        pass
    elif isinstance(v, pyemu.geostats.SphVario):
        vartype = 1
    elif isinstance(v, pyemu.geostats.GauVario):
        vartype = 3
    else:
        raise NotImplementedError("unsupported variogram type: {0}".format(str(type(v))))
    krigtype = 1  #ordinary
    config_df.loc["vartype","value"] = vartype
    config_df.loc["krigtype","value"] = krigtype
    config_df.loc["shape", "value"] = arr_shape

    keys = list(pp_options.keys())
    keys.sort()
    for k in keys:
        config_df.loc[k,"value"] = pp_options[k]

    #config_df.loc["function_call","value"] = fnx_call
    config_df_filename = file_tag + ".config.csv"
    config_df.loc["config_df_filename",:"value"] = config_df_filename

    config_df.to_csv(os.path.join(ws, config_df_filename))
    # this is just a temp input file needed for testing...
    #pp_info.to_csv(os.path.join(ws,pp_filename),sep=" ",header=False)
    pyemu.pp_utils.write_pp_file(os.path.join(ws,pp_filename),pp_info)
    bd = os.getcwd()
    os.chdir(ws)
    try:
        apply_ppu_hyperpars(config_df_filename)
    except Exception as e:
        os.chdir(bd)
        raise RuntimeError(f"apply_ppu_hyperpars() error: {e}")
    os.chdir(bd)
    return config_df


def apply_ppu_hyperpars(config_df_filename):
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception("apply_ppu_hyperpars() error importing pypestutils: '{0}'".format(str(e)))

    config_df = pd.read_csv(config_df_filename,index_col=0)
    config_dict = config_df["value"].to_dict()
    vartransform = config_dict.get("vartransform", "none")
    config_dict = parse_pp_options_with_defaults(config_dict, threads=None, log=vartransform=='log')

    out_filename = config_dict["out_filename"]
    #pp_info = pd.read_csv(config_dict["pp_filename"],sep="\s+")
    pp_info = pyemu.pp_utils.pp_file_to_dataframe(config_dict["pp_filename"])
    grid_df = pd.read_csv(config_dict["gridinfo_filename"])
    corrlen = np.loadtxt(config_dict["corrlen_filename"])
    bearing = np.loadtxt(config_dict["bearing_filename"])
    aniso = np.loadtxt(config_dict["aniso_filename"])
    zone = np.loadtxt(config_dict["zone_filename"])

    lib = PestUtilsLib()
    fac_fname = out_filename+".temp.fac"
    if os.path.exists(fac_fname):
        os.remove(fac_fname)
    fac_ftype = "text"
    npts = lib.calc_kriging_factors_2d(
            pp_info.x.values,
            pp_info.y.values,
            pp_info.zone.values,
            grid_df.x.values.flatten(),
            grid_df.y.values.flatten(),
            zone.flatten().astype(int),
            int(config_dict.get("vartype",1)),
            int(config_dict.get("krigtype",1)),
            corrlen.flatten(),
            aniso.flatten(),
            bearing.flatten(),
            # defaults should be in config_dict -- the fallbacks here should not be hit now
            config_dict.get("search_dist",config_dict.get("search_radius", 1e10)),
            config_dict.get("maxpts_interp",50),
            config_dict.get("minpts_interp",1),
            fac_fname,
            fac_ftype,
        )

    # this is now filled as a default in config_dict if not in config file,
    # default value dependent on vartransform (ie. 1 for log 0 for non log)
    noint = config_dict.get("fill_value",pp_info.loc[:, "parval1"].mean())

    result = lib.krige_using_file(
        fac_fname,
        fac_ftype,
        zone.size,
        int(config_dict.get("krigtype", 1)),
        vartransform,
        pp_info["parval1"].values,
        noint,
        noint,
    )
    assert npts == result["icount_interp"]
    result = result["targval"]
    #shape = tuple([int(s) for s in config_dict["shape"]])
    tup_string = config_dict["shape"]
    shape = tuple(int(x) for x in tup_string[1:-1].split(','))
    result = result.reshape(shape)
    np.savetxt(out_filename,result,fmt="%20.8E")
    os.remove(fac_fname)
    lib.free_all_memory()

    return result
