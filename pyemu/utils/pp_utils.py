"""Pilot point support utilities
"""
from __future__ import division, print_function

import os
import copy
import heapq
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

# ordinary kriging needs at least 3 non-collinear points to avoid a
# singular system - fewer than this and interpolation is not well-posed
MIN_KRIGE_PPOINTS = 3

# only flag the search as having "under-delivered" relative to what
# every_n_cell implies for the zone's active area (active cells /
# every_n_cell**2) if the shortfall is large. The fallback trigger still
# compares against the bounding-box stride request (n_requested), since
# that is what the direct search was asked to deliver
PPOINT_UNDER_DELIVERY_FRACTION = 0.5


def _greedy_fill_ppoints(cells, seeds, radius, max_points, min_points):
    """Pick pilot point cells from a zone's active ``cells`` by greedy
    farthest-point selection.

    Starting from ``seeds`` (the stride hits already found - or, if there
    are none, the active cell nearest the zone centroid), repeatedly add
    the active cell farthest from everything chosen so far, until every
    active cell is within ``radius`` cells of a chosen point,
    ``max_points`` is reached, or all cells are used. ``min_points`` is
    always honoured if the zone has that many cells.

    Unlike slicing the zone's bounding box into buckets, this follows the
    zone's own shape: thin or diagonal zones get evenly spaced points
    along their length and no cell is chosen merely for being the first
    in its bucket.

    Parameters
    ----------
    cells : array-like, shape (n, 2)
        (row, col) of every active cell in the zone.
    seeds : list of (row, col)
        locations to keep and fill around.
    radius : float
        covering radius, in cells: stop once no active cell is further than
        this from a chosen point. ``every_n_cell / sqrt(2)`` is the radius
        of a square lattice with ``every_n_cell`` between points.
    max_points : int
        upper limit on the number of points returned.
    min_points : int
        keep adding points until at least this many (if available).

    Returns
    -------
    list of (row, col) tuples, seeds first then in order of selection
    """
    from scipy.spatial import cKDTree

    cells = np.asarray(cells, dtype=float)
    chosen = [tuple(int(v) for v in s) for s in seeds]
    if not chosen:
        centroid = cells.mean(axis=0)
        first = int(np.argmin(((cells - centroid) ** 2).sum(axis=1)))
        chosen.append(tuple(int(v) for v in cells[first]))
    # squared distance from every cell to its nearest chosen point (cell
    # coordinates are integers, so this is exact after rounding)
    min_d2 = np.rint(cKDTree(np.array(chosen, dtype=float)).query(cells)[0] ** 2)
    radius2 = float(radius) ** 2
    max_points = max(max_points, len(chosen))

    # Picking the cell with the largest min_d2 each round, then updating
    # min_d2 over every cell, costs O(points added x cells). Instead keep a
    # lazy max-heap of the cells still further than ``radius`` from every
    # chosen point (ties go to the lowest index, as argmax would), and after
    # each pick only update the cells within the current worst gap of the new
    # point - nothing further away can be improved by it. A heap entry's key
    # is only ever an upper bound on its cell's current min_d2, so an entry
    # whose key no longer matches is re-queued with the current value rather
    # than trusted. The points chosen are identical to the dense version.
    todo = np.flatnonzero(min_d2 > radius2)
    heap = list(zip((-min_d2[todo]).tolist(), todo.tolist()))
    heapq.heapify(heap)
    tree = cKDTree(cells)
    while len(chosen) < max_points and heap:
        neg_key, idx = heapq.heappop(heap)
        d2 = -neg_key
        if d2 != min_d2[idx]:
            if min_d2[idx] > radius2:
                heapq.heappush(heap, (-min_d2[idx], idx))
            continue
        chosen.append(tuple(int(v) for v in cells[idx]))
        near = np.asarray(tree.query_ball_point(cells[idx], r=d2 ** 0.5 + 1e-9), dtype=int)
        near_d2 = ((cells[near] - cells[idx]) ** 2).sum(axis=1)
        closer = near_d2 < min_d2[near]
        min_d2[near[closer]] = near_d2[closer]

    # honour min_points even if everything is already within ``radius``
    while len(chosen) < max_points and len(chosen) < min_points:
        far = int(np.argmax(min_d2))
        if min_d2[far] == 0:
            break
        chosen.append(tuple(int(v) for v in cells[far]))
        min_d2 = np.minimum(min_d2, ((cells - cells[far]) ** 2).sum(axis=1))
    return chosen


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
                # cycle through zones, searching within each zone's own
                # extent rather than the full array - this way a zone
                # whose active cells never line up with a stride anchored
                # to the full grid still gets sampled (this accounts for
                # MF6-style ibound too, since only >0 cells are active)
                if use_ibound_zones:
                    zone_vals = sorted(z for z in np.unique(ib) if z > 0)
                else:
                    zone_vals = [1] if np.any(ib > 0) else []

                for zone in zone_vals:
                    zone_mask = (ib == zone) if use_ibound_zones else (ib > 0)
                    rows, cols = np.where(zone_mask)
                    row_min, row_max = int(rows.min()), int(rows.max())
                    col_min, col_max = int(cols.min()), int(cols.max())
                    h_row = row_max - row_min + 1
                    h_col = col_max - col_min + 1

                    # if the zone is thinner than a single stride step along
                    # an axis (this also covers x-section models, where the
                    # whole array is a single row or column), just use the
                    # midpoint of that axis instead of striding it
                    if h_row <= every_n_cell:
                        row_cands = [row_min + h_row // 2]
                    else:
                        # allow to run closer to outside edge rather than
                        # leaving a gap
                        row_cands = list(range(
                            row_min + start, row_min + h_row - start // 2, every_n_cell
                        ))
                    if h_col <= every_n_cell:
                        col_cands = [col_min + h_col // 2]
                    else:
                        col_cands = list(range(
                            col_min + start, col_min + h_col - start // 2, every_n_cell
                        ))

                    n_requested = len(row_cands) * len(col_cands)
                    # what every_n_cell implies for the zone's own area (one
                    # point per every_n_cell**2 active cells) - unlike
                    # n_requested this doesn't assume the zone fills its
                    # bounding box
                    n_expected = len(rows) / float(every_n_cell) ** 2
                    hits = [
                        (i, j) for i in row_cands for j in col_cands if zone_mask[i, j]
                    ]
                    fallback_threshold = max(
                        MIN_KRIGE_PPOINTS, n_expected * PPOINT_UNDER_DELIVERY_FRACTION
                    )
                    # also fall back if enough points were found but they
                    # leave part of the zone uncovered - i.e. some active
                    # cell is further than every_n_cell from every direct
                    # hit (a count alone can't see this: e.g. all the hits
                    # may fall in one of several disjoint parts of a zone)
                    too_few = len(hits) < fallback_threshold
                    uncovered = False
                    if not too_few and len(hits) > 0:
                        from scipy.spatial import cKDTree

                        gap = cKDTree(np.array(hits)).query(
                            np.column_stack([rows, cols])
                        )[0].max()
                        uncovered = gap > every_n_cell
                    if too_few or uncovered:
                        # too few of the candidate stride locations landed
                        # on an active cell (below the hard floor, or well
                        # below what every_n_cell implies for this zone's
                        # active area), or they leave part of the zone
                        # uncovered (e.g. a small, irregular/non-
                        # rectangular, periodic, or multi-part zone shape
                        # that "aliases" against the stride, so the direct
                        # hits it does find are few and/or clustered in
                        # only part of the zone) - keep the direct hits
                        # and fill in the gaps from the zone's active
                        # cells, so a zone that actually has enough active
                        # cells to support kriging isn't raised on, or left
                        # with lopsided coverage, just because the coarse
                        # stride under-sampled it. Points are added
                        # greedily (farthest from those already chosen),
                        # so the count follows the zone's own shape rather
                        # than its bounding box, and a large zone that
                        # aliases against the stride doesn't explode into
                        # a pilot point for every active cell (capped at
                        # what the stride requested, but never below
                        # MIN_KRIGE_PPOINTS). How far to fill depends on
                        # what triggered the fallback: too few points means
                        # the fill has to deliver the density every_n_cell
                        # asks for (the covering radius of a square lattice
                        # at that spacing); if there were enough points and
                        # only part of the zone was left uncovered, repair
                        # just that, down to the same every_n_cell tolerance
                        # that triggered it, rather than re-densifying
                        # a zone the stride mostly covered well.
                        n_direct = len(hits)
                        hits = _greedy_fill_ppoints(
                            np.column_stack([rows, cols]),
                            hits,
                            every_n_cell / 2 ** 0.5 if too_few else every_n_cell,
                            max_points=max(n_requested, MIN_KRIGE_PPOINTS),
                            min_points=MIN_KRIGE_PPOINTS,
                        )
                        if too_few:
                            msg = (
                                "setup_pilotpoints_grid(): layer {0}, zone {1}: the "
                                "every_n_cell={2} stride search found only {3} of {4} "
                                "requested candidate location(s) for this zone - falling "
                                "back to filling in from its {5} active cell(s), giving "
                                "{6} pilot point(s)."
                            )
                        else:
                            msg = (
                                "setup_pilotpoints_grid(): layer {0}, zone {1}: the "
                                "every_n_cell={2} stride search found {3} of {4} "
                                "requested candidate location(s) for this zone but left "
                                "part of it more than every_n_cell from any pilot point - "
                                "added pilot points from its {5} active cell(s), giving "
                                "{6} in total."
                            )
                        warnings.warn(
                            msg.format(
                                k, zone, every_n_cell, n_direct, n_requested, len(rows), len(hits)
                            ),
                            PyemuWarning,
                        )

                    if len(hits) < MIN_KRIGE_PPOINTS:
                        raise Exception(
                            "setup_pilotpoints_grid(): layer {0}, zone {1} produced only "
                            "{2} pilot point(s) ({3} active cell(s) in this zone) - fewer "
                            "than the {4} needed for ordinary kriging to be well-posed. "
                            "Merge this zone with a neighboring one, lower every_n_cell, "
                            "or exclude this zone/layer from the pilot point "
                            "parameterization.".format(
                                k, zone, len(hits), len(rows), MIN_KRIGE_PPOINTS
                            )
                        )
                    elif len(hits) < n_expected * PPOINT_UNDER_DELIVERY_FRACTION:
                        warnings.warn(
                            "setup_pilotpoints_grid(): layer {0}, zone {1}: every_n_cell={2} "
                            "implies about {3:.1f} pilot point(s) for this zone's {5} active "
                            "cell(s) (one per every_n_cell**2 cells), but only {4} were "
                            "placed - check the zone's shape, or consider a smaller "
                            "every_n_cell for better coverage.".format(
                                k, zone, every_n_cell, n_expected, len(hits), len(rows)
                            ),
                            PyemuWarning,
                        )

                    for i, j in hits:
                        x = xcentergrid[i, j]
                        y = ycentergrid[i, j]
                        name = "pp_{0:04d}".format(pp_count)
                        parval1 = 1.0
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
            dtype_str = str(dtype)
            # Handle string types (pandas.StringDtype shows as 'str' in Pandas +3.0)
            if dtype_str == 'str' or dtype_str == 'object' or dtype == object:
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
    df.loc[:, "name"] = df.name.str.lower()
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
                        minpts_interp=1,  # todo: check these defaults!!?
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


def prep_pp_hyperpars(file_tag, pp_filename, pp_info,out_filename, grid_dict,
                      geostruct, arr_shape, pp_options, zone_array=None,
                      ws = "."):
    """Make preparation for hyperparameter ppu setup.

    Writes files that to be used at apply time incl:
    ".gridinfo.dat" : target model cell location definition
    ".corrlen.dat", ".bearing.dat", ".aniso.dat", ".zone.dat" : arrays of
    hyperparamters to be used by calc_kriging_factors_2d() at apply time these
    arrays can also be paramterised (e.g. within a PstFrom add_pars setup).

    Args:
        file_tag (str): tag for hyperpar files (will be cleaned of illegal chars)
        pp_filename (str): pilot point file (filled by PEST)
        pp_info (pandas.DataFrame): DataFrame of pilotpoint data -- written to
            pp_filename temporarily for testing apply_ppu_hyperpars func
        out_filename (Path-like): Location of result of applying kriging
            interpolation after hyperpar factor calcs
        grid_dict (dict): Dictionary of {node: (x,y)} for interp target.
            Note, sorted node keys defines the order for output arrays and of
            zone_array (if passed).
        geostruct (pyemu.geostats.GeoStruct): Geostructure used for filling
            initial hyperpar arrays. Note, Geostruct should only contain one
            variogram
        arr_shape (list-like, int): shape of array in output file
        pp_options  (dict): Dictionary of pilot-point definition options to
            propagate to facotr calculation. Incl. "search_radius", "search_dist",
            "maxpts_interp", "minpts_interp", "fill_value"
        zone_array (array-like): array of integer zonal values for target array
            locations. Default will be all target locations (in grid_dict)
            assigned to same zone.
        ws (Path-like): Location of saving files.

    Returns:


    """
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

    if isinstance(arr_shape, int):
        arr_shape = [arr_shape]
    if len(arr_shape) == 1 and isinstance(arr_shape, (tuple, list)):
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
        if k == 'zone_array':
            continue
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
    from pyemu.utils.geostats import ppu_fac2real
    # read config info from file
    config_dict = pd.read_csv(config_df_filename,index_col=0)['value'].to_dict()
    # get transform before filling defaults
    vartransform = config_dict.get("vartransform", "none")
    # get defaults for pp options
    config_dict = parse_pp_options_with_defaults(config_dict, threads=None,
                                                 log=vartransform=='log')
    # pull out variables and read input files
    out_filename = config_dict["out_filename"]
    pp_info = pyemu.pp_utils.pp_file_to_dataframe(config_dict["pp_filename"])
    grid_df = pd.read_csv(config_dict["gridinfo_filename"])
    corrlen = np.loadtxt(config_dict["corrlen_filename"])
    bearing = np.loadtxt(config_dict["bearing_filename"])
    aniso = np.loadtxt(config_dict["aniso_filename"])
    zone = np.loadtxt(config_dict["zone_filename"])
    # this is now filled as a default in config_dict if not in config file,
    # default value dependent on vartransform (ie. 1 for log 0 for non log)
    noint = config_dict.get("fill_value",pp_info.loc[:, "parval1"].mean())

    # file name for temporary factor calculated on the fly here
    fac_fname = out_filename+".temp.fac"
    if os.path.exists(fac_fname):
        os.remove(fac_fname)
    # binary is rapid
    fac_ftype = "binary"

    # currently only working with OrdinaryKriging
    krigtype = int(config_dict.get("krigtype", 1))
    if krigtype != 1:
        raise NotImplementedError("only Ordinary Kriging supported (for now)")

    tup_string = config_dict["shape"]
    shape = tuple(int(x) for x in tup_string[1:-1].split(','))

    # run in context so that we can keep ppulib alive
    with pyemu.geostats.OrdinaryKrigePPU(None, pp_info, express=True) as krige:
        npts = krige.calc_factors(
            targets=grid_df,
            geostruct=dict(corrlen=corrlen,
                           aniso=aniso,
                           bearing=bearing,
                           vartype=config_dict.get("vartype", 2)),
            fac_fname=fac_fname,
            zone_array=zone,
            **config_dict
        )
        result = ppu_fac2real(
            source_vals=pp_info["parval1"].values,
            factor_filename=fac_fname,
            mpts=len(grid_df),
            out_shape=shape,
            ppulib=krige.lib,
            transform=vartransform,
            fac_ftype=fac_ftype,
            krigtype=krigtype,
            noint=noint,
        )

    # save interpolated field
    if out_filename.endswith(".npy"):
        np.save(out_filename, result)
    else:
        np.savetxt(out_filename,result,fmt="%20.8E")
    # kill temp fac file
    os.remove(fac_fname)
    return result
