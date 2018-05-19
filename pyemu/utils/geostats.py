"""Geostatistical analyses within the pyemu framework.
Support for Ordinary Kriging as well as construction of
covariance matrices from (nested) geostistical structures.
Also support for reading GSLIB and SGEMS files
"""
from __future__ import print_function
import os
import copy
from datetime import datetime
import multiprocessing as mp
import warnings
import numpy as np
import scipy.sparse
import pandas as pd
from pyemu.mat.mat_handler import Cov,SparseMatrix
from pyemu.utils.pp_utils import pp_file_to_dataframe

EPSILON = 1.0e-7

# class KrigeFactors(pd.DataFrame):
#     def __init__(self,*args,**kwargs):
#         super(KrigeFactors,self).__init__(*args,**kwargs)
#
#     def to_factors(self,filename,nrow,ncol,
#                    points_file="points.junk",
#                    zone_file="zone.junk"):
#         with open(filename,'w') as f:
#             f.write(points_file+'\n')
#             f.write(zone_file+'\n')
#             f.write("{0} {1}\n".format(ncol,nrow))
#             f.write("{0}\n".format(self.shape[0]))
#
#
#
#     def from_factors(self,filename):
#         raise NotImplementedError()



class GeoStruct(object):
    """a geostatistical structure object.  The object contains
    variograms and (optionally) nugget information.

    Parameters
    ----------
    nugget : (float)
        nugget contribution
    variograms : (list or Vario2d instance)
        variogram(s) associated with this GeoStruct instance
    name : (str)
        name to assign the structure.  Default is "struct1".
    transform : (str)
        the transformation to apply to the GeoStruct.  Can be
        "none" or "log", depending on the transformation of the
        property being represented by the GeoStruct.
        Default is "none

    Attributes
    ----------
    variograms : list
        the Vario2d objects associated with the GeoStruct
    name : str
        the name of the GeoStruct
    transform : str
        the transform of the GeoStruct

    Example
    -------
    ``>>>import pyemu``

    ``>>>v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)``

    ``>>>gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)``

    """
    def __init__(self,nugget=0.0,variograms=[],name="struct1",
                 transform="none"):
        self.name = name
        self.nugget = float(nugget)
        if not isinstance(variograms,list):
            variograms = [variograms]
        for vario in variograms:
            assert isinstance(vario,Vario2d)
        self.variograms = variograms
        transform = transform.lower()
        assert transform in ["none","log"]
        self.transform = transform

    def to_struct_file(self, f):
        """ write a PEST-style structure file

        Parameters
        ----------
        f : (str or file handle)
            file to write the GeoStruct information to

        """
        if isinstance(f, str):
            f = open(f,'w')
        f.write("STRUCTURE {0}\n".format(self.name))
        f.write("  NUGGET {0}\n".format(self.nugget))
        f.write("  NUMVARIOGRAM {0}\n".format(len(self.variograms)))
        for v in self.variograms:
            f.write("  VARIOGRAM {0} {1}\n".format(v.name,v.contribution))
        f.write("  TRANSFORM {0}\n".format(self.transform))
        f.write("END STRUCTURE\n\n")
        for v in self.variograms:
            v.to_struct_file(f)

    def sparse_covariance_matrix(self,x,y,names):
        """build a pyemu.Cov instance from GeoStruct

                Parameters
                ----------
                x : (iterable of floats)
                    x-coordinate locations
                y : (iterable of floats)
                    y-coordinate locations
                names : (iterable of str)
                   (parameter) names of locations.

                Returns
                -------
                sparse : pyemu.SparseMatrix
                    the sparse covariance matrix implied by this GeoStruct for the x,y pairs.

                Example
                -------
                ``>>>pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")``

                ``>>>cov = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)``


                """

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == len(names)

        iidx = [i for i in range(len(names))]
        jidx = list(iidx)
        data = list(np.zeros(x.shape[0])+self.nugget)

        for v in self.variograms:
            v.add_sparse_covariance_matrix(x,y,names,iidx,jidx,data)
        coo = scipy.sparse.coo_matrix((data,(iidx,jidx)),shape=(len(names),len(names)))
        coo.eliminate_zeros()
        coo.sum_duplicates()
        return SparseMatrix(coo,row_names=names,col_names=names)


    def covariance_matrix(self,x,y,names=None,cov=None):
        """build a pyemu.Cov instance from GeoStruct

        Parameters
        ----------
        x : (iterable of floats)
            x-coordinate locations
        y : (iterable of floats)
            y-coordinate locations
        names : (iterable of str)
            names of location. If None, cov must not be None.  Default is None.
        cov : (pyemu.Cov) instance
            an existing Cov instance.  The contribution of this GeoStruct is added
            to cov.  If cov is None, names must not be None. Default is None

        Returns
        -------
        cov : pyemu.Cov
            the covariance matrix implied by this GeoStruct for the x,y pairs.
            cov has row and column names supplied by the names argument unless
            the "cov" argument was passed.

        Note
        ----
        either "names" or "cov" must be passed.  If "cov" is passed, cov.shape
        must equal len(x) and len(y).

        Example
        -------
        ``>>>pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")``

        ``>>>cov = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)``


        """

        if not isinstance(x,np.ndarray):
            x = np.array(x)
        if not isinstance(y,np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]

        if names is not None:
            assert x.shape[0] == len(names)
            c = np.zeros((len(names),len(names)))
            np.fill_diagonal(c,self.nugget)
            cov = Cov(x=c,names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
            c = np.zeros((len(names),1))
            c += self.nugget
            cont = Cov(x=c,names=names,isdiagonal=True)
            cov += cont

        else:
            raise Exception("GeoStruct.covariance_matrix() requires either " +
                            "names or cov arg")
        for v in self.variograms:
            v.covariance_matrix(x,y,cov=cov)
        return cov

    def covariance(self,pt0,pt1):
        """get the covariance between two points implied by the GeoStruct.
        This is used during the ordinary kriging process to get the RHS

        Parameters
        ----------
        pt0 : (iterable length 2 of floats)
        pt1 : (iterable length 2 of floats)

        Returns
        -------
        covariance : float
            the covariance between pt0 and pt1 implied by the GeoStruct

        """
        #raise Exception()
        cov = self.nugget
        for vario in self.variograms:
            cov += vario.covariance(pt0,pt1)
        return cov

    def covariance_points(self,x0,y0,xother,yother):
        """ Get the covariance between point x0,y0 and the points
        contained in xother, yother.

        Parameters
        ----------
        x0 : (float)
            x-coordinate
        y0 : (float)
            y-coordinate
        xother : (iterable of floats)
            x-coordinate of other points
        yother : (iterable of floats)
            y-coordinate of other points

        Returns
        -------
        cov : numpy.ndarray
            a 1-D array of covariance between point x0,y0 and the
            points contained in xother, yother.  len(cov) = len(xother) =
            len(yother)

        """

        cov = np.zeros((len(xother))) + self.nugget
        for v in self.variograms:
            cov += v.covariance_points(x0,y0,xother,yother)
        return cov

    @property
    def sill(self):
        """ get the sill of the GeoStruct

        Return
        ------
        sill : float
            the sill of the (nested) GeoStruct, including nugget and contribution
            from each variogram
        """
        sill = self.nugget
        for v in self.variograms:
            sill += v.contribution
        return sill


    def plot(self,**kwargs):
        """ make a cheap plot of the GeoStruct

        Parameters
        ----------
        **kwargs : (dict)
            keyword arguments to use for plotting.

        Returns
        -------
        ax : matplotlib.pyplot.axis
            the axis with the GeoStruct plot

        Note
        ----
        optional arguments include "ax" (an existing axis),
        "individuals" (plot each variogram on a separate axis),
        "legend" (add a legend to the plot(s)).  All other kwargs
        are passed to matplotlib.pyplot.plot()

        """
        #
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            import matplotlib.pyplot as plt
            ax = plt.subplot(111)
        legend = kwargs.pop("legend",False)
        individuals = kwargs.pop("individuals",False)
        xmx = max([v.a*3.0 for v in self.variograms])
        x = np.linspace(0,xmx,100)
        y = np.zeros_like(x)
        for v in self.variograms:
            yv = v.inv_h(x)
            if individuals:
                ax.plot(x,yv,label=v.name,**kwargs)
            y += yv
        y += self.nugget
        ax.plot(x,y,label=self.name,**kwargs)
        if legend:
            ax.legend()
        ax.set_xlabel("distance")
        ax.set_ylabel("$\gamma$")
        return ax

    def __str__(self):
        """ the str representation of the GeoStruct

        Returns
        -------
        str : str
            the str representation of the GeoStruct
        """
        s = ''
        s += 'name:{0},nugget:{1},structures:\n'.format(self.name,self.nugget)
        for v in self.variograms:
            s += str(v)
        return s


# class LinearUniversalKrige(object):
#     def __init__(self,geostruct,point_data):
#         if isinstance(geostruct,str):
#             geostruct = read_struct_file(geostruct)
#         assert isinstance(geostruct,GeoStruct),"need a GeoStruct, not {0}".\
#             format(type(geostruct))
#         self.geostruct = geostruct
#         if isinstance(point_data,str):
#             point_data = pp_file_to_dataframe(point_data)
#         assert isinstance(point_data,pd.DataFrame)
#         assert 'name' in point_data.columns,"point_data missing 'name'"
#         assert 'x' in point_data.columns, "point_data missing 'x'"
#         assert 'y' in point_data.columns, "point_data missing 'y'"
#         assert "value" in point_data.columns,"point_data missing 'value'"
#         self.point_data = point_data
#         self.point_data.index = self.point_data.name
#         self.interp_data = None
#         self.spatial_reference = None
#         #X, Y = np.meshgrid(point_data.x,point_data.y)
#         #self.point_data_dist = pd.DataFrame(data=np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2),
#         #                                    index=point_data.name,columns=point_data.name)
#         self.point_cov_df = self.geostruct.covariance_matrix(point_data.x,
#                                                             point_data.y,
#                                                             point_data.name).to_dataframe()
#         #for name in self.point_cov_df.index:
#         #    self.point_cov_df.loc[name,name] -= self.geostruct.nugget
#
#
#     def estimate_grid(self,spatial_reference,zone_array=None,minpts_interp=1,
#                           maxpts_interp=20,search_radius=1.0e+10,verbose=False,
#                           var_filename=None):
#
#         self.spatial_reference = spatial_reference
#         self.interp_data = None
#         #assert isinstance(spatial_reference,SpatialReference)
#         try:
#             x = self.spatial_reference.xcentergrid.copy()
#             y = self.spatial_reference.ycentergrid.copy()
#         except Exception as e:
#             raise Exception("spatial_reference does not have proper attributes:{0}"\
#                             .format(str(e)))
#
#         if var_filename is not None:
#             arr = np.zeros((self.spatial_reference.nrow,
#                             self.spatial_reference.ncol)) - 1.0e+30
#
#         df = self.estimate(x.ravel(),y.ravel(),
#                            minpts_interp=minpts_interp,
#                            maxpts_interp=maxpts_interp,
#                            search_radius=search_radius,
#                            verbose=verbose)
#         if var_filename is not None:
#             arr = df.err_var.values.reshape(x.shape)
#             np.savetxt(var_filename,arr,fmt="%15.6E")
#         arr = df.estimate.values.reshape(x.shape)
#         return arr
#
#
#     def estimate(self,x,y,minpts_interp=1,maxpts_interp=20,
#                      search_radius=1.0e+10,verbose=False):
#         assert len(x) == len(y)
#
#         # find the point data to use for each interp point
#         sqradius = search_radius**2
#         df = pd.DataFrame(data={'x':x,'y':y})
#         inames,idist,ifacts,err_var = [],[],[],[]
#         estimates = []
#         sill = self.geostruct.sill
#         pt_data = self.point_data
#         ptx_array = pt_data.x.values
#         pty_array = pt_data.y.values
#         ptnames = pt_data.name.values
#         #if verbose:
#         print("starting interp point loop for {0} points".format(df.shape[0]))
#         start_loop = datetime.now()
#         for idx,(ix,iy) in enumerate(zip(df.x,df.y)):
#             if np.isnan(ix) or np.isnan(iy): #if nans, skip
#                 inames.append([])
#                 idist.append([])
#                 ifacts.append([])
#                 err_var.append(np.NaN)
#                 continue
#             if verbose:
#                 istart = datetime.now()
#                 print("processing interp point:{0} of {1}".format(idx,df.shape[0]))
#             # if verbose == 2:
#             #     start = datetime.now()
#             #     print("calc ipoint dist...",end='')
#
#             #  calc dist from this interp point to all point data...slow
#             dist = pd.Series((ptx_array-ix)**2 + (pty_array-iy)**2,ptnames)
#             dist.sort_values(inplace=True)
#             dist = dist.loc[dist <= sqradius]
#
#             # if too few points were found, skip
#             if len(dist) < minpts_interp:
#                 inames.append([])
#                 idist.append([])
#                 ifacts.append([])
#                 err_var.append(sill)
#                 estimates.append(np.NaN)
#                 continue
#
#             # only the maxpts_interp points
#             dist = dist.iloc[:maxpts_interp].apply(np.sqrt)
#             pt_names = dist.index.values
#             # if one of the points is super close, just use it and skip
#             if dist.min() <= EPSILON:
#                 ifacts.append([1.0])
#                 idist.append([EPSILON])
#                 inames.append([dist.idxmin()])
#                 err_var.append(self.geostruct.nugget)
#                 estimates.append(self.point_data.loc[dist.idxmin(),"value"])
#                 continue
#             # if verbose == 2:
#             #     td = (datetime.now()-start).total_seconds()
#             #     print("...took {0}".format(td))
#             #     start = datetime.now()
#             #     print("extracting pt cov...",end='')
#
#             #vextract the point-to-point covariance matrix
#             point_cov = self.point_cov_df.loc[pt_names,pt_names]
#             # if verbose == 2:
#             #     td = (datetime.now()-start).total_seconds()
#             #     print("...took {0}".format(td))
#             #     print("forming ipt-to-point cov...",end='')
#
#             # calc the interp point to points covariance
#             ptx = self.point_data.loc[pt_names,"x"]
#             pty = self.point_data.loc[pt_names,"y"]
#             interp_cov = self.geostruct.covariance_points(ix,iy,ptx,pty)
#
#             if verbose == 2:
#                 td = (datetime.now()-start).total_seconds()
#                 print("...took {0}".format(td))
#                 print("forming lin alg components...",end='')
#
#             # form the linear algebra parts and solve
#             d = len(pt_names) + 3 # +1 for lagrange mult + 2 for x and y coords
#             npts = len(pt_names)
#             A = np.ones((d,d))
#             A[:npts,:npts] = point_cov.values
#             A[npts,npts] = 0.0 #unbiaised constraint
#             A[-2,:npts] = ptx #x coords for linear trend
#             A[:npts,-2] = ptx
#             A[-1,:npts] = pty #y coords for linear trend
#             A[:npts,-1] = pty
#             A[npts:,npts:] = 0
#             print(A)
#             rhs = np.ones((d,1))
#             rhs[:npts,0] = interp_cov
#             rhs[-2,0] = ix
#             rhs[-1,0] = iy
#             # if verbose == 2:
#             #     td = (datetime.now()-start).total_seconds()
#             #     print("...took {0}".format(td))
#             #     print("solving...",end='')
#             # # solve
#             facs = np.linalg.solve(A,rhs)
#             assert len(facs) - 3 == len(dist)
#             estimate = facs[-3] + (ix * facs[-2]) + (iy * facs[-1])
#             estimates.append(estimate[0])
#             err_var.append(float(sill + facs[-1] - sum([f*c for f,c in zip(facs[:-1],interp_cov)])))
#             inames.append(pt_names)
#
#             idist.append(dist.values)
#             ifacts.append(facs[:-1,0])
#             # if verbose == 2:
#             #     td = (datetime.now()-start).total_seconds()
#             #     print("...took {0}".format(td))
#             if verbose:
#                 td = (datetime.now()-istart).total_seconds()
#                 print("point took {0}".format(td))
#         df["idist"] = idist
#         df["inames"] = inames
#         df["ifacts"] = ifacts
#         df["err_var"] = err_var
#         df["estimate"] = estimates
#         self.interp_data = df
#         td = (datetime.now() - start_loop).total_seconds()
#         print("took {0}".format(td))
#         return df


class OrdinaryKrige(object):
    """ Ordinary Kriging using Pandas and Numpy.

    Parameters
    ----------
    geostruct : (GeoStruct)
        a pyemu.geostats.GeoStruct to use for the kriging
    point_data : (pandas.DataFrame)
        the conditioning points to use for kriging.  point_data must contain
        columns "name", "x", "y".

    Note
    ----
    if point_data is an str, then it is assumed to be a pilot points file
    and is loaded as such using pyemu.pp_utils.pp_file_to_dataframe()

    If zoned interpolation is used for grid-based interpolation, then
    point_data must also contain a "zone" column


    Example
    -------
    ``>>>import pyemu``

    ``>>>v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)``

    ``>>>gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)``

    ``>>>pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")``

    ``>>>ok = pyemu.utils.geostats.OrdinaryKrige(gs,pp_df)``
    """

    def __init__(self,geostruct,point_data):
        if isinstance(geostruct,str):
            geostruct = read_struct_file(geostruct)
        assert isinstance(geostruct,GeoStruct),"need a GeoStruct, not {0}".\
            format(type(geostruct))
        self.geostruct = geostruct
        if isinstance(point_data,str):
            point_data = pp_file_to_dataframe(point_data)
        assert isinstance(point_data,pd.DataFrame)
        assert 'name' in point_data.columns,"point_data missing 'name'"
        assert 'x' in point_data.columns, "point_data missing 'x'"
        assert 'y' in point_data.columns, "point_data missing 'y'"
        #check for duplicates in point data
        unique_name = point_data.name.unique()
        if len(unique_name) != point_data.shape[0]:
            warnings.warn("duplicates detected in point_data..attempting to rectify")
            ux_std = point_data.groupby(point_data.name).std()['x']
            if ux_std.max() > 0.0:
                raise Exception("duplicate point_info entries with name {0} have different x values"
                                .format(uname))
            uy_std = point_data.groupby(point_data.name).std()['y']
            if uy_std.max() > 0.0:
                raise Exception("duplicate point_info entries with name {0} have different y values"
                                .format(uname))

            self.point_data = point_data.drop_duplicates(subset=["name"])
        else:
            self.point_data = point_data.copy()
        self.point_data.index = self.point_data.name
        self.check_point_data_dist()
        self.interp_data = None
        self.spatial_reference = None
        #X, Y = np.meshgrid(point_data.x,point_data.y)
        #self.point_data_dist = pd.DataFrame(data=np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2),
        #                                    index=point_data.name,columns=point_data.name)
        self.point_cov_df = self.geostruct.covariance_matrix(self.point_data.x,
                                                            self.point_data.y,
                                                            self.point_data.name).to_dataframe()
        #for name in self.point_cov_df.index:
        #    self.point_cov_df.loc[name,name] -= self.geostruct.nugget

    def check_point_data_dist(self, rectify=False):
        """ check for point_data entries that are closer than
        EPSILON distance - this will cause a singular kriging matrix.

        Parameters
        ----------
        rectify : (boolean)
            flag to fix the problems with point_data
            by dropping additional points that are
            closer than EPSILON distance.  Default is False

        Note
        ----
        this method will issue warnings for points that are closer
        than EPSILON distance

        """

        ptx_array = self.point_data.x.values
        pty_array = self.point_data.y.values
        ptnames = self.point_data.name.values
        drop = []
        for i in range(self.point_data.shape[0]):
            ix,iy,iname = ptx_array[i],pty_array[i],ptnames[i]
            dist = pd.Series((ptx_array[i+1:] - ix) ** 2 + (pty_array[i+1:] - iy) ** 2, ptnames[i+1:])
            if dist.min() < EPSILON**2:
                print(iname,ix,iy)
                warnings.warn("points {0} and {1} are too close. This will cause a singular kriging matrix ".\
                              format(iname,dist.idxmin()))
                drop_idxs = dist.loc[dist<=EPSILON**2]
                drop.extend([pt for pt in list(drop_idxs.index) if pt not in drop])
        if rectify and len(drop) > 0:
            print("rectifying point data by removing the following points: {0}".format(','.join(drop)))
            print(self.point_data.shape)
            self.point_data = self.point_data.loc[self.point_data.index.map(lambda x: x not in drop),:]
            print(self.point_data.shape)

    #def prep_for_ppk2fac(self,struct_file="structure.dat",pp_file="points.dat",):
    #    pass



    def calc_factors_grid(self,spatial_reference,zone_array=None,minpts_interp=1,
                          maxpts_interp=20,search_radius=1.0e+10,verbose=False,
                          var_filename=None, forgive=False):
        """ calculate kriging factors (weights) for a structured grid.

        Parameters
        ----------
        spatial_reference : (flopy.utils.reference.SpatialReference)
            a spatial reference that describes the orientation and
            spatail projection of the the structured grid
        zone_array : (numpy.ndarray)
            an integer array of zones to use for kriging.  If not None,
            then point_data must also contain a "zone" column.  point_data
            entries with a zone value not found in zone_array will be skipped.
            If None, then all point_data will (potentially) be used for
            interpolating each grid node. Default is None
        minpts_interp : (int)
            minimum number of point_data entires to use for interpolation at
            a given grid node.  grid nodes with less than minpts_interp
            point_data found will be skipped (assigned np.NaN).  Defaut is 1
        maxpts_interp : (int)
            maximum number of point_data entries to use for interpolation at
            a given grid node.  A larger maxpts_interp will yield "smoother"
            interplation, but using a large maxpts_interp will slow the
            (already) slow kriging solution process and may lead to
            memory errors. Default is 20.
        search_radius : (float)
            the size of the region around a given grid node to search for
            point_data entries. Default is 1.0e+10
        verbose : (boolean)
            a flag to  echo process to stdout during the interpolatino process.
            Default is False
        var_filename : (str)
            a filename to save the kriging variance for each interpolated grid node.
            Default is None.
        forgive : (boolean)
            flag to continue if inversion of the kriging matrix failes at one or more
            grid nodes.  Inversion usually fails if the kriging matrix is singular,
            resulting from point_data entries closer than EPSILON distance.  If True,
            warnings are issued for each failed inversion.  If False, an exception
            is raised for failed matrix inversion.

        Returns
        -------
        df : pandas.DataFrame
            a dataframe with information summarizing the ordinary kriging
            process for each grid node

        Note
        ----
        this method calls OrdinaryKrige.calc_factors()


        Example
        -------
        ``>>>import flopy``

        ``>>>import pyemu``

        ``>>>v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)``

        ``>>>gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)``

        ``>>>pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")``

        ``>>>ok = pyemu.utils.geostats.OrdinaryKrige(gs,pp_df)``

        ``>>>m = flopy.modflow.Modflow.load("mymodel.nam")``

        ``>>>df = ok.calc_factors_grid(m.sr,zone_array=m.bas6.ibound[0].array,``

        ``>>>                          var_filename="ok_var.dat")``

        ``>>>ok.to_grid_factor_file("factors.dat")``

        """

        self.spatial_reference = spatial_reference
        self.interp_data = None
        #assert isinstance(spatial_reference,SpatialReference)
        try:
            x = self.spatial_reference.xcentergrid.copy()
            y = self.spatial_reference.ycentergrid.copy()
        except Exception as e:
            raise Exception("spatial_reference does not have proper attributes:{0}"\
                            .format(str(e)))

        if var_filename is not None:
                arr = np.zeros((self.spatial_reference.nrow,
                                self.spatial_reference.ncol)) - 1.0e+30

        # the simple case of no zone array: ignore point_data zones
        if zone_array is None:
            df = self.calc_factors(x.ravel(),y.ravel(),
                               minpts_interp=minpts_interp,
                               maxpts_interp=maxpts_interp,
                               search_radius=search_radius,
                               verbose=verbose, forgive=forgive)
            if var_filename is not None:
                arr = df.err_var.values.reshape(x.shape)
                np.savetxt(var_filename,arr,fmt="%15.6E")

        if zone_array is not None:
            assert zone_array.shape == x.shape
            if "zone" not in self.point_data.columns:
                warnings.warn("'zone' columns not in point_data, assigning generic zone")
                self.point_data.loc[:,"zone"] = 1
            pt_data_zones = self.point_data.zone.unique()
            dfs = []
            for pt_data_zone in pt_data_zones:
                if pt_data_zone not in zone_array:
                    warnings.warn("pt zone {0} not in zone array {1}, skipping".\
                                  format(pt_data_zone,np.unique(zone_array)))
                    continue
                xzone,yzone = x.copy(),y.copy()
                xzone[zone_array!=pt_data_zone] = np.NaN
                yzone[zone_array!=pt_data_zone] = np.NaN
                df = self.calc_factors(xzone.ravel(),yzone.ravel(),
                                       minpts_interp=minpts_interp,
                                       maxpts_interp=maxpts_interp,
                                       search_radius=search_radius,
                                       verbose=verbose,pt_zone=pt_data_zone,
                                       forgive=forgive)
                dfs.append(df)
                if var_filename is not None:
                    a = df.err_var.values.reshape(x.shape)
                    na_idx = np.isfinite(a)
                    arr[na_idx] = a[na_idx]
            if self.interp_data is None or self.interp_data.dropna().shape[0] == 0:
                raise Exception("no interpolation took place...something is wrong")
            df = pd.concat(dfs)
        if var_filename is not None:
            np.savetxt(var_filename,arr,fmt="%15.6E")
        return df

    def calc_factors(self,x,y,minpts_interp=1,maxpts_interp=20,
                     search_radius=1.0e+10,verbose=False,
                     pt_zone=None,forgive=False):
        """ calculate ordinary kriging factors (weights) for the points
        represented by arguments x and y

        Parameters
        ----------
        x : (iterable of floats)
            x-coordinates to calculate kriging factors for
        y : (iterable of floats)
            y-coordinates to calculate kriging factors for
        minpts_interp : (int)
            minimum number of point_data entires to use for interpolation at
            a given x,y interplation point.  interpolation points with less
            than minpts_interp point_data found will be skipped
            (assigned np.NaN).  Defaut is 1
        maxpts_interp : (int)
            maximum number of point_data entries to use for interpolation at
            a given x,y interpolation point.  A larger maxpts_interp will
            yield "smoother" interplation, but using a large maxpts_interp
            will slow the (already) slow kriging solution process and may
            lead to memory errors. Default is 20.
        search_radius : (float)
            the size of the region around a given x,y interpolation point to search for
            point_data entries. Default is 1.0e+10
        verbose : (boolean)
            a flag to  echo process to stdout during the interpolatino process.
            Default is False
        forgive : (boolean)
            flag to continue if inversion of the kriging matrix failes at one or more
            interpolation points.  Inversion usually fails if the kriging matrix is singular,
            resulting from point_data entries closer than EPSILON distance.  If True,
            warnings are issued for each failed inversion.  If False, an exception
            is raised for failed matrix inversion.

        Returns
        -------
        df : pandas.DataFrame
            a dataframe with information summarizing the ordinary kriging
            process for each interpolation points


        """

        assert len(x) == len(y)

        # find the point data to use for each interp point
        sqradius = search_radius**2
        df = pd.DataFrame(data={'x':x,'y':y})
        inames,idist,ifacts,err_var = [],[],[],[]
        sill = self.geostruct.sill
        if pt_zone is None:
            ptx_array = self.point_data.x.values
            pty_array = self.point_data.y.values
            ptnames = self.point_data.name.values
        else:
            pt_data = self.point_data
            ptx_array = pt_data.loc[pt_data.zone==pt_zone,"x"].values
            pty_array = pt_data.loc[pt_data.zone==pt_zone,"y"].values
            ptnames = pt_data.loc[pt_data.zone==pt_zone,"name"].values
        #if verbose:
        print("starting interp point loop for {0} points".format(df.shape[0]))
        start_loop = datetime.now()
        for idx,(ix,iy) in enumerate(zip(df.x,df.y)):
            if np.isnan(ix) or np.isnan(iy): #if nans, skip
                inames.append([])
                idist.append([])
                ifacts.append([])
                err_var.append(np.NaN)
                continue
            if verbose:
                istart = datetime.now()
                print("processing interp point:{0} of {1}".format(idx,df.shape[0]))
            # if verbose == 2:
            #     start = datetime.now()
            #     print("calc ipoint dist...",end='')

            #  calc dist from this interp point to all point data...slow
            dist = pd.Series((ptx_array-ix)**2 + (pty_array-iy)**2,ptnames)
            dist.sort_values(inplace=True)
            dist = dist.loc[dist <= sqradius]

            # if too few points were found, skip
            if len(dist) < minpts_interp:
                inames.append([])
                idist.append([])
                ifacts.append([])
                err_var.append(sill)
                continue

            # only the maxpts_interp points
            dist = dist.iloc[:maxpts_interp].apply(np.sqrt)
            pt_names = dist.index.values
            # if one of the points is super close, just use it and skip
            if dist.min() <= EPSILON:
                ifacts.append([1.0])
                idist.append([EPSILON])
                inames.append([dist.idxmin()])
                err_var.append(self.geostruct.nugget)
                continue
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            #     start = datetime.now()
            #     print("extracting pt cov...",end='')

            #vextract the point-to-point covariance matrix
            point_cov = self.point_cov_df.loc[pt_names,pt_names]
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            #     print("forming ipt-to-point cov...",end='')

            # calc the interp point to points covariance
            interp_cov = self.geostruct.covariance_points(ix,iy,self.point_data.loc[pt_names,"x"],
                                                          self.point_data.loc[pt_names,"y"])

            if verbose == 2:
                td = (datetime.now()-start).total_seconds()
                print("...took {0} seconds".format(td))
                print("forming lin alg components...",end='')

            # form the linear algebra parts and solve
            d = len(pt_names) + 1 # +1 for lagrange mult
            A = np.ones((d,d))
            A[:-1,:-1] = point_cov.values
            A[-1,-1] = 0.0 #unbiaised constraint
            rhs = np.ones((d,1))
            rhs[:-1,0] = interp_cov
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            #     print("solving...",end='')
            # # solve
            try:
                facs = np.linalg.solve(A,rhs)
            except Exception as e:
                print("error solving for factors: {0}".format(str(e)))
                print("point:",ix,iy)
                print("dist:",dist)
                print("A:", A)
                print("rhs:", rhs)
                if forgive:
                    inames.append([])
                    idist.append([])
                    ifacts.append([])
                    err_var.append(np.NaN)
                    continue
                else:
                    raise Exception("error solving for factors:{0}".format(str(e)))
            assert len(facs) - 1 == len(dist)

            err_var.append(float(sill + facs[-1] - sum([f*c for f,c in zip(facs[:-1],interp_cov)])))
            inames.append(pt_names)

            idist.append(dist.values)
            ifacts.append(facs[:-1,0])
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            if verbose:
                td = (datetime.now()-istart).total_seconds()
                print("point took {0}".format(td))
        df["idist"] = idist
        df["inames"] = inames
        df["ifacts"] = ifacts
        df["err_var"] = err_var
        if pt_zone is None:
            self.interp_data = df
        else:
            if self.interp_data is None:
                self.interp_data = df
            else:
                self.interp_data = self.interp_data.append(df)
        td = (datetime.now() - start_loop).total_seconds()
        print("took {0} seconds".format(td))
        return df

    def to_grid_factors_file(self, filename,points_file="points.junk",
                             zone_file="zone.junk"):
        """ write a grid-based PEST-style factors file.  This file can be used with
        the fac2real() method to write an interpolated structured array

        Parameters
        ----------
        filename : (str)
            factor filename
        points_file : (str)
            points filename to add to the header of the factors file.
            Not used by fac2real() method.  Default is "points.junk"
        zone_file : (str)
            zone filename to add to the header of the factors file.
            Not used by fac2real() method.  Default is "zone.junk"

        Note
        ----
        this method should be called after OrdinaryKirge.calc_factors_grid()

        """
        if self.interp_data is None:
            raise Exception("ok.interp_data is None, must call calc_factors_grid() first")
        if self.spatial_reference is None:
            raise Exception("ok.spatial_reference is None, must call calc_factors_grid() first")
        with open(filename, 'w') as f:
            f.write(points_file + '\n')
            f.write(zone_file + '\n')
            f.write("{0} {1}\n".format(self.spatial_reference.ncol, self.spatial_reference.nrow))
            f.write("{0}\n".format(self.point_data.shape[0]))
            [f.write("{0}\n".format(name)) for name in self.point_data.name]
            t = 0
            if self.geostruct.transform == "log":
                t = 1
            pt_names = list(self.point_data.name)
            for idx,names,facts in zip(self.interp_data.index,self.interp_data.inames,self.interp_data.ifacts):
                if len(facts) == 0:
                    continue
                n_idxs = [pt_names.index(name) for name in names]
                f.write("{0} {1} {2} {3:8.5e} ".format(idx+1, t, len(names), 0.0))
                [f.write("{0} {1:12.8g} ".format(i+1, w)) for i, w in zip(n_idxs, facts)]
                f.write("\n")


class Vario2d(object):
    """base class for 2-D variograms.

    Parameters
    ----------
    contribution : (float)
        sill of the variogram
    a : (float)
        (practical) range of correlation
    anisotropy : (float)
        Anisotropy ratio. Default is 1.0
    bearing : (float)
        angle in degrees East of North corresponding to anisotropy ellipse.
        Default is 0.0
    name : (str)
        name of the variogram.  Default is "var1"

    Returns
    -------
    Vario2d : Vario2d

    Note
    ----
    This base class should not be instantiated directly as it does not implement
    an h_function() method.

    """

    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        self.name = name
        self.epsilon = EPSILON
        self.contribution = float(contribution)
        assert self.contribution > 0.0
        self.a = float(a)
        assert self.a > 0.0
        self.anisotropy = float(anisotropy)
        assert self.anisotropy > 0.0
        self.bearing = float(bearing)

    def to_struct_file(self, f):
        """ write the Vario2d to a PEST-style structure file

        Parameters
        ----------
        f : (str or file handle)
            item to write to

        """
        if isinstance(f, str):
            f = open(f,'w')
        f.write("VARIOGRAM {0}\n".format(self.name))
        f.write("  VARTYPE {0}\n".format(self.vartype))
        f.write("  A {0}\n".format(self.a))
        f.write("  ANISOTROPY {0}\n".format(self.anisotropy))
        f.write("  BEARING {0}\n".format(self.bearing))
        f.write("END VARIOGRAM\n\n")

    @property
    def bearing_rads(self):
        """ get the bearing of the Vario2d in radians

        Returns
        -------
        bearing_rads : float
            the Vario2d bearing in radians
        """
        return (np.pi / 180.0 ) * (90.0 - self.bearing)

    @property
    def rotation_coefs(self):
        """ get the rotation coefficents in radians

        Returns
        -------
        rotation_coefs : list
            the rotation coefficients implied by Vario2d.bearing


        """
        return [np.cos(self.bearing_rads),
                np.sin(self.bearing_rads),
                -1.0*np.sin(self.bearing_rads),
                np.cos(self.bearing_rads)]

    def inv_h(self,h):
        """ the inverse of the h_function.  Used for plotting

        Parameters
        ----------
        h : (float)
            the value of h_function to invert

        Returns
        -------
        inv_h : float
            the inverse of h

        """
        return self.contribution - self._h_function(h)

    def plot(self,**kwargs):
        """ get a cheap plot of the Vario2d

        Parameters
        ----------
        **kwargs : (dict)
            keyword arguments to use for plotting

        Returns
        -------
        ax : matplotlib.pyplot.axis

        Note
        ----
        optional arguments in kwargs include
        "ax" (existing matplotlib.pyplot.axis).  Other
        kwargs are passed to matplotlib.pyplot.plot()

        """
        import matplotlib.pyplot as plt
        ax = kwargs.pop("ax",plt.subplot(111))
        x = np.linspace(0,self.a*3,100)
        y = self.inv_h(x)
        ax.set_xlabel("distance")
        ax.set_ylabel("$\gamma$")
        ax.plot(x,y,**kwargs)
        return ax


    def add_sparse_covariance_matrix(self,x,y,names,iidx,jidx,data):

        """build a pyemu.SparseMatrix instance implied by Vario2d

        Parameters
        ----------
        x : (iterable of floats)
            x-coordinate locations
        y : (iterable of floats)
            y-coordinate locations
        names : (iterable of str)
            names of locations. If None, cov must not be None
        iidx : 1-D ndarray
            i row indices
        jidx : 1-D ndarray
            j col indices
        data : 1-D ndarray
            nonzero entries


        Returns
        -------
        None

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]


        assert x.shape[0] == len(names)
        #     c = np.zeros((len(names), len(names)))
        #     np.fill_diagonal(c, self.contribution)
        #     cov = Cov(x=c, names=names)
        # elif cov is not None:
        #     assert cov.shape[0] == x.shape[0]
        #     names = cov.row_names
        #     c = np.zeros((len(names), 1)) + self.contribution
        #     cont = Cov(x=c, names=names, isdiagonal=True)
        #     cov += cont
        #
        # else:
        #     raise Exception("Vario2d.covariance_matrix() requires either" +
        #                     "names or cov arg")
        # rc = self.rotation_coefs
        for i,name in enumerate(names):
            iidx.append(i)
            jidx.append(i)
            data.append(self.contribution)

        for i1, (n1, x1, y1) in enumerate(zip(names, x, y)):
            dx = x1 - x[i1 + 1:]
            dy = y1 - y[i1 + 1:]
            dxx, dyy = self._apply_rotation(dx, dy)
            h = np.sqrt(dxx * dxx + dyy * dyy)

            h[h < 0.0] = 0.0
            cv = self._h_function(h)
            if np.any(np.isnan(cv)):
                raise Exception("nans in cv for i1 {0}".format(i1))
            #cv[h>self.a] = 0.0
            j = list(np.arange(i1+1,x.shape[0]))
            i = [i1] * len(j)
            iidx.extend(i)
            jidx.extend(j)
            data.extend(list(cv))
            # replicate across the diagonal
            iidx.extend(j)
            jidx.extend(i)
            data.extend(list(cv))



    def covariance_matrix(self,x,y,names=None,cov=None):
        """build a pyemu.Cov instance implied by Vario2d

        Parameters
        ----------
        x : (iterable of floats)
            x-coordinate locations
        y : (iterable of floats)
            y-coordinate locations
        names : (iterable of str)
            names of locations. If None, cov must not be None
        cov : (pyemu.Cov)
            an existing Cov instance.  Vario2d contribution is added to cov

        Returns
        -------
        cov : pyemu.Cov

        Note
        ----
        either names or cov must not be None.

        """
        if not isinstance(x,np.ndarray):
            x = np.array(x)
        if not isinstance(y,np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]

        if names is not None:
            assert x.shape[0] == len(names)
            c = np.zeros((len(names),len(names)))
            np.fill_diagonal(c,self.contribution)
            cov = Cov(x=c,names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
            c = np.zeros((len(names),1)) + self.contribution
            cont = Cov(x=c,names=names,isdiagonal=True)
            cov += cont

        else:
            raise Exception("Vario2d.covariance_matrix() requires either" +
                            "names or cov arg")
        rc = self.rotation_coefs
        for i1,(n1,x1,y1) in enumerate(zip(names,x,y)):
            dx = x1 - x[i1+1:]
            dy = y1 - y[i1+1:]
            dxx,dyy = self._apply_rotation(dx,dy)
            h = np.sqrt(dxx*dxx + dyy*dyy)

            h[h<0.0] = 0.0
            h = self._h_function(h)
            if np.any(np.isnan(h)):
                raise Exception("nans in h for i1 {0}".format(i1))
            cov.x[i1,i1+1:] += h
        for i in range(len(names)):
            cov.x[i+1:,i] = cov.x[i,i+1:]
        return cov

    def _apply_rotation(self,dx,dy):
        """ private method to rotate points
        according to Vario2d.bearing and Vario2d.anisotropy

        Parameters
        ----------
        dx : (float or numpy.ndarray)
            x-coordinates to rotate
        dy : (float or numpy.ndarray)
            y-coordinates to rotate

        Returns
        -------
            dxx : (float or numpy.ndarray)
                rotated x-coordinates
            dyy : (float or numpy.ndarray)
                rotated y-coordinates

        """
        if self.anisotropy == 1.0:
            return dx,dy
        rcoefs = self.rotation_coefs
        dxx = (dx * rcoefs[0]) +\
             (dy * rcoefs[1])
        dyy = ((dx * rcoefs[2]) +\
             (dy * rcoefs[3])) *\
             self.anisotropy
        return dxx,dyy

    def covariance_points(self,x0,y0,xother,yother):
        """ get the covariance between base point x0,y0 and
        other points xother,yother implied by Vario2d

        Parameters
        ----------
        x0 : (float)
            x-coordinate of base point
        y0 : (float)
            y-coordinate of base point
        xother : (float or numpy.ndarray)
            x-coordinates of other points
        yother : (float or numpy.ndarray)
            y-coordinates of other points

        Returns
        -------
        cov : numpy.ndarray
            covariance between base point and other points implied by
            Vario2d.

        Note
        ----
        len(cov) = len(xother) = len(yother)

        """
        dx = x0 - xother
        dy = y0 - yother
        dxx,dyy = self._apply_rotation(dx,dy)
        h = np.sqrt(dxx*dxx + dyy*dyy)
        return self._h_function(h)

    def covariance(self,pt0,pt1):
        """ get the covarince between two points implied by Vario2d

        Parameters
        ----------
        pt0 : (iterable of len 2)
            first point x and y
        pt1 : (iterable of len 2)
            second point x and y

        Returns
        -------
        cov : float
            covariance between pt0 and pt1

        """

        x = np.array([pt0[0],pt1[0]])
        y = np.array([pt0[1],pt1[1]])
        names = ["n1","n2"]
        return self.covariance_matrix(x,y,names=names).x[0,1]


    def __str__(self):
        """ get the str representation of Vario2d

        Returns
        -------
        str : str
        """
        s = "name:{0},contribution:{1},a:{2},anisotropy:{3},bearing:{4}\n".\
            format(self.name,self.contribution,self.a,\
                   self.anisotropy,self.bearing)
        return s

class ExpVario(Vario2d):
    """ Exponetial variogram derived type

    Parameters
    ----------
    contribution : (float)
        sill of the variogram
    a : (float)
        (practical) range of correlation
    anisotropy : (float)
        Anisotropy ratio. Default is 1.0
    bearing : (float)
        angle in degrees East of North corresponding to anisotropy ellipse.
        Default is 0.0
    name : (str)
        name of the variogram.  Default is "var1"

    Returns
    -------
    ExpVario : ExpVario

    Example
    -------
    ``>>>import pyemu``

    ``>>>v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)``

    """
    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        super(ExpVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)
        self.vartype = 2

    def _h_function(self,h):
        """ private method exponential variogram "h" function

        Parameters
        ----------
        h : (float or numpy.ndarray)
            distance(s)

        Returns
        -------
        h_function : float or numpy.ndarray
            the value of the "h" function implied by the ExpVario

        """
        return self.contribution * np.exp(-1.0 * h / self.a)

class GauVario(Vario2d):
    """Gaussian variogram derived type

    Parameters
    ----------
    contribution : (float)
        sill of the variogram
    a : (float)
        (practical) range of correlation
    anisotropy : (float)
        Anisotropy ratio. Default is 1.0
    bearing : (float)
        angle in degrees East of North corresponding to anisotropy ellipse.
        Default is 0.0
    name : (str)
        name of the variogram.  Default is "var1"

    Returns
    -------
    GauVario : GauVario

    Note
    ----
    the Gaussian variogram can be unstable (not invertible) for long ranges.

    Example
    -------
    ``>>>import pyemu``

    ``>>>v = pyemu.utils.geostats.GauVario(a=1000,contribution=1.0)``

    """

    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        super(GauVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)
        self.vartype = 3

    def _h_function(self,h):
        """ private method for the gaussian variogram "h" function

        Parameters
        ----------
        h : (float or numpy.ndarray)
            distance(s)

        Returns
        -------
        h_function : float or numpy.ndarray
            the value of the "h" function implied by the GauVario

        """

        hh = -1.0 * (h * h) / (self.a * self.a)
        return self.contribution * np.exp(hh)

class SphVario(Vario2d):
    """Spherical variogram derived type

    Parameters
    ----------
    contribution : (float)
        sill of the variogram
    a : (float)
        (practical) range of correlation
    anisotropy : (float)
        Anisotropy ratio. Default is 1.0
    bearing : (float)
        angle in degrees East of North corresponding to anisotropy ellipse.
        Default is 0.0
    name : (str)
        name of the variogram.  Default is "var1"

    Returns
    -------
    SphVario : SphVario

    Example
    -------
    ``>>>import pyemu``

    ``>>>v = pyemu.utils.geostats.SphVario(a=1000,contribution=1.0)``

    """

    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        super(SphVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)
        self.vartype = 1

    def _h_function(self,h):
        """ private method for the spherical variogram "h" function

        Parameters
        ----------
        h : (float or numpy.ndarray)
            distance(s)

        Returns
        -------
        h_function : float or numpy.ndarray
            the value of the "h" function implied by the SphVario

        """

        hh = h / self.a
        h = self.contribution * (1.0 - (hh * (1.5 - (0.5 * hh * hh))))
        h[hh > 1.0] = 0.0
        return h
        # try:
        #     h[hh < 1.0] = 0.0
        #
        # except TypeError:
        #     if hh > 0.0:
        #         h = 0.0
        #return h
        # if hh < 1.0:
        #     return self.contribution * (1.0 - (hh * (1.5 - (0.5 * hh * hh))))
        # else:
        #     return 0.0





def read_struct_file(struct_file,return_type=GeoStruct):
    """read an existing PEST-type structure file into a GeoStruct instance

    Parameters
    ----------
    struct_file : (str)
        existing pest-type structure file
    return_type :  (object)
        the instance type to return.  Default is GeoStruct

    Returns
    -------
    GeoStruct : list or GeoStruct

    Note
    ----
    if only on structure is listed in struct_file, then return type
    is GeoStruct.  Otherwise, return type is a list of GeoStruct

    Example
    -------
    ``>>>import pyemu``

    ``>>>gs = pyemu.utils.geostats.reads_struct_file("struct.dat")``


    """

    VARTYPE = {1:SphVario,2:ExpVario,3:GauVario,4:None}
    assert os.path.exists(struct_file)
    structures = []
    variograms = []
    with open(struct_file,'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            line = line.strip().lower()
            if line.startswith("structure"):
                name = line.strip().split()[1]
                nugget,transform,variogram_info = _read_structure_attributes(f)
                s = return_type(nugget=nugget,transform=transform,name=name)
                s.variogram_info = variogram_info
                # not sure what is going on, but if I don't copy s here,
                # all the structures end up sharing all the variograms later
                structures.append(copy.deepcopy(s))
            elif line.startswith("variogram"):
                name = line.strip().split()[1].lower()
                vartype,bearing,a,anisotropy = _read_variogram(f)
                if name in variogram_info:
                    v = VARTYPE[vartype](variogram_info[name],a,anisotropy=anisotropy,
                                         bearing=bearing,name=name)
                    variograms.append(v)

    for i,st in enumerate(structures):
        for vname in st.variogram_info:
            vfound = None
            for v in variograms:
                if v.name == vname:
                    vfound = v
                    break
            if vfound is None:
                raise Exception("variogram {0} not found for structure {1}".\
                                format(vname,s.name))

            st.variograms.append(vfound)
    if len(structures) == 1:
        return structures[0]
    return structures



def _read_variogram(f):
    """Function to instantiate a Vario2d from a PEST-style structure file

    Parameters
    ----------
    f : (file handle)
        file handle opened for reading

    Returns
    -------
    Vario2d : Vario2d
        Vario2d derived type

    """

    line = ''
    vartype = None
    bearing = 0.0
    a = None
    anisotropy = 1.0
    while "end variogram" not in line:
        line = f.readline()
        if line == '':
            raise Exception("EOF while read variogram")
        line = line.strip().lower().split()
        if line[0].startswith('#'):
            continue
        if line[0] == "vartype":
            vartype = int(line[1])
        elif line[0] == "bearing":
            bearing = float(line[1])
        elif line[0] == "a":
            a = float(line[1])
        elif line[0] == "anisotropy":
            anisotropy = float(line[1])
        elif line[0] == "end":
            break
        else:
            raise Exception("unrecognized arg in variogram:{0}".format(line[0]))
    return vartype,bearing,a,anisotropy


def _read_structure_attributes(f):
    """ function to read information from a PEST-style structure file

    Parameters
    ----------
    f : (file handle)
        file handle open for reading

    Returns
    -------
    nugget : float
        the GeoStruct nugget
    transform : str
        the GeoStruct transformation
    variogram_info : dict
        dictionary of structure-level variogram information

    """

    line = ''
    variogram_info = {}
    while "end structure" not in line:
        line = f.readline()
        if line == '':
            raise Exception("EOF while reading structure")
        line = line.strip().lower().split()
        if line[0].startswith('#'):
            continue
        if line[0] == "nugget":
            nugget = float(line[1])
        elif line[0] == "transform":
            transform = line[1]
        elif line[0] == "numvariogram":
            numvariograms = int(line[1])
        elif line[0] == "variogram":
            variogram_info[line[1]] = float(line[2])
        elif line[0] == "end":
            break
        else:
            raise Exception("unrecognized line in structure definition:{0}".\
                            format(line[0]))
    assert numvariograms == len(variogram_info)
    return nugget,transform,variogram_info


def read_sgems_variogram_xml(xml_file,return_type=GeoStruct):
    """ function to read an SGEMS-type variogram XML file into
    a GeoStruct

    Parameters
    ----------
    xml_file : (str)
        SGEMS variogram XML file
    return_type :  (object)
        the instance type to return.  Default is GeoStruct

    Returns
    -------
    GeoStruct : GeoStruct


    Example
    -------
    ``>>>import pyemu``

    ``>>>gs = pyemu.utils.geostats.read_sgems_variogram_xml("sgems.xml")``

    """
    try:
        import xml.etree.ElementTree as ET

    except Exception as e:
        print("error import elementtree, skipping...")
    VARTYPE = {1: SphVario, 2: ExpVario, 3: GauVario, 4: None}
    assert os.path.exists(xml_file)
    tree = ET.parse(xml_file)
    gs_model = tree.getroot()
    structures = []
    variograms = []
    nugget = 0.0
    num_struct = 0
    for key,val in gs_model.items():
        #print(key,val)
        if str(key).lower() == "nugget":
            if len(val) > 0:
                nugget = float(val)
        if str(key).lower() == "structures_count":
            num_struct = int(val)
    if num_struct == 0:
        raise Exception("no structures found")
    if num_struct != 1:
        raise NotImplementedError()
    for structure in gs_model:
        vtype, contribution = None, None
        mx_range,mn_range = None, None
        x_angle,y_angle = None,None
        #struct_name = structure.tag
        for key,val in structure.items():
            key = str(key).lower()
            if key == "type":
                vtype = str(val).lower()
                if vtype.startswith("sph"):
                    vtype = SphVario
                elif vtype.startswith("exp"):
                    vtype = ExpVario
                elif vtype.startswith("gau"):
                    vtype = GauVario
                else:
                    raise Exception("unrecognized variogram type:{0}".format(vtype))

            elif key == "contribution":
                contribution = float(val)
            for item in structure:
                if item.tag.lower() == "ranges":
                    mx_range = float(item.attrib["max"])
                    mn_range = float(item.attrib["min"])
                elif item.tag.lower() == "angles":
                    x_angle = float(item.attrib["x"])
                    y_angle = float(item.attrib["y"])

        assert contribution is not None
        assert mn_range is not None
        assert mx_range is not None
        assert x_angle is not None
        assert y_angle is not None
        assert vtype is not None
        v = vtype(contribution=contribution,a=mx_range,
                  anisotropy=mx_range/mn_range,bearing=(180.0/np.pi)*np.arctan2(x_angle,y_angle),
                  name=structure.tag)
        return GeoStruct(nugget=nugget,variograms=[v])


def gslib_2_dataframe(filename,attr_name=None,x_idx=0,y_idx=1):
    """ function to read a GSLIB point data file into a pandas.DataFrame

    Parameters
    ----------
    filename : (str)
        GSLIB file
    attr_name : (str)
        the column name in the dataframe for the attribute.  If None, GSLIB file
        can have only 3 columns.  attr_name must be in the GSLIB file header
    x_idx : (int)
        the index of the x-coordinate information in the GSLIB file. Default is
        0 (first column)
    y_idx : (int)
        the index of the y-coordinate information in the GSLIB file.
        Default is 1 (second column)

    Returns
    -------
    df : pandas.DataFrame

    Raises
    ------
    exception if attr_name is None and GSLIB file has more than 3 columns

    Note
    ----
    assigns generic point names ("pt0, pt1, etc)

    Example
    -------
    ``>>>import pyemu``

    ``>>>df = pyemu.utiils.geostats.gslib_2_dataframe("prop.gslib",attr_name="hk")``


    """
    with open(filename,'r') as f:
        title = f.readline().strip()
        num_attrs = int(f.readline().strip())
        attrs = [f.readline().strip() for _ in range(num_attrs)]
        if attr_name is not None:
            assert attr_name in attrs,"{0} not in attrs:{1}".format(attr_name,','.join(attrs))
        else:
            assert len(attrs) == 3,"propname is None but more than 3 attrs in gslib file"
            attr_name = attrs[2]
        assert len(attrs) > x_idx
        assert len(attrs) > y_idx
        a_idx = attrs.index(attr_name)
        x,y,a = [],[],[]
        while True:
            line = f.readline()
            if line == '':
                break
            raw = line.strip().split()
            try:
                x.append(float(raw[x_idx]))
                y.append(float(raw[y_idx]))
                a.append(float(raw[a_idx]))
            except Exception as e:
                raise Exception("error paring line {0}: {1}".format(line,str(e)))
    df = pd.DataFrame({"x":x,"y":y,"value":a})
    df.loc[:,"name"] = ["pt{0}".format(i) for i in range(df.shape[0])]
    df.index = df.name
    return df


#class ExperimentalVariogram(object):
#    def __init__(self,na)

def load_sgems_exp_var(filename):
    """ read an SGEM experimental variogram into a sequence of
    pandas.DataFrames

    Parameters
    ----------
    filename : (str)
        an SGEMS experimental variogram XML file

    Returns
    -------
    dfs : list
        a list of pandas.DataFrames of x, y, pairs for each
        division in the experimental variogram

    """

    assert os.path.exists(filename)
    import xml.etree.ElementTree as etree
    tree = etree.parse(filename)
    root = tree.getroot()
    dfs = {}
    for variogram in root:
        #print(variogram.tag)
        for attrib in variogram:

            #print(attrib.tag,attrib.text)
            if attrib.tag == "title":
                title = attrib.text.split(',')[0].split('=')[-1]
            elif attrib.tag == "x":
                x = [float(i) for i in attrib.text.split()]
            elif attrib.tag == "y":
                y = [float(i) for i in attrib.text.split()]
            elif attrib.tag == "pairs":
                pairs = [int(i) for i in attrib.text.split()]

            for item in attrib:
                print(item,item.tag)
        df = pd.DataFrame({"x":x,"y":y,"pairs":pairs})
        df.loc[df.y<0.0,"y"] = np.NaN
        dfs[title] = df
    return dfs



def fac2real(pp_file=None,factors_file="factors.dat",out_file="test.ref",
             upper_lim=1.0e+30,lower_lim=-1.0e+30,fill_value=1.0e+30):
    """A python replication of the PEST fac2real utility for creating a
    structure grid array from previously calculated kriging factors (weights)

    Parameters
    ----------
    pp_file : (str)
        PEST-type pilot points file
    factors_file : (str)
        PEST-style factors file
    out_file : (str)
        filename of array to write.  If None, array is returned, else
        value of out_file is returned.  Default is "test.ref".
    upper_lim : (float)
        maximum interpolated value in the array.  Values greater than
        upper_lim are set to fill_value
    lower_lim : (float)
        minimum interpolated value in the array.  Values less than lower_lim
        are set to fill_value
    fill_value : (float)
        the value to assign array nodes that are not interpolated


    Returns
    -------
    arr : numpy.ndarray
        if out_file is None
    out_file : str
        if out_file it not None

    Example
    -------
    ``>>>import pyemu``

    ``>>>pyemu.utils.geostats.fac2real("hkpp.dat",out_file="hk_layer_1.ref")``

    """

    if pp_file is not None and isinstance(pp_file,str):
        assert os.path.exists(pp_file)
        # pp_data = pd.read_csv(pp_file,delim_whitespace=True,header=None,
        #                       names=["name","parval1"],usecols=[0,4])
        pp_data = pp_file_to_dataframe(pp_file)
        pp_data.loc[:,"name"] = pp_data.name.apply(lambda x: x.lower())
    elif pp_file is not None and isinstance(pp_file,pd.DataFrame):
        assert "name" in pp_file.columns
        assert "parval1" in pp_file.columns
        pp_data = pp_file
    else:
        raise Exception("unrecognized pp_file arg: must be str or pandas.DataFrame, not {0}"\
                        .format(type(pp_file)))
    assert os.path.exists(factors_file)
    f_fac = open(factors_file,'r')
    fpp_file = f_fac.readline()
    if pp_file is None and pp_data is None:
        pp_data = pp_file_to_dataframe(fpp_file)
        pp_data.loc[:, "name"] = pp_data.name.apply(lambda x: x.lower())

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

    arr = np.zeros((nrow,ncol),dtype=np.float) + fill_value
    pp_dict = {int(name):val for name,val in zip(pp_data.index,pp_data.parval1)}
    try:
        pp_dict_log = {name:np.log10(val) for name,val in zip(pp_data.index,pp_data.parval1)}
    except:
        pp_dict_log = {}
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

    #print(out_file,arr.min(),pp_data.parval1.min(),lower_lim)

    if out_file is not None:
        np.savetxt(out_file,arr,fmt="%15.6E",delimiter='')
        return out_file
    return arr

def parse_factor_line(line):
    """ function to parse a factor file line.  Used by fac2real()

    Parameters
    ----------
    line : (str)
        a factor line from a factor file

    Returns
    -------
    inode : int
        the inode of the grid node
    itrans : int
        flag for transformation of the grid node
    fac_data : dict
        a dictionary of point number, factor

    """

    raw = line.strip().split()
    inode,itrans,nfac = [int(i) for i in raw[:3]]
    fac_data = {int(raw[ifac])-1:float(raw[ifac+1]) for ifac in range(4,4+nfac*2,2)}
    # fac_data = {}
    # for ifac in range(4,4+nfac*2,2):
    #     pnum = int(raw[ifac]) - 1 #zero based to sync with pandas
    #     fac = float(raw[ifac+1])
    #     fac_data[pnum] = fac
    return inode,itrans,fac_data