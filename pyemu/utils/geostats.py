from __future__ import print_function
import os
import copy
from datetime import datetime
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
from pyemu import Cov
from pyemu.utils.gw_utils import pp_file_to_dataframe
#from pyemu.utils.reference import SpatialReference

#TODO:  plot variogram elipse

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
    """a geostatistical structure object
    Parameters
    ----------
        nugget : float
            nugget contribution
        variograms : list
            list of Vario2d instances
        name : str
            name to assign the structure

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

    def covariance_matrix(self,x,y,names=None,cov=None):
        """build a pyemu.Cov instance from GeoStruct
        Parameters
        ----------
            x : iterable of floats
                x locations
            y : iterable of floats
                y locations
            names : iterable of str (optional)
                names of location. If None, generic names will be used
            cov : pyemu.Cov instance (optional)
                an existing Cov instance to add contribution to
        Returns
        -------
            pyemu.Cov
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
        """get the covariance between two points implied by the GeoStruct
        Parameters
        ----------
            pt0 : iterable length 2 of floats
            pt1 : iterable length 2 of floats
        Returns
            float : covariance
        """
        #raise Exception()
        cov = self.nugget
        for vario in self.variograms:
            cov += vario.covariance(pt0,pt1)
        return cov

    def covariance_points(self,x0,y0,xother,yother):
        cov = np.zeros((len(xother))) + self.nugget
        for v in self.variograms:
            cov += v.covariance_points(x0,y0,xother,yother)
        return cov

    @property
    def sill(self):
        sill = self.nugget
        for v in self.variograms:
            sill += v.contribution
        return sill


    def plot(self,**kwargs):
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
        self.point_data = point_data
        self.point_data.index = self.point_data.name
        self.interp_data = None
        self.spatial_reference = None
        #X, Y = np.meshgrid(point_data.x,point_data.y)
        #self.point_data_dist = pd.DataFrame(data=np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2),
        #                                    index=point_data.name,columns=point_data.name)
        self.point_cov_df = self.geostruct.covariance_matrix(point_data.x,
                                                            point_data.y,
                                                            point_data.name).to_dataframe()
        #for name in self.point_cov_df.index:
        #    self.point_cov_df.loc[name,name] -= self.geostruct.nugget

    def prep_for_ppk2fac(self,struct_file="structure.dat",pp_file="points.dat",):
        pass

    def calc_factors_grid(self,spatial_reference,zone_array=None,minpts_interp=1,
                          maxpts_interp=20,search_radius=1.0e+10,verbose=False,
                          var_filename=None):

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
                               verbose=verbose)
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
                                       verbose=verbose,pt_zone=pt_data_zone)
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
                     pt_zone=None):
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
                print("...took {0}".format(td))
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
            facs = np.linalg.solve(A,rhs)
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
        print("took {0}".format(td))
        return df

    def to_grid_factors_file(self, filename,points_file="points.junk",
                             zone_file="zone.junk"):
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
    """base class for 2-D variograms
    Parameters
    ----------
        contribution : float
            sill of the variogram
        a : float
            (practical) range
        anisotropy : float (optional)
            Anisotropy ratio. If None, 1.0 used
        bearing : float (optional)
            angle in degrees East of North cooresponding to anisotropy ellipse.
            If None, 0.0 used
        name : str (optional)
            name of the variogram
    Returns
    -------
        Vario2d instance
    Note
    ----
        This base class should not be instantiated directly.
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
        return (np.pi / 180.0 ) * (90.0 - self.bearing)

    @property
    def rotation_coefs(self):
        return [np.cos(self.bearing_rads),
                np.sin(self.bearing_rads),
                -1.0*np.sin(self.bearing_rads),
                np.cos(self.bearing_rads)]

    def inv_h(self,h):
        return self.contribution - self._h_function(h)

    def plot(self,ax=None,**kwargs):
        import matplotlib.pyplot as plt
        ax = kwargs.pop("ax",plt.subplot(111))
        x = np.linspace(0,self.a*3,100)
        y = self.inv_h(x)
        ax.set_xlabel("distance")
        ax.set_ylabel("$\gamma$")
        ax.plot(x,y,**kwargs)
        return ax

    def covariance_matrix(self,x,y,names=None,cov=None):
        """build a pyemu.Cov instance from Vario2d
        Parameters
        ----------
            x : iterable of floats
                x locations
            y : iterable of floats
                y locations
            names : iterable of str (optional)
                names of location. If None, generic names will be used
            cov : pyemu.Cov instance (optional)
                an existing Cov instance to add contribution to
        Returns
        -------
            pyemu.Cov
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
        dx = x0 - xother
        dy = y0 - yother
        dxx,dyy = self._apply_rotation(dx,dy)
        h = np.sqrt(dxx*dxx + dyy*dyy)
        return self._h_function(h)

    def covariance(self,pt0,pt1):
        x = np.array([pt0[0],pt1[0]])
        y = np.array([pt0[1],pt1[1]])
        names = ["n1","n2"]
        return self.covariance_matrix(x,y,names=names).x[0,1]


    def __str__(self):
        s = "name:{0},contribution:{1},a:{2},anisotropy:{3},bearing:{4}\n".\
            format(self.name,self.contribution,self.a,\
                   self.anisotropy,self.bearing)
        return s

class ExpVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        """Exponential 2-D variograms
    Parameters
    ----------
        contribution : float
            sill of the variogram
        a : float
            (practical) range
        anisotropy : float (optional)
            Anisotropy ratio. If None, 1.0 used
        bearing : float (optional)
            angle in degrees East of North cooresponding to anisotropy ellipse.
            If None, 0.0 used
        name : str (optional)
            name of the variogram
    Returns
    -------
        ExpVario instance
    """
        super(ExpVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)
        self.vartype = 2

    def _h_function(self,h):
        return self.contribution * np.exp(-1.0 * h / self.a)

class GauVario(Vario2d):
    """Gaussian 2-D variograms
    Parameters
    ----------
        contribution : float
            sill of the variogram
        a : float
            (practical) range
        anisotropy : float (optional)
            Anisotropy ratio. If None, 1.0 used
        bearing : float (optional)
            angle in degrees East of North cooresponding to anisotropy ellipse.
            If None, 0.0 used
        name : str (optional)
            name of the variogram
    Returns
    -------
        GauVario instance
    """

    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        super(GauVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)
        self.vartype = 3

    def _h_function(self,h):
        hh = -1.0 * (h * h) / (self.a * self.a)
        return self.contribution * np.exp(hh)

class SphVario(Vario2d):
    """Spherical 2-D variograms
    Parameters
    ----------
        contribution : float
            sill of the variogram
        a : float
            (practical) range
        anisotropy : float (optional)
            Anisotropy ratio. If None, 1.0 used
        bearing : float (optional)
            angle in degrees East of North cooresponding to anisotropy ellipse.
            If None, 0.0 used
        name : str (optional)
            name of the variogram
    Returns
    -------
        SphVario instance
    """

    def __init__(self,contribution,a,anisotropy=1.0,bearing=0.0,name="var1"):
        super(SphVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)
        self.vartype = 1

    def _h_function(self,h):

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
    """read an existing structure file into a GeoStruct instance
    Parameters
    ----------
        struct_file : str
            existing pest-type structure file
    Returns
    -------
        GeoStruct instance
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
        dfs[title] = df
    return dfs