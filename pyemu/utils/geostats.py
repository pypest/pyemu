from __future__ import print_function
import os
import copy
from datetime import datetime
import numpy as np
import pandas as pd
from pyemu import Cov
from pyemu.utils.gw_utils import pp_file_to_dataframe
from .reference import SpatialReference

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


    def plot(self):
        raise NotImplementedError()

    def __str__(self):
        s = ''
        s += 'name:{0},nugget:{1},structures:\n'.format(self.name,self.nugget)
        for v in self.variograms:
            s += str(v)
        return s


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
    def calc_factors_grid(self,spatial_reference,zone_array=None,minpts_interp=1,
                          maxpts_interp=20,search_radius=1.0e+10,verbose=False):

        #assert isinstance(spatial_reference,SpatialReference)
        try:
            x = spatial_reference.xcentergrid
            y = spatial_reference.ycentergrid
        except Exception as e:
            raise Exception("spatial_reference does not have proper attributes:{0}"\
                            .format(str(e)))

        if zone_array is not None:
            print("only supporting a single zone via zone array - "+\
                  "locations cooresponding to values in zone array > 0 are used")
            assert zone_array.shape == x.shape
            x[zone_array<=0] = np.NaN
            y[zone_array<=0] = np.NaN
            #x = x[~np.isnan(x)]
            #y = y[~np.isnan(y)]
        self.spatial_reference = spatial_reference
        return self.calc_factors(x.ravel(),y.ravel(),
                                 minpts_interp=minpts_interp,
                                 maxpts_interp=maxpts_interp,
                                 search_radius=search_radius,
                                 verbose=verbose)

    def calc_factors(self,x,y,minpts_interp=1,maxpts_interp=20,
                     search_radius=1.0e+10,verbose=False):
        assert len(x) == len(y)

        # find the point data to use for each interp point
        sqradius = search_radius**2
        df = pd.DataFrame(data={'x':x,'y':y})
        inames,idist,ifacts = [],[],[]
        ptx_array = self.point_data.x.values
        pty_array = self.point_data.y.values
        ptnames = self.point_data.name.values
        if verbose: print("starting interp point loop")
        for idx,(ix,iy) in enumerate(zip(df.x,df.y)):
            if np.isnan(ix) or np.isnan(iy): #if nans, skip
                inames.append([])
                idist.append([])
                ifacts.append([])
                continue
            if verbose:
                istart = datetime.now()
                print("processing interp point:{0}:{1}".format(ix,iy))
            if verbose == 2:
                start = datetime.now()
                print("calc ipoint dist...",end='')

            #  calc dist from this interp point to all point data...slow
            dist = pd.Series((ptx_array-ix)**2 + (pty_array-iy)**2,ptnames)
            dist.sort_values(inplace=True)
            dist = dist.loc[dist <= sqradius]

            # if too few points were found, skip
            if len(dist) < minpts_interp:
                inames.append([])
                idist.append([])
                ifacts.append([])
                continue

            # only the maxpts_interp points
            dist = dist.iloc[:maxpts_interp].apply(np.sqrt)
            pt_names = dist.index.values
            # if one of the points is super close, just use it and skip
            if dist.min() <= EPSILON:
                ifacts.append([1.0])
                idist.append([EPSILON])
                inames.append([dist.idxmin()])
                continue
            if verbose == 2:
                td = (datetime.now()-start).total_seconds()
                print("...took {0}".format(td))
                start = datetime.now()
                print("extracting pt cov...",end='')

            #vextract the point-to-point covariance matrix
            point_cov = self.point_cov_df.loc[pt_names,pt_names]
            if verbose == 2:
                td = (datetime.now()-start).total_seconds()
                print("...took {0}".format(td))
                print("forming ipt-to-point cov...",end='')

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
            if verbose == 2:
                td = (datetime.now()-start).total_seconds()
                print("...took {0}".format(td))
                print("solving...",end='')
            # solve
            facs = np.linalg.solve(A,rhs)
            assert len(facs) - 1 == len(dist)
            inames.append(pt_names)
            idist.append(dist.values)
            ifacts.append(facs[:-1,0])
            if verbose == 2:
                td = (datetime.now()-start).total_seconds()
                print("...took {0}".format(td))
            if verbose:
                td = (datetime.now()-istart).total_seconds()
                print("point took {0}".format(td))
        df["idist"] = idist
        df["inames"] = inames
        df["ifacts"] = ifacts
        self.interp_data = df
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
        dxx = (dx * self.rotation_coefs[0]) +\
             (dy * self.rotation_coefs[1])
        dyy = ((dx * self.rotation_coefs[2]) +\
             (dy * self.rotation_coefs[3])) *\
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

    def plot(self):
        raise NotImplementedError()

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