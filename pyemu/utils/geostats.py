import os
import copy
import numpy as np
import pandas as pd
from pyemu import Cov

#TODO:  plot variogram elipse

EPSILON = 1.0e-7


def read_struct_file(struct_file):
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
                s = GeoStruct(nugget=nugget,transform=transform,name=name)
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


# def read_variograms(variogram_info,f):
#     variograms = []
#     while True:
#         line = f.readline()
#         if line == '':
#             raise Exception("EOF while reading variograms")
#         line = line.strip().lower()
#     return variograms


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

    def to_struct_file(self):
        raise NotImplementedError()

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
            c = np.zeros((len(names),1)) + self.nugget
            cont = Cov(x=c,names=names,isdiagonal=True)
            cov += cont

        else:
            raise Exception("GeoStruct.covariance_matrix() requires either" +
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

        cov = self.nugget
        for vario in self.variograms:
            cov += vario.covariance(pt0,pt1)
        return cov

    def plot(self):
        raise NotImplementedError()

    def __str__(self):
        s = ''
        s += 'name:{0},nugget:{1},structures:\n'.format(self.name,self.nugget)
        for v in self.variograms:
            s += str(v)
        return s


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
        self.bearing_rads = (np.pi / 180.0 ) * (90.0 - self.bearing)
        self.rotation_coefs = [np.cos(self.bearing_rads),
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

        for i1,(n1,x1,y1) in enumerate(zip(names,x,y)):
            dx = x1 - x[i1+1:]
            dy = y1 - y[i1+1:]
            dxx = (dx * self.rotation_coefs[0]) +\
                 (dy * self.rotation_coefs[1])
            dyy = ((dx * self.rotation_coefs[2]) +\
                 (dy * self.rotation_coefs[3])) *\
                 self.anisotropy
            h = np.sqrt(dxx*dxx + dyy*dyy)

            h[h<0.0] = 0.0
            h = self.h_function(h)
            if np.any(np.isnan(h)):
                raise Exception("nans in h for i1 {0}".format(i1))
            cov.x[i1,i1+1:] += h
        for i in range(len(names)):
            cov.x[i+1:,i] = cov.x[i,i+1:]
        return cov

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

    def h_function(self,h):
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

    def h_function(self,h):
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

    def h_function(self,h):

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


