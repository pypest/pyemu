import os
import numpy as np
import pandas as pd
from pyemu import Cov

#TODO:  plot variogram elipse

EPSILON = 1.0e-7


def read_struct_file(struct_file):
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
                structures.append(s)
            elif line.startswith("variogram"):

                name = line.strip().split()[1].lower()
                vartype,bearing,a,anisotropy = read_variogram(f)
                v = VARTYPE[vartype](variogram_info[name],a,anisotropy=anisotropy,
                                     bearing=bearing,name=name)
                variograms.append(v)

    for s in structures:
        for vname in s.variogram_info:
            vfound = None
            for v in variograms:
                if v.name == vname:
                    vfound = v
                    break
            if vfound is None:
                raise Exception("variogram {0} not found for structure {1}".\
                                format(vname,s.name))
            s.variograms.append(v)

    return structures


def read_variograms(variogram_info,f):
    variograms = []
    while True:
        line = f.readline()
        if line == '':
            raise Exception("EOF while reading variograms")
        line = line.strip().lower()
    return variograms


def read_variogram(f):
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
        if not isinstance(x,np.ndarray):
            x = np.array(x)
        if not isinstance(y,np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]
        if names is not None:
            assert x.shape[0] == len(names)
            c = np.diag(np.zeros(len(names))) + self.nugget
            cov = Cov(x=c,names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
        else:
            raise Exception("GeoStruct.covariance_matrix() requires either" +
                            "names or cov arg")
        for v in self.variograms:
            v.covariance_matrix(x,y,cov=cov)
        return cov



    def covariance(self,pt0,pt1):
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

    def __init__(self,contribution,a,anisotropy=1.0,bearing=None,name="var1"):
        self.name = name
        self.epsilon = EPSILON
        self.contribution = float(contribution)
        assert self.contribution > 0.0
        self.a = float(a)
        assert self.a > 0.0

        self.anisotropy = float(anisotropy)
        if bearing is None:
            self.rotation_coefs = [1,1,1,1]
            self.bearing = None
        elif anisotropy != 1.0:
            assert anisotropy > 0.0
            self.bearing = float(bearing)
            self.bearing_rads = (np.pi / 180.0 ) * (90.0 - self.bearing)
            self.rotation_coefs = [np.cos(self.bearing_rads),
                                   np.sin(self.bearing_rads),
                                   -1.0*np.sin(self.bearing_rads),
                                   np.cos(self.bearing_rads)]
        pass

    def covariance_matrix(self,x,y,names=None,cov=None):
        if not isinstance(x,np.ndarray):
            x = np.array(x)
        if not isinstance(y,np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]
        if names is not None:
            assert x.shape[0] == len(names)
            c = np.diag(np.zeros(len(names))) + self.contribution
            cov = Cov(x=c,names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
        else:
            raise Exception("Vario2d.covariance_matrix() requires either" +
                            "names or cov arg")

        for i1,(n1,x1,y1) in enumerate(zip(names,x,y)):
            dx = x1 - x[i1+1:]
            dy = y1 - y[i1+1:]
            if self.bearing is not None:
                temp = (dx * self.rotation_coefs[0]) +\
                     (dy * self.rotation_coefs[1])
                dy = ((dx * self.rotation_coefs[2]) +\
                     (dy * self.rotation_coefs[3])) /\
                     self.anisotropy
                dx = temp
            h = np.sqrt(dx*dx + dy*dy)
            h[h<0.0] = 0.0
            cov.x[i1,i1+1:] = self.h_function(h)
        for i in range(len(names)):
            cov.x[i+1:,i] = cov.x[i,i+1:]
        return cov

    def covariance(self,pt0,pt1):
        x = np.array([pt0[0],pt1[0]])
        y = np.array([pt0[1],pt1[1]])
        names = ["n1","n2"]
        return self.covariance_matrix(x,y,names=names).x[0,1]

    # def covariance(self,pt0,pt1):
    #     dx = pt0[0] - pt1[0]
    #     dy = pt0[1] - pt1[1]
    #     if self.bearing is not None:
    #         temp = (dx * self.rotation_coefs[0]) +\
    #              (dy * self.rotation_coefs[1])
    #         dy = ((dx * self.rotation_coefs[2]) +\
    #              (dy * self.rotation_coefs[3])) /\
    #              self.anisotropy
    #         dx = temp
    #     h = np.sqrt(max(dx*dx+dy*dy,0.0))
    #     return self.h_function(h)

    def plot(self):
        raise NotImplementedError()

    def __str__(self):
        s = "name:{0},contribution:{1},a:{2},anisotropy:{3},bearing:{4}\n".\
            format(self.name,self.contribution,self.a,\
                   self.anisotropy,self.bearing)
        return s

class ExpVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,bearing=None,name="var1"):

        super(ExpVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)

    def h_function(self,h):
        return self.contribution * np.exp(-1.0 * h / self.a)

class GauVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,bearing=None,name="var1"):
        super(GauVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      bearing=bearing,name=name)

    def h_function(self,h):
        hh = -1.0 * (h * h) / (self.a * self.a)
        return self.contribution * np.exp(hh)

class SphVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,bearing=None,name="var1"):
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


