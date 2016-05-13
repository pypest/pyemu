import os
import numpy as np
import pandas as pd

#TODO:  plot variogram elipse

EPSILON = 1.0e-7



def read_struct_file(struct_file):
    assert os.path.exists(struct_file)
    structures = []
    with open(struct_file,'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            line = line.strip().lower()
            if line.startswith("structure"):
                nugget,transform,variogram_info = _read_structure_attributes(f)
                variograms = read_variograms(variogram_info,f)



    return structures

def read_variograms(variogram_info,f):
    while True:
        line = f.readline()
        if line == '':
            raise Exception("EOF while reading variograms")
        line = line.strip().lower()
        if line.startswith("variogram"):
            v = read_variogram(f)


def read_variogram(f):
    line = ''
    while "end variogram" not in line:
        line = f.readline()
        if line == '':
            raise Exception("EOF while read variogram")
        line = line.strip().lower().split()
        if line[0] == "vartype":
            pass
        elif line[0] == "bearing":
            pass
        elif line[0] == "a":
            pass
        elif line[0] == "anisotropy":
            pass
        elif line[0] == "end":
            break
        else:
            raise Exception("unrecognized arg in variogram:{0}".format(line[0]))
    

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
    def __init__(self,nugget,variograms,name="struct1"):
        self.name = name
        self.nugget = float(nugget)
        if not isinstance(variograms,list):
            variograms = [variograms]
        for vario in variograms:
            assert isinstance(vario,Vario2d)
        self.variograms = variograms



    def to_struct_file(self):
        raise NotImplementedError()


    def covariance(self,pt0,pt1):
        cov = self.nugget
        for vario in self.variograms:
            cov += vario.covariance(pt0,pt1)
        return cov

    def plot(self):
        raise NotImplementedError()


class Vario2d(object):

    def __init__(self,contribution,a,anisotropy=1.0,angle=None,name="var1"):
        self.name = name
        self.epsilon = EPSILON
        self.contribution = float(contribution)
        assert self.contribution > 0.0
        self.a = float(a)
        assert self.a > 0.0

        self.anisotropy = float(anisotropy)
        if angle is None:
            self.rotation_coefs = [1,1,1,1]
            self.angle = None
        elif anisotropy != 1.0:
            assert anisotropy > 0.0
            self.angle = (np.pi / 180.0 ) * (90.0 - float(angle))
            self.rotation_coefs = [np.cos(self.angle),np.sin(self.angle),
                                   -1.0*np.sin(self.angle),np.cos(self.angle)]
        pass

    def covariance(self,pt0,pt1):
        dx = pt0[0] - pt1[0]
        dy = pt0[1] - pt1[1]
        if self.angle is not None:
            temp = (dx * self.rotation_coefs[0]) +\
                 (dy * self.rotation_coefs[1])
            dy = ((dx * self.rotation_coefs[2]) +\
                 (dy * self.rotation_coefs[3])) /\
                 self.anisotropy
            dx = temp
        h = np.sqrt(max(dx*dx+dy*dy,0.0))
        return self.h_function(h)

    def plot(self):
        raise NotImplementedError()

class ExpVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,angle=None):

        super(ExpVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      angle=angle)

    def h_function(self,h):
        return self.contribution * np.exp(-1.0 * h / self.a)

class GauVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,angle=None):
        super(GauVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      angle=angle)

    def h_function(self,h):
        hh = -1.0 * (h * h) / (self.a * self.a)
        return self.contribution * np.exp(hh)

class SphVario(Vario2d):

    def __init__(self,contribution,a,anisotropy=1.0,angle=None):
        super(SphVario,self).__init__(contribution,a,anisotropy=anisotropy,
                                      angle=angle)

    def h_function(self,h):
        hh = h / self.a
        if hh < 1.0:
            return self.contribution * (1.0 - (hh * (1.5 - (0.5 * hh * hh))))
        else:
            return 0.0


