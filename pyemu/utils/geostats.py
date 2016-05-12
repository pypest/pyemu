import os
import numpy as np
import pandas as pd

#TODO:  plot variogram elipse

EPSILON = 1.0e-7

class GeoStruct(object):

    def __init__(self,nugget):
        pass

    @classmethod
    def from_struct_file(cls,struct_file):

        return cls()

class Vario2d(object):

    def __init__(self,contribution,a,anisotropy=1.0,angle=None):
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
        h = np.sqrt(max(dx*dx+dy*dy),0.0)
        return self.h_function(h)


class ExpVario(Vario2d):

    def __init__(self):

        super(ExpVario,self).__init__()

    def h_function(self,h):
        return self.contribution * np.exp(-1.0 * h / self.a)

class GauVario(Vario2d):

    def __init__(self):
        super(GauVario,self).__init__()

    def h_function(self,h):
        hh = -1.0 * (h * h) / (self.a * self.a)
        return self.contribution * np.exp(hh)

class SphVario(Vario2d):

    def __init__(self):
        super(SphVario,self).__init__()

    def h_function(self,h):
        hh = h / self.a
        return self.contribution * (1.0 - hh * (1.5-0.5 * hh * hh))

