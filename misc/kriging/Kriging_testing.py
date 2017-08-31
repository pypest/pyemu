from __future__ import print_function
import pandas as pd
import pyemu
import scipy
import flopy as fp
import numpy as np
import matplotlib.pyplot as plt
import platform
# make sure to always write out windows-style line endings
if 'window' in platform.platform().lower():
    newln = '\n'
else:
    newln = '\r\n'

class KrigingFactors(object):
    def __init__(self, namfile, structfile, pointsfile, zonefile, factor_file,
                 minpts_interp, maxpts_interp, search_radius, czone):
        print('Initializing Kriging setup')
        # get x y locations for all points in the model
        if minpts_interp > maxpts_interp:
            raise Exception("minpts_interp ({0}) > maxpts_interp ({1})".format(minpts_interp,maxpts_interp))

        # set input filenames
        self.structfile = structfile
        self.pointsfile = pointsfile
        self.zonefile = zonefile
        self.namfile = namfile

        # set output filename for factors
        self.factor_filename = factor_file

        # read in information from the base model
        inmod = fp.modflow.Modflow.load(self.namfile)
        self.x_centers = inmod.sr.get_xcenter_array()
        self.y_centers = inmod.sr.get_ycenter_array()
        self.ncol = inmod.ncol
        self.nrow = inmod.nrow



        # get geostatistical structures
        self.geo_struct = pyemu.utils.geostats.read_struct_file(structfile)


        # get pilot point locations
        self.pp_df = pyemu.utils.gw_utils.pp_file_to_dataframe(pointsfile)

        # get zone map
        self.pp_zones = np.loadtxt(zonefile, dtype=int)

        # keep track of number of points to interpolate to and search radius and current zone of iterest
        self.minpts_interp = minpts_interp
        self.maxpts_interp = maxpts_interp
        self.search_radius = search_radius
        self.czone = czone

        # make a list of point locations where interpolation will take place
        X,Y = np.meshgrid(self.x_centers, self.y_centers)
        X[self.pp_zones != self.czone] = np.nan
        Y[self.pp_zones != self.czone] = np.nan

        self.allpts = pd.DataFrame(list(zip(X.ravel(),Y.ravel())), columns=['x','y'])
        self.allpts.dropna(inplace=True)

    def get_interp_points(self):
        print('Navigating for with pilot points to interpolate from for each point in the grid')
        # loop through all the points in the zone of interest to determine which should be interpolants
        ppslist = []
        for cp in zip(self.allpts.x.values,self.allpts.y.values):
            cpp = self.pp_df.copy()
            # calculate distances
            cpp['dist'] = np.sqrt((cp[0]-cpp.x.values)**2+(cp[1]-cpp.y.values)**2)
            # sort by distances
            cpp.sort_values(by='dist', inplace=True)
            # remove points farther away than the search_radius
            cpp = cpp.loc[cpp.dist <= self.search_radius]
            # if not enough points (less than self.minpst_interp) then bomb
            if len(cpp) < self.minpts_interp:
                raise Exception("Not enough pilot points within search radius\n" \
                                "Check minpts_interp value = {0}".format(self.minpts_interp))
            # keep only the self.maxpst_inter points
            ppslist.append(cpp.iloc[:self.maxpts_interp].index.values)
        self.allpts['pps'] = ppslist

    def kriging_weights(self):
        print('Solving for Kriging Weights for {0} points'.format(len(self.allpts)))
        # solve the Kriging system for each point in the model grid for the zone individually
        allwts = []
        for cind,cx,cy,pps in self.allpts.itertuples():
            # set up matrices for distances among all the relevant pilot points
            allx = self.pp_df.loc[pps].x
            ally = self.pp_df.loc[pps].y
            X, Y = np.meshgrid(allx, ally)
            dist = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)

            # apply the nugget to the diagonal
            Q = np.eye(len(allx)) * self.geo_struct.nugget
            b = np.atleast_2d(np.zeros(len(allx)))
            # now add up the variogram functions for each variogram in the structure
            for cvar in self.geo_struct.variograms:
                # for the Q matrix on LHS
                Q += cvar.h_function(dist)
                # also for the b vector on RHS
                b += cvar.h_function(np.sqrt((allx - cx) ** 2 + (ally - cy) ** 2))


            # form the lefthandside, including accounting for lagrange multiplier (see GSLIB book P. ###)
            fulldim = len(allx) + 1
            A = np.ones((fulldim,fulldim))
            A[-1,-1] = 0
            A[:-1,:-1] = Q
            # complete the RHS
            rhs = np.ones((fulldim,1))
            rhs[:-1] = b.T
            lam = np.linalg.solve(A,rhs)
            allwts.append(lam[:-1])
        self.allpts['factors'] = allwts

    def write_factor_file(self):
        print('Writing out file: {0}'.format(self.factor_filename))
        with open(self.factor_filename, 'w') as ofp:
            ofp.write("{0}{1}".format(self.pointsfile, newln))
            ofp.write("{0}{1}".format(self.zonefile, newln))
            ofp.write("{0} {1}{2}".format(self.ncol, self.nrow, newln))
            ofp.write("{0}{1}".format(len(self.pp_df), newln))
            [ofp.write("{0}{1}".format(i.upper(), newln)) for i in self.pp_df['name'].values]
            if self.geo_struct.transform.lower() == 'log':
                ctrans = 1
            else:
                ctrans = 0
            for cind,cx,cy,cpps,cfacts in self.allpts.itertuples():

                ofp.write("{0} {1} {2} {3:8.5e} ".format(cind+1, ctrans, len(cpps), 0.0))
                [ofp.write("{0} {1:12.8e} ".format(i+1, w)) for i, w in zip(cpps, np.squeeze(cfacts))]
                ofp.write(newln)

if __name__=='__main__':
    test_obj = KrigingFactors('freyberg_pp.nam', 'structure.dat', 'points1.dat', 'lay1zones.dat',
                       'testfactors1.dat',1,8, 1500, 1)
    test_obj.get_interp_points()
    test_obj.kriging_weights()
    test_obj.write_factor_file()