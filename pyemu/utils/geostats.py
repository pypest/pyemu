"""Geostatistics in the PEST(++) realm
"""
from __future__ import print_function
import os
import copy
from datetime import datetime
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
from pyemu.mat.mat_handler import Cov
from pyemu.utils.pp_utils import pp_file_to_dataframe
from ..pyemu_warnings import PyemuWarning

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
    """a geostatistical structure object that mimics the behavior of a PEST
    geostatistical structure.  The object contains variogram instances and
    (optionally) nugget information.

    Args:
        nugget (`float` (optional)): nugget contribution. Default is 0.0
        variograms : ([`pyemu.Vario2d`] (optional)): variogram(s) associated
            with this GeoStruct instance. Default is empty list
        name (`str` (optional)): name to assign the structure.  Default
            is "struct1".
        transform  (`str` (optional)): the transformation to apply to
            the GeoStruct.  Can be "none" or "log", depending on the
            transformation of the property being represented by the `GeoStruct`.
            Default is "none"

    Example::

        v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)
        gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)
        gs.plot()
        # get a covariance matrix implied by the geostruct for three points
        px = [0,1000,2000]
        py = [0,0,0]
        pnames ["p1","p2","p3"]
        cov = gs.covariance_matrix(px,py,names=pnames)

    """

    def __init__(self, nugget=0.0, variograms=[], name="struct1", transform="none"):
        self.name = name
        self.nugget = float(nugget)
        """`float`: the nugget effect contribution"""
        if not isinstance(variograms, list):
            variograms = [variograms]
        for vario in variograms:
            assert isinstance(vario, Vario2d)
        self.variograms = variograms
        """[`pyemu.utils.geostats.Vario2d`]: a list of variogram instances"""
        transform = transform.lower()
        assert transform in ["none", "log"]
        self.transform = transform
        """`str`: the transformation of the `GeoStruct`.  Can be 'log' or 'none'"""

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name

    def same_as_other(self, other):
        """compared to geostructs for similar attributes

        Args:
            other (`pyemu.geostats.Geostruct`): the other one

        Returns:
            same (`bool`): True is the `other` and `self` have the same characteristics


        """
        if self.nugget != other.nugget:
            return False
        if len(self.variograms) != len(other.variograms):
            return False
        for sv, ov in zip(self.variograms, other.variograms):
            if not sv.same_as_other(ov):
                return False
        return True

    def to_struct_file(self, f):
        """write a PEST-style structure file

        Args:
            f (`str`): file to write the GeoStruct information in to.  Can
                also be an open file handle

        """
        if isinstance(f, str):
            f = open(f, "w")
        f.write("STRUCTURE {0}\n".format(self.name))
        f.write("  NUGGET {0}\n".format(self.nugget))
        f.write("  NUMVARIOGRAM {0}\n".format(len(self.variograms)))
        for v in self.variograms:
            f.write("  VARIOGRAM {0} {1}\n".format(v.name, v.contribution))
        f.write("  TRANSFORM {0}\n".format(self.transform))
        f.write("END STRUCTURE\n\n")
        for v in self.variograms:
            v.to_struct_file(f)

    def covariance_matrix(self, x, y, names=None, cov=None):
        """build a `pyemu.Cov` instance from `GeoStruct`

        Args:
            x ([`floats`]): x-coordinate locations
            y ([`float`]): y-coordinate locations
            names ([`str`] (optional)): names of location. If None,
                cov must not be None.  Default is None.
            cov (`pyemu.Cov`): an existing Cov instance.  The contribution
                of this GeoStruct is added to cov.  If cov is None,
                names must not be None. Default is None

        Returns:
            `pyemu.Cov`: the covariance matrix implied by this
            GeoStruct for the x,y pairs. `cov` has row and column
            names supplied by the names argument unless the "cov"
            argument was passed.

        Note:
            either "names" or "cov" must be passed.  If "cov" is passed, cov.shape
            must equal len(x) and len(y).

        Example::

            pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")
            cov = gs.covariance_matrix(pp_df.x,pp_df.y,pp_df.name)
            cov.to_binary("cov.jcb")


        """

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]

        if names is not None:
            assert x.shape[0] == len(names)
            c = np.zeros((len(names), len(names)))
            np.fill_diagonal(c, self.nugget)
            cov = Cov(x=c, names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
            c = np.zeros((len(names), 1))
            c += self.nugget
            cont = Cov(x=c, names=names, isdiagonal=True)
            cov += cont

        else:
            raise Exception(
                "GeoStruct.covariance_matrix() requires either " + "names or cov arg"
            )
        for v in self.variograms:
            v.covariance_matrix(x, y, cov=cov)
        return cov

    def covariance(self, pt0, pt1):
        """get the covariance between two points implied by the `GeoStruct`.
        This is used during the ordinary kriging process to get the RHS

        Args:
            pt0 ([`float`]): xy-pair
            pt1 ([`float`]): xy-pair

        Returns:
            `float`: the covariance between pt0 and pt1 implied
            by the GeoStruct

        Example::

            p1 = [0,0]
            p2 = [1,1]
            v = pyemu.geostats.ExpVario(a=0.1,contribution=1.0)
            gs = pyemu.geostats.Geostruct(variograms=v)
            c = gs.covariance(p1,p2)

        """
        # raise Exception()
        cov = self.nugget
        for vario in self.variograms:
            cov += vario.covariance(pt0, pt1)
        return cov

    def covariance_points(self, x0, y0, xother, yother):
        """Get the covariance between point (x0,y0) and the points
        contained in xother, yother.

        Args:
            x0 (`float`): x-coordinate
            y0 (`float`): y-coordinate
            xother ([`float`]): x-coordinates of other points
            yother ([`float`]): y-coordinates of other points

        Returns:
            `numpy.ndarray`: a 1-D array of covariance between point x0,y0 and the
            points contained in xother, yother.  len(cov) = len(xother) =
            len(yother)

        Example::

            x0,y0 = 1,1
            xother = [2,3,4,5]
            yother = [2,3,4,5]
            v = pyemu.geostats.ExpVario(a=0.1,contribution=1.0)
            gs = pyemu.geostats.Geostruct(variograms=v)
            c = gs.covariance_points(x0,y0,xother,yother)


        """

        cov = np.zeros((len(xother))) + self.nugget
        for v in self.variograms:
            cov += v.covariance_points(x0, y0, xother, yother)
        return cov

    @property
    def sill(self):
        """get the sill of the `GeoStruct`

        Returns:
            `float`: the sill of the (nested) `GeoStruct`, including
            nugget and contribution from each variogram

        """
        sill = self.nugget
        for v in self.variograms:
            sill += v.contribution
        return sill

    def plot(self, **kwargs):
        """make a cheap plot of the `GeoStruct`

        Args:
            **kwargs : (dict)
                keyword arguments to use for plotting.
        Returns:
            `matplotlib.pyplot.axis`: the axis with the GeoStruct plot

        Note:
            optional arguments include "ax" (an existing axis),
            "individuals" (plot each variogram on a separate axis),
            "legend" (add a legend to the plot(s)).  All other kwargs
            are passed to matplotlib.pyplot.plot()

        Example::

            v = pyemu.geostats.ExpVario(a=0.1,contribution=1.0)
            gs = pyemu.geostats.Geostruct(variograms=v)
            gs.plot()

        """
        #
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            try:
                import matplotlib.pyplot as plt
            except Exception as e:
                raise Exception("error importing matplotlib: {0}".format(str(e)))

            ax = plt.subplot(111)
        legend = kwargs.pop("legend", False)
        individuals = kwargs.pop("individuals", False)
        xmx = max([v.a * 3.0 for v in self.variograms])
        x = np.linspace(0, xmx, 100)
        y = np.zeros_like(x)
        for v in self.variograms:
            yv = v.inv_h(x)
            if individuals:
                ax.plot(x, yv, label=v.name, **kwargs)
            y += yv
        y += self.nugget
        ax.plot(x, y, label=self.name, **kwargs)
        if legend:
            ax.legend()
        ax.set_xlabel("distance")
        ax.set_ylabel(r"$\gamma$")
        return ax

    def __str__(self):
        """the `str` representation of the `GeoStruct`

        Returns:
            `str`: the string representation of the GeoStruct
        """
        s = ""
        s += "name:{0},nugget:{1},structures:\n".format(self.name, self.nugget)
        for v in self.variograms:
            s += str(v)
        return s


class SpecSim2d(object):
    """2-D unconditional spectral simulation for regular grids

    Args:
        delx (`numpy.ndarray`): a 1-D array of x-dimension cell centers
            (or leading/trailing edges).  Only the distance between points
            is important
        dely (`numpy.ndarray`): a 1-D array of y-dimension cell centers
            (or leading/trailing edges).  Only the distance between points
            is important
        geostruct (`pyemu.geostats.Geostruct`): geostatistical structure instance

    Example::

        v = pyemu.utils.geostats.ExpVario(a=100,contribution=1.0)
        gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)
        delx,dely = np.ones(150), np.ones(50)
        ss = pyemu.utils.geostats.SpecSim2d(delx,dely,gs)
        arr = np.squeeze(ss.draw_arrays(num_reals=1))*.05 + .08
        plt.imshow(arr)
        plt.colorbar(shrink=.40)

    """

    def __init__(self, delx, dely, geostruct):

        self.geostruct = geostruct
        self.delx = delx
        self.dely = dely
        self.num_pts = np.NaN
        self.sqrt_fftc = np.NaN
        self.effective_variograms = None
        self.initialize()

    @staticmethod
    def grid_is_regular(delx, dely, tol=1.0e-4):
        """check that a grid is regular using delx and dely vectors

        Args:
            delx : `numpy.ndarray`
                a 1-D array of x-dimension cell centers (or leading/trailing edges).  Only the
                distance between points is important
            dely : `numpy.ndarray`
                a 1-D array of y-dimension cell centers (or leading/trailing edges).  Only the
                distance between points is important
            tol : `float` (optional)
                tolerance to determine grid regularity.  Default is 1.0e-4

        Returns:
            `bool`: flag indicating if the grid defined by `delx` and `dely` is regular

        """
        if np.abs((delx.mean() - delx.min()) / delx.mean()) > tol:
            return False
        if (np.abs(dely.mean() - dely.min()) / dely.mean()) > tol:
            return False
        if (np.abs(delx.mean() - dely.mean()) / delx.mean()) > tol:
            return False
        return True

    def initialize(self):
        """prepare for spectral simulation.

        Note:
            `initialize()` prepares for simulation by undertaking
            the fast FFT on the wave number matrix and should be called
            if the `SpecSim2d.geostruct` is changed.
            This method is called by the constructor.


        """
        if not SpecSim2d.grid_is_regular(self.delx, self.dely):
            raise Exception("SpectSim2d() error: grid not regular")

        for v in self.geostruct.variograms:
            if v.bearing % 90.0 != 0.0:
                raise Exception("SpecSim2d only supports grid-aligned anisotropy...")

        # since we checked for grid regularity, we now can work in unit space
        # and use effective variograms:
        self.effective_variograms = []
        dist = self.delx[0]
        for v in self.geostruct.variograms:
            eff_v = type(v)(
                contribution=v.contribution,
                a=v.a / dist,
                bearing=v.bearing,
                anisotropy=v.anisotropy,
            )
            self.effective_variograms.append(eff_v)
        # pad the grid with 3X max range
        mx_a = -1.0e10
        for v in self.effective_variograms:
            mx_a = max(mx_a, v.a)
        mx_dim = max(self.delx.shape[0], self.dely.shape[0])
        freq_pad = int(np.ceil(mx_a * 3))
        freq_pad = int(np.ceil(freq_pad / 8.0) * 8.0)
        # use the max dimension so that simulation grid is square
        full_delx = np.ones((mx_dim + (2 * freq_pad)))
        full_dely = np.ones_like(full_delx)
        print(
            "SpecSim.initialize() summary: full_delx X full_dely: {0} X {1}".format(
                full_delx.shape[0], full_dely.shape[0]
            )
        )

        xdist = np.cumsum(full_delx)
        ydist = np.cumsum(full_dely)
        xdist -= xdist.min()
        ydist -= ydist.min()
        xgrid = np.zeros((ydist.shape[0], xdist.shape[0]))
        ygrid = np.zeros_like(xgrid)
        for j, d in enumerate(xdist):
            xgrid[:, j] = d
        for i, d in enumerate(ydist):
            ygrid[i, :] = d
        grid = np.array((xgrid, ygrid))
        domainsize = np.array((full_dely.shape[0], full_delx.shape[0]))
        for i in range(2):
            domainsize = domainsize[:, np.newaxis]
        grid = np.min((grid, np.array(domainsize) - grid), axis=0)
        # work out the contribution from each effective variogram and nugget
        c = np.zeros_like(xgrid)
        for v in self.effective_variograms:
            c += v._specsim_grid_contrib(grid)
        if self.geostruct.nugget > 0.0:
            h = ((grid ** 2).sum(axis=0)) ** 0.5
            c[np.where(h == 0)] += self.geostruct.nugget
        # fft components
        fftc = np.abs(np.fft.fftn(c))
        self.num_pts = np.prod(xgrid.shape)
        self.sqrt_fftc = np.sqrt(fftc / self.num_pts)

    def draw_arrays(self, num_reals=1, mean_value=1.0):
        """draw realizations

        Args:
            num_reals (`int`): number of realizations to generate
            mean_value (`float`): the mean value of the realizations

        Returns:
            `numpy.ndarray`: a 3-D array of realizations.  Shape
            is (num_reals,self.dely.shape[0],self.delx.shape[0])
        Note:
            log transformation is respected and the returned `reals` array is
            in linear space

        """
        reals = []

        for ireal in range(num_reals):
            real = np.random.standard_normal(size=self.sqrt_fftc.shape)
            imag = np.random.standard_normal(size=self.sqrt_fftc.shape)
            epsilon = real + 1j * imag
            rand = epsilon * self.sqrt_fftc
            real = np.real(np.fft.ifftn(rand)) * self.num_pts
            real = real[: self.dely.shape[0], : self.delx.shape[0]]
            reals.append(real)
        reals = np.array(reals)
        if self.geostruct.transform == "log":
            reals += np.log10(mean_value)
            reals = 10 ** reals

        else:
            reals += mean_value
        return reals

    def grid_par_ensemble_helper(
        self, pst, gr_df, num_reals, sigma_range=6, logger=None
    ):
        """wrapper around `SpecSim2d.draw()` designed to support `PstFromFlopy`
        and `PstFrom` grid-based parameters

        Args:
            pst (`pyemu.Pst`): a control file instance
            gr_df (`pandas.DataFrame`): a dataframe listing `parval1`,
                `pargp`, `i`, `j` for each grid based parameter
            num_reals (`int`): number of realizations to generate
            sigma_range (`float` (optional)): number of standard deviations
                implied by parameter bounds in control file. Default is 6
            logger (`pyemu.Logger` (optional)): a logger instance for logging

        Returns:
            `pyemu.ParameterEnsemble`: an untransformed parameter ensemble of
            realized grid-parameter values

        Note:
            the method processes each unique `pargp` value in `gr_df` and resets the sill of `self.geostruct` by
            the maximum bounds-implied variance of each `pargp`.  This method makes repeated calls to
            `self.initialize()` to deal with the geostruct changes.

        """

        if "i" not in gr_df.columns:
            print(gr_df.columns)
            raise Exception(
                "SpecSim2d.grid_par_ensmeble_helper() error: 'i' not in gr_df"
            )
        if "j" not in gr_df.columns:
            print(gr_df.columns)
            raise Exception(
                "SpecSim2d.grid_par_ensmeble_helper() error: 'j' not in gr_df"
            )
        if len(self.geostruct.variograms) > 1:
            raise Exception(
                "SpecSim2D grid_par_ensemble_helper() error: only a single variogram can be used..."
            )
        gr_df.loc[:, ["i", "j"]] = gr_df[["i", "j"]].astype(int)

        # scale the total contrib
        org_var = self.geostruct.variograms[0].contribution
        org_nug = self.geostruct.nugget
        new_var = org_var
        new_nug = org_nug
        if self.geostruct.sill != 1.0:
            print(
                "SpecSim2d.grid_par_ensemble_helper() warning: scaling contribution and nugget to unity"
            )
            tot = org_var + org_nug
            new_var = org_var / tot
            new_nug = org_nug / tot
            self.geostruct.variograms[0].contribution = new_var
            self.geostruct.nugget = new_nug

        gr_grps = gr_df.pargp.unique()
        pst.add_transform_columns()
        par = pst.parameter_data

        # real and name containers
        real_arrs, names = [], []
        for gr_grp in gr_grps:

            gp_df = gr_df.loc[gr_df.pargp == gr_grp, :]

            gp_par = par.loc[gp_df.parnme, :]
            # use the parval1 as the mean
            mean_arr = np.zeros((self.dely.shape[0], self.delx.shape[0])) + np.NaN
            mean_arr[gp_df.i, gp_df.j] = gp_par.parval1
            # fill missing mean values
            mean_arr[np.isnan(mean_arr)] = gp_par.parval1.mean()

            # use the max upper and min lower (transformed) bounds for the variance
            mx_ubnd = gp_par.parubnd_trans.max()
            mn_lbnd = gp_par.parlbnd_trans.min()
            var = ((mx_ubnd - mn_lbnd) / sigma_range) ** 2

            # update the geostruct
            self.geostruct.variograms[0].contribution = var * new_var
            self.geostruct.nugget = var * new_nug
            # print(gr_grp, var,new_var,mx_ubnd,mn_lbnd)
            # reinitialize and draw
            if logger is not None:
                logger.log(
                    "SpecSim: drawing {0} realization for group {1} with {4} pars, (log) variance {2} (sill {3})".format(
                        num_reals, gr_grp, var, self.geostruct.sill, gp_df.shape[0]
                    )
                )
            self.initialize()
            reals = self.draw_arrays(num_reals=num_reals, mean_value=mean_arr)
            # put the pieces into the par en
            reals = reals[:, gp_df.i, gp_df.j].reshape(num_reals, gp_df.shape[0])
            real_arrs.append(reals)
            names.extend(list(gp_df.parnme.values))
            if logger is not None:
                logger.log(
                    "SpecSim: drawing {0} realization for group {1} with {4} pars, (log) variance {2} (sill {3})".format(
                        num_reals, gr_grp, var, self.geostruct.sill, gp_df.shape[0]
                    )
                )

        # get into a dataframe
        reals = real_arrs[0]
        for r in real_arrs[1:]:
            reals = np.append(reals, r, axis=1)
        pe = pd.DataFrame(data=reals, columns=names)
        # reset to org conditions
        self.geostruct.nugget = org_nug
        self.geostruct.variograms[0].contribution = org_var
        self.initialize()

        return pe

    def draw_conditional(
        self,
        seed,
        obs_points,
        sg,
        base_values_file,
        local=True,
        factors_file=None,
        num_reals=1,
        mean_value=1.0,
        R_factor=1.0,
    ):

        """Generate a conditional, correlated random field using the Spec2dSim
            object, a set of observation points, and a factors file.

            The conditional field is made by generating an unconditional correlated random
            field that captures the covariance in the variogram and conditioning it by kriging
            a second surface using the value of the random field as observations.
            This second conditioning surface provides an estimate of uncertainty (kriging error)
            away from the observation points. At the observation points, the kriged surface is
            equal to (less nugget effects) the observation. The conditioned correlated field
            is then generated using: T(x) = Z(x) + [S(x) − S∗(x)]
            where T(x) is the conditioned simulation, Z(x) is a kriging estimator of the
            unknown field, S(x) is an unconditioned random field with the same covariance
            structure as the desired field, and S∗(x) is a kriging estimate of the unconditioned
            random field using its values at the observation points (pilot points).
            [S(x) − S∗(x)] is an estimate of the kriging error.

            This approach makes T(x) match the observed values at the observation points
            (x_a, y_z), T(a) = Z(a), and have a structure away from the observation points that
            follows the variogram used to generate Z, S, and S∗.

            Chiles, J-P, and Delfiner, P., Geostatistics- Modeling Spatial Uncertainty: Wiley,
                London, 695 p.

        Args:
            seed (`int`): integer used for random seed.  If seed is used as a PEST parameter,
                then passing the same value for seed will yield the same
                conditioned random fields. This allows runs to be recreated
                given an ensemble of seeds.
            obs_points (`str` or `dataframe`): locations for observation points.
                Either filename in pyemupilot point file format:
                ["name","x","y","zone","parval1"] ora dataframe with these columns.
                Note that parval1 is not used.
            base_values_file (`str`): filename containing 2d array with the base
                parameter values from which the random field will depart (Z(x)) above.
                Values of Z(x) are used for conditioning, not parval1 in the
                observation point file.
            factors_file (`str`): name of the factors file generated using the
                locations of the observation points and the target grid.
                If None this file will be generated and called conditional_factors.dat;
                but this is a slow step and should not generally be called for every simulation.
            sg: flopy StructuredGrid object
            local (`boolean`): whether coordinates in obs_points are in local (model) or map coordinates
            num_reals (`int`): number of realizations to generate
            mean_value (`float`): the mean value of the realizations
            R_factor (`float`): a factor to scale the field, sometimes the variation from the
                                geostruct parameters is larger or smaller than desired.

        Returns:
            `numpy.ndarray`: a 3-D array of realizations.  Shape is
                (num_reals, self.dely.shape[0], self.delx.shape[0])
        Note:
            log transformation is respected and the returned `reals`
                array is in arithmetic space
        """

        # get a dataframe for the observation points, from file unless passed
        if isinstance(obs_points, str):
            obs_points = pp_file_to_dataframe(obs_points)
        assert isinstance(obs_points, pd.DataFrame), "need a DataFrame, not {0}".format(
            type(obs_points)
        )

        # if factors_file is not passed, generate one from the geostruct associated
        # with the calling object and the observation points dataframe
        if factors_file is None:
            ok = OrdinaryKrige(self.geostruct, obs_points)
            ok.calc_factors_grid(sg, zone_array=None, var_filename=None)
            ok.to_grid_factors_file("conditional_factors.dat")
            factors_file = "conditional_factors.dat"

        # read in the base values, Z(x), assume these are not log-transformed
        values_krige = np.loadtxt(base_values_file)

        np.random.seed(int(seed))

        # draw random fields for num_reals
        unconditioned = self.draw_arrays(num_reals=num_reals, mean_value=mean_value)

        # If geostruct is log transformed, then work with log10 of field
        if self.geostruct.transform == "log":
            unconditioned = np.log10(unconditioned)

        # now do the conditioning by making another kriged surface with the
        # values of the unconditioned random fields at the pilot points
        conditioning_df = obs_points.copy()
        # need to row and column for the x and y values in the observation
        # dataframe, regular grid is tested when object is instantiated
        # so grid spacing can be used.
        conditioning_df["row"] = conditioning_df.apply(
            lambda row: sg.intersect(row["x"], row["y"], local=local)[0], axis=1
        )
        conditioning_df["col"] = conditioning_df.apply(
            lambda row: sg.intersect(row["x"], row["y"], local=local)[1], axis=1
        )
        reals = []
        for layer in range(0, num_reals):
            unconditioned[layer] = (
                unconditioned[layer] * R_factor
            )  # scale the unconditioned values
            conditioning_df["unconditioned"] = conditioning_df.apply(
                lambda row: unconditioned[layer][row["row"], row["col"]], axis=1
            )
            conditioning_df.to_csv(
                "unconditioned.dat",
                columns=["name", "x", "y", "zone", "unconditioned"],
                sep=" ",
                header=False,
                index=False,
            )
            # krige a surface using unconditioned observations to make the conditioning surface
            fac2real(
                pp_file="unconditioned.dat",
                factors_file=factors_file,
                out_file="conditioning.dat",
            )
            conditioning = np.loadtxt("conditioning.dat")

            if self.geostruct.transform == "log":
                conditioned = np.log10(values_krige) + (
                    unconditioned[layer] - conditioning
                )
                conditioned = np.power(10, conditioned)
            else:
                conditioned = values_krige + (unconditioned[layer] - conditioning)

            reals.append(conditioned)

        reals = np.array(reals)
        return reals


class OrdinaryKrige(object):
    """Ordinary Kriging using Pandas and Numpy.

    Args:
        geostruct (`GeoStruct`): a pyemu.geostats.GeoStruct to use for the kriging
        point_data (`pandas.DataFrame`): the conditioning points to use for kriging.
            `point_data` must contain columns "name", "x", "y".

    Note:
        if `point_data` is an `str`, then it is assumed to be a pilot points file
        and is loaded as such using `pyemu.pp_utils.pp_file_to_dataframe()`

        If zoned interpolation is used for grid-based interpolation, then
        `point_data` must also contain a "zone" column


    Example::

        import pyemu
        v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)
        gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)
        pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")
        ok = pyemu.utils.geostats.OrdinaryKrige(gs,pp_df)

    """

    def __init__(self, geostruct, point_data):
        if isinstance(geostruct, str):
            geostruct = read_struct_file(geostruct)
        assert isinstance(geostruct, GeoStruct), "need a GeoStruct, not {0}".format(
            type(geostruct)
        )
        self.geostruct = geostruct
        if isinstance(point_data, str):
            point_data = pp_file_to_dataframe(point_data)
        assert isinstance(point_data, pd.DataFrame)
        assert "name" in point_data.columns, "point_data missing 'name'"
        assert "x" in point_data.columns, "point_data missing 'x'"
        assert "y" in point_data.columns, "point_data missing 'y'"
        # check for duplicates in point data
        unique_name = point_data.name.unique()
        if len(unique_name) != point_data.shape[0]:
            warnings.warn(
                "duplicates detected in point_data..attempting to rectify", PyemuWarning
            )
            ux_std = point_data.groupby(point_data.name).std()["x"]
            if ux_std.max() > 0.0:
                raise Exception("duplicate point_info entries with different x values")
            uy_std = point_data.groupby(point_data.name).std()["y"]
            if uy_std.max() > 0.0:
                raise Exception("duplicate point_info entries with different y values")

            self.point_data = point_data.drop_duplicates(subset=["name"])
        else:
            self.point_data = point_data.copy()
        self.point_data.index = self.point_data.name
        self.check_point_data_dist()
        self.interp_data = None
        self.spatial_reference = None
        # X, Y = np.meshgrid(point_data.x,point_data.y)
        # self.point_data_dist = pd.DataFrame(data=np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2),
        #                                    index=point_data.name,columns=point_data.name)
        self.point_cov_df = self.geostruct.covariance_matrix(
            self.point_data.x, self.point_data.y, self.point_data.name
        ).to_dataframe()
        # for name in self.point_cov_df.index:
        #    self.point_cov_df.loc[name,name] -= self.geostruct.nugget

    def check_point_data_dist(self, rectify=False):
        """check for point_data entries that are closer than
        EPSILON distance - this will cause a singular kriging matrix.

        Args:
            rectify (`bool`): flag to fix the problems with point_data
                by dropping additional points that are
                closer than EPSILON distance.  Default is False

        Note:
            this method will issue warnings for points that are closer
            than EPSILON distance

        """

        ptx_array = self.point_data.x.values
        pty_array = self.point_data.y.values
        ptnames = self.point_data.name.values
        drop = []
        for i in range(self.point_data.shape[0]):
            ix, iy, iname = ptx_array[i], pty_array[i], ptnames[i]
            dist = pd.Series(
                (ptx_array[i + 1 :] - ix) ** 2 + (pty_array[i + 1 :] - iy) ** 2,
                ptnames[i + 1 :],
            )
            if dist.min() < EPSILON ** 2:
                print(iname, ix, iy)
                warnings.warn(
                    "points {0} and {1} are too close. This will cause a singular kriging matrix ".format(
                        iname, dist.idxmin()
                    ),
                    PyemuWarning,
                )
                drop_idxs = dist.loc[dist <= EPSILON ** 2]
                drop.extend([pt for pt in list(drop_idxs.index) if pt not in drop])
        if rectify and len(drop) > 0:
            print(
                "rectifying point data by removing the following points: {0}".format(
                    ",".join(drop)
                )
            )
            print(self.point_data.shape)
            self.point_data = self.point_data.loc[
                self.point_data.index.map(lambda x: x not in drop), :
            ]
            print(self.point_data.shape)

    # def prep_for_ppk2fac(self,struct_file="structure.dat",pp_file="points.dat",):
    #    pass

    def calc_factors_grid(
        self,
        spatial_reference,
        zone_array=None,
        minpts_interp=1,
        maxpts_interp=20,
        search_radius=1.0e10,
        verbose=False,
        var_filename=None,
        forgive=False,
        num_threads=1,
    ):
        """calculate kriging factors (weights) for a structured grid.

        Args:
            spatial_reference (`flopy.utils.reference.SpatialReference`): a spatial
                reference that describes the orientation and
                spatail projection of the the structured grid
            zone_array (`numpy.ndarray`): an integer array of zones to use for kriging.
                If not None, then `point_data` must also contain a "zone" column.  `point_data`
                entries with a zone value not found in zone_array will be skipped.
                If None, then all `point_data` will (potentially) be used for
                interpolating each grid node. Default is None
            minpts_interp (`int`): minimum number of `point_data` entires to use for interpolation at
                a given grid node.  grid nodes with less than `minpts_interp`
                `point_data` found will be skipped (assigned np.NaN).  Defaut is 1
            maxpts_interp (`int`) maximum number of `point_data` entries to use for interpolation at
                a given grid node.  A larger `maxpts_interp` will yield "smoother"
                interplation, but using a large `maxpts_interp` will slow the
                (already) slow kriging solution process and may lead to
                memory errors. Default is 20.
            search_radius (`float`) the size of the region around a given grid node to search for
                `point_data` entries. Default is 1.0e+10
            verbose : (`bool`): a flag to  echo process to stdout during the interpolatino process.
                Default is False
            var_filename (`str`): a filename to save the kriging variance for each interpolated grid node.
                Default is None.
            forgive (`bool`):  flag to continue if inversion of the kriging matrix failes at one or more
                grid nodes.  Inversion usually fails if the kriging matrix is singular,
                resulting from `point_data` entries closer than EPSILON distance.  If True,
                warnings are issued for each failed inversion.  If False, an exception
                is raised for failed matrix inversion.
            num_threads (`int`): number of multiprocessing workers to use to try to speed up
                kriging in python.  Default is 1.

        Returns:
            `pandas.DataFrame`: a dataframe with information summarizing the ordinary kriging
            process for each grid node

        Note:
            this method calls OrdinaryKrige.calc_factors()
            this method is the main entry point for grid-based kriging factor generation


        Example::

            import flopy
            v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)
            gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)
            pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")
            ok = pyemu.utils.geostats.OrdinaryKrige(gs,pp_df)
            m = flopy.modflow.Modflow.load("mymodel.nam")
            df = ok.calc_factors_grid(m.sr,zone_array=m.bas6.ibound[0].array,
                                      var_filename="ok_var.dat")
            ok.to_grid_factor_file("factors.dat")

        """

        self.spatial_reference = spatial_reference
        self.interp_data = None
        # assert isinstance(spatial_reference,SpatialReference)
        try:
            x = self.spatial_reference.xcentergrid.copy()
            y = self.spatial_reference.ycentergrid.copy()
        except Exception as e:
            raise Exception(
                "spatial_reference does not have proper attributes:{0}".format(str(e))
            )

        if var_filename is not None:
            arr = (
                np.zeros((self.spatial_reference.nrow, self.spatial_reference.ncol))
                - 1.0e30
            )

        # the simple case of no zone array: ignore point_data zones
        if zone_array is None:

            df = self.calc_factors(
                x.ravel(),
                y.ravel(),
                minpts_interp=minpts_interp,
                maxpts_interp=maxpts_interp,
                search_radius=search_radius,
                verbose=verbose,
                forgive=forgive,
                num_threads=num_threads,
            )

            if var_filename is not None:
                arr = df.err_var.values.reshape(x.shape)
                np.savetxt(var_filename, arr, fmt="%15.6E")

        if zone_array is not None:
            assert zone_array.shape == x.shape
            if "zone" not in self.point_data.columns:
                warnings.warn(
                    "'zone' columns not in point_data, assigning generic zone",
                    PyemuWarning,
                )
                self.point_data.loc[:, "zone"] = 1
            pt_data_zones = self.point_data.zone.unique()
            dfs = []
            for pt_data_zone in pt_data_zones:
                if pt_data_zone not in zone_array:
                    warnings.warn(
                        "pt zone {0} not in zone array {1}, skipping".format(
                            pt_data_zone, np.unique(zone_array)
                        ),
                        PyemuWarning,
                    )
                    continue
                # cutting list of cell positions to just in zone
                xzone = x[zone_array == pt_data_zone].copy()
                yzone = y[zone_array == pt_data_zone].copy()
                idx = np.arange(
                    len(zone_array.ravel())
                )[(zone_array == pt_data_zone).ravel()]
                # xzone[zone_array != pt_data_zone] = np.NaN
                # yzone[zone_array != pt_data_zone] = np.NaN

                df = self.calc_factors(
                    xzone,
                    yzone,
                    idx_vals=idx,  # need to pass if xzone,yzone is not all x,y
                    minpts_interp=minpts_interp,
                    maxpts_interp=maxpts_interp,
                    search_radius=search_radius,
                    verbose=verbose,
                    pt_zone=pt_data_zone,
                    forgive=forgive,
                    num_threads=num_threads,
                )

                dfs.append(df)
                if var_filename is not None:
                    # rebuild full df so we can build array, as per
                    fulldf = pd.DataFrame(data={"x": x.ravel(), "y": y.ravel()})
                    fulldf[['idist', 'inames', 'ifacts', 'err_var']] = np.array(
                        [[[], [], [], np.nan]] * len(fulldf),
                        dtype=object)
                    fulldf = fulldf.set_index(['x', 'y'])
                    fulldf.loc[df.set_index(['x', 'y']).index] = df.set_index(['x', 'y'])
                    fulldf = fulldf.reset_index()
                    a = fulldf.err_var.values.reshape(x.shape)
                    na_idx = np.isfinite(a.astype(float))
                    arr[na_idx] = a[na_idx]
            if self.interp_data is None or self.interp_data.dropna().shape[0] == 0:
                raise Exception("no interpolation took place...something is wrong")
            df = pd.concat(dfs)
        if var_filename is not None:
            np.savetxt(var_filename, arr, fmt="%15.6E")
        return df

    def _dist_calcs(self, ix, iy, ptx_array, pty_array, ptnames, sqradius):
        """private: find nearby points"""
        #  calc dist from this interp point to all point data...slow
        dist = pd.Series((ptx_array - ix) ** 2 + (pty_array - iy) ** 2, ptnames)
        dist.sort_values(inplace=True)
        dist = dist.loc[dist <= sqradius]
        return dist

    def _remove_neg_factors(self):
        """
        private function to remove negative kriging factors and
        renormalize remaining positive factors following the
        method of Deutsch (1996):
        https://doi.org/10.1016/0098-3004(96)00005-2


        """
        newd, newn, newf = (
            [],
            [],
            [],
        )
        for d, n, f in zip(
            self.interp_data.idist.values,
            self.interp_data.inames.values,
            self.interp_data.ifacts.values,
        ):
            # if the factor list is empty, no changes are made
            # if the factor list has only one value, it is 1.0 so no changes
            # if more than one factor, remove negatives and renormalize
            if len(f) > 1:
                # only keep dist, names, and factors of factor > 0
                d = np.array(d)
                n = np.array(n)
                f = np.array(f)
                d = d[f > 0]
                n = n[f > 0]
                f = f[f > 0]
                f /= f.sum()  # renormalize to sum to unity
            newd.append(d)
            newn.append(n)
            newf.append(f)
        # update the interp_data dataframe
        self.interp_data.idist = newd
        self.interp_data.inames = newn
        self.interp_data.ifacts = newf

    def _cov_points(self, ix, iy, pt_names):
        """private: get covariance between points"""
        interp_cov = self.geostruct.covariance_points(
            ix,
            iy,
            self.point_data.loc[pt_names, "x"],
            self.point_data.loc[pt_names, "y"],
        )

        return interp_cov

    def _form(self, pt_names, point_cov, interp_cov):
        """private: form the kriging equations"""

        d = len(pt_names) + 1  # +1 for lagrange mult
        A = np.ones((d, d))
        A[:-1, :-1] = point_cov.values
        A[-1, -1] = 0.0  # unbiaised constraint
        rhs = np.ones((d, 1))
        rhs[:-1, 0] = interp_cov
        return A, rhs

    def _solve(self, A, rhs):
        return np.linalg.solve(A, rhs)

    def calc_factors(
        self,
        x,
        y,
        minpts_interp=1,
        maxpts_interp=20,
        search_radius=1.0e10,
        verbose=False,
        pt_zone=None,
        forgive=False,
        num_threads=1,
        idx_vals=None,
        remove_negative_factors=True,
    ):
        """calculate ordinary kriging factors (weights) for the points
        represented by arguments x and y

        Args:
            x ([`float`]):  x-coordinates to calculate kriging factors for
            y (([`float`]): y-coordinates to calculate kriging factors for
            minpts_interp (`int`): minimum number of point_data entires to use for interpolation at
                a given x,y interplation point.  interpolation points with less
                than `minpts_interp` `point_data` found will be skipped
                (assigned np.NaN).  Defaut is 1
            maxpts_interp (`int`): maximum number of point_data entries to use for interpolation at
                a given x,y interpolation point.  A larger `maxpts_interp` will
                yield "smoother" interplation, but using a large `maxpts_interp`
                will slow the (already) slow kriging solution process and may
                lead to memory errors. Default is 20.
            search_radius (`float`): the size of the region around a given x,y
                interpolation point to search for `point_data` entries. Default is 1.0e+10
            verbose (`bool`): a flag to  echo process to stdout during the interpolatino process.
                Default is False
            forgive (`bool`): flag to continue if inversion of the kriging matrix failes at one or more
                interpolation points.  Inversion usually fails if the kriging matrix is singular,
                resulting from `point_data` entries closer than EPSILON distance.  If True,
                warnings are issued for each failed inversion.  If False, an exception
                is raised for failed matrix inversion.
            num_threads (`int`): number of multiprocessing workers to use to try to speed up
                kriging in python.  Default is 1.
            idx_vals (iterable of `int`): optional index values to use in the interpolation dataframe.  This is
                used to set the proper node number in the factors file for unstructured grids.
            remove_negative_factors (`bool`): option to remove negative Kriging factors, following the method of
                Deutsch (1996) https://doi.org/10.1016/0098-3004(96)00005-2. Default is True
        Returns:
            `pandas.DataFrame`: a dataframe with information summarizing the ordinary kriging
            process for each interpolation points

        Note:
            this method calls either `OrdinaryKrige.calc_factors_org()` or
            `OrdinaryKrige.calc_factors_mp()` depending on the value of `num_threads`

        Example::

            v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)
            gs = pyemu.utils.geostats.GeoStruct(variograms=v,nugget=0.5)
            pp_df = pyemu.pp_utils.pp_file_to_dataframe("hkpp.dat")
            ok = pyemu.utils.geostats.OrdinaryKrige(gs,pp_df)
            x = np.arange(100)
            y = np.ones_like(x)
            zone_array = y.copy()
            zone_array[:zone_array.shape[0]/2] = 2
            # only calc factors for the points in zone 1
            ok.calc_factors(x,y,pt_zone=1)
            ok.to_grid_factors_file("zone_1.fac",ncol=x.shape[0])



        """
        # can do this up here, as same between org and mp method
        assert len(x) == len(y)
        if idx_vals is not None and len(idx_vals) != len(x):
            raise Exception("len(idx_vals) != len(x)")
        # find the point data to use for each interp point
        df = pd.DataFrame(data={"x": x, "y": y})
        if idx_vals is not None:
            df.index = np.array(idx_vals).astype(int)
        # now can just pass df (contains x and y)
        # trunc to just deal with pp locations in zones
        pt_data = self.point_data
        if pt_zone is None:
            ptx_array = self.point_data.x.values
            pty_array = self.point_data.y.values
            ptnames = self.point_data.name.values
        else:
            ptx_array = pt_data.loc[pt_data.zone == pt_zone, "x"].values
            pty_array = pt_data.loc[pt_data.zone == pt_zone, "y"].values
            ptnames = pt_data.loc[pt_data.zone == pt_zone, "name"].values
            # pt_data = pt_data.loc[ptnames]
        if num_threads == 1:
            return self._calc_factors_org(
                df,
                ptx_array,
                pty_array,
                ptnames,
                minpts_interp,
                maxpts_interp,
                search_radius,
                verbose,
                pt_zone,
                forgive,
                remove_negative_factors,
            )
        else:
            return self._calc_factors_mp(
                df,
                ptx_array,
                pty_array,
                ptnames,
                minpts_interp,
                maxpts_interp,
                search_radius,
                verbose,
                pt_zone,
                forgive,
                num_threads,
                remove_negative_factors,
            )

    def _calc_factors_org(
        self,
        df,
        ptx_array,
        pty_array,
        ptnames,
        minpts_interp=1,
        maxpts_interp=20,
        search_radius=1.0e10,
        verbose=False,
        pt_zone=None,
        forgive=False,
        remove_negative_factors=True,
    ):
        # assert len(x) == len(y)
        # if idx_vals is not None and len(idx_vals) != len(x):
        #     raise Exception("len(idx_vals) != len(x)")
        #
        # df = pd.DataFrame(data={"x": x, "y": y})
        # if idx_vals is not None:
        #     df.index = [int(i) for i in idx_vals]
        inames, idist, ifacts, err_var = [], [], [], []
        sill = self.geostruct.sill
        # ptnames = ptd.name.values
        # find the point data to use for each interp point
        sqradius = search_radius ** 2
        print("starting interp point loop for {0} points".format(df.shape[0]))
        start_loop = datetime.now()
        for idx, (ix, iy) in enumerate(zip(df.x, df.y)):
            if np.isnan(ix) or np.isnan(iy):  # if nans, skip
                inames.append([])
                idist.append([])
                ifacts.append([])
                err_var.append(np.NaN)
                continue
            if verbose:
                istart = datetime.now()
                print("processing interp point:{0} of {1}".format(idx, df.shape[0]))
            if verbose == 2:
                start = datetime.now()
                print("calc ipoint dist...", end='')

            #  calc dist from this interp point to all point data...slow
            # dist = pd.Series((ptx_array-ix)**2 + (pty_array-iy)**2,ptnames)
            # dist.sort_values(inplace=True)
            # dist = dist.loc[dist <= sqradius]
            # def _dist_calcs(self, ix, iy, ptx_array, pty_array, ptnames, sqradius):
            dist = self._dist_calcs(ix, iy, ptx_array, pty_array, ptnames,
                                    sqradius)

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

            # vextract the point-to-point covariance matrix
            point_cov = self.point_cov_df.loc[pt_names, pt_names]
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            #     print("forming ipt-to-point cov...",end='')

            # calc the interp point to points covariance
            # interp_cov = self.geostruct.covariance_points(ix,iy,self.point_data.loc[pt_names,"x"],
            #                                               self.point_data.loc[pt_names,"y"])
            interp_cov = self._cov_points(ix, iy, pt_names)
            if verbose == 2:
                td = (datetime.now() - start).total_seconds()
                print("...took {0} seconds".format(td))
                print("forming lin alg components...", end="")

            # form the linear algebra parts and solve
            # d = len(pt_names) + 1 # +1 for lagrange mult
            # A = np.ones((d,d))
            # A[:-1,:-1] = point_cov.values
            # A[-1,-1] = 0.0 #unbiaised constraint
            # rhs = np.ones((d,1))
            # rhs[:-1,0] = interp_cov
            A, rhs = self._form(pt_names, point_cov, interp_cov)

            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            #     print("solving...",end='')
            # # solve
            try:
                facs = self._solve(A, rhs)
            except Exception as e:
                print("error solving for factors: {0}".format(str(e)))
                print("point:", ix, iy)
                print("dist:", dist)
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

            err_var.append(
                float(
                    sill
                    + facs[-1]
                    - sum([f * c for f, c in zip(facs[:-1], interp_cov)])
                )
            )
            inames.append(pt_names)

            idist.append(dist.values)
            ifacts.append(facs[:-1, 0])
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))
            if verbose:
                td = (datetime.now() - istart).total_seconds()
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
        # correct for negative kriging factors, if requested
        if remove_negative_factors == True:
            self._remove_neg_factors()
        td = (datetime.now() - start_loop).total_seconds()
        print("took {0} seconds".format(td))
        return df

    def _calc_factors_mp(
        self,
        df,
        ptx_array,
        pty_array,
        ptnames,
        minpts_interp=1,
        maxpts_interp=20,
        search_radius=1.0e10,
        verbose=False,
        pt_zone=None,
        forgive=False,
        num_threads=1,
        remove_negative_factors=True,
    ):
        sqradius = search_radius ** 2
        print("starting interp point loop for {0} points".format(df.shape[0]))
        start_loop = datetime.now()
        # ensure same order as point data and just pass array
        # ptnames = ptd.name
        # point_data = self.point_data.loc[self.point_data.zone == pt_zone]
        point_cov_data = self.point_cov_df.loc[ptnames, ptnames].values
        point_pairs = [(i, xx, yy) for i, (xx, yy) in enumerate(zip(df.x, df.y))]
        idist = [[]] * len(df.x)
        inames = [[]] * len(df.x)
        ifacts = [[]] * len(df.x)
        err_var = [np.NaN] * len(df.x)
        with mp.Manager() as manager:
            point_pairs = manager.list(point_pairs)
            idist = manager.list(idist)
            inames = manager.list(inames)
            ifacts = manager.list(ifacts)
            err_var = manager.list(err_var)
            lock = mp.Lock()
            procs = []
            for i in range(num_threads):
                print("starting", i)
                p = mp.Process(
                    target=OrdinaryKrige._worker,
                    args=(
                        ptx_array,
                        pty_array,
                        ptnames,
                        point_pairs,
                        inames,
                        idist,
                        ifacts,
                        err_var,
                        point_cov_data,
                        self.geostruct,
                        EPSILON,
                        sqradius,
                        minpts_interp,
                        maxpts_interp,
                        lock,
                    ),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

            df[["idist", "inames", "ifacts"]] = pd.DataFrame(
                [[s[0], n[0], f[0]] for s, n, f in zip(idist, inames, ifacts)],
                columns=["idist", "inames", "ifacts"], index=df.index)

            df["err_var"] = [
                float(e[0]) if not isinstance(e[0], list) else float(e[0][0])
                for e in err_var
            ]

        if pt_zone is None:
            self.interp_data = df
        else:
            if self.interp_data is None:
                self.interp_data = df
            else:
                self.interp_data = pd.concat([self.interp_data, df])
        # correct for negative kriging factors, if requested
        if remove_negative_factors == True:
            self._remove_neg_factors()
        td = (datetime.now() - start_loop).total_seconds()
        print("took {0} seconds".format(td))
        return df

    @staticmethod
    def _worker(
        ptx_array,
        pty_array,
        ptnames,
        point_pairs,
        inames,
        idist,
        ifacts,
        err_var,
        full_point_cov,
        geostruct,
        epsilon,
        sqradius,
        minpts_interp,
        maxpts_interp,
        lock,
    ):
        sill = geostruct.sill
        while True:
            if len(point_pairs) == 0:
                return
            else:
                try:
                    idx, ix, iy = point_pairs.pop(0)
                except IndexError:
                    return

            # if idx % 1000 == 0 and idx != 0:
            #    print (ithread, idx,"done",datetime.now())
            if np.isnan(ix) or np.isnan(iy):  # if nans, skip
                ifacts[idx] = [[]]
                idist[idx] = [[]]
                inames[idx] = [[]]
                err_var[idx] = [np.NaN]
                continue

            # calc dist from this interp point to all point data...
            # can we just use a numpy approach...?
            dist = (ptx_array - ix) ** 2 + (pty_array - iy) ** 2
            sortorder = np.argsort(dist)
            dist = dist[sortorder]
            pt_names = ptnames[sortorder]
            trunc = dist <= sqradius
            dist = dist[trunc]
            pt_names = pt_names[trunc]
            sortorder = sortorder[trunc]

            # if too few points were found, skip
            if len(dist) < minpts_interp:
                ifacts[idx] = [[]]
                idist[idx] = [[]]
                inames[idx] = [[]]
                err_var[idx] = [sill]
                continue

            # only the maxpts_interp points
            dist = np.sqrt(dist[:maxpts_interp])
            pt_names = pt_names[:maxpts_interp]
            sortorder = sortorder[:maxpts_interp]

            # if one of the points is super close, just use it and skip
            if dist[0] <= epsilon:
                ifacts[idx] = [[1.0]]
                idist[idx] = [[epsilon]]
                inames[idx] = [[pt_names[0]]]
                err_var[idx] = [[geostruct.nugget]]
                continue

            # vextract the point-to-point covariance matrix
            # point_cov = full_point_cov.loc[pt_names, pt_names]
            point_cov = full_point_cov[tuple([sortorder[:, None], sortorder])]
            # calc the interp point to points covariance
            interp_cov = geostruct.covariance_points(
                ix, iy, ptx_array[sortorder], pty_array[sortorder]
            )

            # form the linear algebra parts and solve
            d = len(pt_names) + 1  # +1 for lagrange mult
            A = np.ones((d, d))
            A[:-1, :-1] = point_cov  # .values
            A[-1, -1] = 0.0  # unbiaised constraint
            rhs = np.ones((d, 1))
            rhs[:-1, 0] = interp_cov

            try:
                facs = np.linalg.solve(A, rhs)
            except Exception as e:
                print("error solving for factors: {0}".format(str(e)))
                print("point:", ix, iy)
                print("dist:", dist)
                print("A:", A)
                print("rhs:", rhs)
                continue

            assert len(facs) - 1 == len(dist)

            err_var[idx] = [
                float(
                    sill
                    + facs[-1]
                    - sum([f * c for f, c in zip(facs[:-1], interp_cov)])
                )
            ]
            inames[idx] = [pt_names.tolist()]
            idist[idx] = [dist.tolist()]
            ifacts[idx] = [facs[:-1, 0].tolist()]
            # if verbose == 2:
            #     td = (datetime.now()-start).total_seconds()
            #     print("...took {0}".format(td))

    def to_grid_factors_file(
        self, filename, points_file="points.junk", zone_file="zone.junk", ncol=None
    ):
        """write a PEST-style factors file.  This file can be used with
        the fac2real() method to write an interpolated structured or unstructured array

        Args:
            filename (`str`): factor filename
            points_file (`str`): points filename to add to the header of the factors file.
                This is not used by the fac2real() method.  Default is "points.junk"
            zone_file (`str`): zone filename to add to the header of the factors file.
                This is not used by the fac2real() method.  Default is "zone.junk"

            ncol (`int`) column value to write to factors file.  This is normally determined
                from the spatial reference and should only be passed for unstructured grids -
                it should be equal to the number of nodes in the current property file. Default is None.
                Required for unstructured grid models.

        Note:
            this method should be called after OrdinaryKrige.calc_factors_grid() for structured
            models or after OrdinaryKrige.calc_factors() for unstructured models.

        """
        if self.interp_data is None:
            raise Exception(
                "ok.interp_data is None, must call calc_factors_grid() first"
            )
        if self.spatial_reference is None:
            print(
                "OrdinaryKrige.to_grid_factors_file(): spatial_reference attr is None, assuming unstructured grid"
            )
            if ncol is None:
                raise Exception("'ncol' arg must be passed for unstructured grids")
            nrow = 1
            if ncol < self.interp_data.shape[0]:
                raise Exception("something is wrong")
        else:
            nrow = self.spatial_reference.nrow
            ncol = self.spatial_reference.ncol
        with open(filename, "w") as f:
            f.write(points_file + "\n")
            f.write(zone_file + "\n")
            f.write("{0} {1}\n".format(ncol, nrow))
            f.write("{0}\n".format(self.point_data.shape[0]))
            [f.write("{0}\n".format(name)) for name in self.point_data.name]
            t = 0
            if self.geostruct.transform == "log":
                t = 1
            pt_names = list(self.point_data.name)
            for idx, names, facts in zip(
                self.interp_data.index, self.interp_data.inames, self.interp_data.ifacts
            ):
                if len(facts) == 0:
                    continue
                n_idxs = [pt_names.index(name) for name in names]
                f.write("{0} {1} {2} {3:8.5e} ".format(idx + 1, t, len(names), 0.0))
                [
                    f.write("{0} {1:12.8g} ".format(i + 1, w))
                    for i, w in zip(n_idxs, facts)
                ]
                f.write("\n")


class Vario2d(object):
    """base class for 2-D variograms.

    Args:
        contribution (float): sill of the variogram
        a (`float`): (practical) range of correlation
        anisotropy (`float`, optional): Anisotropy ratio. Default is 1.0
        bearing : (`float`, optional): angle in degrees East of North corresponding
            to anisotropy ellipse. Default is 0.0
        name (`str`, optinoal): name of the variogram.  Default is "var1"

    Note:
        This base class should not be instantiated directly as it does not implement
        an h_function() method.

    """

    def __init__(self, contribution, a, anisotropy=1.0, bearing=0.0, name="var1"):
        self.name = name
        self.epsilon = EPSILON
        self.contribution = float(contribution)
        assert self.contribution > 0.0
        self.a = float(a)
        assert self.a > 0.0
        self.anisotropy = float(anisotropy)
        assert self.anisotropy > 0.0
        self.bearing = float(bearing)

    def same_as_other(self, other):
        if type(self) != type(other):
            return False
        if self.contribution != other.contribution:
            return False
        if self.anisotropy != other.anisotropy:
            return False
        if self.a != other.a:
            return False
        if self.bearing != other.bearing:
            return False
        return True

    def to_struct_file(self, f):
        """write the `Vario2d` to a PEST-style structure file

        Args:
            f (`str`): filename to write to.  `f` can also be an open
                file handle.

        """
        if isinstance(f, str):
            f = open(f, "w")
        f.write("VARIOGRAM {0}\n".format(self.name))
        f.write("  VARTYPE {0}\n".format(self.vartype))
        f.write("  A {0}\n".format(self.a))
        f.write("  ANISOTROPY {0}\n".format(self.anisotropy))
        f.write("  BEARING {0}\n".format(self.bearing))
        f.write("END VARIOGRAM\n\n")

    @property
    def bearing_rads(self):
        """get the bearing of the Vario2d in radians

        Returns:
            `float`: the Vario2d bearing in radians
        """
        return (np.pi / 180.0) * (90.0 - self.bearing)

    @property
    def rotation_coefs(self):
        """get the rotation coefficents in radians

        Returns:
            [`float`]: the rotation coefficients implied by `Vario2d.bearing`


        """
        return [
            np.cos(self.bearing_rads),
            np.sin(self.bearing_rads),
            -1.0 * np.sin(self.bearing_rads),
            np.cos(self.bearing_rads),
        ]

    def inv_h(self, h):
        """the inverse of the h_function.  Used for plotting

        Args:
            h (`float`): the value of h_function to invert

        Returns:
            `float`: the inverse of h

        """
        return self.contribution - self._h_function(h)

    def plot(self, **kwargs):
        """get a cheap plot of the Vario2d

        Args:
            **kwargs (`dict`): keyword arguments to use for plotting

        Returns:
            `matplotlib.pyplot.axis`

        Note:
            optional arguments in kwargs include
            "ax" (existing `matplotlib.pyplot.axis`).  Other
            kwargs are passed to `matplotlib.pyplot.plot()`

        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise Exception("error importing matplotlib: {0}".format(str(e)))

        ax = kwargs.pop("ax", plt.subplot(111))
        x = np.linspace(0, self.a * 3, 100)
        y = self.inv_h(x)
        ax.set_xlabel("distance")
        ax.set_ylabel(r"$\gamma$")
        ax.plot(x, y, **kwargs)
        return ax

    def covariance_matrix(self, x, y, names=None, cov=None):
        """build a pyemu.Cov instance implied by Vario2d

        Args:
            x ([`float`]): x-coordinate locations
            y ([`float`]): y-coordinate locations
            names ([`str`]): names of locations. If None, cov must not be None
            cov (`pyemu.Cov`): an existing Cov instance.  Vario2d contribution is added to cov
            in place

        Returns:
            `pyemu.Cov`: the covariance matrix for `x`, `y` implied by `Vario2d`

        Note:
            either `names` or `cov` must not be None.

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]

        if names is not None:
            assert x.shape[0] == len(names)
            c = np.zeros((len(names), len(names)))
            np.fill_diagonal(c, self.contribution)
            cov = Cov(x=c, names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
            c = np.zeros((len(names), 1)) + self.contribution
            cont = Cov(x=c, names=names, isdiagonal=True)
            cov += cont

        else:
            raise Exception(
                "Vario2d.covariance_matrix() requires either" + "names or cov arg"
            )
        rc = self.rotation_coefs
        for i1, (n1, x1, y1) in enumerate(zip(names, x, y)):
            dx = x1 - x[i1 + 1 :]
            dy = y1 - y[i1 + 1 :]
            dxx, dyy = self._apply_rotation(dx, dy)
            h = np.sqrt(dxx * dxx + dyy * dyy)

            h[h < 0.0] = 0.0
            h = self._h_function(h)
            if np.any(np.isnan(h)):
                raise Exception("nans in h for i1 {0}".format(i1))
            cov.x[i1, i1 + 1 :] += h
        for i in range(len(names)):
            cov.x[i + 1 :, i] = cov.x[i, i + 1 :]
        return cov

    def _specsim_grid_contrib(self, grid):
        rot_grid = grid
        if self.bearing % 90.0 != 0:
            dx, dy = self._apply_rotation(grid[0, :, :], grid[1, :, :])
            rot_grid = np.array((dx, dy))
        h = ((rot_grid ** 2).sum(axis=0)) ** 0.5
        c = self._h_function(h)
        return c

    def _apply_rotation(self, dx, dy):
        """private method to rotate points
        according to Vario2d.bearing and Vario2d.anisotropy


        """
        if self.anisotropy == 1.0:
            return dx, dy
        rcoefs = self.rotation_coefs
        dxx = (dx * rcoefs[0]) + (dy * rcoefs[1])
        dyy = ((dx * rcoefs[2]) + (dy * rcoefs[3])) * self.anisotropy
        return dxx, dyy

    def covariance_points(self, x0, y0, xother, yother):
        """get the covariance between base point (x0,y0) and
        other points xother,yother implied by `Vario2d`

        Args:
            x0 (`float`): x-coordinate
            y0 (`float`): y-coordinate
            xother ([`float`]): x-coordinates of other points
            yother ([`float`]): y-coordinates of other points

        Returns:
            `numpy.ndarray`: a 1-D array of covariance between point x0,y0 and the
            points contained in xother, yother.  len(cov) = len(xother) =
            len(yother)

        """
        dxx = x0 - xother
        dyy = y0 - yother
        dxx, dyy = self._apply_rotation(dxx, dyy)
        h = np.sqrt(dxx * dxx + dyy * dyy)
        return self._h_function(h)

    def covariance(self, pt0, pt1):
        """get the covarince between two points implied by Vario2d

        Args:
            pt0 : ([`float`]): first point x and y
            pt1 : ([`float`]): second point x and y

        Returns:
            `float`: covariance between pt0 and pt1

        """

        x = np.array([pt0[0], pt1[0]])
        y = np.array([pt0[1], pt1[1]])
        names = ["n1", "n2"]
        return self.covariance_matrix(x, y, names=names).x[0, 1]

    def __str__(self):
        """get the str representation of Vario2d

        Returns:
            `str`: string rep
        """
        s = "name:{0},contribution:{1},a:{2},anisotropy:{3},bearing:{4}\n".format(
            self.name, self.contribution, self.a, self.anisotropy, self.bearing
        )
        return s


class ExpVario(Vario2d):
    """Exponential variogram derived type

    Args:
        contribution (float): sill of the variogram
        a (`float`): (practical) range of correlation
        anisotropy (`float`, optional): Anisotropy ratio. Default is 1.0
        bearing : (`float`, optional): angle in degrees East of North corresponding
            to anisotropy ellipse. Default is 0.0
        name (`str`, optinoal): name of the variogram.  Default is "var1"

    Example::

        v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)


    """

    def __init__(self, contribution, a, anisotropy=1.0, bearing=0.0, name="var1"):
        super(ExpVario, self).__init__(
            contribution, a, anisotropy=anisotropy, bearing=bearing, name=name
        )
        self.vartype = 2

    def _h_function(self, h):
        """private method exponential variogram "h" function"""
        return self.contribution * np.exp(-1.0 * h / self.a)


class GauVario(Vario2d):
    """Gaussian variogram derived type

    Args:
        contribution (float): sill of the variogram
        a (`float`): (practical) range of correlation
        anisotropy (`float`, optional): Anisotropy ratio. Default is 1.0
        bearing : (`float`, optional): angle in degrees East of North corresponding
            to anisotropy ellipse. Default is 0.0
        name (`str`, optinoal): name of the variogram.  Default is "var1"

    Example::

        v = pyemu.utils.geostats.GauVario(a=1000,contribution=1.0)

    Note:
        the Gaussian variogram can be unstable (not invertible) for long ranges.

    """

    def __init__(self, contribution, a, anisotropy=1.0, bearing=0.0, name="var1"):
        super(GauVario, self).__init__(
            contribution, a, anisotropy=anisotropy, bearing=bearing, name=name
        )
        self.vartype = 3

    def _h_function(self, h):
        """private method for the gaussian variogram "h" function"""

        hh = -1.0 * (h * h) / (self.a * self.a)
        return self.contribution * np.exp(hh)


class SphVario(Vario2d):
    """Spherical variogram derived type

    Args:
         contribution (float): sill of the variogram
         a (`float`): (practical) range of correlation
         anisotropy (`float`, optional): Anisotropy ratio. Default is 1.0
         bearing : (`float`, optional): angle in degrees East of North corresponding
             to anisotropy ellipse. Default is 0.0
         name (`str`, optinoal): name of the variogram.  Default is "var1"

     Example::

         v = pyemu.utils.geostats.SphVario(a=1000,contribution=1.0)

    """

    def __init__(self, contribution, a, anisotropy=1.0, bearing=0.0, name="var1"):
        super(SphVario, self).__init__(
            contribution, a, anisotropy=anisotropy, bearing=bearing, name=name
        )
        self.vartype = 1

    def _h_function(self, h):
        """private method for the spherical variogram "h" function"""

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
        # return h
        # if hh < 1.0:
        #     return self.contribution * (1.0 - (hh * (1.5 - (0.5 * hh * hh))))
        # else:
        #     return 0.0


def read_struct_file(struct_file, return_type=GeoStruct):
    """read an existing PEST-type structure file into a GeoStruct instance

    Args:
        struct_file (`str`): existing pest-type structure file
        return_type (`object`): the instance type to return.  Default is GeoStruct

    Returns:
        [`pyemu.GeoStruct`]: list of `GeoStruct` instances.  If
        only one `GeoStruct` is in the file, then a `GeoStruct` is returned


    Example::

        gs = pyemu.utils.geostats.reads_struct_file("struct.dat")


    """

    VARTYPE = {1: SphVario, 2: ExpVario, 3: GauVario, 4: None}
    assert os.path.exists(struct_file)
    structures = []
    variograms = []
    with open(struct_file, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip().lower()
            if line.startswith("structure"):
                name = line.strip().split()[1]
                nugget, transform, variogram_info = _read_structure_attributes(f)
                s = return_type(nugget=nugget, transform=transform, name=name)
                s.variogram_info = variogram_info
                # not sure what is going on, but if I don't copy s here,
                # all the structures end up sharing all the variograms later
                structures.append(copy.deepcopy(s))
            elif line.startswith("variogram"):
                name = line.strip().split()[1].lower()
                vartype, bearing, a, anisotropy = _read_variogram(f)
                if name in variogram_info:
                    v = VARTYPE[vartype](
                        variogram_info[name],
                        a,
                        anisotropy=anisotropy,
                        bearing=bearing,
                        name=name,
                    )
                    variograms.append(v)

    for i, st in enumerate(structures):
        for vname in st.variogram_info:
            vfound = None
            for v in variograms:
                if v.name == vname:
                    vfound = v
                    break
            if vfound is None:
                raise Exception(
                    "variogram {0} not found for structure {1}".format(vname, s.name)
                )

            st.variograms.append(vfound)
    if len(structures) == 1:
        return structures[0]
    return structures


def _read_variogram(f):
    """Function to instantiate a Vario2d from a PEST-style structure file"""

    line = ""
    vartype = None
    bearing = 0.0
    a = None
    anisotropy = 1.0
    while "end variogram" not in line:
        line = f.readline()
        if line == "":
            raise Exception("EOF while read variogram")
        line = line.strip().lower().split()
        if line[0].startswith("#"):
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
    return vartype, bearing, a, anisotropy


def _read_structure_attributes(f):
    """function to read information from a PEST-style structure file"""

    line = ""
    variogram_info = {}
    while "end structure" not in line:
        line = f.readline()
        if line == "":
            raise Exception("EOF while reading structure")
        line = line.strip().lower().split()
        if line[0].startswith("#"):
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
        elif line[0] == "mean":
            warnings.warn("'mean' attribute not supported, skipping", PyemuWarning)
        else:
            raise Exception(
                "unrecognized line in structure definition:{0}".format(line[0])
            )
    assert numvariograms == len(variogram_info)
    return nugget, transform, variogram_info


def read_sgems_variogram_xml(xml_file, return_type=GeoStruct):
    """function to read an SGEMS-type variogram XML file into
    a `GeoStruct`

    Args:
        xml_file (`str`): SGEMS variogram XML file
        return_type (`object`): the instance type to return.  Default is `GeoStruct`

    Returns:
        gs : `GeoStruct`


    Example::

        gs = pyemu.utils.geostats.read_sgems_variogram_xml("sgems.xml")

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
    for key, val in gs_model.items():
        # print(key,val)
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
        mx_range, mn_range = None, None
        x_angle, y_angle = None, None
        # struct_name = structure.tag
        for key, val in structure.items():
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
        v = vtype(
            contribution=contribution,
            a=mx_range,
            anisotropy=mx_range / mn_range,
            bearing=(180.0 / np.pi) * np.arctan2(x_angle, y_angle),
            name=structure.tag,
        )
        return GeoStruct(nugget=nugget, variograms=[v])


def gslib_2_dataframe(filename, attr_name=None, x_idx=0, y_idx=1):
    """function to read a GSLIB point data file into a pandas.DataFrame

    Args:
        filename (`str`): GSLIB file
        attr_name (`str`): the column name in the dataframe for the attribute.
            If None, GSLIB file can have only 3 columns.  `attr_name` must be in
            the GSLIB file header
        x_idx (`int`): the index of the x-coordinate information in the GSLIB file. Default is
            0 (first column)
        y_idx (`int`): the index of the y-coordinate information in the GSLIB file.
            Default is 1 (second column)

    Returns:
        `pandas.DataFrame`: a dataframe of info from the GSLIB file

    Note:
        assigns generic point names ("pt0, pt1, etc)

    Example::

        df = pyemu.utiils.geostats.gslib_2_dataframe("prop.gslib",attr_name="hk")


    """
    with open(filename, "r") as f:
        title = f.readline().strip()
        num_attrs = int(f.readline().strip())
        attrs = [f.readline().strip() for _ in range(num_attrs)]
        if attr_name is not None:
            assert attr_name in attrs, "{0} not in attrs:{1}".format(
                attr_name, ",".join(attrs)
            )
        else:
            assert (
                len(attrs) == 3
            ), "propname is None but more than 3 attrs in gslib file"
            attr_name = attrs[2]
        assert len(attrs) > x_idx
        assert len(attrs) > y_idx
        a_idx = attrs.index(attr_name)
        x, y, a = [], [], []
        while True:
            line = f.readline()
            if line == "":
                break
            raw = line.strip().split()
            try:
                x.append(float(raw[x_idx]))
                y.append(float(raw[y_idx]))
                a.append(float(raw[a_idx]))
            except Exception as e:
                raise Exception("error paring line {0}: {1}".format(line, str(e)))
    df = pd.DataFrame({"x": x, "y": y, "value": a})
    df.loc[:, "name"] = ["pt{0}".format(i) for i in range(df.shape[0])]
    df.index = df.name
    return df


# class ExperimentalVariogram(object):
#    def __init__(self,na)


def load_sgems_exp_var(filename):
    """read an SGEM experimental variogram into a sequence of
    pandas.DataFrames

    Args:
        filename (`str`): an SGEMS experimental variogram XML file

    Returns:
        [`pandas.DataFrame`]: a list of pandas.DataFrames of x, y, pairs for each
        division in the experimental variogram

    """

    assert os.path.exists(filename)
    import xml.etree.ElementTree as etree

    tree = etree.parse(filename)
    root = tree.getroot()
    dfs = {}
    for variogram in root:
        # print(variogram.tag)
        for attrib in variogram:

            # print(attrib.tag,attrib.text)
            if attrib.tag == "title":
                title = attrib.text.split(",")[0].split("=")[-1]
            elif attrib.tag == "x":
                x = [float(i) for i in attrib.text.split()]
            elif attrib.tag == "y":
                y = [float(i) for i in attrib.text.split()]
            elif attrib.tag == "pairs":
                pairs = [int(i) for i in attrib.text.split()]

            for item in attrib:
                print(item, item.tag)
        df = pd.DataFrame({"x": x, "y": y, "pairs": pairs})
        df.loc[df.y < 0.0, "y"] = np.NaN
        dfs[title] = df
    return dfs


def fac2real(
    pp_file=None,
    factors_file="factors.dat",
    out_file="test.ref",
    upper_lim=1.0e30,
    lower_lim=-1.0e30,
    fill_value=1.0e30,
):
    """A python replication of the PEST fac2real utility for creating a
    structure grid array from previously calculated kriging factors (weights)

    Args:
        pp_file (`str`): PEST-type pilot points file
        factors_file (`str`): PEST-style factors file
        out_file (`str`): filename of array to write.  If None, array is returned, else
            value of out_file is returned.  Default is "test.ref".
        upper_lim (`float`): maximum interpolated value in the array.  Values greater than
            `upper_lim` are set to fill_value
        lower_lim (`float`): minimum interpolated value in the array.  Values less than
            `lower_lim` are set to fill_value
        fill_value (`float`): the value to assign array nodes that are not interpolated


    Returns:
        `numpy.ndarray`: if out_file is None

        `str`: if out_file it not None

    Example::

        pyemu.utils.geostats.fac2real("hkpp.dat",out_file="hk_layer_1.ref")

    """

    if pp_file is not None and isinstance(pp_file, str):
        assert os.path.exists(pp_file)
        # pp_data = pd.read_csv(pp_file,delim_whitespace=True,header=None,
        #                       names=["name","parval1"],usecols=[0,4])
        pp_data = pp_file_to_dataframe(pp_file)
        pp_data.loc[:, "name"] = pp_data.name.apply(lambda x: x.lower())
    elif pp_file is not None and isinstance(pp_file, pd.DataFrame):
        assert "name" in pp_file.columns
        assert "parval1" in pp_file.columns
        pp_data = pp_file
    else:
        raise Exception(
            "unrecognized pp_file arg: must be str or pandas.DataFrame, not {0}".format(
                type(pp_file)
            )
        )
    assert os.path.exists(factors_file), "factors file not found"
    f_fac = open(factors_file, "r")
    fpp_file = f_fac.readline()
    if pp_file is None and pp_data is None:
        pp_data = pp_file_to_dataframe(fpp_file)
        pp_data.loc[:, "name"] = pp_data.name.apply(lambda x: x.lower())

    fzone_file = f_fac.readline()
    ncol, nrow = [int(i) for i in f_fac.readline().strip().split()]
    npp = int(f_fac.readline().strip())
    pp_names = [f_fac.readline().strip().lower() for _ in range(npp)]

    # check that pp_names is sync'd with pp_data
    diff = set(list(pp_data.name)).symmetric_difference(set(pp_names))
    if len(diff) > 0:
        raise Exception(
            "the following pilot point names are not common "
            + "between the factors file and the pilot points file "
            + ",".join(list(diff))
        )

    arr = np.zeros((nrow, ncol), dtype=np.float64) + fill_value
    pp_dict = {int(name): val for name, val in zip(pp_data.index, pp_data.parval1)}
    try:
        pp_dict_log = {
            name: np.log10(val) for name, val in zip(pp_data.index, pp_data.parval1)
        }
    except:
        pp_dict_log = {}
    # for i in range(nrow):
    #    for j in range(ncol):
    while True:
        line = f_fac.readline()
        if len(line) == 0:
            # raise Exception("unexpected EOF in factors file")
            break
        try:
            inode, itrans, fac_data = _parse_factor_line(line)
        except Exception as e:
            raise Exception("error parsing factor line {0}:{1}".format(line, str(e)))
        # fac_prods = [pp_data.loc[pp,"value"]*fac_data[pp] for pp in fac_data]
        if itrans == 0:
            fac_sum = sum([pp_dict[pp] * fac_data[pp] for pp in fac_data])
        else:
            fac_sum = sum([pp_dict_log[pp] * fac_data[pp] for pp in fac_data])
        if itrans != 0:
            fac_sum = 10 ** fac_sum
        # col = ((inode - 1) // nrow) + 1
        # row = inode - ((col - 1) * nrow)
        row = ((inode - 1) // ncol) + 1
        col = inode - ((row - 1) * ncol)
        # arr[row-1,col-1] = np.sum(np.array(fac_prods))
        arr[row - 1, col - 1] = fac_sum
    arr[arr < lower_lim] = lower_lim
    arr[arr > upper_lim] = upper_lim

    # print(out_file,arr.min(),pp_data.parval1.min(),lower_lim)

    if out_file is not None:
        np.savetxt(out_file, arr, fmt="%15.6E", delimiter="")
        return out_file
    return arr


def _parse_factor_line(line):
    """function to parse a factor file line.  Used by fac2real()"""

    raw = line.strip().split()
    inode, itrans, nfac = [int(i) for i in raw[:3]]
    fac_data = {
        int(raw[ifac]) - 1: float(raw[ifac + 1]) for ifac in range(4, 4 + nfac * 2, 2)
    }
    # fac_data = {}
    # for ifac in range(4,4+nfac*2,2):
    #     pnum = int(raw[ifac]) - 1 #zero based to sync with pandas
    #     fac = float(raw[ifac+1])
    #     fac_data[pnum] = fac
    return inode, itrans, fac_data
