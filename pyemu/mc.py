from __future__ import print_function, division
import os
import numpy as np
from pyemu.la import LinearAnalysis
from pyemu.en import ObservationEnsemble, ParameterEnsemble
from pyemu.mat import Cov
#from pyemu.utils.helpers import zero_order_tikhonov

class MonteCarlo(LinearAnalysis):
    """LinearAnalysis derived type for monte carlo analysis

       Note: requires a pest control file, which can be
             derived from a jco argument
             MonteCarlo.project_parsensemble also
             requires a jacobian

    """
    def __init__(self,**kwargs):
        super(MonteCarlo,self).__init__(**kwargs)
        assert self.pst is not None, \
            "monte carlo requires a pest control file"
        self.parensemble = ParameterEnsemble(pst=self.pst)
        self.obsensemble = ObservationEnsemble(pst=self.pst)

    @property
    def num_reals(self):
        return self.parensemble.shape[0]

    def get_nsing(self,epsilon=1.0e-4):
        """ get the number of solution space dimensions given
            a ratio between the largest and smallest singular
            values

        Parameters:
            epsilon: ratio
        Returns : integer (or None)
            number of singular components above the epsilon ratio threshold
            If nsing == nadj_par, then None is returned
        """
        mx = self.xtqx.shape[0]
        nsing = mx - np.searchsorted(
                np.sort((self.xtqx.s.x / self.xtqx.s.x.max())[:,0]),epsilon)
        if nsing == mx:
            self.logger.warn("optimal nsing=npar")
            nsing = None
        return nsing

    def get_null_proj(self,nsing=None):
        """ get a null-space projection matrix of XTQX

        Parameters:
        ----------
            nsing: optional number of singular components to use
                      if none, call self.get_nsing()
        Returns:
        -------
            Matrix instance : V2V2^T
        """
        if nsing is None:
            nsing = self.get_nsing()
        if nsing is None:
            raise Exception("nsing is None")
        print("using {0} singular components".format(nsing))
        self.log("forming null space projection matrix with " +\
                 "{0} of {1} singular components".format(nsing,self.jco.shape[1]))

        v2_proj = (self.xtqx.v[:,nsing:] * self.xtqx.v[:,nsing:].T)
        self.log("forming null space projection matrix with " +\
                 "{0} of {1} singular components".format(nsing,self.jco.shape[1]))

        return v2_proj

    def draw(self, num_reals=1, par_file = None, obs=False,
             enforce_bounds=False,cov=None, how="gaussian"):
        """draw stochastic realizations of parameters and
           optionally observations

        Parameters:
        ----------
            num_reals (int): number of realization to generate

            par_file (str): parameter file to use as mean values

            obs (bool): add a realization of measurement noise to obs

            enforce_bounds (bool): enforce parameter bounds in control file

            how (str): type of distribution.  Must be in ["gaussian","uniform"]
        Returns:
            None
        Raises:
            None
        """
        if par_file is not None:
            self.pst.parrep(par_file)
        how = how.lower().strip()
        assert how in ["gaussian","uniform"]

        if cov is not None:
            assert isinstance(cov,Cov)
            if how == "uniform":
                raise Exception("MonteCarlo.draw() error: 'how'='uniform'," +\
                                " 'cov' arg cannot be passed")
        else:
            cov = self.parcov

        self.parensemble = ParameterEnsemble(pst=self.pst)
        self.obsensemble = ObservationEnsemble(pst=self.pst)
        self.log("generating {0:d} parameter realizations".format(num_reals))
        self.parensemble.draw(cov,num_reals=num_reals, how=how,
                              enforce_bounds=enforce_bounds)
        #if enforce_bounds:
        #    self.parensemble.enforce()
        self.log("generating {0:d} parameter realizations".format(num_reals))
        if obs:
            self.log("generating {0:d} observation realizations".format(num_reals))
            self.obsensemble.draw(self.obscov,num_reals=num_reals)
            self.log("generating {0:d} observation realizations".format(num_reals))




    def project_parensemble(self,par_file=None,nsing=None,
                            inplace=True,enforce_bounds='reset'):
        """ perform the null-space projection operations for null-space monte carlo

        Parameters:
            par_file: str
                an optional file of parameter values to use
            nsing: int
                number of singular values to in forming null subspace matrix
            inplace: bool
                overwrite the existing parameter ensemble with the
                projected values
            enforce_bounds: str
                how to enforce parameter bounds.  can be None, 'reset', or 'drop'
        Returns:
        -------
            if inplace is False, ParameterEnsemble instance, otherwise None
        """
        assert self.jco is not None,"MonteCarlo.project_parensemble()" +\
                                    "requires a jacobian attribute"
        if par_file is not None:
            assert os.path.exists(par_file),"monte_carlo.draw() error: par_file not found:" +\
                par_file
            self.parensemble.pst.parrep(par_file)

        # project the ensemble
        self.log("projecting parameter ensemble")
        en = self.parensemble.project(self.get_null_proj(nsing),inplace=inplace,log=self.log)
        self.log("projecting parameter ensemble")
        return en

    def write_psts(self,prefix,existing_jco=None,noptmax=None):
        """ write parameter and optionally observation realizations
            to pest control files
        Parameters:
        ----------
            prefix: str
                pest control file prefix
            existing_jco: str
                filename of an existing jacobian matrix to add to the
                pest++ options in the control file.  This is useful for
                NSMC since this jco can be used to get the first set of
                parameter upgrades for free!  Needs to be the path the jco
                file as seen from the location where pest++ will be run
            noptmax: int
                value of NOPTMAX to set in new pest control files
        Returns:
        -------
            None
        """
        self.log("writing realized pest control files")
        # get a copy of the pest control file
        pst = self.pst.get(par_names=self.pst.par_names,obs_names=self.pst.obs_names)

        if noptmax is not None:
            pst.control_data.noptmax = noptmax
            pst.control_data.noptmax = noptmax

        if existing_jco is not None:
            pst.pestpp_options["BASE_JACOBIAN"] = existing_jco

        # set the indices
        pst.parameter_data.index = pst.parameter_data.parnme
        pst.observation_data.index = pst.observation_data.obsnme

        if self.parensemble.istransformed:
            par_en = self.parensemble._back_transform(inplace=False)
        else:
            par_en = self.parensemble

        for i in range(self.num_reals):
            pst_name = prefix + "{0:d}.pst".format(i)
            self.log("writing realized pest control file " + pst_name)
            pst.parameter_data.loc[par_en.columns,"parval1"] = par_en.iloc[i, :].T

            # reset the regularization
            #if pst.control_data.pestmode == "regularization":
                #pst.zero_order_tikhonov(parbounds=True)
                #zero_order_tikhonov(pst,parbounds=True)
            # add the obs noise realization if needed
            if self.obsensemble.shape[0] == self.num_reals:
                pst.observation_data.loc[self.obsensemble.columns,"obsval"] = \
                    self.obsensemble.iloc[i, :].T

            # write
            pst.write(pst_name)
            self.log("writing realized pest control file " + pst_name)
        self.log("writing realized pest control files")


