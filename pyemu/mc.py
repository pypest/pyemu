from __future__ import print_function, division
import os
import numpy as np
from pyemu.la import LinearAnalysis
from pyemu.en import Ensemble, ParameterEnsemble


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
        self.obsensemble = Ensemble(mean_values=self.pst.observation_data.values,
                                    columns=self.pst.observation_data.obsnme)

    @property
    def num_reals(self):
        return self.parensemble.shape[0]

    def get_nsing(self,epsilon=1.0e-6):
        """ get the number of solution space dimensions given
            a machine floating point precision (epsilon)

        :param epsilon: machine floating point precision
        :return: integer
        """
        nsing = self.xtqx.shape[0] - np.searchsorted(
                np.sort((self.xtqx.s.x / self.xtqx.s.x.max())[:,0]),epsilon)
        return nsing

    def get_null_proj(self,nsing=None):
        """ get a null-space projection matrix of XTQX

        :param nsing: optional number of singular components to use
                      if none, call self.get_nsing()
        :return: Matrix instance
        """
        if nsing is None:
            nsing = self.get_nsing()

        v2_proj = (self.xtqx.v[:,nsing:] * self.xtqx.v[:,nsing:].T)
        #v2_proj = (self.qhalfx.v[:,nsing:] * self.qhalfx.v[:,nsing:].T)
        #self.__parcov = self.parcov.identity
        return v2_proj

    def draw(self, num_reals=1, par_file = None, obs=False,
             enforce_bounds=False,cov=None):
        """draw stochastic realizations of parameters and
           optionally observations

        Parameters:
        ----------
            num_reals (int): number of realization to generate

            par_file (str): parameter file to use as mean values

            obs (bool): add a realization of measurement noise to obs

            enforce_bounds (bool): enforce parameter bounds in control file


        Returns:
            None
        Raises:
            None
        """

        self.log("generating {0:d} parameter realizations".format(num_reals))
        self.parensemble.draw(self.parcov,num_reals=num_reals)
        if enforce_bounds:
            self.parensemble.enforce()
        self.log("generating {0:d} parameter realizations".format(num_reals))
        if obs:
            raise NotImplementedError()

    def project_parensemble(self,par_file=None,nsing=None,
                            inplace=True):
        """ perform the null-space projection operations for null-space monte carlo

        :param par_file: an optional file of parameter values to use
        :param nsing: number of singular values to in forming null subspace matrix
        :param inplace: overwrite the existing parameter ensemble with the
                        projected values
        :return: is inplace is False, ParameterEnsemble instance, otherwise None
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

    def write_psts(self,prefix):
        """ write parameter and optionally observation realizations
            to pest control files
        :param prefix: pest control file prefix
        :return: None
        """
        self.log("writing realized pest control files")
        # get a copy of the pest control file
        pst = self.pst.get(par_names=self.pst.par_names,obs_names=self.pst.obs_names)

        # set the indices
        pst.parameter_data.index = pst.parameter_data.parnme
        pst.observation_data.index = pst.observation_data.obsnme

        par_en = self.parensemble.back_transform(inplace=False)

        for i in range(self.num_reals):
            pst_name = prefix + "{0:04d}.pst".format(i)
            self.log("writing realized pest control file " + pst_name)
            pst.parameter_data.loc[par_en.columns,"parval1"] = par_en.iloc[i, :].T
            if self.obsensemble.shape[0] == self.num_reals:
                pst.observation_data.loc[self.obsensemble.columns,"obsval"] = \
                    self.obsensemble.iloc[i, :].T
            pst_name = prefix + "{0:04d}.pst".format(i)
            pst.write(pst_name)
            self.log("writing realized pest control file " + pst_name)
        self.log("writing realized pest control files")


    @staticmethod
    def test():
        jco = os.path.join('..',"verification","henry","pest.jco")
        pst = jco.replace(".jco",".pst")

        #write testing
        mc = MonteCarlo(jco=jco,verbose=True)
        mc.draw(10)
        mc.write_psts(os.path.join("tests","mc","real_"))

        mc = MonteCarlo(jco=jco,verbose=True)
        mc.draw(500)
        print("prior ensemble variance:",
              np.var(mc.parensemble.loc[:,"mult1"]))
        projected_en = mc.project_parensemble(inplace=False)
        print("projected ensemble variance:",
              np.var(projected_en.loc[:,"mult1"]))

        import pyemu
        sc = pyemu.Schur(jco=jco)

        mc = MonteCarlo(pst=pst,parcov=sc.posterior_parameter,verbose=True)
        mc.draw(500)
        print("posterior ensemble variance:",
              np.var(mc.parensemble.loc[:,"mult1"]))

        #import matplotlib.pyplot as plt
        #ax = mc.parensemble.loc[:,"mult1"].plot(kind="hist",bins=50,alpha=0.5)
        #projected_en.loc[:,"mult1"].plot(ax=ax,kind="hist",bins=50,
        #                                     facecolor="none",hatch='/',alpha=0.5)

        #mc.write_psts(os.path.join("montecarlo_test","real"))
        #plt.show()

if __name__ == "__main__":
    MonteCarlo.test()
