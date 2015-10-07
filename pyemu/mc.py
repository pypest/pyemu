from __future__ import print_function, division
import os
import numpy as np
from pyemu.la import LinearAnalysis
from pyemu.en import Ensemble, ParameterEnsemble


class MonteCarlo(LinearAnalysis):
    """derived type for monte carlo analysis
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

    def draw(self, num_reals=1, par_file = None, obs=False, enforce_bounds=False):
        """draw stochastic realizations of parameters and optionally observations
        Args:
            num_reals (int): number of realization to generate
            par_file (str): parameter file to use as mean values
            obs (bool): add a realization of measurement noise to obs
            enforce_bounds (bool): enforce parameter bounds in control file
        Returns:
            None
        Raises:
            None
        """

        self.log("generating parameter realizations")
        self.parensemble.draw(self.parcov,num_reals=num_reals)
        if enforce_bounds:
            self.parensemble.enforce()
        self.log("generating parameter realizations")
        if obs:
            raise NotImplementedError()
            self.log("generating noise realizations")
            self.log("generating noise realizations")

    def project_parensemble(self,par_file=None,nsing=None):
        assert self.jco is not None,"MonteCarlo.project_parensemble()" +\
                                    "requires a jacobian attribute"
        if par_file is not None:
            assert os.path.exists(par_file),"monte_carlo.draw() error: par_file not found:" +\
                par_file
            self.parensemble.pst.parrep(par_file)

        self.log("projecting parameter ensemble")
        # work out number of singular values to use based on machine precision
        if nsing is None:
            nsing = self.xtqx.shape[0] - np.searchsorted(
                np.sort((self.xtqx.s.x / self.xtqx.s.x.max())[:,0]),1.0e-6)

        # form null space projection matrix
        v2_proj = (self.xtqx.v[:,nsing:] * self.xtqx.v[:,nsing:].T)

        # project the ensemble
        self.parensemble.project(v2_proj)
        self.log("projecting parameter ensemble")

    def write_psts(self,prefix):
        pst = self.pst.get(par_names=self.pst.par_names,obs_names=self.pst.obs_names)
        pst.parameter_data.index = pst.parameter_data.parnme
        pst.observation_data.index = pst.observation_data.obsnme
        for i in range(self.num_reals):
            pst.parameter_data.loc[self.parensemble.columns,"parval1"] = self.parensemble.loc[i,:].T
            if self.obsensemble is not None:
                pst.observation_data.loc[self.obsensemble.columns,"obsval"] = self.obsensemble.loc[i,:].T
            pst_name = prefix + "{0:04d}.pst".format(i)
            pst.write(pst_name)
        self.log("writing realized pest control files")

    @staticmethod
    def test():
        mc = MonteCarlo(jco=os.path.join('..',"verification","henry","pest.jcb"),verbose=True)
        mc.draw(500)
        print(mc.parensemble.loc[:,"mult1"])

        import matplotlib.pyplot as plt
        ax = mc.parensemble.loc[:,"mult1"].plot(kind="hist",bins=50,alpha=0.5)

        mc.project_parensemble()
        print(mc.parensemble.loc[:,"mult1"])
        mc.parensemble.loc[:,"mult1"].plot(ax=ax,kind="hist",bins=50,
                                             facecolor="none",hatch='/',alpha=0.5)
        #mc.write_psts(os.path.join("montecarlo_test","real"))
        plt.show()


if __name__ == "__main__":
    MonteCarlo.test()
