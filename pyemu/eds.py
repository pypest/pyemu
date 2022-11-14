from __future__ import print_function, division
import os
import copy
import shutil
from datetime import datetime
import warnings
from .pyemu_warnings import PyemuWarning
import numpy as np
import pandas as pd
from pyemu.en import ObservationEnsemble
from pyemu.mat.mat_handler import Matrix, Jco, Cov
from pyemu.pst.pst_handler import Pst
from pyemu.utils.os_utils import _istextfile,run
from .logger import Logger



class EnDS(object):
    """Ensemble Data Space Analysis using the approach of He et al (2018) (https://doi.org/10.2118/182609-PA)

    Args:
        pst (varies): something that can be cast into a `pyemu.Pst`.  Can be an `str` for a
            filename or an existing `pyemu.Pst`.
        sim_ensemble (varies): something that can be cast into a `pyemu.ObservationEnsemble`.  Can be
            an `str` for a  filename or `pd.DataFrame` or an existing `pyemu.ObservationEnsemble`.
        noise_ensemble (varies): something that can be cast into a `pyemu.ObservationEnsemble` that is the
            obs+noise realizations.  If not passed, a noise ensemble is generated using either `obs_cov` or the
            information in `pst` (i.e. weights or standard deviations)
        obscov (varies, optional): observation noise covariance matrix.  If `str`, a filename is assumed and
            the noise covariance matrix is loaded from a file using
            the file extension (".jcb"/".jco" for binary, ".cov"/".mat" for PEST-style ASCII matrix,
            or ".unc" for uncertainty files).  If `None`, the noise covariance matrix is
            constructed from the observation weights (and optionally "standard_deviation")
            .  Can also be a `pyemu.Cov` instance
        forecasts (enumerable of `str`): the names of the entries in `pst.osbervation_data` (and in `ensemble`).
        verbose (`bool`): controls screen output.  If `str`, a filename is assumed and
                and log file is written.
        
    Note:

    Example::

        #assumes "my.pst" exists
        ends = pyemu.EnDS(ensemble="my.0.obs.jcb",forecasts=["fore1","fore2"])
        ends.get_posterior_prediction_moments() #similar to Schur-style data worth
        ends.prep_for_dsi() #setup a new pest interface() based on the DSI approach


    """

    def __init__(
        self,
        pst=None,
        sim_ensemble=None,
        noise_ensemble=None,
        obscov=None,
        predictions=None,
        verbose=False,
    ):
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.sim_en_arg = sim_ensemble
        # if jco is None:
        self.__sim_en = sim_ensemble

        self.pst_arg = pst
        if obscov is None and pst is not None:
            obscov = pst

        # private attributes - access is through @decorated functions
        self.__pst = None
        self.__obscov = None
        self.__sim_ensemble = None
        self.__noise_ensemble = None


        self.log("pre-loading base components")
        self.sim_ensemble_arg = sim_ensemble
        if sim_ensemble is not None:
            self.__sim_ensemble = self.__load_ensemble(self.sim_ensemble_arg)
        if self.sim_ensemble is None:
            raise Exception("sim_ensemble is required for EnDS")
        self.noise_ensemble_arg = noise_ensemble
        if noise_ensemble is not None:
            self.__noise_ensemble = self.__load_ensemble(self.noise_ensemble_arg)
        if pst is not None:
            self.__load_pst()
        if pst is None:
            raise Exception("pst is required for EnDS")
        self.obscov_arg = obscov
        if obscov is not None:
            self.__load_obscov()

        self.predictions = predictions
        if predictions is None and self.pst is not None:
            if self.pst.forecast_names is not None:
                self.predictions = self.pst.forecast_names
        if self.predictions is None:
            raise Exception("predictions are required for EnDS")
        if isinstance(self.predictions,list):
            self.predictions = [p.strip().lower() for p in self.predictions]

        self.log("pre-loading base components")

        # automatically do some things that should be done
        assert type(self.obscov) == Cov

    def __fromfile(self, filename, astype=None):
        """a private method to deduce and load a filename into a matrix object.
        Uses extension: 'jco' or 'jcb': binary, 'mat','vec' or 'cov': ASCII,
        'unc': pest uncertainty file.

        """
        assert os.path.exists(filename), (
            "LinearAnalysis.__fromfile(): " + "file not found:" + filename
        )
        ext = filename.split(".")[-1].lower()
        if ext in ["jco", "jcb"]:
            self.log("loading jcb format: " + filename)
            if astype is None:
                astype = Jco
            m = astype.from_binary(filename)
            self.log("loading jcb format: " + filename)
        elif ext in ["mat", "vec"]:
            self.log("loading ascii format: " + filename)
            if astype is None:
                astype = Matrix
            m = astype.from_ascii(filename)
            self.log("loading ascii format: " + filename)
        elif ext in ["cov"]:
            self.log("loading cov format: " + filename)
            if astype is None:
                astype = Cov
            if _istextfile(filename):
                m = astype.from_ascii(filename)
            else:
                m = astype.from_binary(filename)
            self.log("loading cov format: " + filename)
        elif ext in ["unc"]:
            self.log("loading unc file format: " + filename)
            if astype is None:
                astype = Cov
            m = astype.from_uncfile(filename)
            self.log("loading unc file format: " + filename)
        elif ext in ["csv"]:
            self.log("loading csv format: " + filename)
            if astype is None:
                astype = ObservationEnsemble
            m = astype.from_csv(self.pst,filename=filename)
            self.log("loading csv format: " + filename)
        else:
            raise Exception(
                "EnDS.__fromfile(): unrecognized"
                + " filename extension:"
                + str(ext)
            )
        return m

    def __load_pst(self):
        """private method set the pst attribute"""
        if self.pst_arg is None:
            return None
        if isinstance(self.pst_arg, Pst):
            self.__pst = self.pst_arg
            return self.pst
        else:
            try:
                self.log("loading pst: " + str(self.pst_arg))
                self.__pst = Pst(self.pst_arg)
                self.log("loading pst: " + str(self.pst_arg))
                return self.pst
            except Exception as e:
                raise Exception(
                    "EnDS.__load_pst(): error loading"
                    + " pest control from argument: "
                    + str(self.pst_arg)
                    + "\n->"
                    + str(e)
                )

    def __load_ensemble(self,arg):
        """private method to set the ensemble attribute from a file or a dataframe object"""
        if self.sim_ensemble_arg is None:
            return None
            # raise Exception("linear_analysis.__load_jco(): jco_arg is None")
        if isinstance(arg, Matrix):
            ensemble = ObservationEnsemble(pst=self.pst, df=arg.to_dataframe())
        elif isinstance(self.sim_ensemble_arg, str):
            ensemble = self.__fromfile(arg, astype=ObservationEnsemble)
        elif isinstance(arg, ObservationEnsemble):
            ensemble = arg.copy()
        else:
            raise Exception(
                "EnDS.__load_ensemble(): arg must "
                + "be a matrix object, dataframe, or a file name: "
                + str(arg)
            )
        return ensemble



    def __load_obscov(self):
        """private method to set the obscov attribute from:
        a pest control file (observation weights)
        a pst object
        a matrix object
        an uncert file
        an ascii matrix file
        """
        # if the obscov arg is None, but the pst arg is not None,
        # reset and load from obs weights
        self.log("loading obscov")
        if not self.obscov_arg:
            if self.pst_arg:
                self.obscov_arg = self.pst_arg
            else:
                raise Exception(
                    "linear_analysis.__load_obscov(): " + "obscov_arg is None"
                )
        if isinstance(self.obscov_arg, Matrix):
            self.__obscov = self.obscov_arg
            return

        elif isinstance(self.obscov_arg, str):
            if self.obscov_arg.lower().endswith(".pst"):
                self.__obscov = Cov.from_obsweights(self.obscov_arg)
            else:
                self.__obscov = self.__fromfile(self.obscov_arg, astype=Cov)
        elif isinstance(self.obscov_arg, Pst):
            self.__obscov = Cov.from_observation_data(self.obscov_arg)
        else:
            raise Exception(
                "EnDS.__load_obscov(): "
                + "obscov_arg must be a "
                + "matrix object or a file name: "
                + str(self.obscov_arg)
            )
        self.log("loading obscov")



    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a reference - cheap, but can be dangerous

    @property
    def sim_ensemble(self):
        return self.__sim_en

    @property
    def obscov(self):
        """get the observation noise covariance matrix attribute

        Returns:
            `pyemu.Cov`: a reference to the `LinearAnalysis.obscov` attribute

        """
        if not self.__obscov:
            self.__load_obscov()
        return self.__obscov


    @property
    def pst(self):
        """the pst attribute

        Returns:
            `pyemu.Pst`: the pst attribute

        """
        if self.__pst is None and self.pst_arg is None:
            raise Exception(
                "linear_analysis.pst: can't access self.pst:"
                + "no pest control argument passed"
            )
        elif self.__pst:
            return self.__pst
        else:
            self.__load_pst()
            return self.__pst

    def reset_pst(self, arg):
        """reset the EnDS.pst attribute

        Args:
            arg (`str` or `pyemu.Pst`): the value to assign to the pst attribute

        """
        self.logger.statement("resetting pst")
        self.__pst = None
        self.pst_arg = arg

    def reset_obscov(self, arg=None):
        """reset the obscov attribute to None

        Args:
            arg (`str` or `pyemu.Matrix`): the value to assign to the obscov
                attribute.  If None, the private __obscov attribute is cleared
                but not reset
        """
        self.logger.statement("resetting obscov")
        self.__obscov = None
        if arg is not None:
            self.obscov_arg = arg

    def get_posterior_prediction_convergence_summary(self,num_realization_sequence,num_replicate_sequence,
                                             obslist_dict=None):
        """repeatedly run `EnDS.get_predictive_posterior_moments() with less than all the possible
        realizations to evaluate whether the uncertainty estimates have converged

        Args:
            num_realization_sequence (`[int']): the sequence of realizations to test.
            num_replicate_sequence (`[int]`): The number of replicates of randomly selected realizations to test
                for each `num_realization_sequence` value.  For example, if num_realization_sequence is [10,100,1000]
                and num_replicated_sequence is [4,5,6], then `EnDS.get_predictive posterior_moments()` is called 4
                times using 10 randomly selected realizations (new realizations selected 4 times), 5 times using
                100 randomly selected realizations, and then 6 times using 1000 randomly selected realizations.
            obslist_dict (`dict`, optional): a nested dictionary-list of groups of observations
                to pass to `EnDS.get_predictive_posterior_moments()`.

        Returns:
             `dict`: a dictionary of num_reals: `pd.DataFrame` pairs, where the dataframe is the mean
                predictive standard deviation results from calling `EnDS.get_predictive_posterior_moments()` for the
                desired number of replicates.

         Example::

            ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",predictions=["predhead","predflux"])
            obslist_dict = {"hds":["head1","head2"],"flux":["flux1","flux2"]}
            num_reals_seq = [10,20,30,100,1000] # assuming there are 1000 reals in "my.0.obs.csv"]
            num_reps_seq = [5,5,5,5,5]
            mean_dfs = sc.get_posterior_prediction_convergence_summary(num_reals_seq,num_reps_seq,
                obslist_dict=obslist_dict)

        """
        real_idx = np.arange(self.sim_ensemble.shape[0],dtype=int)
        results = {}
        for nreals,nreps in zip(num_realization_sequence,num_replicate_sequence):
            rep_results = []
            print("-->testing ",nreals)
            for rep in range(nreps):
                rreals = np.random.choice(real_idx,nreals,False)
                sim_ensemble = self.sim_ensemble.iloc[rreals,:].copy()
                _,dfstd,_ = self.get_posterior_prediction_moments(obslist_dict=obslist_dict,
                                                                 sim_ensemble=sim_ensemble,
                                                                 include_first_moment=False)
                rep_results.append(dfstd)
            results[nreals] = rep_results

        means = {}
        for nreals,dfs in results.items():
            mn = dfs[0]
            for df in dfs[1:]:
                mn += df
            mn /= len(dfs)
            means[nreals] = mn

        return means


    def get_posterior_prediction_moments(self, obslist_dict=None,sim_ensemble=None,include_first_moment=True):
        """A dataworth method to analyze the posterior (expected) mean and uncertainty as a result of conditioning with
         some additional observations not used in the conditioning of the current ensemble results.

        Args:
            obslist_dict (`dict`, optional): a nested dictionary-list of groups of observations
                that are to be treated as gained/collected.  key values become
                row labels in returned dataframe. If `None`, then every zero-weighted
                observation is tested sequentially. Default is `None`
            sim_ensemble (`pyemu.ObservationEnsemble`): the simulation results ensemble to use.
                If `None`, `self.sim_ensemble` is used.  Default is `None`

            include_first_moment (`bool`): flag to include calculations of the predictive first moments.
                This can slow things down,so if not needed, better to skip.  Default is `True`

        Returns:
            tuple containing

            - **dict**: dictionary of first-moment dataframes. Keys are `obslist_dict` keys.  If `include_first_moment`
                is None, this is an empty dict.
            - **pd.DataFrame**: prediction standard deviation summary
            - **pd.DataFrame**: precent prediction standard deviation reduction summary


        Example::

            ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",predictions=["predhead","predflux"])
            obslist_dict = {"hds":["head1","head2"],"flux":["flux1","flux2"]}
            mean_dfs,dfstd,dfpercent = sc.get_posterior_prediction_moments(obslist_dict=obslist_dict)

        """


        if obslist_dict is not None:
            if type(obslist_dict) == list:
                obslist_dict = dict(zip(obslist_dict, obslist_dict))
            obs = self.pst.observation_data
            onames = []
            [onames.extend(names) for gp,names in obslist_dict.items()]
            oobs = obs.loc[onames,:]
            zobs = oobs.loc[oobs.weight==0,"obsnme"].tolist()

            if len(zobs) > 0:
                raise Exception(
                    "Observations in obslist_dict must have "
                    + "nonzero weight. The following observations "
                    + "violate that condition: "
                    + ",".join(zobs)
                )
        else:
            obslist_dict = dict(zip(self.pst.nnz_obs_names, self.pst.nnz_obs_names))
            onames = self.pst.nnz_obs_names

        if "posterior" not in obslist_dict:
            obslist_dict["posterior"] = onames

        names = onames
        names.extend(self.predictions)
        self.logger.log("getting deviations")
        if sim_ensemble is None:
            sim_ensemble = self.sim_ensemble
        oe = sim_ensemble.loc[:,names].get_deviations() / np.sqrt(float(sim_ensemble.shape[0] - 1))
        self.logger.log("getting deviations")
        self.logger.log("forming cov matrix")
        cmat = (np.dot(oe.values.transpose(),oe.values))
        cdf = pd.DataFrame(cmat, index=names, columns=names)
        self.logger.log("forming cov matrix")

        var_data = {}
        prior_var = sim_ensemble.loc[:, self.predictions].std() ** 2
        prior_mean = sim_ensemble.loc[:, self.predictions].mean()
        var_data["prior"] = prior_var
        groups = list(obslist_dict.keys())
        groups.sort()

        mean_dfs = {}
        for group in groups:
            self.logger.log("processing "+group)
            onames = obslist_dict[group]
            self.logger.log("extracting blocks from cov matrix")

            dd = cdf.loc[onames,onames].values.copy()
            ccov = cdf.loc[onames,self.predictions].values.copy()
            self.logger.log("extracting blocks from cov matrix")

            self.logger.log("adding noise cov to data block")
            dd += self.obscov.get(onames,onames).x
            self.logger.log("adding noise cov to data block")

            #todo: test for inveribility and shrink if needed...
            self.logger.log("inverting data cov block")
            dd = np.linalg.inv(dd)
            self.logger.log("inverting data cov block")
            pt_var = []
            pt_mean = {}


            if include_first_moment:
                self.logger.log("preping first moment pieces")
                omean = sim_ensemble.loc[:, onames].mean().values
                reals = sim_ensemble.index.values
                innovation_vecs = {real: sim_ensemble.loc[real, onames].values - omean for real in sim_ensemble.index}
                self.logger.log("preping first moment pieces")

            for i,p in enumerate(self.predictions):
                self.logger.log("calc second moment for "+p)
                ccov_vec = ccov[:,i]
                first_term = np.dot(ccov_vec.transpose(), dd)
                schur = np.dot(first_term,ccov_vec)
                post_var = prior_var[i] - schur
                pt_var.append(post_var)
                self.logger.log("calc second moment for " + p)
                if include_first_moment:
                    self.logger.log("calc first moment values for "+p)
                    mean_vals = []
                    prmn = prior_mean[p]
                    for real in reals:
                        mn = prmn + np.dot(first_term,innovation_vecs[real])
                        mean_vals.append(mn)
                    pt_mean[p] = np.array(mean_vals)
                    self.logger.log("calc first moment values for "+p)
            if include_first_moment:
                mean_df = pd.DataFrame(pt_mean,index=reals)
                mean_dfs[group] = mean_df
            var_data[group] = pt_var

            self.logger.log("processing " + group)

        dfstd = pd.DataFrame(var_data,index=self.predictions).apply(np.sqrt).T
        dfper = dfstd.copy()
        prior_std = prior_var.apply(np.sqrt)
        for p in self.predictions:
            dfper.loc[groups,p] = 100 * (1-(dfstd.loc[groups,p].values/prior_std.loc[p]))
        dfper = dfper.loc[groups,self.predictions]

        return mean_dfs,dfstd,dfper


    def prep_for_dsi(self,sim_ensemble=None,t_d="dsi_template"):
        """setup a new PEST interface for the data-space inversion process

        """
        if sim_ensemble is None:
            sim_ensemble = self.sim_ensemble.copy()

        if os.path.exists(t_d):
            self.logger.warn("EnDS.prep_for_dsi(): t_d '{0}' exists, removing...".format(t_d))
            shutil.rmtree(t_d)
        os.makedirs(t_d)

        self.logger.log("getting deviations")
        nz_names = self.pst.nnz_obs_names
        snz_names = set(nz_names)
        z_names = [n for n in self.pst.obs_names if n not in snz_names]
        names = z_names.copy()
        names.extend(nz_names)
        oe = sim_ensemble.get_deviations() / np.sqrt(float(sim_ensemble.shape[0] - 1))
        oe = oe.loc[:,names]
        self.logger.log("getting deviations")

        self.logger.log("pseudo inv of deviations matrix")
        deltad = Matrix.from_dataframe(oe).T
        U,S,V = deltad.pseudo_inv_components(maxsing=self.pst.svd_data.maxsing,eigthresh=self.pst.svd_data.eigthresh)
        self.logger.log("pseudo inv of deviations matrix")

        self.logger.log("saving proj mat")
        pmat = U * S
        proj_name = "dsi_proj_mat.jcb" # dont change this name!!!
        proj_path = os.path.join(t_d,proj_name)
        pmat.to_coo(proj_path)
        self.logger.statement("projection matrix dimensions:"+str(pmat.shape))
        self.logger.statement("projection matrix saved to "+proj_path)
        self.logger.log("saving proj mat")


        self.logger.log("creating tpl files")
        dsi_in_file = os.path.join(t_d,"dsi_pars.csv")
        dsi_tpl_file = dsi_in_file+".tpl"
        ftpl = open(dsi_tpl_file,'w')
        fin = open(dsi_in_file,'w')
        ftpl.write("ptf ~\n")
        fin.write("parnme,parval1\n")
        ftpl.write("parnme,parval1\n")
        npar = S.shape[0]
        for i in range(npar):
            pname = "dsi_par{0:04d}".format(i)
            fin.write("{0},0.0\n".format(pname))
            ftpl.write("{0},~   {0}   ~\n".format(pname,pname))
        fin.close()
        ftpl.close()

        mn_vec = sim_ensemble.mean(axis=0)
        mn_in_file = os.path.join(t_d, "dsi_pr_mean.csv")
        mn_tpl_file = mn_in_file+".tpl"
        fin = open(mn_in_file, 'w')
        ftpl = open(mn_tpl_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("obsnme,mn\n")
        ftpl.write("obsnme,pn\n")
        mn_dict = {}
        for oname in names:
            pname = "dsi_prmn_{0}".format(oname)
            fin.write("{0},{1}\n".format(oname,mn_vec[oname]))
            ftpl.write("{0},~   {0}   ~\n".format(pname, pname))
            mn_dict[pname] = mn_vec[oname]
        fin.close()
        ftpl.close()
        self.logger.log("creating tpl files")

        # this is the dsi forward run function - it is harded coded below!
        def dsi_forward_run():
            import numpy as np
            import pandas as pd
            import pyemu
            pmat = pyemu.Matrix.from_binary("dsi_proj_mat.jcb")
            pvals = pd.read_csv("dsi_pars.csv",index_col=0)
            ovals = pd.read_csv("dsi_pr_mean.csv",index_col=0)
            sim_vals = ovals + np.dot(pmat.x,pvals.values)
            print(sim_vals)
            sim_vals.to_csv("dsi_sim_vals.csv")

        self.logger.log("test run")
        b_d = os.getcwd()
        os.chdir(t_d)
        dsi_forward_run()
        os.chdir(b_d)
        self.logger.log("test run")

        self.logger.log("creating ins file")
        out_file = os.path.join(t_d,"dsi_sim_vals.csv")
        ins_file = out_file + ".ins"
        sdf = pd.read_csv(out_file,index_col=0)
        print(sdf)
        with open(ins_file,'w') as f:
            f.write("pif ~\n")
            f.write("l1\n")
            for oname in sdf.index.values:
                f.write("l1 ~,~ !{0}!\n".format(oname))
        self.logger.log("creating ins file")

        self.logger.log("creating Pst")
        pst = Pst.from_io_files([mn_tpl_file,dsi_tpl_file],[mn_in_file,dsi_in_file],[ins_file],[out_file],pst_path=".")

        par = pst.parameter_data
        dsi_pars = par.loc[par.parnme.str.startswith("dsi_par"),"parnme"]
        par.loc[dsi_pars,"parval1"] = 0
        par.loc[dsi_pars,"parubnd"] = 2.5
        par.loc[dsi_pars,"parlbnd"] = -2.5
        par.loc[dsi_pars,"partrans"] = "none"
        mn_pars = par.loc[par.parnme.str.startswith("dsi_prmn"),"parnme"]
        par.loc[mn_pars,"partrans"] = "fixed"
        for pname,pval in mn_dict.items():
            par.loc[pname,"parval1"] = pval
            par.loc[pname, "parubnd"] = pval + 1000
            par.loc[pname, "parlbnd"] = pval - 1000

        obs = pst.observation_data
        print(obs)
        org_obs = self.pst.observation_data
        for col in org_obs.columns:
            obs.loc[org_obs.obsnme,col] = org_obs.loc[:,col]
        pst.control_data.noptmax = 0
        pst.model_command = "python forward_run.py"
        self.logger.log("creating Pst")

        with open(os.path.join(t_d,"forward_run.py"),'w') as f:
            lines = [line.strip() for line in """ import numpy as np
            import pandas as pd
            import pyemu
            pmat = pyemu.Matrix.from_binary("dsi_proj_mat.jcb")
            pvals = pd.read_csv("dsi_pars.csv",index_col=0)
            ovals = pd.read_csv("dsi_pr_mean.csv",index_col=0)
            sim_vals = ovals + np.dot(pmat.x,pvals.values)
            print(sim_vals)
            sim_vals.to_csv("dsi_sim_vals.csv")""".split("\n")]
            for line in lines:
                f.write(line+"\n")
        pst.write(os.path.join(t_d,"dsi.pst"),version=2)
        self.logger.statement("saved pst to {0}".format(os.path.join(t_d,"dsi.pst")))
        try:
            run("pestpp-ies dsi.pst",cwd=t_d)
        except Exception as e:
            self.logger.warn("error testing noptmax=0 run:{0}".format(str(e)))

        return pst






        






