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
from pyemu.utils.helpers import normal_score_transform,randrealgen_optimized
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
            - **pd.DataFrame**: percent prediction standard deviation reduction summary


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

            #todo: test for invertibility and shrink if needed...
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
                post_var = prior_var.iloc[i] - schur
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


    def prep_for_dsi(self,sim_ensemble=None,t_d="dsi_template",
                     apply_normal_score_transform=False,nst_extrap=None,
                     use_ztz=False,energy=1.0):
        """Setup a new PEST interface for the data-space inversion process.
        If the observation data in the Pst object has a "obstransform" column, then observations for which "log" is specified will be subject to log-transformation. 
        If the `apply_normal_score_transform` flag is set to `True`, then the observations and predictions will be subject to a normal score transform.

        Args:

            sim_ensemble (`pyemu.ObservationEnsemble`): observation ensemble to use for DSI latent space
                variables.  If `None`, use `self.sim_ensemble`.  Default is `None`
            t_d (`str`): template directory to setup the DSI model + pest files in.  Default is `dsi_template`
            apply_normal_score_transform (`bool`): flag to apply a normal score transform to the observations
                and predictions.  Default is `False`
            nst_extrap (`str`): flag to apply extrapolation to the normal score transform. Can be None, 'linear' or 'quadratic'. Default is None. 
            use_ztz (`bool`): flag to use the condensed ZtZ matrix for SVD. The ZtZ matrix has dimensions nreal*nreal, instead of the nreal*nobs dimensions of Z. 
                This makes the SVD computation faster and more memory efficient when nobs >> nreal. 
                Default is `False`
            energy (`float`): energy threshold for truncating the sqrt(C) matrix.  Default is `1.0` which applies no truncation. 

        Example::

        #assumes "my.pst" exists
        ends = pyemu.EnDS(ensemble="my.0.obs.jcb",forecasts=["fore1","fore2"])
        ends.prep_for_dsi() #setup a new pest interface() based on the DSI approach
        pyemu.os_utils.start_workers("pestpp-ies","my.pst","dsi_template",num_workers=20,
                                      master_dir="dsi_master")
                                      


        """
        if sim_ensemble is None:
            sim_ensemble = self.sim_ensemble.copy()

        if nst_extrap is not None:
            assert nst_extrap in ["linear","quadratic"], "nst_extrap must be None, 'linear' or 'quadratic'"

        if os.path.exists(t_d):
            self.logger.warn("EnDS.prep_for_dsi(): t_d '{0}' exists, removing...".format(t_d))
            shutil.rmtree(t_d)
        os.makedirs(t_d)

        
        nz_names = self.pst.nnz_obs_names
        snz_names = set(nz_names)
        z_names = [n for n in self.pst.obs_names if n not in snz_names]
        names = z_names.copy()
        names.extend(nz_names)
        names.sort()

        # make sure names are sorted
        sim_ensemble = sim_ensemble.loc[:,names]
        
        self.logger.log("applying transformations")
        # implement log-transform/offset and normal score transform
        transf_names = nz_names.copy()
        transf_names.extend(self.predictions)
        
        if "obstransform" in self.pst.observation_data.columns:
            obs = self.pst.observation_data.copy()
            #make sure names are ordered
            obs = obs.loc[names,:]
            #TODO: deal with "scale" and user-specified "offset"
            obs["offset"] = 0.0 #TODO: more elegant? in case all 'none' are passed...
            obsnmes = obs.loc[obs.obstransform=='log'].obsnme.values
            if len(obsnmes) > 0:
                for name in obsnmes:
                    #TODO: make more efficient
                    self.logger.log("applying obs log-transform to:"+name)
                    values = sim_ensemble.loc[:,name].astype(float).values
                    offset = abs(min(values))+1.0 #arbitrary; enforce positive values
                    values+=offset
                    assert min(values)>0, "values must be positive. min value is "+str(min(values))
                    sim_ensemble.loc[:,name] = np.log10(values)
                    obs.loc[obs.obsnme==name,'offset'] = offset
            obs[['obsnme','obsval','obstransform','offset']].to_csv(os.path.join(t_d,"dsi_obs_transform.csv"),index=False)
            #numpy binary for i/o speed
            np.save(os.path.join(t_d,"dsi_obs_offset.npy"),
                    obs.offset.values, 
                    allow_pickle=False, fix_imports=True)
            obs['flag'] = 0
            obs.loc[obs.obstransform=='log', "flag"] = 1
            np.save(os.path.join(t_d,"dsi_obs_log.npy"),
                    obs.flag.values, 
                    allow_pickle=False, fix_imports=True)
            
        if apply_normal_score_transform:
            # prepare for normal score transform
            nstval = randrealgen_optimized(sim_ensemble.shape[0])
            back_transform_df = pd.DataFrame()
            self.logger.log("applying normal score transform to non-zero obs and predictions")
            #TODO: make more efficient
            for name in  transf_names:
                print("transforming:",name)
                values = sim_ensemble._df.loc[:,name].copy()
                values.sort_values(inplace=True)
                if values.iloc[0] != values.iloc[-1]:
                    # apply smoothing as per DSI2; window sizes are arbitrary...                
                    window_size=3   
                    if values.shape[0]>40:
                        window_size=5                    
                    if values.shape[0]>90:
                        window_size=7
                    if values.shape[0]>200:
                        window_size=9            
                    #print("window size:",window_size,values.shape[0])     
                    values.loc[:] = moving_average_with_endpoints(values.values, window_size)
                    transformed_values = [normal_score_transform(nstval, values.values, v)[0] for v in values.values]
                    #transformed_values, sorted_values, sorted_idxs = normal_score_transform(values) #transformed data retains the same order as the original data
                elif values.iloc[0] == values.iloc[-1]:
                    print("all values are the same, skipping nst")
                    transformed_values = values.values
                sim_ensemble.loc[values.index,name] = transformed_values
                df = pd.DataFrame()
                df['real'] = values.index
                df['sorted_values'] = values.values
                df['transformed_values'] = transformed_values
                df['nstval'] = nstval
                df['obsnme'] = name
                back_transform_df=pd.concat([back_transform_df,df],ignore_index=True)
            #back_transform_df.to_csv(os.path.join(t_d,"dsi_obs_backtransform.csv"),index=False)
            #numpy binary for speed
            np.save(os.path.join(t_d,"dsi_obs_backtransformvals.npy"),
                    back_transform_df[['sorted_values',"nstval"]].values, 
                    allow_pickle=False, fix_imports=True)
            np.save(os.path.join(t_d,"dsi_obs_backtransformobsnmes.npy"),
                    back_transform_df['obsnme'].values, 
                    allow_pickle=True, fix_imports=True)
        
        self.logger.log("applying transformations")

        self.logger.log("computing projection matrix")
        if use_ztz:
            self.logger.log("using ztz approach...")
            pmat, s = compute_using_ztz(sim_ensemble)
            self.logger.log("using ztz approach...")
        else:
            self.logger.log("using z approach...")
            pmat, s = compute_using_z(sim_ensemble)
            self.logger.log("using z approach...")
        self.logger.log("computing projection matrix")

        self.logger.log("applying truncation...")
        apply_energy_based_truncation(energy,s,pmat)
        self.logger.log("applying truncation...")

        self.logger.log("creating tpl files")
        dsi_in_file = os.path.join(t_d, "dsi_pars.csv")
        dsi_tpl_file = dsi_in_file + ".tpl"
        ftpl = open(dsi_tpl_file, 'w')
        fin = open(dsi_in_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("parnme,parval1\n")
        ftpl.write("parnme,parval1\n")
        npar = s.shape[0]
        dsi_pnames = []
        for i in range(npar):
            pname = "dsi_par{0:04d}".format(i)
            dsi_pnames.append(pname)
            fin.write("{0},0.0\n".format(pname))
            ftpl.write("{0},~   {0}   ~\n".format(pname, pname))
        fin.close()
        ftpl.close()

        mn_vec = sim_ensemble.mean(axis=0)
        # check that sim_ensemble has names ordered
        assert (mn_vec.index.values == names).all(), "sim_ensemble names are not ordered"
        mn_in_file = os.path.join(t_d, "dsi_pr_mean.csv")
        mn_tpl_file = mn_in_file + ".tpl"
        fin = open(mn_in_file, 'w')
        ftpl = open(mn_tpl_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("obsnme,mn\n")
        ftpl.write("obsnme,mn\n")
        mn_dict = {}
        for oname in names:
            pname = "dsi_prmn_{0}".format(oname)
            fin.write("{0},{1}\n".format(oname, mn_vec[oname]))
            ftpl.write("{0},~   {1}   ~\n".format(oname, pname))
            mn_dict[pname] = mn_vec[oname]
        fin.close()
        ftpl.close()
        self.logger.log("creating tpl files")

        self.logger.log("saving proj mat")
        #row_names = ["sing_vec_{0}".format(i) for i in range(pmat.shape[0])]
        pmat = Matrix(x=pmat,col_names=dsi_pnames,row_names=names)
        pmat.col_names = dsi_pnames
        #proj_name = "dsi_proj_mat.jcb" # dont change this name!!!
        proj_name = "dsi_proj_mat.npy" # dont change this name!!!
        proj_path = os.path.join(t_d,proj_name)
        #pmat.to_coo(proj_path)
        # use numpy for speed
        np.save(os.path.join(t_d,proj_name), pmat.x, allow_pickle=False, fix_imports=True)

        self.logger.statement("projection matrix dimensions:"+str(pmat.shape))
        self.logger.statement("projection matrix saved to "+proj_path)
        self.logger.log("saving proj mat")


        # this is the dsi forward run function - it is harded coded below!
        def dsi_forward_run():
            import os
            import numpy as np
            import pandas as pd
            from pyemu.utils.helpers import inverse_normal_score_transform
            pmat = np.load("dsi_proj_mat.npy")
            pvals = pd.read_csv("dsi_pars.csv",index_col=0)
            ovals = pd.read_csv("dsi_pr_mean.csv",index_col=0)
            sim_vals = ovals + np.dot(pmat,pvals.values)
            filename = "dsi_obs_backtransformvals.npy"
            if os.path.exists(filename):
                print("applying back-transform")
                backtransformvals = np.load("dsi_obs_backtransformvals.npy")
                backtransformobsnmes = np.load("dsi_obs_backtransformobsnmes.npy",allow_pickle=True)
                obsnmes = np.unique(backtransformobsnmes)
                back_vals = [
                            inverse_normal_score_transform(
                                                backtransformvals[np.where(backtransformobsnmes==o)][:,1],
                                                backtransformvals[np.where(backtransformobsnmes==o)][:,0],
                                                sim_vals.loc[o].mn,
                                                extrap=None
                                                )[0] 
                            for o in obsnmes
                            ]     
                sim_vals.loc[obsnmes,'mn'] = back_vals
            if os.path.exists("dsi_obs_transform.csv"):
                print("reversing log-transform")
                offset = np.load("dsi_obs_offset.npy")
                log_trans = np.load("dsi_obs_log.npy")
                assert log_trans.shape[0] == sim_vals.mn.values.shape[0], f"log transform shape mismatch: {log_trans.shape[0]},{sim_vals.mn.values.shape[0]}"
                assert offset.shape[0] == sim_vals.mn.values.shape[0], f"offset transform shape mismatch: {offset.shape[0]},{sim_vals.mn.values.shape[0]}"
                vals = sim_vals.mn.values
                vals[np.where(log_trans==1)] = 10**vals[np.where(log_trans==1)]
                vals-= offset
                sim_vals.loc[:,'mn'] = vals
            #print(sim_vals)
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
        par.loc[dsi_pars,"parubnd"] = 10.0
        par.loc[dsi_pars,"parlbnd"] = -10.0
        par.loc[dsi_pars,"partrans"] = "none"
        with open(os.path.join(t_d,"dsi.unc"),'w') as f:
            f.write("START STANDARD_DEVIATION\n")
            for p in dsi_pars:
                f.write("{0} 1.0\n".format(p))
            f.write("END STANDARD_DEVIATION")
        pst.pestpp_options['parcov'] = "dsi.unc"

        mn_pars = par.loc[par.parnme.str.startswith("dsi_prmn"),"parnme"]
        par.loc[mn_pars,"partrans"] = "fixed"
        for pname,pval in mn_dict.items():
            par.loc[pname,"parval1"] = pval
            par.loc[pname, "parubnd"] = pval + 1000
            par.loc[pname, "parlbnd"] = pval - 1000

        obs = pst.observation_data
        org_obs = self.pst.observation_data
        for col in org_obs.columns:
            obs.loc[org_obs.obsnme,col] = org_obs.loc[:,col]
        pst.control_data.noptmax = 0
        pst.model_command = "python forward_run.py"
        self.logger.log("creating Pst")
        import inspect
        #print([l for l in inspect.getsource(dsi_forward_run).split("\n")])
        lines = [line[12:] for line in inspect.getsource(dsi_forward_run).split("\n")][1:]
        with open(os.path.join(t_d,"forward_run.py"),'w') as f:
            for line in lines:
                if nst_extrap is not None:
                    if "extrap=None" in line:
                        line = line.replace("None",f"'{nst_extrap}'") 
                f.write(line+"\n")
        pst.write(os.path.join(t_d,"dsi.pst"),version=2)
        self.logger.statement("saved pst to {0}".format(os.path.join(t_d,"dsi.pst")))
        try:
            run("pestpp-ies dsi.pst",cwd=t_d)
        except Exception as e:
            self.logger.warn("error testing noptmax=0 run:{0}".format(str(e)))

        return pst


def compute_using_z(sim_ensemble):
    z = sim_ensemble.get_deviations() / np.sqrt(float(sim_ensemble._df.shape[0] - 1))
    z = z.values
    u, s, v = np.linalg.svd(z, full_matrices=False)
    us = np.dot(v.T, np.diag(s))
    return us,s

def compute_using_ztz(sim_ensemble):
    # rval are the transformed obs values
    rval = sim_ensemble._df.copy()
    #mu2 is the mean of the transformed obs values
    mu2 = rval.mean()
    #adjust rval by subtracting mu2
    rval -= mu2
    #divide rval by the sqrt of nreal-1
    nreal = rval.shape[0]
    rval = rval*np.sqrt(1/(nreal-1))
    # rval.T to match pest utils implementation
    z = rval.T.values
    # Compute the ZtZ matrix
    ztz = np.dot(z.T,z)
    assert ztz.shape[0] == z.shape[1], "ZtZ matrix is not square"
    assert ztz.shape[0] == sim_ensemble.shape[0], "ZtZ matrix is not nreal*nreal"

    #We now do SVD on ZtZ.
    print("doing SVD on ZtZ")
    u, s2, v = np.linalg.svd(ztz, full_matrices=False)
    s =  np.sqrt(s2)
    s[z.shape[0]:] = 0  #truncation to match compute_using_z()

    # formulate the sqrt of the covariance matrix
    us = np.dot(z,u)
    return us, s

def apply_energy_based_truncation(energy,s,us):
    if energy >= 1.0:
        print("Warning: energy>=1.0, no truncation applied")
        return us
    # Determine where to truncate
    # Determine nn
    if us.shape[0]==us.shape[1]:
        nn = us.shape[0] - 1
    else:
        nobs = us.shape[0]
        nreal = us.shape[1]
        nn = min(nobs, nreal) - 1
    # Compute total_energy
    total_energy = np.sum((np.sqrt(s))[:nn])
    # Find energy truncation point
    ntrunc = np.where((np.sqrt(s)).cumsum()/total_energy<=energy)[0].shape[0]
    # Initialize threshold
    #s1 = s[0]
    #thresh = 1.0e-7 * s1 #NOTE: JDoh's implementation uses an additional level of truncation
    #ntrunc = min(np.where(s>=thresh)[0][0], ntrunc)+1
    ntrunc=ntrunc+1
    if ntrunc>=us.shape[1]:
        print("ntrunc>=us.shape[1], no truncation applied")
    else:
        print("truncating to {0} singular values".format(ntrunc))
        # Apply threshold logic
        us = us[:,:ntrunc]
    return us

def moving_average_with_endpoints(y_values, window_size):
    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    # Calculate half-window size
    half_window = window_size // 2
    # Initialize the output array
    smoothed_y = np.zeros_like(y_values)
    # Handle the endpoints
    for i in range(0,half_window):
        # Start
        smoothed_y[i] = np.mean(y_values[:i + half_window ])
    for i in range(1,half_window+1):
        # End
        smoothed_y[-i] = np.mean(y_values[::-1][:i + half_window +1])
    # Handle the middle part with full window
    for i in range(half_window, len(y_values) - half_window):
        smoothed_y[i] = np.mean(y_values[i - half_window:i + half_window])
    #Enforce endpoints
    smoothed_y[0] = y_values[0]
    smoothed_y[-1] = y_values[-1]
    # Ensure uniqueness by adding small increments if values are duplicated
    #NOTE: this is a hack to ensure uniqueness in the normal score transform
    smoothed_y = make_unique(smoothed_y, delta=1e-10)
    return smoothed_y


def make_unique(arr, delta=1e-10):
    """
    Modifies a sorted numpy array in-place to ensure all elements are unique.
    
    Parameters:
    arr (np.ndarray): The sorted numpy array.
    delta (float): The minimum increment to apply to duplicate elements. 
                   Default is a very small value (1e-10).
    """
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            arr[i] = arr[i - 1] + delta

    return arr
