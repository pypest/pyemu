import os
import copy
from datetime import datetime
import numpy as np
import pandas
import mat_handler as mhand
import pst_handler as phand

class logger(object):
    """ a basic class for logging events during the linear analysis calculations
        if filename is passed, then an file handle is opened
    Args:
        filename (bool or string): if string, it is the log file to write
            if a bool, then log is written to the screen
        echo (bool): a flag to force screen output
    Attributes:
        items (dict) : tracks when something is started.  If a log entry is
            not in items, then it is treated as a new entry with the string
            being the key and the datetime as the value.  If a log entry is
            in items, then the end time and delta time are written and
            the item is popped from the keys

    """
    def __init__(self,filename, echo=False):
        self.items = {}
        self.echo = bool(echo)
        if filename == True:
            self.echo = True
            self.filename = None
        elif filename:
            self.f = open(filename, 'w', 0) #unbuffered
            self.t = datetime.now()
            self.log("opening " + str(filename) + " for logging")
        else:
            self.filename = None


    def log(self,phrase):
        """log something that happened
        Args:
            phrase (str) : the thing that happened
        Returns:
            None
        Raises:
            None
        """
        pass
        t = datetime.now()
        if phrase in self.items.keys():
            s = str(t) + ' finished: ' + str(phrase) + " took: " + \
                str(t - self.items[phrase]) + '\n'
            if self.echo:
                print s,
            if self.filename:
                self.f.write(s)
            self.items.pop(phrase)
        else:
            s = str(t) + ' starting: ' + str(phrase) + '\n'
            if self.echo:
                print s,
            if self.filename:
                self.f.write(s)
            self.items[phrase] = copy.deepcopy(t)

    def warn(self,message):
        """write a warning to the log file
        Args:
            message (str) : the warning text
        Returns:
            None
        Raises:
            None
        """
        s = str(datetime.now()) + " WARNING: " + message + '\n'
        if self.echo:
            print s,
        if self.filename:
            self.f.write(s)



class linear_analysis(object):
    """ the super class for linear analysis.  Can be used for prior analyses
        only.  The derived types (schur and errvar) are for posterior analyses
        this class tries hard to not load items until they are needed
        all arguments are optional

        Args:
            jco ([enumerable of] [string,ndarray,matrix objects]) : jacobian
            pst (pst object) : the pest control file object
            parcov ([enumerable of] [string,ndarray,matrix objects]) :
                parameter covariance matrix
            obscov ([enumerable of] [string,ndarray,matrix objects]):
                observation noise covariance matrix
            predictions ([enumerable of] [string,ndarray,matrix objects]) :
                prediction sensitivity vectors
            ref_var (float) : reference variance
            verbose (either bool or string) : controls log file / screen output
        Attributes:
            too many to list...just figure it out
        Notes:
            the class makes heavy use of property decorator to encapsulate
            private attributes
    """
    def __init__(self, jco=None, pst=None, parcov=None, obscov=None,
                 predictions=None, ref_var=1.0, verbose=True,
                 resfile=None, forecasts=None,**kwargs):
        self.logger = logger(verbose)
        self.log = self.logger.log
        self.jco_arg = jco
        if jco is None:
            self.__jco = mhand.jco()
        if pst is None:
            if isinstance(jco, str):
                pst_case = jco.replace(".jco", ".pst").replace(".jcb",".pst")
                if os.path.exists(pst_case):
                    pst = pst_case
        self.pst_arg = pst
        if parcov is None and pst is not None:
            parcov = pst
        self.parcov_arg = parcov
        if obscov is None and pst is not None:
            obscov = pst
        self.obscov_arg = obscov
        self.ref_var = ref_var
        if forecasts is not None and predictions is not None:
            raise Exception("can't pass both forecasts and predictions")
        if forecasts is not None:
            predictions = forecasts
        self.prediction_arg = predictions

        #private attributes - access is through @decorated functions
        self.__pst = None
        self.__parcov = None
        self.__obscov = None
        self.__predictions = None
        self.__qhalf = None
        self.__fehalf = None
        self.__prior_prediction = None

        self.log("pre-loading base components")
        if jco is not None:
            self.__load_jco()
        if pst is not None:
            self.__load_pst()
        if parcov is not None:
            self.__load_parcov()
        if obscov is not None:
            self.__load_obscov()

        if predictions is not None:
            self.__load_predictions()
        self.log("pre-loading base components")
        if len(kwargs.keys()) > 0:
            self.logger.warn("unused kwargs in type " +
                             str(self.__class__.__name__) +
                             " : " + str(kwargs))
            raise Exception("unused kwargs" +
                             " : " + str(kwargs))
        # automatically do some things that should be done
        self.log("dropping prior information")
        pi = None
        try:
            pi = self.pst.prior_information
        except:
            self.logger.warn("unable to access self.pst: can't tell if " +
                             " any prior information needs to be dropped.")
        if pi is not None:
            self.drop_prior_information()
        self.log("dropping prior information")


        if resfile != False:
            self.log("scaling obscov by residual phi components")
            try:
                self.adjust_obscov_resfile(resfile=resfile)
            except:
                self.logger.warn("unable to a find a residuals file for " +\
                                " scaling obscov")
                self.resfile = None
                self.res = None
            self.log("scaling obscov by residual phi components")


    def __fromfile(self, filename):
        """a private method to deduce and load a filename into a matrix object

            Args:
                filename (str) : the name of the file
            Returns:
                mat (or cov) object
            Raises:
                Exception if filename extension is not in [jco,mat,vec,cov,unc]

        """
        ext = filename.split('.')[-1].lower()
        if ext in ["jco", "jcb"]:
            self.log("loading jco: "+filename)
            m = mhand.jco()
            m.from_binary(filename)
            self.log("loading jco: "+filename)
        elif ext in ["mat","vec"]:
            self.log("loading ascii: "+filename)
            m = mhand.matrix()
            m.from_ascii(filename)
            self.log("loading ascii: "+filename)
        elif ext in ["cov"]:
            self.log("loading cov: "+filename)
            m = mhand.cov()
            m.from_ascii(filename)
            self.log("loading cov: "+filename)
        elif ext in["unc"]:
            self.log("loading unc: "+filename)
            m = mhand.cov()
            m.from_uncfile(filename)
            self.log("loading unc: "+filename)
        else:
            raise Exception("linear_analysis.__fromfile(): unrecognized" +
                            " filename extension:" + str(ext))
        return m


    def __load_pst(self):
        """private: set the pst attribute
        Args:
            None
        Returns:
            None
        Raises:
            Exception from instantiating a pst object
        """
        if self.pst_arg is None:
            return None
        if isinstance(self.pst_arg, phand.pst):
            self.__pst = self.pst_arg
            return self.pst
        else:
            try:
                self.log("loading pst: " + str(self.pst_arg))
                self.__pst = phand.pst(self.pst_arg)
                self.log("loading pst: " + str(self.pst_arg))
                return self.pst
            except Exception as e:
                raise Exception("linear_analysis.__load_pst(): error loading"+\
                                " pest control from argument: " +
                                str(self.pst_arg) + '\n->' + str(e))


    def __load_jco(self):
        """private :set the jco attribute from a file or a matrix object
        Args:
            None
        Returns:
            None
        Raises:
            Exception if the jco_arg is not a matrix object or str
        """
        if self.jco_arg is None:
            return None
            #raise Exception("linear_analysis.__load_jco(): jco_arg is None")
        if isinstance(self.jco_arg, mhand.matrix):
            self.__jco = self.jco_arg
        elif isinstance(self.jco_arg, str):
            self.__jco = self.__fromfile(self.jco_arg)
        else:
            raise Exception("linear_analysis.__load_jco(): jco_arg must " +
                            "be a matrix object or a file name: " +
                            str(self.jco_arg))


    def __load_parcov(self):
        """private: set the parcov attribute from:
                a pest control file (parameter bounds)
                a pst object
                a matrix object
                an uncert file
                an ascii matrix file
        Args:
            None
        Returns:
            None
        Raises:
            Exception is the parcov_arg is not a matrix object or string
        """
        # if the parcov arg was not passed but the pst arg was,
        # reset and use parbounds to build parcov
        if not self.parcov_arg:
            if self.pst_arg:
                self.parcov_arg = self.pst_arg
            else:
                raise Exception("linear_analysis.__load_parcov(): " +
                                "parcov_arg is None")
        if isinstance(self.parcov_arg, mhand.matrix):
            self.__parcov = self.parcov_arg
            return
        if isinstance(self.parcov_arg, np.ndarray):
            # if the passed array is a vector,
            # then assume it is the diagonal of the parcov matrix
            if len(self.parcov_arg.shape) == 1:
                assert self.parcov_arg.shape[0] == self.jco.shape[1]
                isdiagonal = True
            else:
                assert self.parcov_arg.shape[0] == self.jco.shape[1]
                assert self.parcov_arg.shape[1] == self.jco.shape[1]
                isdiagonal = False
            self.logger.warn("linear_analysis.__load_parcov(): " +
                             "instantiating parcov from ndarray, can't " +
                             "verify parameters alignment with jco")
            self.__parcov = mhand.matrix(x=self.parcov_arg,
                                         isdiagonal=isdiagaonal,
                                         row_names=self.jco.col_names,
                                         col_names=self.jco.col_names)
        self.log("loading parcov")
        if isinstance(self.parcov_arg,str):
            # if the arg is a string ending with "pst"
            # then load parcov from parbounds
            if self.parcov_arg.lower().endswith(".pst"):
                self.__parcov = mhand.cov()
                self.__parcov.from_parbounds(self.parcov_arg)
            else:
                self.__parcov = self.__fromfile(self.parcov_arg)
        #--if the arg is a pst object
        elif isinstance(self.parcov_arg,phand.pst):
            self.__parcov = mhand.cov()
            self.__parcov.from_parameter_data(self.parcov_arg)
        else:
            raise Exception("linear_analysis.__load_parcov(): " +
                            "parcov_arg must be a " +
                            "matrix object or a file name: " +
                            str(self.parcov_arg))
        self.log("loading parcov")


    def __load_obscov(self):
        """private: method to set the obscov attribute from:
                a pest control file (observation weights)
                a pst object
                a matrix object
                an uncert file
                an ascii matrix file
        Args:
            None
        Returns:
            None
        Raises:
            Exception if the obscov_arg is not a matrix object or string
        """
        # if the obscov arg is None, but the pst arg is not None,
        # reset and load from obs weights
        if not self.obscov_arg:
            if self.pst_arg:
                self.obscov_arg = self.pst_arg
            else:
                raise Exception("linear_analysis.__load_obscov(): " +
                                "obscov_arg is None")
        if isinstance(self.obscov_arg,mhand.matrix):
            self.__obscov = self.obscov_arg
            return
        if isinstance(self.obscov_arg,np.ndarray):
            # if the ndarray arg is a vector,
            # assume it is the diagonal of the obscov matrix
            if len(self.obscov_arg.shape) == 1:
                assert self.parcov_arg.shape[0] == self.jco.shape[1]
                isdiagonal = True
            else:
                assert self.obscov_arg.shape[0] == self.jco.shape[0]
                assert self.obscov_arg.shape[1] == self.jco.shape[0]
                isdiagonal = False
            self.logger.warn("linear_analysis.__load_obscov(): " +
                             "instantiating obscov from ndarray,  " +
                             "can't verify observation alignment with jco")
            self.__parcov = mhand.matrix(x=self.obscov_arg,
                                         isdiagonal=isdiagaonal,
                                         row_names=self.jco.row_names,
                                         col_names=self.jco.row_names)
        self.log("loading obscov")
        if isinstance(self.obscov_arg, str):
            if self.obscov_arg.lower().endswith(".pst"):
                self.__obscov = mhand.cov()
                self.__obscov.from_obsweights(self.obscov_arg)
            else:
                self.__obscov = self.__fromfile(self.obscov_arg)
        elif isinstance(self.obscov_arg, phand.pst):
            self.__obscov = mhand.cov()
            self.__obscov.from_observation_data(self.obscov_arg)
        else:
            raise Exception("linear_analysis.__load_obscov(): " +
                            "obscov_arg must be a " +
                            "matrix object or a file name: " +
                            str(self.obscov_arg))
        self.log("loading obscov")


    def __load_predictions(self):
        """private: set the predictions attribute from:
                mixed list of row names, matrix files and ndarrays
                a single row name
                an ascii file
            can be none if only interested in parameters.

            linear_analysis.__predictions is stored as a list of column vectors

        Args:
            None
        Returns:
            None
        Raises:
            Assertion error if prediction matrix object is not aligned with
                jco attribute
        """
        if self.prediction_arg is None:
            self.__predictions = None
            return
        self.log("loading forecasts")
        if not isinstance(self.prediction_arg, list):
            self.prediction_arg = [self.prediction_arg]

        row_names = []
        vecs = []
        for arg in self.prediction_arg:
            if isinstance(arg, mhand.matrix):
                #--a vector
                if arg.shape[1] == 1:
                    vecs.append(arg)
                else:
                    assert arg.shape[1] == self.jco.shape[1],\
                    "linear_analysis.__load_predictions(): " +\
                    "multi-prediction matrix(npred,npar) not aligned " +\
                    "with jco(nobs,npar): " + str(arg.shape) +\
                    ' ' + str(self.jco.shape)
                    for pred_name in arg.row_names:
                        vecs.append(arg.extract(row_names=pred_name).T)
            elif isinstance(arg, str):
                if arg.lower() in self.jco.row_names:
                    row_names.append(arg.lower())
                else:
                    pred_mat = self.__fromfile(arg)
                    #--vector
                    if pred_mat.shape[1] == 1:
                        vecs.append(pred_mat)
                    else:
                        for pred_name in pred_mat.row_names:
                            vecs.append(pred_mat.get(row_names=pred_name))
            elif isinstance(arg, np.ndarray):
                self.logger.warn("linear_analysis.__load_predictions(): " +
                                "instantiating prediction matrix from " +
                                "ndarray, can't verify alignment")
                self.logger.warn("linear_analysis.__load_predictions(): " +
                                 "instantiating prediction matrix from " +
                                 "ndarray, generating generic prediction names")
                pred_names = []
                [pred_names.append("pred_" + str(i + 1))
                 for i in xrange(self.prediction_arg.shape[0])]

                if self.jco:
                    names = self.jco.col_names
                elif self.parcov:
                    names = self.parcov.col_names
                else:
                    raise Exception("linear_analysis.__load_predictions(): " +
                                    "ndarray passed for predicitons " +
                                    "requires jco or parcov to get " +
                                    "parameter names")
                pred_matrix = mhand.matrix(x=self.prediction_arg,
                                           row_names=pred_names,
                                           col_names=names)
                for pred_name in pred_names:
                    vecs.append(pred_matrix.extract(row_names=pred_name).T)
            else:
                raise Exception("unrecognized predictions argument: " +
                                str(arg))
        if len(row_names) > 0:
            for row_name in row_names:
                vecs.append(self.jco.extract(row_names=row_name).T)
            # call obscov to load __obscov so that __obscov
            # (priavte) can be manipulated
            self.obscov
            self.__obscov.drop(row_names, axis=0)
        self.__predictions = vecs
        self.log("loading forecasts")
        return self.__predictions

    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a reference - cheap, but can be dangerous


    @property
    def parcov(self):
        if not self.__parcov:
            self.__load_parcov()
        return self.__parcov


    @property
    def obscov(self):
        if not self.__obscov:
            self.__load_obscov()
        return self.__obscov


    @property
    def jco(self):
        if not self.__jco:
            self.__load_jco()
        return self.__jco


    @property
    def predictions(self):
        if not self.__predictions:
            self.__load_predictions()
        return self.__predictions

    @property
    def forecasts(self):
        return self.predictions

    @property
    def pst(self):
        if self.__pst is None and self.pst_arg is None:
            raise Exception("linear_analysis.pst: can't access self.pst:" +
                            "no pest control argument passed")
        elif self.__pst:
            return self.__pst
        else:
            self.__load_pst()


    @property
    def fehalf(self):
        """set the KL parcov scaling matrix attribute
        """
        if self.__fehalf != None:
            return self.__fehalf
        self.log("fehalf")
        self.__fehalf = self.parcov.u * (self.parcov.s ** (0.5))
        self.log("fehalf")
        return self.__fehalf


    @property
    def qhalf(self):
        """set the square root of the cofactor matrix attribute
        """
        if self.__qhalf != None:
            return self.__qhalf
        self.log("qhalf")
        self.__qhalf = self.obscov ** (-0.5)
        self.log("qhalf")
        return self.__qhalf


    @property
    def prior_parameter(self):
        return self.parcov

    @property
    def prior_forecast(self):
        return self.prior_prediction

    @property
    def prior_prediction(self):
        """get a dict of prior prediction variances
        Args:
            None
        Returns
            dict{prediction name(str):prior variance(float)}
        Raises:
            None
        """
        if self.__prior_prediction is not None:
            return self.__prior_prediction
        else:
            if self.predictions is not None:
                self.log("propagating prior to predictions")
                pred_dict = {}
                for prediction in self.predictions:
                    var = (prediction.T * self.parcov * prediction).x[0, 0]
                    pred_dict[prediction.col_names[0]] = var
                self.__prior_prediction = pred_dict
                self.log("propagating prior to predictions")
            else:
                self.__prior_prediction = {}
            return self.__prior_prediction


    def apply_karhunen_loeve_scaling(self):
        """apply karhuene-loeve scaling to the jacobian matrix.

            This scaling is not necessary for analyses using Schur's
            complement, but can be very important for error variance
            analyses.  This operation effectively transfers prior knowledge
            specified in the parcov to the jacobian and reset parcov to the
            identity matrix.
        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        cnames = copy.deepcopy(self.jco.col_names)
        self.__jco *= self.fehalf
        self.__jco.col_names = cnames
        self.__parcov = self.parcov.identity


    def clean(self):
        """drop regularization and prior information observation from the jco
        """
        if self.pst_arg is None:
            self.logger.warn("linear_analysis.clean(): not pst object")
            return
        if not self.pst.estimation and self.pst.nprior > 0:
            self.drop_prior_information()


    def reset_parcov(self,arg=None):
        """reset the parcov attribute to None
        Args:
            arg (str or matrix) : the value to assign to the parcov_arg attrib
        Returns:
            None
        Raises:
            None
        """
        self.logger.warn("resetting parcov")
        self.__parcov = None
        if arg is not None:
            self.parcov_arg = arg


    def reset_obscov(self,arg=None):
        """reset the obscov attribute to None
        Args:
            arg (str or matrix) : the value to assign to the obscov_arg attrib
        Returns:
            None
        Raises:
            None
        """
        self.logger.warn("resetting obscov")
        self.__obscov = None
        if arg is not None:
            self.obscov_arg = arg


    def drop_prior_information(self):
        """drop the prior information from the jco and pst attributes
        """
        nprior_str = str(self.pst.nprior)
        self.log("removing " + nprior_str + " prior info from jco, pst, and " +
                                            "obs cov")
        #pi_names = list(self.pst.prior_information.pilbl.values)
        pi_names = self.pst.prior_names
        self.__jco.drop(pi_names, axis=0)
        self.__pst.prior_information = self.pst.null_prior
        #self.__obscov.drop(pi_names,axis=0)
        self.log("removing " + nprior_str + " prior info from jco, pst, and " +
                                            "obs cov")


    def get(self,par_names=None,obs_names=None,astype=None):
        """method to get a new linear_analysis class using a
             subset of parameters and/or observations
         Args:
            par_names (enumerable of str) : par names for new object
            obs_names (enumerable of str) : obs names for new object
            astype (either schur or errvar type) : type to cast the new object
        Returns:
            linear_analysis object
        Raises:
            None
        """
        # make sure we aren't fooling with unwanted prior information
        self.clean()
        #--if there is nothing to do but copy
        if par_names is None and obs_names is None:
            if astype is not None:
                self.logger.warn("linear_analysis.get(): astype is not None, " +
                                 "but par_names and obs_names are None so" +
                                 "\n  ->Omitted attributes will not be " +
                                 "propagated to new instance")
            else:
                return copy.deepcopy(self)
        #--make sure the args are lists
        if par_names is not None and not isinstance(par_names, list):
            par_names = [par_names]
        if obs_names is not None and not isinstance(obs_names, list):
            obs_names = [obs_names]

        if par_names is None:
            par_names = self.jco.par_names
        if obs_names is None:
            obs_names = self.jco.obs_names
        #--if possible, get a new parcov
        if self.parcov:
            new_parcov = self.parcov.get(col_names=par_names)
        else:
            new_parcov = None
        #--if possible, get a new obscov
        if self.obscov_arg is not None:
            new_obscov = self.obscov.get(row_names=obs_names)
        else:
            new_obscov = None
        #--if possible, get a new pst
        if self.pst_arg is not None:
            new_pst = self.pst.get(par_names=par_names,obs_names=obs_names)
        else:
            new_pst = None
        if self.predictions:
            new_preds = []
            for prediction in self.predictions:
                new_preds.append(prediction.get(row_names=par_names))
        else:
            new_preds = None
        if self.jco_arg is not None:
            new_jco = self.jco.get(row_names=obs_names, col_names=par_names)
        else:
            new_jco = None
        if astype is not None:
            return astype(jco=new_jco, pst=new_pst, parcov=new_parcov,
                          obscov=new_obscov, predictions=new_preds,
                          verbose=False)
        else:
            #--return a new object of the same type
            return type(self)(jco=new_jco, pst=new_pst, parcov=new_parcov,
                              obscov=new_obscov, predictions=new_preds,
                              verbose=False)


    def draw(self, pst_prefix=None, num_reals=1, add_noise=True):
        """draw stochastic realizations and write to pst
        Args:
            pst_prefix (str): realized pst output prefix
            num_reals (int): number of realization to generate
            add_noise (bool): add a realization of measurement noise to obs
        Returns:
            None
        Raises:
            None
        TODO: check parameter bounds, handle log transform
        """
        if pst_prefix is None:
            pst_prefix = "real."
        pi = self.pst.prior_information
        self.drop_prior_information()
        pst = self.pst.get(self.parcov.row_names, self.obscov.row_names)
        mean_pars = pst.parameter_data.parval1


        islog = pst.parameter_data.partrans == "log"
        islog = islog.values
        ub = pst.parameter_data.parubnd.values
        lb = pst.parameter_data.parlbnd.values

        #log transform
        mean_pars[islog] = np.log10(mean_pars[islog])
        self.log("generating parameter realizations")
        par_vals = np.random.multivariate_normal(mean_pars, self.parcov.as_2d,
                                                 num_reals)
        #back log transform
        par_vals[:, islog] = 10.0**(par_vals[:, islog])

        #apply parameter bounds
        for i in xrange(num_reals):
            par_vals[i, np.where(par_vals[i] > ub)] = \
                ub[np.where(par_vals[i] > ub)]
            par_vals[i, np.where(par_vals[i] < lb)] = \
                ub[np.where(par_vals[i] < lb)]


        self.log("generating parameter realizations")
        if add_noise:
            self.log("generating noise realizations")
            nz_idx = []
            weights = self.pst.observation_data.weight.values
            for iw,w in enumerate(weights):
                if w > 0.0:
                    nz_idx.append(iw)
            obscov = self.obscov.get(self.pst.nnz_obs_names)
            noise_vals = np.random.multivariate_normal(
                np.zeros(pst.nnz_obs), obscov.as_2d,
                num_reals)
            self.log("generating noise realizations")
        self.log("writing realized pest control files")
        pst.prior_information = pi
        obs_vals = pst.observation_data.obsval.values
        for i in xrange(num_reals):
            pst.parameter_data.parval1 = par_vals[i, :]
            if add_noise:
                ovs = obs_vals
                ovs[nz_idx] += noise_vals[i,:]
                pst.observation_data.obsval = ovs
            pst_name = pst_prefix + "{0:04d}.pst".format(i)
            pst.write(pst_name)
        self.log("writing realized pest control files")

    def adjust_obscov_resfile(self, resfile=None):
        """reset the elements of obscov by scaling the implied weights
        based on the phi components in res_file
        """
        self.pst.adjust_weights_resfile(resfile)
        self.__obscov.from_observation_data(self.pst)
        



class schur(linear_analysis):
    """derived type for posterior covariance analysis using Schur's complement
    """
    def __init__(self,jco,**kwargs):
        self.__posterior_prediction = None
        self.__posterior_parameter = None
        super(schur,self).__init__(jco,**kwargs)


    def plot(self):
        raise NotImplementedError("need to do this!!!")


    @property
    def pandas(self):
        """get a pandas dataframe of prior and posterior for all predictions
        """
        names,prior,posterior = [],[],[]
        for iname,name in enumerate(self.posterior_parameter.row_names):
            names.append(name)
            posterior.append(np.sqrt(float(
                self.posterior_parameter[iname, iname]. x)))
            iprior = self.parcov.row_names.index(name)
            prior.append(np.sqrt(float(self.parcov[iprior, iprior].x)))
        for pred_name, pred_var in self.posterior_prediction.iteritems():
            names.append(pred_name)
            posterior.append(np.sqrt(pred_var))
            prior.append(self.prior_prediction[pred_name])
        return pandas.DataFrame({"posterior": posterior, "prior": prior},
                                index=names)


    @property
    def posterior_parameter(self):
        """get the posterior parameter covariance matrix
        """
        if self.__posterior_parameter is not None:
            return self.__posterior_parameter
        else:
            self.clean()
            self.log("Schur's complement")
            self.__posterior_parameter = \
                ((self.jco.transpose * self.obscov ** (-1) *
                  self.jco) + self.parcov.inv).inv
            self.log("Schur's complement")
            return self.__posterior_parameter


    @property
    def posterior_forecast(self):
        return self.posterior_prediction

    @property
    def posterior_prediction(self):
        """get a dict of posterior prediction variances
        """
        if self.__posterior_prediction is not None:
            return self.__posterior_prediction
        else:
            if self.predictions is not None:
                self.log("propagating posterior to predictions")
                pred_dict = {}
                for prediction in self.predictions:
                    var = (prediction.T * self.posterior_parameter
                           * prediction).x[0, 0]
                    pred_dict[prediction.col_names[0]] = var
                self.__posterior_prediction = pred_dict
                self.log("propagating posterior to predictions")
            else:
                self.__posterior_prediction = {}
            return self.__posterior_prediction


    def contribution_from_parameters(self, parameter_names):
        """get the prior and posterior uncertainty reduction as a result of
        some parameter becoming perfectly known
        Args:
            parameter_names (list of str) : parameter that are perfectly known
        Returns:
            dict{prediction name : [% prior uncertainty reduction,
                % posterior uncertainty reduction]}
        Raises:
            Exception if no predictions are set
            Exception if one or more parameter_names are not in jco
            Exception if no parameter remain
        """
        if not isinstance(parameter_names, list):
            parameter_names = [parameter_names]

        for iname, name in enumerate(parameter_names):
            parameter_names[iname] = name.lower()
            assert name.lower() in self.jco.par_names,\
                "contribution parameter " + name + " not found jco"
        keep_names = []
        for name in self.jco.par_names:
            if name not in keep_names:
                keep_names.append(name)
        if len(keep_names) == 0:
            raise Exception("schur.contribution_from_parameters: " +
                            "atleast one parameter must remain uncertain")
        #get the reduced predictions
        if self.predictions is None:
            raise Exception("schur.contribution_from_parameters: " +
                            "no predictions have been set")
        cond_preds = []
        for pred in self.predictions:
            cond_preds.append(pred.get(keep_names, pred.col_names))
        la_cond = schur(jco=self.jco.get(self.jco.obs_names, keep_names),
                        parcov=self.parcov.condition_on(parameter_names),
                        obscov=self.obscov, predictions=cond_preds,verbose=False)

        #get the prior and posterior for the base case
        bprior,bpost = self.prior_prediction, self.posterior_prediction
        #get the prior and posterior for the conditioned case
        cprior,cpost = la_cond.prior_prediction, la_cond.posterior_prediction

        # pack the results into a dict{pred_name:[prior_%_reduction,
        # posterior_%_reduction]}
        results = {}
        for pname in bprior.keys():
            prior_reduc = 100. * ((bprior[pname] - cprior[pname]) /
                                  bprior[pname])
            post_reduc = 100. * ((bpost[pname] - cpost[pname]) / bpost[pname])
            results[pname] = [prior_reduc, post_reduc]
        return results


    def importance_of_observations(self,observation_names):
        """get the importance of some observations for reducing the
        posterior uncertainty
        Args:
            observation_names (list of str) : observations to analyze
        Returns:
            dict{prediction_name:% posterior reduction}
        Raises:
            Exception if one or more names not in jco obs names
            Exception if all obs are in observation names
            Exception if predictions are not set

        """
        if not isinstance(observation_names,list):
            observation_names = [observation_names]
        for iname,name in enumerate(observation_names):
            observation_names[iname] = name.lower()
            if name.lower() not in self.jco.obs_names:
                raise Exception("schur.importance_of_observations: " +
                                "obs name not found in jco: " + name)

        keep_names = []
        for name in self.jco.obs_names:
            if name not in observation_names:
                keep_names.append(name)
        if len(keep_names) == 0:
            raise Exception("schur.importance_of_observations: " +
                            " atleast one observation must remain")
        if self.predictions is None:
            raise Exception("schur.importance_of_observations: " +
                            "no predictions have been set")


        la_reduced = self.get(par_names=self.jco.par_names,
                              obs_names=keep_names)
        rpost = la_reduced.posterior_prediction
        bpost = self.posterior_prediction

        results = {}
        for pname in rpost.keys():
            post_reduc = 100. * ((rpost[pname] - bpost[pname]) / rpost[pname])
            results[pname] = post_reduc
        return results


class errvar(linear_analysis):
    """child class for error variance analysis
        todo: add KL parameter scaling with parcov -> identity reset
    """
    def __init__(self,jco,**kwargs):
        """there are some additional keyword args that can be passed to active
            the 3-term error variance calculation
        Args:
            omitted_parameters (list of str): argument that identifies
                parameters that will be treated as omitted
            omitted_parcov (matrix or str): argument that identifies
                omitted parameter parcov
            omitted_predictions (matrix or str): argument that identifies
            omitted prediction vectors

        Note: if only omitted_parameters is passed, then the omitted_parameter
            argument must be a string or list of strings that identifies
            parameters that are in the linear_analysis attributes that will
             extracted
        """
        self.__need_omitted = False
        if "omitted_parameters" in kwargs.keys():
             self.omitted_par_arg = kwargs["omitted_parameters"]
             kwargs.pop("omitted_parameters")
             self.__need_omitted = True
        else:
             self.omitted_par_arg = None
        if "omitted_parcov" in kwargs.keys():
            self.omitted_parcov_arg = kwargs["omitted_parcov"]
            kwargs.pop("omitted_parcov")
            self.__need_omitted = True
        else:
            self.omitted_parcov_arg = None

        if "omitted_forecasts" in kwargs.keys():
            self.omitted_predictions_arg = kwargs["omitted_forecasts"]
            kwargs.pop("omitted_forecasts")
            self.__need_omitted = True
        elif "omitted_predictions" in kwargs.keys():
            self.omitted_predictions_arg = kwargs["omitted_predictions"]
            kwargs.pop("omitted_predictions")
            self.__need_omitted = True
        else:
            self.omitted_predictions_arg = None

        kl = False
        if "kl" in kwargs.keys():
            kl = bool(kwargs["kl"])
            kwargs.pop("kl")


        self.__qhalfx = None
        self.__R = None
        self.__R_sv = None
        self.__G = None
        self.__G_sv = None
        self.__I_R = None
        self.__I_R_sv = None
        self.__omitted_jco = None
        self.__omitted_parcov = None
        self.__omitted_predictions = None

        #--instantiate the parent class
        super(errvar, self).__init__(jco, **kwargs)
        if self.__need_omitted:
            self.log("pre-loading omitted components")
            #self._linear_analysis__load_jco()
            #self._linear_analysis__load_parcov()
            #self._linear_analysis__load_obscov()
            #if self.prediction_arg is not None:
            #    self._linear_analysis__load_predictions()
            self.__load_omitted_jco()
            self.__load_omitted_parcov()
            if self.prediction_arg is not None:
                self.__load_omitted_predictions()
            self.log("pre-loading omitted components")
        if kl:
            self.log("applying KL scaling")
            self.apply_karhunen_loeve_scaling()
            self.log("applying KL scaling")

        self.valid_terms = ["null","solution", "omitted", "all"]
        self.valid_return_types = ["parameters", "predictions"]



    def __load_omitted_predictions(self):
        """private: set the omitted_predictions attribute
        """
        #--if there are no base predictions
        if self.predictions is None:
            raise Exception("errvar.__load_omitted_predictions(): " +
                            "no 'included' predictions is None")
        if self.omitted_predictions_arg is None and \
                        self.omitted_par_arg is None:
            raise Exception("errvar.__load_omitted_predictions: " +
                            "both omitted args are None")
        # try to set omitted_predictions by
        # extracting from existing predictions
        if self.omitted_predictions_arg is None and \
                        self.omitted_par_arg is not None:
            #--check to see if omitted par names are in each predictions
            found = True
            missing_par,missing_pred = None, None
            for par_name in self.omitted_jco.par_names:
                for prediction in self.predictions:
                    if par_name not in prediction.row_names:
                        found = False
                        missing_par = par_name
                        missing_pred = prediction.col_names[0]
                        break
            if found:
                opreds = []
                # need to access the attribute directly,
                # not a view of attribute
                for prediction in self._linear_analysis__predictions:
                    opred = prediction.extract(self.omitted_jco.par_names)
                    opreds.append(opred)
                self.__omitted_predictions = opreds
            else:
                raise Exception("errvar.__load_omitted_predictions(): " +
                                " omitted parameter " + str(missing_par) +\
                                " not found in prediction vector " +
                                str(missing_pred))
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()


    def __load_omitted_parcov(self):
        """private: set the omitted_parcov attribute
        """
        if self.omitted_parcov_arg is None and self.omitted_par_arg is None:
            raise Exception("errvar.__load_omitted_parcov: " +
                            "both omitted args are None")
        # try to set omitted_parcov by extracting from base parcov
        if self.omitted_parcov_arg is None and self.omitted_par_arg is not None:
            # check to see if omitted par names are in parcov
            found = True
            for par_name in self.omitted_jco.par_names:
                if par_name not in self.parcov.col_names:
                    found = False
                    break
            if found:
                #--need to access attribute directly, not view of attribute
                self.__omitted_parcov = \
                    self._linear_analysis__parcov.extract(
                        row_names=self.omitted_jco.par_names)
            else:
                self.logger.warn("errvar.__load_omitted_parun: " +
                                 "no omitted parcov arg passed: " +
                        "setting omitted parcov as identity matrix")
                self.__omitted_parcov = mhand.cov(
                    x=np.ones(self.omitted_jco.shape[1]),
                    names=self.omitted_jco.par_names, isdiagonal=True)
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()


    def __load_omitted_jco(self):
        """private: set the omitted jco attribute
        """
        if self.omitted_par_arg is None:
            raise Exception("errvar.__load_omitted: omitted_arg is None")
        if isinstance(self.omitted_par_arg,str):
            if self.omitted_par_arg in self.jco.par_names:
                #--need to access attribute directly, not view of attribute
                self.__omitted_jco = \
                    self._linear_analysis__jco.extract(
                        col_names=self.omitted_par_arg)
            else:
                # must be a filename
                self.__omitted_jco = self.__fromfile(self.omitted_par_arg)
        # if the arg is an already instantiated matrix (or jco) object
        elif isinstance(self.omitted_par_arg,mhand.jco) or \
                isinstance(self.omitted_par_arg,mhand.matrix):
            self.__omitted_jco = \
                mhand.jco(x=self.omitted_par_arg.newx(),
                          row_names=self.omitted_par_arg.row_names,
                          col_names=self.omitted_par_arg.col_names)
        # if it is a list, then it must be a list
        # of parameter names in self.jco
        elif isinstance(self.omitted_par_arg,list):
            for arg in self.omitted_par_arg:
                if isinstance(arg,str):
                    assert arg in self.jco.par_names,\
                        "errvar.__load_omitted_jco: omitted_jco " +\
                        "arg str not in jco par_names: " + str(arg)
            self.__omitted_jco = \
                self._linear_analysis__jco.extract(col_names=self.omitted_par_arg)


    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a view only - cheap, but can be dangerous


    @property
    def omitted_predictions(self):
        if self.__omitted_predictions is None:
            self.log("loading omitted_predictions")
            self.__load_omitted_predictions()
            self.log("loading omitted_predictions")
        return self.__omitted_predictions


    @property
    def omitted_jco(self):
        if self.__omitted_jco is None:
            self.log("loading omitted_jco")
            self.__load_omitted_jco()
            self.log("loading omitted_jco")
        return self.__omitted_jco


    @property
    def omitted_parcov(self):
        if self.__omitted_parcov is None:
            self.log("loading omitted_parcov")
            self.__load_omitted_parcov()
            self.log("loading omitted_parcov")
        return self.__omitted_parcov


    @property
    def qhalfx(self):
        if self.__qhalfx is None:
            self.__qhalfx = self.qhalf * self.jco
        return self.__qhalfx


    def get_errvar_dataframe(self, singular_values):
        """get a pandas dataframe of error variance results indexed
            on singular value and (prediction name,<term>)
        Args:
            singular_values (list of int) : singular values to test
        Returns:
            multi-indexed pandas dataframe
        Raises:
            None
        """
        if not isinstance(singular_values, list) and \
                not isinstance(singular_values, np.ndarray):
            singular_values = [singular_values]
        results = {}
        for singular_value in singular_values:
            sv_results = self.variance_at(singular_value)
            for key, val in sv_results.iteritems():
                if key not in results.keys():
                    results[key] = []
                results[key].append(val)
        return pandas.DataFrame(results, index=singular_values)


    def get_identifiability_dataframe(self,singular_value):
        """get the parameter identifiability as a pandas dataframe
        Aergs:
            singular_value (int) : the truncation point
        Returns:
            A pandas dataframe of the V_1**2 matrix with the
             identifiability in the column labeled "ident"
         Raises:
            None
        """
        v1_df = self.qhalfx.v[:, :singular_value].to_dataframe() ** 2
        v1_df["ident"] = v1_df.sum(axis=1)
        return v1_df


    def variance_at(self, singular_value):
        """get the error variance of all three terms
        Args:
            singular_value (int) : singular value to test
        Returns:
            dict{[<term>,prediction_name]:standard_deviation}
        Raises:
            None
        """
        results = {}
        results.update(self.first_prediction(singular_value))
        results.update(self.second_prediction(singular_value))
        results.update(self.third_prediction(singular_value))
        return results


    def R(self, singular_value):
        """get resolution matrix at a singular value
             V_1 * V_1^T
        Args:
            singular_value (int) : singular value to calc R at
        Returns:
            R at singular_value
        Raises:
            None
        """
        if self.__R is not None and singular_value == self.__R_sv:
            return self.__R

        elif singular_value > self.jco.npar:
            self.__R_sv = self.jco.npar
            return self.parcov.identity
        else:
            self.log("calc R @" + str(singular_value))
            v1 = self.qhalfx.v[:, :singular_value]
            self.__R = v1 * v1.T
            self.__R_sv = singular_value
            self.log("calc R @" + str(singular_value))
            return self.__R


    def I_minus_R(self,singular_value):
        """get I - R at singular value
             V_2 * V_2^T
         Args:
            singular_value (int) : singular value to calc I - R at
        Returns:
            I - R at singular_value
        Raises:
            None
        """
        if self.__I_R is not None and singular_value == self.__I_R_sv:
            return self.__I_R
        else:
            if singular_value > self.jco.npar:
                return self.parcov.zero
            else:
                v2 = self.qhalfx.v[:, singular_value:]
                self.__I_R = v2 * v2.T
                self.__I_R_sv = singular_value
                return self.__I_R


    def G(self, singular_value):
        """get the parameter solution matrix at a singular value
            V_1 * S_1^(_1) * U_1^T
        Args:
            singular_value (int) : singular value to calc G at
        Returns:
            G at singular_value
        Raises:
            None
        """
        if self.__G is not None and singular_value == self.__G_sv:
            return self.__G

        if singular_value == 0:
            self.__G_sv = 0
            self.__G = mhand.matrix(
                x=np.zeros((self.jco.npar,self.jco.nobs)),
                row_names=self.jco.col_names, col_names=self.jco.row_names)
            return self.__G
        if singular_value > min(self.pst.npar_adj,self.pst.nnz_obs):
            self.logger.warn(
                "errvar.G(): singular_value > min(npar,nobs):" +
                "resetting to min(npar,nobs): " +
                str(min(self.pst.npar_adj, self.pst.nnz_obs)))
            singular_value = min(self.pst.npar_adj, self.pst.nnz_obs)
        self.log("calc G @" + str(singular_value))
        v1 = self.qhalfx.v[:, :singular_value]
        s1 = ((self.qhalfx.s[:singular_value]) ** 2).inv
        self.__G = v1 * s1 * v1.T * self.jco.T * self.obscov.inv
        self.__G_sv = singular_value
        self.__G.row_names = self.jco.col_names
        self.__G.col_names = self.jco.row_names
        self.__G.autoalign = True
        self.log("calc G @" + str(singular_value))
        return self.__G


    def first_forecast(self,singular_value):
        return self.first_prediction(singular_value)

    def first_prediction(self, singular_value):
        """get the null space term contribution to prediction error variance
            at a singular value
        Args:
            singular_value (int) : singular value to calc first term at
        Returns:
            dict{["first",prediction_names]:error variance} at singular_value
        Raises:
            Exception if no predictions are set
        """
        if not self.predictions:
            raise Exception("errvar.first(): no predictions are set")
        if singular_value > self.jco.npar:
            zero_preds = {}
            for pred in self.predictions:
                zero_preds[("first", pred.col_names[0])] = 0.0
            return zero_preds
        self.log("calc first term parameter @" + str(singular_value))
        first_term = self.I_minus_R(singular_value).T * self.parcov *\
                     self.I_minus_R(singular_value)
        if self.predictions:
            results = {}
            for prediction in self.predictions:
                results[("first",prediction.col_names[0])] = \
                    float((prediction.T * first_term * prediction).x)
            self.log("calc first term parameter @" + str(singular_value))
            return results


    def first_parameter(self, singular_value):
        """get the null space term contribution to parameter error variance
            at a singular value
        Args:
            singular_value (int) : singular value to calc first term at
        Returns:
            Cov object of first term error variance
        Raises:
            None
        """
        self.log("calc first term parameter @" + str(singular_value))
        first_term = self.I_minus_R(singular_value) * self.parcov * \
                     self.I_minus_R(singular_value)
        self.log("calc first term parameter @" + str(singular_value))
        return first_term


    def second_forecast(self,singular_value):
        return self.second_prediction(singular_value)


    def second_prediction(self, singular_value):
        """get the solution space contribution to predictive error variance
            at a singular value
            y^t * G * obscov * G^T * y
        Args:
            singular_value (int) : singular value to calc second term at
        Returns:
             dict{["second",prediction_names]:error variance} at singular_value
        Raises:
            Exception if no predictions are set
        """
        if not self.predictions:
            raise Exception("errvar.second(): not predictions are set")
        self.log("calc second term prediction @" + str(singular_value))

        if singular_value > min(self.pst.npar_adj, self.pst.nnz_obs):
            inf_pred = {}
            for pred in self.predictions:
                inf_pred[("second",pred.col_names[0])] = 1.0E+35
            return inf_pred
        # elif singular_value == 0:
        #     zero_preds = {}
        #     for pred in self.predictions:
        #         zero_preds[("second", pred.col_names[0])] = 0.0
        #     return zero_preds
        else:
            second_term = self.G(singular_value) * self.obscov * \
                          self.G(singular_value).T
            results = {}
            for prediction in self.predictions:
                results[("second",prediction.col_names[0])] = \
                    float((prediction.T * second_term * prediction).x)
            self.log("calc second term prediction @" + str(singular_value))
            return results


    def second_parameter(self, singular_value):
        """get the solution space contribution to parameter error variance
             at a singular value
            G * obscov * G^T
        Args:
            singular_value (int) : singular value to calc second term at
        Returns:
            Cov object of second term error variance
        Raises:
            None
        """
        self.log("calc second term parameter @" + str(singular_value))
        result = self.G(singular_value) * self.obscov * self.G(singular_value).T
        self.log("calc second term parameter @" + str(singular_value))
        return result


    def third_forecast(self,singular_value):
        return self.third_prediction(singular_value)

    def third_prediction(self,singular_value):
        """get the omitted parameter contribution to error variance at a singular value
            predictions:
                p * Simga_(omitted_pars) * p^T
                p = prediction^T * G * omitted_jco - omitted_prediction^T
        Args:
            singular_value (int) : singular value to calc third term at
        Returns:
            dict{["third",prediction_names]:error variance} at singular_value
        Raises:
            Exception if no predictions are set
        """
        if not self.predictions:
            raise Exception("errvar.third(): not predictions are set")
        if self.__need_omitted is False:
            zero_preds = {}
            for pred in self.predictions:
                zero_preds[("third", pred.col_names[0])] = 0.0
            return zero_preds
        self.log("calc third term prediction @" + str(singular_value))
        if singular_value > min(self.pst.npar_adj, self.pst.nnz_obs):
            inf_pred = {}
            for pred in self.predictions:
                inf_pred[("third",pred.col_names[0])] = 1.0E+35
            return inf_pred
        else:
            results = {}
            for prediction,omitted_prediction in \
                    zip(self.predictions, self.omitted_predictions):
                #--comes out as row vector, but needs to be a column vector
                p = ((prediction.T * self.G(singular_value) * self.omitted_jco)
                     - omitted_prediction.T).T
                result = float((p.T * self.omitted_parcov * p).x)
                results[("third", prediction.col_names[0])] = result
            self.log("calc third term prediction @" + str(singular_value))
            return results


    def third_parameter(self, singular_value):
        """get the omitted parameter contribution to parameter error variance
             at a singular value
                G * omitted_jco * Sigma_(omitted_pars) * omitted_jco^T * G^T
        Args:
            singular_value (int) : singular value to calc third term at
        Returns:
            0.0 if need_omitted is False
            Cov object of third term error variance
        Raises:
            None
        """
        if self.__need_omitted is False:
            return 0.0
        self.log("calc third term parameter @" + str(singular_value))
        GZo = self.G(singular_value) * self.omitted_jco
        result = GZo * self.omitted_parcov * GZo.T
        self.log("calc third term parameter @" + str(singular_value))
        return result



if __name__ == "__main__":
    #la = linear_analysis(jco="pest.jcb")
    #forecasts = ["C_obs13_2","c_obs10_2","c_obs05_2"]
    #forecasts = ["pd_one","pd_ten","pd_half"]
    la = schur(jco=os.path.join("for_nick", "tseriesVERArad.jco"))
    print la.posterior_parameter

    #ev = errvar(jco=os.path.join("henry", "pest.jco"), forecasts=forecasts,verbose=False,omitted_parameters="mult1",)
    #df = ev.get_errvar_dataframe(singular_values=[0])
    #print df


