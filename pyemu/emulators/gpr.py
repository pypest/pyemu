"""
Gaussian Process Regression (GPR) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import shutil
import inspect
from .base import Emulator
from .transformers import AutobotsAssemble
from sklearn.gaussian_process import GaussianProcessRegressor
from pyemu.utils import run

from pyemu.pst import Pst


class GPR(Emulator):
    """
    Gaussian Process Regression (GPR) emulator class.
    
    This class implements a GPR-based emulator that trains separate Gaussian Process
    models for each output variable. It supports various kernel types, feature
    transformations, and provides uncertainty quantification.
    
    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input and output features for training.
    input_names : list of str, optional
        Names of input features to use. If None, all columns in input_data are used.
    output_names : list of str, optional
        Names of output variables to emulate. If None, all columns in output_data are used.
    kernel : sklearn kernel object, optional
        Kernel to use for GP regression. If None, defaults to Matern kernel.
    transforms : list of dict, optional. Defaults to [{'type': 'standard_scaler'}]
    n_restarts_optimizer : int, optional
        Number of restarts for kernel hyperparameter optimization. Default is 10.
    return_std : bool, optional
        Whether to return prediction uncertainties. Default is True.
    verbose : bool, optional
        Enable verbose logging. Default is True.
    """
    
    def __init__(self, 
                 data,
                 input_names=None,
                 output_names=None,
                 kernel=None,
                 transforms=[{'type': 'standard_scaler'}],
                 n_restarts_optimizer=10,
                 return_std=True,
                 verbose=True):
        """Initialize the GPR emulator."""
        
        super().__init__(verbose=verbose)
        
        # Store initialization parameters
        # check data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        self.data = data.copy()

        # Check input and output names
        # check input_names and output_names are lists or None
        if input_names is not None and not isinstance(input_names, list):
            raise ValueError("input_names must be a list or None")
        if output_names is not None and not isinstance(output_names, list):
            raise ValueError("output_names must be a list or None")
        self.input_names = input_names
        self.output_names = output_names

        self.kernel = kernel
        self.transforms = transforms
        self.n_restarts_optimizer = n_restarts_optimizer
        self.return_std = return_std
        
        # Initialize data
        self.data = data
        
        # Model storage
        self.models = {}
        self.model_info = None
        self.verification_results = {}
        
        # PEST++ integration
        self.template_dir = None
        
        # Validate transforms parameter
        if transforms is not None:
            self._validate_transforms(transforms)
            self._validate_transforms_for_gpr()
         
    def _validate_transforms_for_gpr(self):
        """Validate transforms parameter for GPR. Make sure transforms are only applied to input data."""
        # Validate transforms parameter
        transforms = self.transforms
        if transforms is not None:
            # For the speicif case of GPR, we only transform input data    
            for t in transforms:
                if 'columns' in t:
                    # check if any columns are in output_names
                    if self.output_names is not None:
                        common_cols = set(t['columns']).intersection(self.output_names)
                        if common_cols:
                            self.logger.statement(f"Transform {t['type']} will not be applied to output columns: {common_cols}")
                            # remove these columns from transforms
                            t['columns'] = [col for col in t['columns'] if col not in common_cols]
                            if not t['columns']:
                                self.logger.statement(f"Transform {t['type']} has no columns left after removing output columns: {common_cols}")
                                # remove this transform
                                self.logger.statement(f"Removing transform {t['type']} as it has no columns left")
                                self.transforms.remove(t)
                else:
                    self.logger.statement(f"Transform {t['type']} has no specified columns, applying to all input columns")
                    t['columns'] = self.input_names if self.input_names is not None else []
        return transforms   

#    def _combine_input_output_data(self, input_data, output_data):
#        """Combine input and output data into a single DataFrame."""
#        if input_data.shape[0] != output_data.shape[0]:
#            raise ValueError("Input and output data must have the same number of rows")
#        
#        combined = input_data.copy()
#        for col in output_data.columns:
#            if col not in combined.columns:
#                combined[col] = output_data[col]
#            else:
#                self.logger.statement(f"Warning: column '{col}' exists in both input and output data, using output data")
#                combined[col] = output_data[col]
#        
#        return combined
    
    def _setup_kernel(self):
        """Set up the GP kernel if not provided."""
        if self.kernel is None:
            try:
                from sklearn.gaussian_process.kernels import Matern,ConstantKernel,RBF
                self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                                                                length_scale=np.ones(len(self.input_names)) * 2.0,
                                                                length_scale_bounds=(1e-4, 1e4),
                                                                nu=1.5)
                self.logger.statement("Using default Matern kernel")
            except ImportError:
                raise ImportError("scikit-learn is required for GPR emulator")

        # Log kernel hyperparameters
        self.logger.statement(f"Using kernel: {self.kernel}")

    
    def _prepare_training_data(self):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        self : GPR
            The GPR emulator instance containing the data and configuration.
            
        Returns
        -------
        pandas.DataFrame
            Processed data ready for model fitting.
        """

        if self.data is None:
            raise ValueError("No data provided and no data stored in the emulator")
        data = self.data
        
        # Apply feature transformations if specified
        if self.transforms is not None:
            self._validate_transforms_for_gpr()
            self.logger.statement("applying feature transforms")
            self.data_transformed = self._fit_transformer_pipeline(data, self.transforms)
        else:
            # Still need to set up a dummy transformer for consistency
            from .transformers import AutobotsAssemble
            self.transformer_pipeline = AutobotsAssemble(data.copy())
            self.data_transformed = data.copy()
        
        return self.data_transformed
    

    def fit(self):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        self: GPR
            The GPR emulator instance containing the data and configuration.
            
        Returns
        -------
        self : GPR
            Fitted GPR emulator instance.
        """
        
        if self.data_transformed is None:
            self.logger.statement("transforming training data")
            self.data_transformed = self._prepare_training_data()
        if self.kernel is None:
            self._setup_kernel()
        # transformed input data
        X_transformed = self.data_transformed.loc[:,self.input_names].copy()
        y_transformed = self.data_transformed.loc[:,self.output_names].copy() #Note that these are actualy not transformed

        assert X_transformed.shape[0] == y_transformed.shape[0], \
            "Input and output data must have the same number of rows"
        assert X_transformed.shape[1] > 0, "Input data must have at least one feature"
        assert y_transformed.shape[1] > 0, "Output data must have at least one variable"

        # Create and fit separate GPR model for each output
        self.gpr_models = {}
        for output_name in self.output_names:
            gpr = GaussianProcessRegressor(
                kernel=self.kernel,
                #alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
                #random_state=self.random_state
            )
            
            # Fit the GPR model for this output
            gpr.fit(X_transformed.loc[:,self.input_names].values, y_transformed.loc[:,output_name].values)
            self.gpr_models[output_name] = gpr
        
        self.fitted = True
        return self

    def predict(self, X, return_std=False):
        """
        Make predictions using the fitted GPR emulators.

        Parameters
        ----------
        X : pandas.DataFrame 
            Input features for prediction
        return_std : bool, default False
            Whether to return prediction standard deviation

        Returns
        -------
        predictions : pandas.DataFrame
            Predicted values for each output
        std : pandas.DataFrame, optional
            Prediction standard deviations (if return_std=True)
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before making predictions")
        
        if not hasattr(self, 'transformer_pipeline') or self.transformer_pipeline is None:
            raise ValueError("Emulator must be fitted and have valid transformations before prediction")
        
        # Apply same transforms as training data
        X_transformed = self.transformer_pipeline.transform(X.copy())

        
        # Make predictions for each output
        predictions_dict = {}
        std_dict = {}
        
        for output_name in self.output_names:
            gpr = self.gpr_models[output_name]
            
            if return_std:
                pred, std = gpr.predict(X_transformed.values, return_std=True)
                predictions_dict[output_name] = pred
                std_dict[output_name] = std
            else:
                pred = gpr.predict(X_transformed.values)
                predictions_dict[output_name] = pred
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions_dict, index=X.index)
        
        if return_std:
            std_df = pd.DataFrame(std_dict, index=X.index)
            return predictions_df, std_df
        else:
            return predictions_df



        

    def prepare_pestpp(self,pst_dir,casename,gpr_t_d="gpr_template"):
        """
        Prepare a PEST++ template directory for the GPR emulator.
        
        Parameters
        ----------
        gpr_t_d : str
            Path to the PEST++ template directory.
        pst_fpath : str
            Path to an existing PEST control file (PST). The assumption is that an existing PST setup exists for the process-based model.
        
        Returns
        -------
        None
            """
        

        
        #TODO: it may be more logical to pass in a Pst object instead of a file path; assume the user loads Pst and training data before hand???
        # Give Emulators a "harvest" function that returns a Pst object with the necessary information?

        # what are the things we need to get from Pst?
        # 1. decivsion variable names (parameters) a.k.a input_names
        # 2. observation names (outputs) aka output_names
        # 3. which obs are objectives; subset of output_names
        # 4. which obs are constraints; subset of output_names

        pst, input_names, output_names, objs, constraints = scrape_pst_dir(pst_dir,casename)


        # check that all input_names ar ein par data
        if self.input_names is None:
            raise ValueError("input_names must be provided")
        missing_inputs = set(self.input_names) - set(pst.parameter_data.index)
        if missing_inputs:
            raise ValueError(f"Input names {missing_inputs} not found in parameter data")
        # check that all input names are adjsutable
        fixed_inputs = pst.parameter_data.loc[self.input_names, "partrans"].str.contains("fixed|tied", case=False, na=False)
        if fixed_inputs.any():
            raise ValueError(f"Input names {self.input_names[fixed_inputs]} cannot be fixed or tied")
        self.logger.statement(f"Decision variable parameter names: {self.input_names}")

        # check that all self.output_names are in observation_data
        if self.output_names is None:
            raise ValueError("output_names must be provided")
        missing_outputs = set(self.output_names) - set(pst.observation_data.index)
        if missing_outputs:
            raise ValueError(f"Output names {missing_outputs} not found in observation data")
        self.logger.statement(f"Observation names: {self.output_names}")


        # preapre the GPR template directory
        if os.path.exists(gpr_t_d):
            self.logger.statement(f"Removing existing template directory {gpr_t_d}")
            shutil.rmtree(gpr_t_d)
        self.logger.statement(f"Creating template directory {gpr_t_d}")
        os.makedirs(gpr_t_d)


        # preapre template files
        self.logger.statement("Preparing PEST++ template files")
        
        #write a template file
        tpl_fname = os.path.join(gpr_t_d,"gpr_input.csv.tpl")
        with open(tpl_fname,'w') as f:
            f.write("ptf ~\nparnme,parval1\n")
            for input_name in self.input_names:
                f.write("{0},~  {0}   ~\n".format(input_name))
        # keep track of other non-decvar parameters
        other_pars = list(set(pst.parameter_data.parnme.tolist())-set(self.input_names))
        aux_tpl_fname = None
        if len(other_pars) > 0:
            aux_tpl_fname = os.path.join(gpr_t_d,"aux_par.csv.tpl")
            print("writing aux par tpl file: ",aux_tpl_fname)
            with open(aux_tpl_fname,'w') as f:
                f.write("ptf ~\n")
                for input_name in other_pars:
                    f.write("{0},~  {0}   ~\n".format(input_name))

        #write an ins file
        ins_fname = os.path.join(gpr_t_d,"gpr_output.csv.ins")
        with open(ins_fname,'w') as f:
            f.write("pif ~\nl1\n")
            for output_name in self.output_names:
                if self.return_std:
                    f.write("l1 ~,~ !{0}! ~,~ !{0}_gprstd!\n".format(output_name))
                else:
                    f.write("l1 ~,~ !{0}!\n".format(output_name))

        # build the GPR Pst object
        self.logger.statement("Building PEST++ control file")
        tpl_list = [tpl_fname]
        if aux_tpl_fname is not None:
            tpl_list.append(aux_tpl_fname)
        input_list = [f.replace(".tpl","") for f in tpl_list]
        gpst = Pst.from_io_files(tpl_list,input_list,
                                    [ins_fname],[ins_fname.replace(".ins","")],pst_path=".")
        

        def fix_df_col_type(orgdf,fixdf):
            for col in orgdf.columns:
                # this gross thing is to avoid a future error warning in pandas - 
                # why is it getting so strict?!  isn't python duck-typed?
                if col in fixdf.columns and\
                fixdf.dtypes[col] != orgdf.dtypes[col]:
                    fixdf[col] = fixdf[col].astype(orgdf.dtypes[col])
                fixdf.loc[orgdf.index,col] = orgdf.loc[orgdf.index,col].values
            return

        fix_df_col_type(orgdf=pst.parameter_data,fixdf=gpst.parameter_data)
        fix_df_col_type(orgdf=pst.observation_data,fixdf=gpst.observation_data)

        if self.return_std:
            stdobs = [o for o in gpst.obs_names if o.endswith("_gprstd")]
            assert len(stdobs) > 0
            gpst.observation_data.loc[stdobs,"weight"] = 0.0

        gpst.pestpp_options = pst.pestpp_options
        gpst.prior_information = pst.prior_information.copy()

        gpst.model_command = "python forward_run.py"
        frun_lines = inspect.getsource(gpr_forward_run)
        with open(os.path.join(gpr_t_d, "forward_run.py"), 'w') as f:
            f.write("\n")
            for import_name in ["pandas as pd","os","numpy as np"]:
                f.write("import {0}\n".format(import_name))
            for line in frun_lines:
                f.write(line)
            f.write("if __name__ == '__main__':\n")
            f.write("    gpr_forward_run()\n")

        # pickle
        self.save(os.path.join(gpr_t_d, "gpr_emulator.pkl"))
        self.logger.statement(f"Saved GPR emulator to {os.path.join(gpr_t_d, 'gpr_emulator.pkl')}")
        
        gpst.control_data.noptmax = 0
        
        gpst_fname = f"{casename}_gpr.pst"
        gpst.write(os.path.join(gpr_t_d,gpst_fname),version=2)
        print("saved gpr pst:",gpst_fname,"in gpr_t_d",gpr_t_d)
        try:
            run("pestpp-mou {0}".format(gpst_fname),cwd=gpr_t_d)
        except Exception as e:
            print("WARNING: pestpp-mou test run failed: {0}".format(str(e)))
        gpst.control_data.noptmax = pst.control_data.noptmax
        gpst.write(os.path.join(gpr_t_d, gpst_fname), version=2)



        return
    
def gpr_forward_run():
    """the function to evaluate a set of inputs thru the GPR emulators.\
    This function gets added programmatically to the forward run process"""
    import pandas as pd
    from pyemu.emulators import GPR
    input_df = pd.read_csv("gpr_input.csv",index_col=0)
    gpr = GPR.load("gpr_emulator.pkl")
    simdf = pd.DataFrame(index=gpr.output_names,columns=["sim","sim_std"])
    simdf.index.name = "output_name"
    if gpr.return_std:
        predmean,predstdv = gpr.predict(input_df.loc[gpr.input_names].T, return_std=True)
        simdf.loc[:,"sim"] = predmean[simdf.index].values
        simdf.loc[:,"sim_std"] = predstdv[simdf.index].values
    else:
        predmean = gpr.predict(input_df.loc[gpr.input_names].T)
        simdf.loc[:,"sim"] = predmean[simdf.index].values
    simdf.to_csv("gpr_output.csv",index=True)
    return simdf

def scrape_pst_dir(pst_dir,casename):

    if not os.path.exists(pst_dir):
        raise FileNotFoundError(f"PEST control file {pst_dir} does not exist")
    
    pst = Pst(os.path.join(pst_dir,casename + ".pst"))

    # work out input variable names
    input_groups = pst.pestpp_options.get("opt_dec_var_groups",None)
    par = pst.parameter_data
    if input_groups is None:
        print("using all adjustable parameters as inputs")
        input_names = pst.adj_par_names
    else:
        input_groups = set([i.strip() for i in input_groups.lower().strip().split(",")])
        print("input groups:",input_groups)
        adj_par = par.loc[pst.adj_par_names,:].copy()
        adj_par = adj_par.loc[adj_par.pargp.apply(lambda x: x in input_groups),:]
        input_names = adj_par.parnme.tolist()
    print("input names:",input_names)

    #work out constraints and objectives
    ineq_names = pst.less_than_obs_constraints.tolist()
    ineq_names.extend(pst.greater_than_obs_constraints.tolist())
    obs = pst.observation_data
    objs = pst.pestpp_options.get("mou_objectives",None)
    constraints = []

    if objs is None:
        print("'mou_objectives' not found in ++ options, using all ineq tagged non-zero weighted obs as objectives")
        objs = ineq_names
    else:
        objs = objs.lower().strip().split(',')
        constraints = [n for n in ineq_names if n not in objs]

    print("objectives:",objs)
    print("constraints:",constraints)
    output_names = objs
    output_names.extend(constraints)

    return pst, input_names, output_names, objs, constraints