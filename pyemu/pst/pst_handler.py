from __future__ import print_function, division
import os
import glob
import re
import copy
import shutil
import time
import warnings
import numpy as np
from numpy.lib.type_check import real_if_close
import pandas as pd


pd.options.display.max_colwidth = 100
pd.options.mode.use_inf_as_na = True
import pyemu
from ..pyemu_warnings import PyemuWarning
from pyemu.pst.pst_controldata import ControlData, SvdData, RegData
from pyemu.pst import pst_utils
from pyemu.plot import plot_utils

# from pyemu.utils.os_utils import run


class Pst(object):
    """All things PEST(++) control file

    Args:
        filename (`str`):  the name of the control file
        load (`bool`, optional): flag to load the control file. Default is True
        resfile (`str`, optional): corresponding residual file.  If `None`, a residual file
            with the control file base name is sought.  Default is `None`

    Note:
        This class is the primary mechanism for dealing with PEST control files.  Support is provided
        for constructing new control files as well as manipulating existing control files.

    Example::

        pst = pyemu.Pst("my.pst")
        pst.control_data.noptmax = -1
        pst.write("my_new.pst")

    """

    def __init__(self, filename, load=True, resfile=None):

        self.parameter_data = None
        """pandas.DataFrame:  '* parameter data' information.  Columns are 
        standard PEST variable names
        
        Example::
            
            pst.parameter_data.loc[:,"partrans"] = "log"
            pst.parameter_data.loc[:,"parubnd"] = 10.0
        
        """
        self.observation_data = None
        """pandas.DataFrame:  '* observation data' information.  Columns are standard PEST
        variable names
        
        Example::
        
            pst.observation_data.loc[:,"weight"] = 1.0
            pst.observation_data.loc[:,"obgnme"] = "obs_group"
        
        """
        self.prior_information = None
        """pandas.DataFrame:  '* prior information' data.  Columns are standard PEST
        variable names"""

        self.model_input_data = pst_utils.pst_config["null_model_io"].copy()
        self.model_output_data = pst_utils.pst_config["null_model_io"].copy()

        self.filename = filename
        self.resfile = resfile
        self.__res = None
        self.__pi_count = 0
        self.with_comments = False
        self.comments = {}
        self.other_sections = {}
        self.new_filename = None
        for key, value in pst_utils.pst_config.items():
            self.__setattr__(key, copy.copy(value))
        # self.tied = None
        self.control_data = ControlData()
        """pyemu.pst.pst_controldata.ControlData:  '* control data' information.  
        Access with standard PEST variable names 
        
        Example:: 
            
            pst.control_data.noptmax = 2
            pst.control_data.pestmode = "estimation"
            
            
        """
        self.svd_data = SvdData()
        """pyemu.pst.pst_controldata.SvdData: '* singular value decomposition' section information.  
        Access with standard PEST variable names
        
        Example::
        
            pst.svd_data.maxsing = 100
            
        
        """
        self.reg_data = RegData()
        """pyemu.pst.pst_controldata.RegData: '* regularization' section information.
        Access with standard PEST variable names.
        
        Example:: 
        
            pst.reg_data.phimlim = 1.00 #yeah right!

        
        """
        if load:
            if not os.path.exists(filename):
                raise Exception("pst file not found:{0}".format(filename))

            self.load(filename)

    def __setattr__(self, key, value):
        if key == "model_command":
            if isinstance(value, str):
                value = [value]
        super(Pst, self).__setattr__(key, value)

    @classmethod
    def from_par_obs_names(cls, par_names=["par1"], obs_names=["obs1"]):
        """construct a shell `Pst` instance from parameter and observation names

        Args:
            par_names ([`str`]): list of parameter names.  Default is [`par1`]
            obs_names ([`str`]): list of observation names.  Default is [`obs1`]

        Note:
            While this method works, it does not make template or instruction files.
            Users are encouraged to use `Pst.from_io_files()` for more usefulness

        Example::

            par_names = ["par1","par2"]
            obs_names = ["obs1","obs2"]
            pst = pyemu.Pst.from_par_obs_names(par_names,obs_names)

        """
        return pst_utils.generic_pst(par_names=par_names, obs_names=obs_names)

    @staticmethod
    def get_constraint_tags(ltgt='lt'):
        if ltgt == 'lt':
            return "l_", "less", ">@"
        else:
            return "g_", "greater", "<@"

    @property
    def phi(self):
        """get the weighted total objective function.

        Returns:
            `float`: sum of squared residuals

        Note:
            Requires `Pst.res` (the residuals file) to be available

        """
        psum = 0.0
        for _, contrib in self.phi_components.items():
            psum += contrib
        return psum

    @property
    def phi_components(self):
        """get the individual components of the total objective function

        Returns:
            `dict`: dictionary of observation group, contribution to total phi


        Note:
            Requires `Pst.res` (the residuals file) to be available

        """

        # calculate phi components for each obs group
        components = {}
        ogroups = self.observation_data.groupby("obgnme").groups
        rgroups = self.res.groupby("group").groups
        self.res.index = self.res.name
        for og, onames in ogroups.items():
            # assert og in rgroups.keys(),"Pst.phi_componentw obs group " +\
            #    "not found: " + str(og)
            # og_res_df = self.res.ix[rgroups[og]]
            og_res_df = self.res.loc[onames, :].dropna(axis=1)
            # og_res_df.index = og_res_df.name
            og_df = self.observation_data.loc[ogroups[og], :]
            og_df.index = og_df.obsnme
            # og_res_df = og_res_df.loc[og_df.index,:]
            assert og_df.shape[0] == og_res_df.shape[0], (
                " Pst.phi_components error: group residual dataframe row length"
                + "doesn't match observation data group dataframe row length"
                + str(og_df.shape)
                + " vs. "
                + str(og_res_df.shape)
                + ","
                + og
            )
            if "modelled" not in og_res_df.columns:
                print(og_res_df)
                m = self.res.loc[onames, "modelled"]
                print(m.loc[m.isna()])
                raise Exception("'modelled' not in res df columns for group " + og)
            # components[og] = np.sum((og_res_df["residual"] *
            #                          og_df["weight"]) ** 2)
            mod_vals = og_res_df.loc[og_df.obsnme, "modelled"]
            if og.lower().startswith(self.get_constraint_tags('gt')):
                mod_vals.loc[mod_vals >= og_df.loc[:, "obsval"]] = og_df.loc[
                    :, "obsval"
                ]
            elif og.lower().startswith(self.get_constraint_tags('lt')):
                mod_vals.loc[mod_vals <= og_df.loc[:, "obsval"]] = og_df.loc[
                    :, "obsval"
                ]
            components[og] = np.sum(
                ((og_df.loc[:, "obsval"] - mod_vals) * og_df.loc[:, "weight"]) ** 2
            )
        if (
            not self.control_data.pestmode.startswith("reg")
            and self.prior_information.shape[0] > 0
        ):
            ogroups = self.prior_information.groupby("obgnme").groups
            for og in ogroups.keys():
                if og not in rgroups.keys():
                    raise Exception(
                        "Pst.adjust_weights_res() obs group " + "not found: " + str(og)
                    )
                og_res_df = self.res.loc[rgroups[og], :]
                og_res_df.index = og_res_df.name
                og_df = self.prior_information.loc[ogroups[og], :]
                og_df.index = og_df.pilbl
                og_res_df = og_res_df.loc[og_df.index, :].copy()
                if og_df.shape[0] != og_res_df.shape[0]:
                    raise Exception(
                        " Pst.phi_components error: group residual dataframe row length"
                        + "doesn't match observation data group dataframe row length"
                        + str(og_df.shape)
                        + " vs. "
                        + str(og_res_df.shape)
                    )
                if og.lower().startswith(self.get_constraint_tags('gt')):
                    gidx = og_res_df.loc[:, "residual"] >= 0
                    og_res_df.loc[gidx, "residual"] = 0
                elif og.lower().startswith(self.get_constraint_tags('lt')):
                    lidx = og_res_df.loc[:, "residual"] <= 0
                    og_res_df.loc[lidx, "residual"] = 0
                components[og] = np.sum((og_res_df["residual"] * og_df["weight"]) ** 2)

        return components

    @property
    def phi_components_normalized(self):
        """get the individual components of the total objective function
            normalized to the total PHI being 1.0

        Returns:
            `dict`:  dictionary of observation group,
            normalized contribution to total phi

        Note:
            Requires `Pst.res` (the residuals file) to be available


        """
        # use a dictionary comprehension to go through and normalize each component of phi to the total
        phi = self.phi
        comps = self.phi_components
        norm = {i: c / phi for i, c in comps.items()}
        #print(phi, comps, norm)

        return norm

    def set_res(self, res):
        """reset the private `Pst.res` attribute.

        Args:
            res : (`pandas.DataFrame` or `str`): something to use as Pst.res attribute.
                If `res` is `str`, a dataframe is read from file `res`


        """
        if isinstance(res, str):
            res = pst_utils.read_resfile(res)
        self.__res = res

    @property
    def res(self):
        """get the residuals dataframe attribute

        Returns:
            `pandas.DataFrame`: a dataframe containing the
            residuals information.

        Note:
            if the Pst.__res attribute has not been loaded,
            this call loads the res dataframe from a file

        Example::

            # print the observed and simulated values for non-zero weighted obs
            print(pst.res.loc[pst.nnz_obs_names,["modelled","measured"]])

        """
        if self.__res is not None:
            return self.__res
        else:
            if self.resfile is not None:
                if not os.path.exists(self.resfile):
                    raise Exception(
                        "Pst.res: self.resfile " + str(self.resfile) + " does not exist"
                    )
            else:
                self.resfile = self.filename.replace(".pst", ".res")
                if not os.path.exists(self.resfile):
                    self.resfile = self.resfile.replace(".res", ".rei")
                    if not os.path.exists(self.resfile):
                        self.resfile = self.resfile.replace(".rei", ".base.rei")
                        if not os.path.exists(self.resfile):
                            if self.new_filename is not None:
                                self.resfile = self.new_filename.replace(".pst", ".res")
                                if not os.path.exists(self.resfile):
                                    self.resfile = self.resfile.replace(".res", ".rei")
                                    if not os.path.exists(self.resfile):
                                        raise Exception(
                                            "Pst.res: "
                                            + "could not residual file case.res"
                                            + " or case.rei"
                                            + " or case.base.rei"
                                            + " or case.obs.csv"
                                        )

            res = pst_utils.read_resfile(self.resfile)
            missing_bool = self.observation_data.obsnme.apply(
                lambda x: x not in res.name
            )
            missing = self.observation_data.obsnme[missing_bool]
            if missing.shape[0] > 0:
                raise Exception(
                    "Pst.res: the following observations "
                    + "were not found in "
                    + "{0}:{1}".format(self.resfile, ",".join(missing))
                )
            self.__res = res
            return self.__res

    @property
    def nprior(self):
        """number of prior information equations

        Returns:
            `int`: the number of prior info equations

        """
        self.control_data.nprior = self.prior_information.shape[0]
        return self.control_data.nprior

    @property
    def nnz_obs(self):
        """get the number of non-zero weighted observations

        Returns:
            `int`: the number of non-zeros weighted observations

        """
        nnz = 0
        for w in self.observation_data.weight:
            if w > 0.0:
                nnz += 1
        return nnz

    @property
    def nobs(self):
        """get the number of observations

        Returns:
            `int`: the number of observations

        """
        self.control_data.nobs = self.observation_data.shape[0]
        return self.control_data.nobs

    @property
    def npar_adj(self):
        """get the number of adjustable parameters (not fixed or tied)

        Returns:
            `int`: the number of adjustable parameters

        """
        pass
        np = 0
        for t in self.parameter_data.partrans:
            if t not in ["fixed", "tied"]:
                np += 1
        return np

    @property
    def npar(self):
        """get number of parameters

        Returns:
            `int`: the number of parameters

        """
        self.control_data.npar = self.parameter_data.shape[0]
        return self.control_data.npar

    @property
    def forecast_names(self):
        """get the forecast names from the pestpp options (if any).
        Returns None if no forecasts are named

        Returns:
            [`str`]: a list of forecast names.

        """
        if "forecasts" in self.pestpp_options.keys():
            if isinstance(self.pestpp_options["forecasts"], str):
                return self.pestpp_options["forecasts"].lower().split(",")
            else:
                return [f.lower() for f in self.pestpp_options["forecasts"]]
        elif "predictions" in self.pestpp_options.keys():
            if isinstance(self.pestpp_options["predictions"], str):
                return self.pestpp_options["predictions"].lower().split(",")
            else:
                return [f.lower() for f in self.pestpp_options["predictions"]]
        else:
            return None

    @property
    def obs_groups(self):
        """get the observation groups

        Returns:
            [`str`]: a list of unique observation groups

        """
        og = self.observation_data.obgnme.unique().tolist()
        return og

    @property
    def nnz_obs_groups(self):
        """get the observation groups that contain at least one non-zero weighted
         observation

        Returns:
            [`str`]: a list of observation groups that contain at
            least one non-zero weighted observation

        """
        obs = self.observation_data
        og = obs.loc[obs.weight > 0.0, "obgnme"].unique().tolist()
        return og

    @property
    def adj_par_groups(self):
        """get the parameter groups with atleast one adjustable parameter

        Returns:
            [`str`]: a list of parameter groups with
            at least one adjustable parameter

        """
        par = self.parameter_data
        tf = set(["tied", "fixed"])
        adj_pargp = par.loc[par.partrans.apply(lambda x: x not in tf), "pargp"].unique()
        return adj_pargp.tolist()

    @property
    def par_groups(self):
        """get the parameter groups

        Returns:
            [`str`]: a list of parameter groups

        """
        return self.parameter_data.pargp.unique().tolist()

    @property
    def prior_groups(self):
        """get the prior info groups

        Returns:
            [`str`]: a list of prior information groups

        """
        og = self.prior_information.obgnme.unique().tolist()
        return og

    @property
    def prior_names(self):
        """get the prior information names

        Returns:
            [`str`]: a list of prior information names

        """
        return self.prior_information.pilbl.tolist()

    @property
    def par_names(self):
        """get the parameter names

        Returns:
            [`str`]: a list of parameter names
        """
        return self.parameter_data.parnme.tolist()

    @property
    def adj_par_names(self):
        """get the adjustable (not fixed or tied) parameter names

        Returns:
            [`str`]: list of adjustable (not fixed or tied)
            parameter names

        """
        par = self.parameter_data
        tf = set(["tied", "fixed"])
        adj_names = par.loc[par.partrans.apply(lambda x: x not in tf), "parnme"]
        return adj_names.tolist()

    @property
    def obs_names(self):
        """get the observation names

        Returns:
            [`str`]: a list of observation names

        """
        return self.observation_data.obsnme.tolist()

    @property
    def nnz_obs_names(self):
        """get the non-zero weight observation names

        Returns:
            [`str`]: a list of non-zero weighted observation names

        """
        obs = self.observation_data
        nz_names = obs.loc[obs.weight > 0.0, "obsnme"]
        return nz_names.tolist()

    @property
    def zero_weight_obs_names(self):
        """get the zero-weighted observation names

        Returns:
            [`str`]: a list of zero-weighted observation names

        """
        obs = self.observation_data
        return obs.loc[obs.weight == 0.0, "obsnme"].tolist()

    @property
    def estimation(self):
        """check if the control_data.pestmode is set to estimation

        Returns:
            `bool`: True if `control_data.pestmode` is estmation, False otherwise

        """
        return self.control_data.pestmode == "estimation"

    @property
    def tied(self):
        """list of tied parameter names

        Returns:
            `pandas.DataFrame`: a dataframe of tied parameter information.
            Columns of `tied` are `parnme` and `partied`.  Returns `None` if
            no tied parameters are found.

        """
        par = self.parameter_data
        tied_pars = par.loc[par.partrans == "tied", "parnme"]
        if tied_pars.shape[0] == 0:
            return None
        if "partied" not in par.columns:
            par.loc[:, "partied"] = np.NaN
        tied = par.loc[tied_pars, ["parnme", "partied"]]
        return tied

    @property
    def template_files(self):
        """list of template file names

        Returns:
            `[str]`: a list of template file names, extracted from
                `Pst.model_input_data.pest_file`.  Returns `None` if this
                attribute is `None`

        Note:
            Use `Pst.model_input_data` to access the template-input file information for writing/modification

        """
        if (
            self.model_input_data is not None
            and "pest_file" in self.model_input_data.columns
        ):
            return self.model_input_data.pest_file.to_list()
        else:
            return None

    @property
    def input_files(self):
        """list of model input file names

        Returns:
            `[str]`: a list of model input file names, extracted from
                `Pst.model_input_data.model_file`.  Returns `None` if this
                attribute is `None`

        Note:
            Use `Pst.model_input_data` to access the template-input file information for writing/modification

        """
        if (
            self.model_input_data is not None
            and "model_file" in self.model_input_data.columns
        ):
            return self.model_input_data.model_file.to_list()
        else:
            return None

    @property
    def instruction_files(self):
        """list of instruction file names
        Returns:
            `[str]`: a list of instruction file names, extracted from
                `Pst.model_output_data.pest_file`.  Returns `None` if this
                 attribute is `None`

        Note:
            Use `Pst.model_output_data` to access the instruction-output file information for writing/modification

        """
        if (
            self.model_output_data is not None
            and "pest_file" in self.model_output_data.columns
        ):
            return self.model_output_data.pest_file.to_list()
        else:
            return None

    @property
    def output_files(self):
        """list of model output file names
        Returns:
            `[str]`: a list of model output file names, extracted from
                `Pst.model_output_data.model_file`.  Returns `None` if this
                attribute is `None`

        Note:
            Use `Pst.model_output_data` to access the instruction-output file information for writing/modification

        """
        if (
            self.model_output_data is not None
            and "model_file" in self.model_output_data.columns
        ):
            return self.model_output_data.model_file.to_list()
        else:
            return None

    @staticmethod
    def _read_df(f, nrows, names, converters, defaults=None):
        """a private method to read part of an open file into a pandas.DataFrame.

        Args:
            f (`file`): open file handle
            nrows (`int`): number of rows to read
            names ([`str`]): names to set the columns of the dataframe with
            converters (`dict`): dictionary of lambda functions to convert strings
                to numerical format
            defaults (`dict`): dictionary of default values to assign columns.
                Default is None

        Returns:
            `pandas.DataFrame`: dataframe of control file section info

        """
        seek_point = f.tell()
        line = f.readline()
        raw = line.strip().split()
        if raw[0].lower() == "external":
            filename = raw[1]
            if not os.path.exists(filename):
                raise Exception(
                    "Pst._read_df() error: external file '{0}' not found".format(
                        filename
                    )
                )
            df = pd.read_csv(filename, index_col=False, comment="#")
            df.columns = df.columns.str.lower()
            for name in names:
                if name not in df.columns:
                    raise Exception(
                        "Pst._read_df() error: name"
                        + "'{0}' not in external file '{1}' columns".format(
                            name, filename
                        )
                    )
                if name in converters:
                    df.loc[:, name] = df.loc[:, name].apply(converters[name])
            if defaults is not None:
                for name in names:
                    df.loc[:, name] = df.loc[:, name].fillna(defaults[name])
        else:
            if nrows is None:
                raise Exception(
                    "Pst._read_df() error: non-external sections require nrows"
                )
            f.seek(seek_point)
            df = pd.read_csv(
                f,
                header=None,
                names=names,
                nrows=nrows,
                delim_whitespace=True,
                converters=converters,
                index_col=False,
                comment="#",
            )

            # in case there was some extra junk at the end of the lines
            if df.shape[1] > len(names):
                df = df.iloc[:, len(names)]
                df.columns = names

            if defaults is not None:
                for name in names:
                    df.loc[:, name] = df.loc[:, name].fillna(defaults[name])

            elif np.any(pd.isnull(df).values.flatten()):
                raise Exception("NANs found")
            f.seek(seek_point)
            extras = []
            for _ in range(nrows):
                line = f.readline()
                extra = np.NaN
                if "#" in line:
                    raw = line.strip().split("#")
                    extra = " # ".join(raw[1:])
                extras.append(extra)

            df.loc[:, "extra"] = extras

        return df

    def _read_line_comments(self, f, forgive):
        """private method to read comment lines from a control file"""
        comments = []
        while True:
            org_line = f.readline()
            line = org_line.lower().strip()

            self.lcount += 1
            if org_line == "":
                if forgive:
                    return None, comments
                else:
                    raise Exception("unexpected EOF")
            if line.startswith("++") and line.split("++")[1].strip()[0] != "#":
                self._parse_pestpp_line(line)
            elif "++" in line:
                comments.append(line.strip())
            elif line.startswith("#"):
                comments.append(line.strip())
            else:
                break
        return org_line.strip(), comments

    def _read_section_comments(self, f, forgive):
        """private method to read comments from a section of the control file"""
        lines = []
        section_comments = []
        while True:
            line, comments = self._read_line_comments(f, forgive)
            section_comments.extend(comments)
            if line is None or line.startswith("*"):
                break
            if len(line.strip()) == 0:
                continue

            lines.append(line)
        return line, lines, section_comments

    @staticmethod
    def _parse_external_line(line, pst_path="."):
        """private method to parse a file for external file info"""
        raw = line.strip().split()
        existing_path, filename = Pst._parse_path_agnostic(raw[0])
        if pst_path is not None:
            if pst_path != ".":
                filename = os.path.join(pst_path, filename)
        else:
            filename = os.path.join(existing_path, filename)

        raw = line.lower().strip().split()
        options = {}
        if len(raw) > 1:
            if len(raw) % 2 == 0:
                s = "wrong number of entries on 'external' line:'{0}\n".format(line)
                s += "Should include 'filename', then pairs of key-value options"
                raise Exception(s)
            options = {k.lower(): v.lower() for k, v in zip(raw[1:-1], raw[2:])}
        return filename, options

    @staticmethod
    def _parse_path_agnostic(filename):
        """private method to parse a file path for any os sep"""
        filename = filename.replace("\\", os.sep).replace("/", os.sep)
        return os.path.split(filename)

    @staticmethod
    def _cast_df_from_lines(
        section, lines, fieldnames, converters, defaults, alias_map={}, pst_path="."
    ):
        """private method to cast a pandas dataframe from raw control file lines"""
        # raw = lines[0].strip().split()
        # if raw[0].lower() == "external":
        if section.lower().strip().split()[-1] == "external":
            dfs = []
            for line in lines:
                filename, options = Pst._parse_external_line(line, pst_path)
                if not os.path.exists(filename):
                    raise Exception(
                        "Pst._cast_df_from_lines() error: external file '{0}' not found".format(
                            filename
                        )
                    )
                sep = options.get("sep", ",")

                missing_vals = options.get("missing_values", None)
                if sep.lower() == "w":
                    df = pd.read_csv(
                        filename, delim_whitespace=True, na_values=missing_vals
                    )
                else:
                    df = pd.read_csv(filename, sep=sep, na_values=missing_vals)
                df.columns = df.columns.str.lower()
                for easy, hard in alias_map.items():
                    if easy in df.columns and hard in df.columns:
                        raise Exception(
                            "fieldname '{0}' and its alias '{1}' both in '{2}'".format(
                                hard, easy, filename
                            )
                        )
                    if easy in df.columns:
                        df.loc[:, hard] = df.pop(easy)
                dfs.append(df)

            df = pd.concat(dfs, axis=0, ignore_index=True)

        else:
            extra = []
            raw = []

            for iline, line in enumerate(lines):
                line = line.lower()
                if "#" in line:
                    er = line.strip().split("#")
                    extra.append("#".join(er[1:]))
                    r = er[0].split()
                else:
                    r = line.strip().split()
                    extra.append(np.NaN)

                raw.append(r[: len(defaults)])

            found_fieldnames = fieldnames[: len(raw[0])]
            df = pd.DataFrame(raw, columns=found_fieldnames)

            df.loc[:, "extra"] = extra

        for col in fieldnames:
            if col not in df.columns:
                df.loc[:, col] = np.NaN
            if col in defaults:
                df.loc[:, col] = df.loc[:, col].fillna(defaults[col])
            if col in converters:

                df.loc[:, col] = df.loc[:, col].apply(converters[col])

        return df

    def _cast_prior_df_from_lines(self, section, lines, pst_path="."):
        """private method to cast prior information lines to a dataframe"""
        if pst_path == ".":
            pst_path = ""
        if section.strip().split()[-1].lower() == "external":
            dfs = []
            for line in lines:
                filename, options = Pst._parse_external_line(line, pst_path)
                if not os.path.exists(filename):
                    raise Exception(
                        "Pst._cast_prior_df_from_lines() error: external file '{0}' not found".format(
                            filename
                        )
                    )
                sep = options.get("sep", ",")

                missing_vals = options.get("missing_values", None)
                if sep.lower() == "w":
                    df = pd.read_csv(
                        filename, delim_whitespace=True, na_values=missing_vals
                    )
                else:
                    df = pd.read_csv(filename, sep=sep, na_values=missing_vals)
                df.columns = df.columns.str.lower()

                for field in pst_utils.pst_config["prior_fieldnames"]:
                    if field not in df.columns:
                        raise Exception(
                            "Pst._cast_prior_df_from_lines() error: external file"
                            + "'{0}' missing required field '{1}'".format(
                                filename, field
                            )
                        )
                dfs.append(df)
            df = pd.concat(dfs, axis=0, ignore_index=True)
            self.prior_information = df
            self.prior_information.index = self.prior_information.pilbl

        else:
            pilbl, obgnme, weight, equation = [], [], [], []
            extra = []
            for line in lines:
                if "#" in line:
                    er = line.split("#")
                    raw = er[0].split()
                    extra.append("#".join(er[1:]))
                else:
                    extra.append(np.NaN)
                    raw = line.split()
                pilbl.append(raw[0].lower())
                obgnme.append(raw[-1].lower())
                weight.append(float(raw[-2]))
                eq = " ".join(raw[1:-2])
                equation.append(eq)

            self.prior_information = pd.DataFrame(
                {
                    "pilbl": pilbl,
                    "equation": equation,
                    "weight": weight,
                    "obgnme": obgnme,
                }
            )
            self.prior_information.index = self.prior_information.pilbl
            self.prior_information.loc[:, "extra"] = extra

    def _load_version2(self, filename):
        """private method to load a version 2 control file"""
        self.lcount = 0
        self.comments = {}
        self.prior_information = self.null_prior
        assert os.path.exists(filename), "couldn't find control file {0}".format(
            filename
        )
        f = open(filename, "r")
        try:
            pst_path, _ = Pst._parse_path_agnostic(filename)
            last_section = ""
            req_sections = {
                "* parameter data",
                "* observation data",
                "* model command line",
                "* control data",
            }
            sections_found = set()
            while True:

                next_section, section_lines, comments = self._read_section_comments(f, True)

                if "* control data" in last_section.lower():
                    iskeyword = False
                    if "keyword" in last_section.lower():
                        iskeyword = True
                    self.pestpp_options = self.control_data.parse_values_from_lines(
                        section_lines, iskeyword=iskeyword
                    )
                    if len(self.pestpp_options) > 0:
                        ppo = self.pestpp_options
                        svd_opts = ["svdmode", "eigthresh", "maxsing", "eigwrite"]
                        for svd_opt in svd_opts:
                            if svd_opt in ppo:
                                self.svd_data.__setattr__(svd_opt, ppo.pop(svd_opt))
                        for reg_opt in self.reg_data.should_write:
                            if reg_opt in ppo:
                                self.reg_data.__setattr__(reg_opt, ppo.pop(reg_opt))

                elif "* singular value decomposition" in last_section.lower():
                    self.svd_data.parse_values_from_lines(section_lines)

                elif "* observation groups" in last_section.lower():
                    pass

                elif "* parameter groups" in last_section.lower():
                    self.parameter_groups = self._cast_df_from_lines(
                        last_section,
                        section_lines,
                        self.pargp_fieldnames,
                        self.pargp_converters,
                        self.pargp_defaults,
                        pst_path=pst_path,
                    )
                    self.parameter_groups.index = self.parameter_groups.pargpnme

                elif "* parameter data" in last_section.lower():
                    # check for tied pars
                    ntied = 0
                    if "external" not in last_section.lower():
                        for line in section_lines:
                            if "tied" in line.lower():
                                ntied += 1
                    if ntied > 0:
                        slines = section_lines[:-ntied]
                    else:
                        slines = section_lines
                    self.parameter_data = self._cast_df_from_lines(
                        last_section,
                        slines,
                        self.par_fieldnames,
                        self.par_converters,
                        self.par_defaults,
                        self.par_alias_map,
                        pst_path=pst_path,
                    )

                    self.parameter_data.index = self.parameter_data.parnme
                    if ntied > 0:
                        tied_pars, partied = [], []
                        for line in section_lines[-ntied:]:
                            raw = line.strip().split()
                            tied_pars.append(raw[0].strip().lower())
                            partied.append(raw[1].strip().lower())
                        self.parameter_data.loc[:, "partied"] = np.NaN
                        self.parameter_data.loc[tied_pars, "partied"] = partied

                elif "* observation data" in last_section.lower():
                    self.observation_data = self._cast_df_from_lines(
                        last_section,
                        section_lines,
                        self.obs_fieldnames,
                        self.obs_converters,
                        self.obs_defaults,
                        alias_map=self.obs_alias_map,
                        pst_path=pst_path,
                    )
                    self.observation_data.index = self.observation_data.obsnme

                elif "* model command line" in last_section.lower():
                    for line in section_lines:
                        self.model_command.append(line.strip())

                elif "* model input/output" in last_section.lower():
                    if "* control data" not in sections_found:
                        raise Exception(
                            "attempting to read '* model input/output' before reading "
                            + "'* control data' - need NTPLFLE counter for this..."
                        )
                    if (
                        len(section_lines)
                        != self.control_data.ntplfle + self.control_data.ninsfle
                    ):
                        raise Exception(
                            "didnt find the right number of '* model input/output' lines,"
                            + "expecting {0} template files and {1} instruction files".format(
                                self.control_data.ntplfle, self.control_data.ninsfle
                            )
                        )
                    template_files, input_files = [], []
                    for i in range(self.control_data.ntplfle):
                        raw = section_lines[i].strip().split()
                        template_files.append(raw[0])
                        input_files.append(raw[1])
                    self.model_input_data = pd.DataFrame(
                        {"pest_file": template_files, "model_file": input_files},
                        index=template_files,
                    )

                    instruction_files, output_files = [], []
                    for j in range(self.control_data.ninsfle):
                        raw = section_lines[i + j + 1].strip().split()
                        instruction_files.append(raw[0])
                        output_files.append(raw[1])
                    self.model_output_data = pd.DataFrame(
                        {"pest_file": instruction_files, "model_file": output_files},
                        index=instruction_files,
                    )

                elif "* model input" in last_section.lower():
                    if last_section.strip().split()[-1].lower() == "external":
                        self.model_input_data = self._cast_df_from_lines(
                            last_section,
                            section_lines,
                            ["pest_file", "model_file"],
                            [],
                            [],
                            pst_path=pst_path,
                        )
                        # self.template_files.extend(io_df.pest_file.tolist())
                        # self.input_files.extend(io_df.model_file.tolist())

                    else:
                        template_files, input_files = [], []
                        for line in section_lines:
                            raw = line.split()
                            template_files.append(raw[0])
                            input_files.append(raw[1])
                        self.model_input_data = pd.DataFrame(
                            {"pest_file": template_files, "model_file": input_files},
                            index=template_files,
                        )

                elif "* model output" in last_section.lower():
                    if last_section.strip().split()[-1].lower() == "external":
                        self.model_output_data = self._cast_df_from_lines(
                            last_section,
                            section_lines,
                            ["pest_file", "model_file"],
                            [],
                            [],
                            pst_path=pst_path,
                        )
                        # self.instruction_files.extend(io_df.pest_file.tolist())
                        # self.output_files.extend(io_df.model_file.tolist())

                    else:
                        instruction_files, output_files = [], []
                        for iline, line in enumerate(section_lines):
                            raw = line.split()
                            instruction_files.append(raw[0])
                            output_files.append(raw[1])
                        self.model_output_data = pd.DataFrame(
                            {"pest_file": instruction_files, "model_file": output_files},
                            index=instruction_files,
                        )

                elif "* prior information" in last_section.lower():
                    self._cast_prior_df_from_lines(
                        last_section, section_lines, pst_path=pst_path
                    )
                    # self.prior_information = Pst._cast_df_from_lines(last_section,section_lines,self.prior_fieldnames,
                    #                                                 self.prior_format,{},pst_path=pst_path)

                elif (
                    last_section.lower() == "* regularization"
                    or last_section.lower() == "* regularisation"
                ):
                    raw = section_lines[0].strip().split()
                    self.reg_data.phimlim = float(raw[0])
                    self.reg_data.phimaccept = float(raw[1])
                    raw = section_lines[1].strip().split()
                    self.reg_data.wfinit = float(raw[0])

                elif len(last_section) > 0:
                    print(
                        "Pst._load_version2() warning: unrecognized section: ", last_section
                    )
                    self.comments[last_section] = section_lines

                if next_section is None or len(section_lines) == 0:
                    break
                next_section_generic = (
                    next_section.replace("external", "")
                    .replace("keyword", "")
                    .strip()
                    .lower()
                )
                if next_section_generic in sections_found:
                    f.close()
                    raise Exception(
                        "duplicate control file sections for '{0}'".format(
                            next_section_generic
                        )
                    )
                sections_found.add(next_section_generic)

                last_section = next_section
        except Exception as e:
            f.close()
            raise Exception("error reading ctrl file '{0}': {1}".format(filename,str(e)))

        f.close()
        not_found = []
        for section in req_sections:
            if section not in sections_found:
                not_found.append(section)
        if len(not_found) > 0:
            raise Exception(
                "Pst._load_version2() error: the following required sections were "
                + "not found:{0}".format(",".join(not_found))
            )
        if "* model input/output" in sections_found and (
            "* model input" in sections_found or "* model output" in sections_found
        ):
            raise Exception(
                "'* model input/output cant be used with '* model input' or '* model output'"
            )

    def load(self, filename):
        """entry point load the pest control file.

        Args:
            filename (`str`): pst filename

        Note:
            This method is called from the `Pst` construtor unless the `load` arg is `False`.



        """
        if not os.path.exists(filename):
            raise Exception("couldn't find control file {0}".format(filename))
        f = open(filename, "r")

        while True:
            line = f.readline()
            if line == "":
                raise Exception(
                    "Pst.load() error: EOF when trying to find first line - #sad"
                )
            if line.strip().split()[0].lower() == "pcf":
                break
        if not line.startswith("pcf"):
            raise Exception(
                "Pst.load() error: first non-comment line must start with 'pcf', not '{0}'".format(
                    line
                )
            )

        self._load_version2(filename)
        self._try_load_longnames()
        self.try_parse_name_metadata()
        self._reset_file_paths_os()

    def _reset_file_paths_os(self):
        for df in [self.model_output_data,self.model_input_data]:
            for col in ["pest_file","model_file"]:
                df.loc[:,col] = df.loc[:,col].apply(lambda x: os.path.sep.join(x.replace("\\","/").split("/")))

    def _try_load_longnames(self):
        from pathlib import Path
        d = Path(self.filename).parent
        for df, fnme in ((self.parameter_data, "parlongname.map"),
                         (self.observation_data, "obslongname.map")):
            try:
                mapr = pd.read_csv(Path(d, fnme), index_col=0)['longname']
                df['longname'] = df.index.map(mapr.to_dict())
            except Exception:
                pass
        if hasattr(self, "parameter_groups"):
            df, fnme = (self.parameter_groups, "pglongname.map")
            try:
                mapr = pd.read_csv(Path(d, fnme), index_col=0)['longname']
                df['longname'] = df.index.map(mapr.to_dict())
            except Exception:
                pass


    def _parse_pestpp_line(self, line):
        # args = line.replace('++','').strip().split()
        args = line.replace("++", "").strip().split(")")
        args = [a for a in args if a != ""]
        # args = ['++'+arg.strip() for arg in args]
        # self.pestpp_lines.extend(args)
        keys = [arg.split("(")[0] for arg in args]
        values = [arg.split("(")[1].replace(")", "") for arg in args if "(" in arg]
        for _ in range(len(values) - 1, len(keys)):
            values.append("")
        for key, value in zip(keys, values):
            if key in self.pestpp_options:
                print(
                    "Pst.load() warning: duplicate pest++ option found:" + str(key),
                    PyemuWarning,
                )
            self.pestpp_options[key.lower()] = value

    def _update_control_section(self):
        """private method to synchronize the control section counters with the
        various parts of the control file.  This is usually called during the
        Pst.write() method.

        """
        self.control_data.npar = self.npar
        self.control_data.nobs = self.nobs
        self.control_data.npargp = self.parameter_groups.shape[0]
        self.control_data.nobsgp = (
            self.observation_data.obgnme.value_counts().shape[0]
            + self.prior_information.obgnme.value_counts().shape[0]
        )

        self.control_data.nprior = self.prior_information.shape[0]
        # self.control_data.ntplfle = len(self.template_files)
        self.control_data.ntplfle = self.model_input_data.shape[0]
        # self.control_data.ninsfle = len(self.instruction_files)
        self.control_data.ninsfle = self.model_output_data.shape[0]
        self.control_data.numcom = len(self.model_command)

    def rectify_pgroups(self):
        """synchronize parameter groups section with the parameter data section

        Note:
            This method is called during `Pst.write()` to make sure all parameter
            groups named in `* parameter data` are included.  This is so users
            don't have to manually keep this section up.  This method can also be
            called during control file modifications to see what parameter groups
            are present and prepare for modifying the default values in the `* parameter
            group` section

        Example::

            pst = pyemu.Pst("my.pst")
            pst.parameter_data.loc["par1","pargp"] = "new_group"
            pst.rectify_groups()
            pst.parameter_groups.loc["new_group","derinc"] = 1.0


        """
        # add any parameters groups
        pdata_groups = list(self.parameter_data.loc[:, "pargp"].value_counts().keys())
        # print(pdata_groups)
        need_groups = []

        if hasattr(self, "parameter_groups"):
            existing_groups = list(self.parameter_groups.pargpnme)
        else:
            existing_groups = []
            self.parameter_groups = pd.DataFrame(columns=self.pargp_fieldnames)

        for pg in pdata_groups:
            if pg not in existing_groups:
                need_groups.append(pg)
        if len(need_groups) > 0:
            # print(need_groups)
            defaults = copy.copy(pst_utils.pst_config["pargp_defaults"])
            for grp in need_groups:
                defaults["pargpnme"] = grp
                self.parameter_groups = pd.concat(
                    [self.parameter_groups, pd.DataFrame([defaults])],
                    ignore_index=True
                )

        # now drop any left over groups that aren't needed
        for gp in self.parameter_groups.loc[:, "pargpnme"]:
            if gp in pdata_groups and gp not in need_groups:
                need_groups.append(gp)
        self.parameter_groups.index = self.parameter_groups.pargpnme
        self.parameter_groups = self.parameter_groups.loc[need_groups, :]
        idx = self.parameter_groups.index.drop_duplicates()
        if idx.shape[0] != self.parameter_groups.shape[0]:
            warnings.warn("dropping duplicate parameter groups", PyemuWarning)
            self.parameter_groups = self.parameter_groups.loc[
                ~self.parameter_groups.index.duplicated(keep="first"), :
            ]

    def _parse_pi_par_names(self):
        """private method to get the parameter names from prior information
        equations.  Sets a 'names' column in Pst.prior_information that is a list
        of parameter names


        """
        if self.prior_information.shape[0] == 0:
            return
        if "names" in self.prior_information.columns:
            self.prior_information.pop("names")
        if "rhs" in self.prior_information.columns:
            self.prior_information.pop("rhs")

        def parse(eqs):
            raw = eqs.split("=")
            # rhs = float(raw[1])
            raw = [
                i
                for i in re.split(
                    "[###]",
                    raw[0].lower().strip().replace(" + ", "###").replace(" - ", "###"),
                )
                if i != ""
            ]
            # in case of a leading '-' or '+'
            if len(raw[0]) == 0:
                raw = raw[1:]
            # pnames = []
            # for r in raw:
            #     if '*' not in r:
            #         continue
            #     pname =  r.split('*')[1].replace("log(", '').replace(')', '').strip()
            #     pnames.append(pname)
            # return pnames
            return [
                r.split("*")[1].replace("log(", "").replace(")", "").strip()
                for r in raw
                if "*" in r
            ]

        self.prior_information.loc[:, "names"] = self.prior_information.equation.apply(
            lambda x: parse(x)
        )

    def add_pi_equation(
        self,
        par_names,
        pilbl=None,
        rhs=0.0,
        weight=1.0,
        obs_group="pi_obgnme",
        coef_dict={},
    ):
        """a helper to construct a new prior information equation.

        Args:
            par_names ([`str`]): parameter names in the equation
            pilbl (`str`): name to assign the prior information equation.  If None,
                a generic equation name is formed. Default is None
            rhs (`float`): the right-hand side of the pi equation
            weight (`float`): the weight of the equation
            obs_group (`str`): the observation group for the equation. Default is 'pi_obgnme'
            coef_dict (`dict`): a dictionary of parameter name, coefficient pairs to assign
                leading coefficients for one or more parameters in the equation.
                If a parameter is not listed, 1.0 is used for its coefficients.
                Default is {}

        Example::

            pst = pyemu.Pst("pest.pst")
            # add a pi equation for the first adjustable parameter
            pst.add_pi_equation(pst.adj_par_names[0],pilbl="pi1",rhs=1.0)
            # add a pi equation for 1.5 times the 2nd and 3 times the 3rd adj pars to sum together to 2.0
            names = pst.adj_par_names[[1,2]]
            pst.add_pi_equation(names,coef_dict={names[0]:1.5,names[1]:3})


        """
        if pilbl is None:
            pilbl = "pilbl_{0}".format(self.__pi_count)
            self.__pi_count += 1
        missing, fixed = [], []

        for par_name in par_names:
            if par_name not in self.parameter_data.parnme:
                missing.append(par_name)
            elif self.parameter_data.loc[par_name, "partrans"] in ["fixed", "tied"]:
                fixed.append(par_name)
        if len(missing) > 0:
            raise Exception(
                "Pst.add_pi_equation(): the following pars "
                + " were not found: {0}".format(",".join(missing))
            )
        if len(fixed) > 0:
            raise Exception(
                "Pst.add_pi_equation(): the following pars "
                + " were are fixed/tied: {0}".format(",".join(fixed))
            )
        eqs_str = ""
        sign = ""
        for i, par_name in enumerate(par_names):
            coef = coef_dict.get(par_name, 1.0)
            if coef < 0.0:
                sign = "-"
                coef = np.abs(coef)
            elif i > 0:
                sign = "+"
            if self.parameter_data.loc[par_name, "partrans"] == "log":
                par_name = "log({})".format(par_name)
            eqs_str += " {0} {1} * {2} ".format(sign, coef, par_name)
        eqs_str += " = {0}".format(rhs)
        self.prior_information.loc[pilbl, "pilbl"] = pilbl
        self.prior_information.loc[pilbl, "equation"] = eqs_str
        self.prior_information.loc[pilbl, "weight"] = weight
        self.prior_information.loc[pilbl, "obgnme"] = obs_group

    def rectify_pi(self):
        """rectify the prior information equation with the current state of the
        parameter_data dataframe.


        Note:
            Equations that list fixed, tied or missing parameters
            are removed completely even if adjustable parameters are also
            listed in the equation. This method is called during Pst.write()

        """
        if self.prior_information.shape[0] == 0:
            return
        self._parse_pi_par_names()
        adj_names = self.adj_par_names

        def is_good(names):
            for n in names:
                if n not in adj_names:
                    return False
            return True

        keep_idx = self.prior_information.names.apply(lambda x: is_good(x))
        self.prior_information = self.prior_information.loc[keep_idx, :]

    def _write_df(self, name, f, df, formatters, columns):
        """private method to write a dataframe to a control file"""
        if name.startswith("*"):
            f.write(name + "\n")
        if self.with_comments:
            for line in self.comments.get(name, []):
                f.write(line + "\n")
        if df.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            df.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )

        def ext_fmt(x):
            if pd.notnull(x):
                return " # {0}".format(x)
            return ""

        if self.with_comments and "extra" in df.columns:
            df.loc[:, "extra_str"] = df.extra.apply(ext_fmt)
            columns.append("extra_str")
            # formatters["extra"] = lambda x: " # {0}".format(x) if pd.notnull(x) else 'test'
            # formatters["extra"] = lambda x: ext_fmt(x)

        # only write out the dataframe if it contains data - could be empty
        if len(df) > 0:
            f.write(
                df.to_string(
                    col_space=0,
                    formatters=formatters,
                    columns=columns,
                    justify="right",
                    header=False,
                    index=False,
                )
                + "\n"
            )

    def sanity_checks(self, forgive=False):
        """some basic check for strangeness

        Args:
            forgive (`bool`): flag to forgive (warn) for issues.  Default is False


        Note:
            checks for duplicate names, atleast 1 adjustable parameter
            and at least 1 non-zero-weighted observation

            Not nearly as comprehensive as pestchek

        Example::

            pst = pyemu.Pst("pest.pst")
            pst.sanity_checks()


        """

        dups = self.parameter_data.parnme.value_counts()
        dups = dups.loc[dups > 1]
        if dups.shape[0] > 0:
            if forgive:
                warnings.warn(
                    "duplicate parameter names: {0}".format(",".join(list(dups.index))),
                    PyemuWarning,
                )
            else:
                raise Exception(
                    "Pst.sanity_check() error: duplicate parameter names: {0}".format(
                        ",".join(list(dups.index))
                    )
                )

        dups = self.observation_data.obsnme.value_counts()
        dups = dups.loc[dups > 1]
        if dups.shape[0] > 0:
            if forgive:
                warnings.warn(
                    "duplicate observation names: {0}".format(
                        ",".join(list(dups.index))
                    ),
                    PyemuWarning,
                )
            else:
                raise Exception(
                    "Pst.sanity_check() error: duplicate observation names: {0}".format(
                        ",".join(list(dups.index))
                    )
                )

        if self.npar_adj == 0:
            warnings.warn("no adjustable pars", PyemuWarning)

        if self.nnz_obs == 0:
            warnings.warn("no non-zero weight obs", PyemuWarning)

        if self.tied is not None and len(self.tied) > 0:
            sadj = set(self.adj_par_names)
            spar = set(self.par_names)

            tpar_dict = self.parameter_data.partied.to_dict()

            for tpar, ptied in tpar_dict.items():
                if pd.isna(ptied):
                    continue
                if tpar == ptied:
                    if forgive:
                        warnings.warn(
                            "tied parameter '{0}' tied to itself".format(tpar),
                            PyemuWarning,
                        )
                    else:
                        raise Exception(
                            "Pst.sanity_check() error: tied parameter '{0}' tied to itself".format(
                                tpar
                            )
                        )
                elif ptied not in spar:
                    if forgive:
                        warnings.warn(
                            "tied parameter '{0}' tied to unknown parameter '{1}'".format(
                                tpar, ptied
                            ),
                            PyemuWarning,
                        )
                    else:
                        raise Exception(
                            "Pst.sanity_check() error: tied parameter '{0}' tied to unknown parameter '{1}'".format(
                                tpar, ptied
                            )
                        )
                elif ptied not in sadj:
                    if forgive:
                        warnings.warn(
                            "tied parameter '{0}' tied to non-adjustable parameter '{1}'".format(
                                tpar, ptied
                            ),
                            PyemuWarning,
                        )
                    else:
                        raise Exception(
                            "Pst.sanity_check() error: tied parameter '{0}' tied to non-adjustable parameter '{1}'".format(
                                tpar, ptied
                            )
                        )

        # print("noptmax: {0}".format(self.control_data.noptmax))

    def _write_version2(self, new_filename, use_pst_path=True, pst_rel_path="."):
        """private method to write a version 2 control file"""
        pst_path = None
        new_filename = str(new_filename)  # ensure convert to str
        if use_pst_path:
            pst_path, _ = Pst._parse_path_agnostic(new_filename)
        if pst_rel_path == ".":
            pst_rel_path = ""

        self.new_filename = new_filename
        self.rectify_pgroups()
        self.rectify_pi()
        self._rectify_parchglim()
        self._update_control_section()
        self.sanity_checks()

        f_out = open(new_filename, "w")
        if self.with_comments:
            for line in self.comments.get("initial", []):
                f_out.write(line + "\n")
        f_out.write("pcf version=2\n")
        self.control_data.write_keyword(f_out)

        if self.with_comments:
            for line in self.comments.get("* singular value decomposition", []):
                f_out.write(line)
        self.svd_data.write_keyword(f_out)

        if self.control_data.pestmode.lower().startswith("r"):
            self.reg_data.write_keyword(f_out)

        for k, v in self.pestpp_options.items():
            if isinstance(v, list) or isinstance(v, tuple):
                v = ",".join([str(vv) for vv in list(v)])
            f_out.write("{0:30} {1}\n".format(k, v))

        # parameter groups
        name = "pargp_data"
        columns = self.pargp_fieldnames
        if self.parameter_groups.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.parameter_groups.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        f_out.write("* parameter groups external\n")
        pargp_filename = new_filename.lower().replace(".pst", ".{0}.csv".format(name))
        if pst_path is not None:
            pargp_filename = os.path.join(pst_path, os.path.split(pargp_filename)[-1])
        self.parameter_groups.to_csv(pargp_filename, index=False)
        pargp_filename = os.path.join(pst_rel_path, os.path.split(pargp_filename)[-1])
        f_out.write("{0}\n".format(pargp_filename))

        # parameter data
        name = "par_data"
        columns = self.par_fieldnames
        if self.parameter_data.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.parameter_data.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        f_out.write("* parameter data external\n")
        par_filename = new_filename.lower().replace(".pst", ".{0}.csv".format(name))
        if pst_path is not None:
            par_filename = os.path.join(pst_path, os.path.split(par_filename)[-1])
        self.parameter_data.to_csv(par_filename, index=False)
        par_filename = os.path.join(pst_rel_path, os.path.split(par_filename)[-1])
        f_out.write("{0}\n".format(par_filename))

        # observation data
        name = "obs_data"
        columns = self.obs_fieldnames
        if self.observation_data.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.observation_data.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        f_out.write("* observation data external\n")
        obs_filename = new_filename.lower().replace(".pst", ".{0}.csv".format(name))
        if pst_path is not None:
            obs_filename = os.path.join(pst_path, os.path.split(obs_filename)[-1])
        self.observation_data.to_csv(obs_filename, index=False)
        obs_filename = os.path.join(pst_rel_path, os.path.split(obs_filename)[-1])
        f_out.write("{0}\n".format(obs_filename))

        f_out.write("* model command line\n")
        for mc in self.model_command:
            f_out.write("{0}\n".format(mc))

        # model input
        name = "tplfile_data"
        columns = self.model_io_fieldnames
        if self.model_input_data.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.model_input_data.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        f_out.write("* model input external\n")
        io_filename = new_filename.lower().replace(".pst", ".{0}.csv".format(name))
        if pst_path is not None:
            io_filename = os.path.join(pst_path, os.path.split(io_filename)[-1])
        self.model_input_data.to_csv(io_filename, index=False)
        io_filename = os.path.join(pst_rel_path, os.path.split(io_filename)[-1])
        f_out.write("{0}\n".format(io_filename))

        # model output
        name = "insfile_data"
        columns = self.model_io_fieldnames
        if self.model_output_data.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.model_output_data.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        f_out.write("* model output external\n")
        io_filename = new_filename.lower().replace(".pst", ".{0}.csv".format(name))
        if pst_path is not None:
            io_filename = os.path.join(pst_path, os.path.split(io_filename)[-1])
        self.model_output_data.to_csv(io_filename, index=False)
        io_filename = os.path.join(pst_rel_path, os.path.split(io_filename)[-1])
        f_out.write("{0}\n".format(io_filename))

        # prior info
        if self.prior_information.shape[0] > 0:
            name = "pi_data"
            columns = self.prior_fieldnames
            if self.prior_information.loc[:, columns].isnull().values.any():
                # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
                csv_name = "pst.{0}.nans.csv".format(
                    name.replace(" ", "_").replace("*", "")
                )
                self.prior_information.to_csv(csv_name)
                raise Exception(
                    "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
                )
            f_out.write("* prior information external\n")
            pi_filename = new_filename.lower().replace(".pst", ".{0}.csv".format(name))
            if pst_path is not None:
                pi_filename = os.path.join(pst_path, os.path.split(pi_filename)[-1])
            self.prior_information.to_csv(pi_filename, index=False)
            pi_filename = os.path.join(pst_rel_path, os.path.split(pi_filename)[-1])
            f_out.write("{0}\n".format(pi_filename))

        f_out.close()

    def write(self, new_filename, version=None):
        """main entry point to write a pest control file.

        Args:
            new_filename (`str`): name of the new pest control file

            version (`int`): flag for which version of control file to write (must be 1 or 2).
                if None, uses the number of pars to decide: if number of pars iis greater than 10,000,
                version 2 is used.

        Example::

            pst = pyemu.Pst("my.pst")
            pst.parrep("my.par")
            pst.write(my_new.pst")
            #write a version 2 control file
            pst.write("my_new_v2.pst",version=2)

        """

        vstring = "noptmax:{0}, npar_adj:{1}, nnz_obs:{2}".format(
            self.control_data.noptmax, self.npar_adj, self.nnz_obs
        )
        print(vstring)

        if version is None:
            if self.npar > 10000:
                version = 2
            else:
                version = 1

        if version == 1:
            return self._write_version1(new_filename=new_filename)
        elif version == 2:
            return self._write_version2(new_filename=new_filename)
        else:
            raise Exception(
                "Pst.write() error: version must be 1 or 2, not '{0}'".format(version)
            )

    def _rectify_parchglim(self):
        """private method to just fix the parchglim vs cross zero issue"""
        par = self.parameter_data
        need_fixing = par.loc[par.parubnd > 0, :].copy()
        need_fixing = need_fixing.loc[par.parlbnd <= 0, "parnme"]

        self.parameter_data.loc[need_fixing, "parchglim"] = "relative"

    def _write_version1(self, new_filename):
        """private method to write a version 1 pest control file"""
        self.new_filename = new_filename
        self.rectify_pgroups()
        self.rectify_pi()
        self._update_control_section()
        self._rectify_parchglim()
        self.sanity_checks()

        f_out = open(new_filename, "w")
        if self.with_comments:
            for line in self.comments.get("initial", []):
                f_out.write(line + "\n")
        f_out.write("pcf\n* control data\n")
        self.control_data.write(f_out)

        # for line in self.other_lines:
        #     f_out.write(line)
        if self.with_comments:
            for line in self.comments.get("* singular value decompisition", []):
                f_out.write(line)
        self.svd_data.write(f_out)

        # f_out.write("* parameter groups\n")

        # to catch the byte code ugliness in python 3
        pargpnme = self.parameter_groups.loc[:, "pargpnme"].copy()
        self.parameter_groups.loc[:, "pargpnme"] = self.parameter_groups.pargpnme.apply(
            self.pargp_format["pargpnme"]
        )

        self._write_df(
            "* parameter groups",
            f_out,
            self.parameter_groups,
            self.pargp_format,
            self.pargp_fieldnames,
        )
        self.parameter_groups.loc[:, "pargpnme"] = pargpnme

        self._write_df(
            "* parameter data",
            f_out,
            self.parameter_data,
            self.par_format,
            self.par_fieldnames,
        )

        if self.tied is not None:
            self._write_df(
                "tied parameter data",
                f_out,
                self.tied,
                self.tied_format,
                self.tied_fieldnames,
            )

        f_out.write("* observation groups\n")
        for group in self.obs_groups:
            try:
                group = group.decode()
            except Exception as e:
                pass
            f_out.write(pst_utils.SFMT(str(group)) + "\n")
        for group in self.prior_groups:
            try:
                group = group.decode()
            except Exception as e:
                pass
            f_out.write(pst_utils.SFMT(str(group)) + "\n")
        self._write_df(
            "* observation data",
            f_out,
            self.observation_data,
            self.obs_format,
            self.obs_fieldnames,
        )

        f_out.write("* model command line\n")
        for cline in self.model_command:
            f_out.write(cline + "\n")

        name = "tplfile_data"
        columns = self.model_io_fieldnames
        if self.model_input_data.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.model_input_data.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        name = "insfile_data"
        columns = self.model_io_fieldnames
        if self.model_output_data.loc[:, columns].isnull().values.any():
            # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
            csv_name = "pst.{0}.nans.csv".format(
                name.replace(" ", "_").replace("*", "")
            )
            self.model_output_data.to_csv(csv_name)
            raise Exception(
                "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
            )
        f_out.write("* model input/output\n")
        for tplfle, infle in zip(
            self.model_input_data.pest_file, self.model_input_data.model_file
        ):
            f_out.write("{0} {1}\n".format(tplfle, infle))
        for insfle, outfle in zip(
            self.model_output_data.pest_file, self.model_output_data.model_file
        ):
            f_out.write("{0} {1}\n".format(insfle, outfle))

        if self.nprior > 0:
            name = "pi_data"
            columns = self.prior_fieldnames
            if self.prior_information.loc[:, columns].isnull().values.any():
                # warnings.warn("WARNING: NaNs in {0} dataframe".format(name))
                csv_name = "pst.{0}.nans.csv".format(
                    name.replace(" ", "_").replace("*", "")
                )
                self.prior_information.to_csv(csv_name)
                raise Exception(
                    "NaNs in {0} dataframe, csv written to {1}".format(name, csv_name)
                )
            f_out.write("* prior information\n")
            # self.prior_information.index = self.prior_information.pop("pilbl")
            max_eq_len = self.prior_information.equation.apply(lambda x: len(x)).max()
            eq_fmt_str = " {0:<" + str(max_eq_len) + "s} "
            eq_fmt_func = lambda x: eq_fmt_str.format(x)
            #  17/9/2016 - had to go with a custom writer loop b/c pandas doesn't want to
            # output strings longer than 100, even with display.max_colwidth
            # f_out.write(self.prior_information.to_string(col_space=0,
            #                                  columns=self.prior_fieldnames,
            #                                  formatters=pi_formatters,
            #                                  justify="right",
            #                                  header=False,
            #                                 index=False) + '\n')
            # self.prior_information["pilbl"] = self.prior_information.index
            # for idx,row in self.prior_information.iterrows():
            #     f_out.write(pst_utils.SFMT(row["pilbl"]))
            #     f_out.write(eq_fmt_func(row["equation"]))
            #     f_out.write(pst_utils.FFMT(row["weight"]))
            #     f_out.write(pst_utils.SFMT(row["obgnme"]) + '\n')
            for _, row in self.prior_information.iterrows():
                f_out.write(pst_utils.SFMT(row["pilbl"]))
                f_out.write(eq_fmt_func(row["equation"]))
                f_out.write(pst_utils.FFMT(row["weight"]))
                f_out.write(pst_utils.SFMT(row["obgnme"]))
                if self.with_comments and "extra" in row:
                    f_out.write(" # {0}".format(row["extra"]))
                f_out.write("\n")

        if self.control_data.pestmode.startswith("regul"):
            # f_out.write("* regularisation\n")
            # if update_regul or len(self.regul_lines) == 0:
            #    f_out.write(self.regul_section)
            # else:
            #    [f_out.write(line) for line in self.regul_lines]
            self.reg_data.write(f_out)

        for line in self.other_lines:
            f_out.write(line + "\n")

        for key, value in self.pestpp_options.items():
            if isinstance(value, list) or isinstance(value, tuple):
                value = ",".join([str(v) for v in list(value)])
            f_out.write("++{0}({1})\n".format(str(key), str(value)))

        if self.with_comments:
            for line in self.comments.get("final", []):
                f_out.write(line + "\n")

        f_out.close()

    def bounds_report(self, iterations=None):
        """report how many parameters are at bounds. If ensemble, the base enbsemble member is evaluated

        Args:
            iterations ([`int`]): a list of iterations for which a bounds report is requested
                If None, all iterations for which `par` files are located are reported. Default
                is None

        Returns:
            `df`: a pandas DataFrame object with rows being parameter groups and columns
                <iter>_num_at_ub, <iter>_num_at_lb, and <iter>_total_at_bounds
                row 0 is total at bounds, subsequent rows correspond with groups

        Example:
            pst = pyemu.Pst("my.pst")
            df = pst.bound_report(iterations=[0,2,3])

        """
        # sort out which files are parameter files and parse pstroot from pst directory
        pstroot = self.filename
        if pstroot.lower().endswith(".pst"):
            pstroot = pstroot[:-4]
        pstdir = os.path.dirname(pstroot)
        if len(pstdir) == 0:
            pstdir = "."
        pstroot = os.path.basename(pstroot)

        # find all the par files
        parfiles = glob.glob(os.path.join(pstdir, "{}*.par".format(pstroot)))

        # exception if no par files found
        if len(parfiles) == 0:
            raise Exception(
                "no par files with root {} in directory {}".format(pstdir, pstroot)
            )

        is_ies = any(["base" in i.lower() for i in parfiles])
        # decide which iterations we care about
        if is_ies:
            iters = [
                os.path.basename(cf).replace(pstroot, "").split(".")[1]
                for cf in parfiles
                if "base" in cf.lower()
            ]
            iters = [int(i) for i in iters if i != "base"]
            parfiles = [i for i in parfiles if "base" in i]
        else:
            iters = [
                os.path.basename(cf).replace(pstroot, "").split(".")[1]
                for cf in parfiles
                if "base" not in cf.lower()
            ]
            iters = [int(i) for i in iters if i != "par"]
            parfiles = [i for i in parfiles if "base" not in i]

        if iterations is None:
            iterations = iters

        if isinstance(iterations, tuple):
            iterations = list(iterations)

        if not isinstance(iterations, list):
            iterations = [iterations]

        # sort the iterations to go through them in order
        iterations.sort()

        # set up a DataFrame with bounds and into which to put the par values
        allpars = self.parameter_data[["parlbnd", "parubnd", "pargp"]].copy()

        # loop over iterations and calculate which are at upper and lower bounds
        for citer in iterations:
            try:
                tmp = pd.read_csv(
                    os.path.join(pstdir, "{}.{}.base.par".format(pstroot, citer)),
                    skiprows=1,
                    index_col=0,
                    usecols=[0, 1],
                    delim_whitespace=True,
                    header=None,
                )
            except FileNotFoundError:
                raise Exception(
                    "iteration {} does not have a paramter file associated with it in {}".format(
                        citer, pstdir
                    )
                )
            tmp.columns = ["pars_iter_{}".format(citer)]
            allpars = allpars.merge(tmp, left_index=True, right_index=True)
            allpars["at_upper_bound_{}".format(citer)] = (
                allpars["pars_iter_{}".format(citer)] >= allpars["parubnd"]
            )
            allpars["at_lower_bound_{}".format(citer)] = (
                allpars["pars_iter_{}".format(citer)] <= allpars["parlbnd"]
            )

        # sum up by groups
        df = (
            allpars.groupby("pargp")
            .sum()[[i for i in allpars.columns if i.startswith("at_")]]
            .astype(int)
        )

        # add the total
        df.loc["total"] = df.sum()

        # sum up upper and lower bounds
        cols = []
        for citer in iterations:
            df["at_either_bound_{}".format(citer)] = (
                df["at_upper_bound_{}".format(citer)]
                + df["at_lower_bound_{}".format(citer)]
            )
            cols.extend(
                [
                    "at_either_bound_{}".format(citer),
                    "at_lower_bound_{}".format(citer),
                    "at_upper_bound_{}".format(citer),
                ]
            )

        # reorder by iterations and return
        return df[cols]

        # loop over the iterations and count the pars at bounds

    def get(self, par_names=None, obs_names=None):
        """get a new pst object with subset of parameters and/or observations

        Args:
            par_names ([`str`]): a list of parameter names to have in the new Pst instance.
                If None, all parameters are in the new Pst instance. Default
                is None
            obs_names ([`str`]): a list of observation names to have in the new Pst instance.
                If None, all observations are in teh new Pst instance. Default
                is None

        Returns:
            `Pst`: a new Pst instance

        Note:
            passing `par_names` as `None` and `obs_names` as `None` effectively
            generates a copy of the current `Pst`

            Does not modify model i/o files - this is just a method for performing pyemu operations

        Example::

            pst = pyemu.Pst("pest.pst")
            new_pst = pst.get(pst.adj_par_names[0],pst.obs_names[:10])

        """

        # if par_names is None and obs_names is None:
        #    return copy.deepcopy(self)
        if par_names is None:
            par_names = self.parameter_data.parnme
        if obs_names is None:
            obs_names = self.observation_data.obsnme

        new_par = self.parameter_data.copy()
        if par_names is not None:
            new_par.index = new_par.parnme
            new_par = new_par.loc[par_names, :]
        new_obs = self.observation_data.copy()
        new_res = None

        if obs_names is not None:
            new_obs.index = new_obs.obsnme
            new_obs = new_obs.loc[obs_names]
            if self.__res is not None:
                new_res = copy.deepcopy(self.res)
                new_res.index = new_res.name
                new_res = new_res.loc[obs_names, :]

        self.rectify_pgroups()
        new_pargp = self.parameter_groups.copy()
        new_pargp.index = new_pargp.pargpnme.apply(str.strip)
        new_pargp_names = new_par.pargp.value_counts().index
        new_pargp.reindex(new_pargp_names)

        new_pst = Pst(self.filename, resfile=self.resfile, load=False)
        new_pst.parameter_data = new_par
        new_pst.observation_data = new_obs
        new_pst.parameter_groups = new_pargp
        new_pst.__res = new_res
        new_pst.prior_information = self.prior_information
        new_pst.rectify_pi()
        new_pst.control_data = self.control_data.copy()

        new_pst.model_command = self.model_command
        new_pst.model_input_data = self.model_input_data.copy()
        new_pst.model_output_data = self.model_output_data.copy()
        if self.tied is not None:
            warnings.warn(
                "Pst.get() not checking for tied parameter "
                + "compatibility in new Pst instance",
                PyemuWarning,
            )
            # new_pst.tied = self.tied.copy()
        new_pst.other_lines = self.other_lines
        new_pst.pestpp_options = self.pestpp_options
        new_pst.regul_lines = self.regul_lines

        return new_pst

    def parrep(
        self,
        parfile=None,
        enforce_bounds=True,
        real_name=None,
        noptmax=0,
        binary_ens_file=False,
    ):
        """replicates the pest parrep util. replaces the parval1 field in the
            parameter data section dataframe with values in a PEST parameter file
            or a single realization from an ensemble parameter csv file

        Args:
            parfile (`str`, optional): parameter file to use.  If None, try to find and use
                a parameter file that corresponds to the case name.
                If parfile has extension '.par' a single realization parameter file is used
                If parfile has extention '.csv' an ensemble parameter file is used which invokes real_name
                If parfile has extention '.jcb' a binary ensemble parameter file is used which invokes real_name
                Default is None
            enforce_bounds (`bool`, optional): flag to enforce parameter bounds after parameter values are updated.
                This is useful because PEST and PEST++ round the parameter values in the
                par file, which may cause slight bound violations.  Default is `True`
            real_name (`str` or `int`, optional): name of the ensemble realization to use for updating the
                parval1 value in the parameter data section dataframe. If None, try using "base". If "base"
                not present, use the real_name with smallest index number.
                Ignored if parfile is of the PEST parameter file format (e.g. not en ensemble)
            noptmax (`int`, optional): Value with which to update the pst.control_data.noptmax value
                Default is 0.
            binary_ens_file (`bool`): If True, use binary format to load ensemble file, else assume it's a CSV file
        Example::

            pst = pyemu.Pst("pest.pst")
            pst.parrep("pest.1.base.par")
            pst.control_data.noptmax = 0
            pst.write("pest_1.pst")

        """

        if parfile is None:
            parfile = self.filename.replace(".pst", ".par")
        # first handle the case of a single parameter realization in a PAR file
        if parfile.lower().endswith(".par"):
            print("Updating parameter values from {0}".format(parfile))
            par_df = pst_utils.read_parfile(parfile)
            self.parameter_data.index = self.parameter_data.parnme
            par_df.index = par_df.parnme
            self.parameter_data.parval1 = par_df.parval1
            self.parameter_data.scale = par_df.scale
            self.parameter_data.offset = par_df.offset

        # next handle ensemble case
        if parfile.lower()[-4:] in [".jcb", ".bin"]:
            binary_ens_file = True
        if parfile.lower()[-4:] in [".jcb", ".bin", ".csv"]:
            if parfile.lower().endswith(".csv"):
                parens = pd.read_csv(parfile, index_col=0)
            if binary_ens_file == True:
                parens = pyemu.ParameterEnsemble.from_binary(
                    pst=self, filename=parfile
                )._df
            # cast the parens.index to string to be sure indexing is cool
            parens.index = [str(i).lower() for i in parens.index]
            # handle None case (potentially) for real_name
            if real_name is None:
                if "base" in parens.index:
                    real_name = "base"
                else:
                    real_name = str(min([int(i) for i in parens.index]))
            # cast the real_name to string to be sure indexing is cool
            real_name = str(real_name)

            # now update with a little pandas trickery
            print(
                "updating parval1 using realization:'{}' from ensemble file {}".format(
                    real_name, parfile
                )
            )
            self.parameter_data.parval1 = parens.loc[real_name].T.loc[
                self.parameter_data.parnme
            ]

        if enforce_bounds:
            par = self.parameter_data
            idx = par.loc[par.parval1 > par.parubnd, "parnme"]
            par.loc[idx, "parval1"] = par.loc[idx, "parubnd"]
            idx = par.loc[par.parval1 < par.parlbnd, "parnme"]
            par.loc[idx, "parval1"] = par.loc[idx, "parlbnd"]
        print("parrep: updating noptmax to {}".format(int(noptmax)))
        self.control_data.noptmax = int(noptmax)

    def adjust_weights_discrepancy(
        self, resfile=None, original_ceiling=True, bygroups=False
    ):
        """adjusts the weights of each non-zero weight observation based
        on the residual in the pest residual file so each observations contribution
        to phi is 1.0 (e.g. Mozorov's discrepancy principal)

        Args:
            resfile (`str`): residual file name.  If None, try to use a residual file
                with the Pst case name.  Default is None
            original_ceiling (`bool`): flag to keep weights from increasing - this is
                generally a good idea. Default is True
            bygroups (`bool`): flag to adjust weights by groups. If False, the weight
                of each non-zero weighted observation is adjusted individually. If True,
                intergroup weighting is preserved (the contribution to each group is used)
                but this may result in some strangeness if some observations in a group have
                a really low phi already.

        Example::

            pst = pyemu.Pst("my.pst")
            print(pst.phi) #assumes "my.res" is colocated with "my.pst"
            pst.adjust_weights_discrepancy()
            print(pst.phi) # phi should equal number of non-zero observations

        """
        if resfile is not None:
            self.resfile = resfile
            self.__res = None
        if bygroups:
            phi_comps = self.phi_components
            self._adjust_weights_by_phi_components(phi_comps, original_ceiling)
        else:
            names = self.nnz_obs_names
            obs = self.observation_data.loc[names, :]
            # "Phi should equal nnz - nnzobs that satisfy inequ"
            res = self.res.loc[names, :].residual
            og = obs.obgnme
            res.loc[
                (og.str.startswith(self.get_constraint_tags('gt'))) &
                (res <= 0)] = 0
            res.loc[
                (og.str.startswith(self.get_constraint_tags('lt'))) &
                (res >= 0)] = 0
            swr = (res * obs.weight) ** 2
            factors = (1.0 / swr).apply(np.sqrt)
            if original_ceiling:
                factors = factors.apply(lambda x: 1.0 if x > 1.0 else x)

            w = self.observation_data.weight
            w.loc[names] *= factors.values

    def _adjust_weights_by_phi_components(self, components, original_ceiling):
        """private method that resets the weights of observations by group to account for
        residual phi components.

        Args:
            components (`dict`): a dictionary of obs group:phi contribution pairs
            original_ceiling (`bool`): flag to keep weights from increasing.

        """
        obs = self.observation_data
        nz_groups = obs.groupby(obs["weight"].map(lambda x: x == 0)).groups
        ogroups = obs.groupby("obgnme").groups
        for ogroup, idxs in ogroups.items():
            if (
                self.control_data.pestmode.startswith("regul")
                and "regul" in ogroup.lower()
            ):
                continue
            og_phi = components[ogroup]
            nz_groups = (
                obs.loc[idxs, :]
                .groupby(obs.loc[idxs, "weight"].map(lambda x: x == 0))
                .groups
            )
            og_nzobs = 0
            if False in nz_groups.keys():
                og_nzobs = len(nz_groups[False])
            if og_nzobs == 0 and og_phi > 0:
                raise Exception(
                    "Pst.adjust_weights_by_phi_components():"
                    " no obs with nonzero weight,"
                    + " but phi > 0 for group:"
                    + str(ogroup)
                )
            if og_phi > 0:
                factor = np.sqrt(float(og_nzobs) / float(og_phi))
                if original_ceiling:
                    factor = min(factor, 1.0)
                obs.loc[idxs, "weight"] = obs.weight[idxs] * factor
        self.observation_data = obs

    def __reset_weights(self, target_phis, res_idxs, obs_idxs):
        """private method to reset weights based on target phi values
        for each group.  This method should not be called directly

        Args:
            target_phis (`dict`): target phi contribution for groups to reweight
            res_idxs (`dict`): the index positions of each group of interest
                in the res dataframe
            obs_idxs (`dict`): the index positions of each group of interest
                in the observation data dataframe

        """

        obs = self.observation_data
        res = self.res
        for item in target_phis.keys():
            if item not in res_idxs.keys():
                raise Exception(
                    "Pst.__reset_weights(): "
                    + str(item)
                    + " not in residual group indices"
                )
            if item not in obs_idxs.keys():
                raise Exception(
                    "Pst.__reset_weights(): "
                    + str(item)
                    + " not in observation group indices"
                )
            # actual_phi = ((self.res.loc[res_idxs[item], "residual"] *
            #               self.observation_data.loc
            #               [obs_idxs[item], "weight"])**2).sum()
            tmpobs = obs.loc[obs_idxs[item]]
            resid = (
                    tmpobs.obsval
                    - res.loc[res_idxs[item], "modelled"]
            ).loc[tmpobs.index]
            og = tmpobs.obgnme
            resid.loc[
                (og.str.startswith(self.get_constraint_tags('gt'))) &
                (resid <= 0)] = 0
            resid.loc[
                (og.str.startswith(self.get_constraint_tags('lt'))) &
                (resid >= 0)] = 0

            actual_phi = np.sum(
                (
                    resid
                    * obs.loc[obs_idxs[item], "weight"]
                )
                ** 2
            )
            if actual_phi > 0.0:
                weight_mult = np.sqrt(target_phis[item] / actual_phi)
                obs.loc[obs_idxs[item], "weight"] *= weight_mult
            else:
                (
                    "Pst.__reset_weights() warning: phi group {0} has zero phi, skipping...".format(
                        item
                    )
                )

    def _adjust_weights_by_list(self, obslist, weight):
        """a private method to reset the weight for a list of observation names.  Supports the
        data worth analyses in pyemu.Schur class.  This method only adjusts
        observation weights in the current weight is nonzero.  User beware!

        Args:
            obslist ([`str`]): list of observation names
            weight (`float`): new weight to assign
        """

        obs = self.observation_data
        if not isinstance(obslist, list):
            obslist = [obslist]
        obslist = set([str(i).lower() for i in obslist])
        # groups = obs.groupby([lambda x:x in obslist,
        #                     obs.weight.apply(lambda x:x==0.0)]).groups
        # if (True,True) in groups:
        #    obs.loc[groups[True,True],"weight"] = weight
        reset_names = obs.loc[
            obs.apply(lambda x: x.obsnme in obslist and x.weight == 0, axis=1), "obsnme"
        ]
        if len(reset_names) > 0:
            obs.loc[reset_names, "weight"] = weight

    def adjust_weights(self, obs_dict=None, obsgrp_dict=None):
        """reset the weights of observations or observation groups to contribute a specified
        amount to the composite objective function

        Args:
            obs_dict (`dict`, optional): dictionary of observation name,new contribution pairs
            obsgrp_dict (`dict`, optional): dictionary of obs group name,contribution pairs

        Notes:
            If a group is assigned a contribution of 0, all observations in that group will be assigned
            zero weight.

            If a group is assigned a nonzero contribution AND all observations in that group start
            with zero weight, the observations will be assigned weight of 1.0 to allow balancing.

            If groups obsgrp_dict is not passed, all nonzero
            

        Example::

            pst = pyemu.Pst("my.pst")

            # adjust a single observation
            pst.adjust_weights(obs_dict={"obs1":10})

            # adjust a single observation group
            pst.adjust_weights(obsgrp_dict={"group1":100.0})

            # make all non-zero weighted groups have a contribution of 100.0
            balanced_groups = {grp:100 for grp in pst.nnz_obs_groups}
            pst.adjust_weights(obsgrp_dict=balanced_groups)

        """
        if (obsgrp_dict is not None) and (obs_dict is not None):
            
            raise Exception(
                "Pst.asjust_weights(): "
                + "Both obsgrp_dict and obs_dict passed "
                + "Must choose one or the other"
            )

        self.observation_data.index = self.observation_data.obsnme
        self.res.index = self.res.name

        if obsgrp_dict is not None:
            obs = self.observation_data
            # first zero-weight all obs in groups specified to have 0 contrib to phi
            for grp, contrib in obsgrp_dict.items():
                if contrib==0:
                    obs.loc[obs.obgnme == grp, "weight"] = 0.0
                    # drop zero- contribution groups
                    del obsgrp_dict[grp]
            # reset groups with all zero weights
            for grp in obsgrp_dict.keys():
                if obs.loc[obs.obgnme == grp, "weight"].sum() == 0.0:
                    obs.loc[obs.obgnme == grp, "weight"] = 1.0
            self.res.loc[obs.index, 'group'] = obs.obgnme.values
            self.res.loc[obs.index, 'weight'] = obs.weight.values 
            res_groups = self.res.groupby("group").groups
            obs_groups = self.observation_data.groupby("obgnme").groups
            self.__reset_weights(obsgrp_dict, res_groups, obs_groups)
        if obs_dict is not None:
            # reset obs with zero weight
            obs = self.observation_data
            for oname in obs_dict.keys():
                if obs.loc[oname, "weight"] == 0.0:
                    obs.loc[oname, "weight"] = 1.0

            # res_groups = self.res.groupby("name").groups
            res_groups = self.res.groupby(self.res.index).groups
            # obs_groups = self.observation_data.groupby("obsnme").groups
            obs_groups = self.observation_data.groupby(
                self.observation_data.index
            ).groups
            self.__reset_weights(obs_dict, res_groups, obs_groups)

    def proportional_weights(self, fraction_stdev=1.0, wmax=100.0, leave_zero=True):
        """setup  weights inversely proportional to the observation value

        Args:
            fraction_stdev (`float`, optional): the fraction portion of the observation
                val to treat as the standard deviation.  set to 1.0 for
                inversely proportional.  Default is 1.0
            wmax (`float`, optional): maximum weight to allow.  Default is 100.0

            leave_zero (`bool`, optional): flag to leave existing zero weights.
                Default is True

        Example::

            pst = pyemu.Pst("pest.pst")
            # set the weights of the observations to 20% of the observed value
            pst.proportional_weights(fraction_stdev=0.2,wmax=10)
            pst.write("pest_propo.pst")

        """
        new_weights = []
        for oval, ow in zip(self.observation_data.obsval, self.observation_data.weight):
            if leave_zero and ow == 0.0:
                ow = 0.0
            elif oval == 0.0:
                ow = wmax
            else:
                nw = 1.0 / (np.abs(oval) * fraction_stdev)
                ow = min(wmax, nw)
            new_weights.append(ow)
        self.observation_data.weight = new_weights

    def calculate_pertubations(self):
        """experimental method to calculate finite difference parameter
        pertubations.

        Note:

            The pertubation values are added to the
            `Pst.parameter_data` attribute - user beware!

        """
        self.build_increments()
        self.parameter_data.loc[:, "pertubation"] = (
            self.parameter_data.parval1 + self.parameter_data.increment
        )

        self.parameter_data.loc[:, "out_forward"] = (
            self.parameter_data.loc[:, "pertubation"]
            > self.parameter_data.loc[:, "parubnd"]
        )

        out_forward = self.parameter_data.groupby("out_forward").groups
        if True in out_forward:
            self.parameter_data.loc[out_forward[True], "pertubation"] = (
                self.parameter_data.loc[out_forward[True], "parval1"]
                - self.parameter_data.loc[out_forward[True], "increment"]
            )

            self.parameter_data.loc[:, "out_back"] = (
                self.parameter_data.loc[:, "pertubation"]
                < self.parameter_data.loc[:, "parlbnd"]
            )
            out_back = self.parameter_data.groupby("out_back").groups
            if True in out_back:
                still_out = out_back[True]
                print(self.parameter_data.loc[still_out, :], flush=True)

                raise Exception(
                    "Pst.calculate_pertubations(): "
                    + "can't calc pertubations for the following "
                    + "Parameters {0}".format(",".join(still_out))
                )

    def build_increments(self):
        """experimental method to calculate parameter increments for use
        in the finite difference pertubation calculations

        Note:
            user beware!

        """
        self.enforce_bounds()
        self.add_transform_columns()
        par_groups = self.parameter_data.groupby("pargp").groups
        inctype = self.parameter_groups.groupby("inctyp").groups
        for itype, inc_groups in inctype.items():
            pnames = []
            for group in inc_groups:
                pnames.extend(par_groups[group])
                derinc = self.parameter_groups.loc[group, "derinc"]
                self.parameter_data.loc[par_groups[group], "derinc"] = derinc
            if itype == "absolute":
                self.parameter_data.loc[pnames, "increment"] = self.parameter_data.loc[
                    pnames, "derinc"
                ]
            elif itype == "relative":
                self.parameter_data.loc[pnames, "increment"] = (
                    self.parameter_data.loc[pnames, "derinc"]
                    * self.parameter_data.loc[pnames, "parval1"]
                )
            elif itype == "rel_to_max":
                mx = self.parameter_data.loc[pnames, "parval1"].max()
                self.parameter_data.loc[pnames, "increment"] = (
                    self.parameter_data.loc[pnames, "derinc"] * mx
                )
            else:
                raise Exception(
                    "Pst.get_derivative_increments(): "
                    + "unrecognized increment type:{0}".format(itype)
                )

        # account for fixed pars
        isfixed = self.parameter_data.partrans == "fixed"
        self.parameter_data.loc[isfixed, "increment"] = self.parameter_data.loc[
            isfixed, "parval1"
        ]

    def add_transform_columns(self):
        """add transformed values to the `Pst.parameter_data` attribute

        Note:
            adds `parval1_trans`, `parlbnd_trans` and `parubnd_trans` to
            `Pst.parameter_data`


        Example::

            pst = pyemu.Pst("pest.pst")
            pst.add_transform_columns()
            print(pst.parameter_data.parval1_trans


        """
        for col in ["parval1", "parlbnd", "parubnd", "increment"]:
            if col not in self.parameter_data.columns:
                continue
            self.parameter_data.loc[:, col + "_trans"] = (
                self.parameter_data.loc[:, col] * self.parameter_data.scale
            ) + self.parameter_data.offset
            # isnotfixed = self.parameter_data.partrans != "fixed"
            islog = self.parameter_data.partrans == "log"
            self.parameter_data.loc[islog, col + "_trans"] = self.parameter_data.loc[
                islog, col + "_trans"
            ].apply(lambda x: np.log10(x))

    def enforce_bounds(self):
        """enforce bounds violation

        Note:
            cheap enforcement of simply bringing violators back in bounds


        Example::

            pst = pyemu.Pst("pest.pst")
            pst.parrep("random.par")
            pst.enforce_bounds()
            pst.write("pest_rando.pst")


        """
        too_big = (
            self.parameter_data.loc[:, "parval1"]
            > self.parameter_data.loc[:, "parubnd"]
        )
        self.parameter_data.loc[too_big, "parval1"] = self.parameter_data.loc[
            too_big, "parubnd"
        ]

        too_small = (
            self.parameter_data.loc[:, "parval1"]
            < self.parameter_data.loc[:, "parlbnd"]
        )
        self.parameter_data.loc[too_small, "parval1"] = self.parameter_data.loc[
            too_small, "parlbnd"
        ]

    @classmethod
    def from_io_files(
        cls, tpl_files, in_files, ins_files, out_files, pst_filename=None, pst_path=None
    ):
        """create a Pst instance from model interface files.

        Args:
            tpl_files ([`str`]): list of template file names
            in_files ([`str`]): list of model input file names (pairs with template files)
            ins_files ([`str`]): list of instruction file names
            out_files ([`str`]): list of model output file names (pairs with instruction files)
            pst_filename (`str`): name of control file to write.  If None, no file is written.
                Default is None
            pst_path ('str'): the path from the control file to the IO files.  For example, if the
                control will be in the same directory as the IO files, then `pst_path` should be '.'.
                Default is None, which doesnt do any path manipulation on the I/O file names


        Returns:
            `Pst`: new control file instance with parameter and observation names
            found in `tpl_files` and `ins_files`, repsectively.

        Note:
            calls `pyemu.helpers.pst_from_io_files()`

            Assigns generic values for parameter info.  Tries to use INSCHEK
            to set somewhat meaningful observation values

            all file paths are relatively to where python is running.


        Example::

            tpl_files = ["my.tpl"]
            in_files = ["my.in"]
            ins_files = ["my.ins"]
            out_files = ["my.out"]
            pst = pyemu.Pst.from_io_files(tpl_files,in_files,ins_files,out_files)
            pst.control_data.noptmax = 0
            pst.write("my.pst)



        """
        from pyemu import helpers

        return helpers.pst_from_io_files(
            tpl_files=tpl_files,
            in_files=in_files,
            ins_files=ins_files,
            out_files=out_files,
            pst_filename=pst_filename,
            pst_path=pst_path,
        )

    def add_parameters(self, template_file, in_file=None, pst_path=None):
        """add new parameters to an existing control file

        Args:
            template_file (`str`): template file with (possibly) some new parameters
            in_file (`str`): model input file. If None, template_file.replace('.tpl','') is used.
                Default is None.
            pst_path (`str`): the path to append to the template_file and in_file in the control file.  If
                not None, then any existing path in front of the template or in file is split off
                and pst_path is prepended.  If python is being run in a directory other than where the control
                file will reside, it is useful to pass `pst_path` as `.`.  Default is None

        Returns:
            `pandas.DataFrame`: the data for the new parameters that were added.
            If no new parameters are in the new template file, returns None

        Note:
            populates the new parameter information with default values

        Example::

            pst = pyemu.Pst(os.path.join("template","my.pst"))
            pst.add_parameters(os.path.join("template","new_pars.dat.tpl",pst_path=".")
            pst.write(os.path.join("template","my_new.pst")

        """
        if not os.path.exists(template_file):
            raise Exception("template file '{0}' not found".format(template_file))
        if template_file == in_file:
            raise Exception("template_file == in_file")
        # get the parameter names in the template file
        parnme = pst_utils.parse_tpl_file(template_file)

        parval1 = pst_utils.try_read_input_file_with_tpl(template_file, in_file)

        # find "new" parameters that are not already in the control file
        new_parnme = [p for p in parnme if p not in self.parameter_data.parnme]

        if len(new_parnme) == 0:
            warnings.warn(
                "no new parameters found in template file {0}".format(template_file),
                PyemuWarning,
            )
            new_par_data = None
        else:
            # extend pa
            # rameter_data
            new_par_data = pst_utils._populate_dataframe(
                new_parnme,
                pst_utils.pst_config["par_fieldnames"],
                pst_utils.pst_config["par_defaults"],
                pst_utils.pst_config["par_dtype"],
            )
            new_par_data.loc[new_parnme, "parnme"] = new_parnme
            self.parameter_data = pd.concat([self.parameter_data, new_par_data])
            if parval1 is not None:
                parval1 = parval1.loc[new_par_data.parnme]
                new_par_data.loc[parval1.parnme, "parval1"] = parval1.parval1
        if in_file is None:
            in_file = template_file.replace(".tpl", "")
        if pst_path is not None:
            template_file = os.path.join(pst_path, os.path.split(template_file)[-1])
            in_file = os.path.join(pst_path, os.path.split(in_file)[-1])

        # self.template_files.append(template_file)
        # self.input_files.append(in_file)
        self.model_input_data.loc[template_file, "pest_file"] = template_file
        self.model_input_data.loc[template_file, "model_file"] = in_file
        print(
            "{0} pars added from template file {1}".format(
                len(new_parnme), template_file
            )
        )
        return new_par_data

    def drop_observations(self, ins_file, pst_path=None):
        """remove observations in an instruction file from the control file

        Args:
            ins_file (`str`): instruction file to remove
            pst_path (`str`): the path to append to the instruction file in the control file.  If
                not None, then any existing path in front of the instruction is split off
                and pst_path is prepended.  If python is being run in a directory other than where the control
                file will reside, it is useful to pass `pst_path` as `.`. Default is None

        Returns:
            `pandas.DataFrame`: the observation data for the observations that were removed.

        Example::

            pst = pyemu.Pst(os.path.join("template", "my.pst"))
            pst.remove_observations(os.path.join("template","some_obs.dat.ins"), pst_path=".")
            pst.write(os.path.join("template", "my_new_with_less_obs.pst")

        """

        if not os.path.exists(ins_file):
            raise Exception("couldn't find instruction file '{0}'".format(ins_file))
        pst_ins_file = ins_file
        if pst_path is not None:
            pst_ins_file = os.path.join(pst_path, os.path.split(ins_file)[1])
        if pst_ins_file not in self.model_output_data.pest_file.to_list():
            if pst_path == ".":
                pst_ins_file = os.path.split(ins_file)[1]
                if pst_ins_file not in self.model_output_data.pest_file.to_list():
                    raise Exception(
                        "ins_file '{0}' not found in Pst.model_output_data.pest_file".format(
                            pst_ins_file
                        )
                    )
            else:
                raise Exception(
                    "ins_file '{0}' not found in Pst.model_output_data.pest_file".format(
                        pst_ins_file
                    )
                )
        i = pst_utils.InstructionFile(ins_file)
        drop_obs = i.obs_name_set

        # if len(drop_obs) == self.nobs:
        #    raise Exception("cannot drop all observations")

        obs_names = set(self.obs_names)
        drop_obs_present = [o for o in drop_obs if o in obs_names]
        dropped_obs = self.observation_data.loc[drop_obs_present, :].copy()
        self.observation_data = self.observation_data.loc[
            self.observation_data.obsnme.apply(lambda x: x not in drop_obs), :
        ]
        self.model_output_data = self.model_output_data.loc[
            self.model_output_data.pest_file != pst_ins_file
        ]
        print(
            "{0} obs dropped from instruction file {1}".format(len(drop_obs), ins_file)
        )
        return dropped_obs

    def drop_parameters(self, tpl_file, pst_path=None):
        """remove parameters in a template file from the control file

        Args:
            tpl_file (`str`): template file to remove
            pst_path (`str`): the path to append to the template file in the control file.  If
                not None, then any existing path in front of the template or in file is split off
                and pst_path is prepended.  If python is being run in a directory other than where the control
                file will reside, it is useful to pass `pst_path` as `.`. Default is None

        Returns:
            `pandas.DataFrame`: the parameter data for the parameters that were removed.

        Note:
            This method does not check for multiple occurences of the same parameter name(s) in
            across template files so if you have the same parameter in multiple template files,
            this is not the method you are looking for

        Example::

            pst = pyemu.Pst(os.path.join("template", "my.pst"))
            pst.remove_parameters(os.path.join("template","boring_zone_pars.dat.tpl"), pst_path=".")
            pst.write(os.path.join("template", "my_new_with_less_pars.pst")

        """

        if not os.path.exists(tpl_file):
            raise Exception("couldn't find template file '{0}'".format(tpl_file))
        pst_tpl_file = tpl_file
        if pst_path is not None:
            pst_tpl_file = os.path.join(pst_path, os.path.split(tpl_file)[1])
        if pst_tpl_file not in self.model_input_data.pest_file.to_list():
            if pst_path == ".":
                pst_tpl_file = os.path.split(tpl_file)[1]
                if pst_tpl_file not in self.model_input_data.pest_file.to_list():
                    raise Exception(
                        "tpl_file '{0}' not found in Pst.model_input_data.pest_file".format(
                            pst_tpl_file
                        )
                    )
            else:
                raise Exception(
                    "tpl_file '{0}' not found in Pst.model_input_data.pest_file".format(
                        pst_tpl_file
                    )
                )
        drop_pars = pst_utils.parse_tpl_file(tpl_file)
        if len(drop_pars) == self.npar:
            raise Exception("cannot drop all parameters")

        # get a list of drop pars that are in parameter_data
        par_names = set(self.par_names)
        drop_pars_present = [p for p in drop_pars if p in par_names]

        # check that other pars arent tied to the dropping pars
        if "partied" in self.parameter_data.columns:
            par_tied = set(
                self.parameter_data.loc[
                    self.parameter_data.partrans == "tied", "partied"
                ].to_list()
            )

            par_tied = par_tied.intersection(drop_pars_present)
            if len(par_tied) > 0:
                raise Exception(
                    "the following pars to be dropped are 'tied' to: {0}".format(
                        str(par_tied)
                    )
                )

        dropped_par = self.parameter_data.loc[drop_pars_present, :].copy()
        self.parameter_data = self.parameter_data.loc[
            self.parameter_data.parnme.apply(lambda x: x not in drop_pars_present), :
        ]
        self.rectify_pi()
        self.model_input_data = self.model_input_data.loc[
            self.model_input_data.pest_file != pst_tpl_file
        ]
        print(
            "{0} pars dropped from template file {1}".format(len(drop_pars), tpl_file)
        )
        return dropped_par

    def add_observations(self, ins_file, out_file=None, pst_path=None, inschek=True):
        """add new observations to a control file

        Args:
            ins_file (`str`): instruction file with exclusively new observation names
            out_file (`str`): model output file.  If None, then ins_file.replace(".ins","") is used.
                Default is None
            pst_path (`str`): the path to append to the instruction file and out file in the control file.  If
                not None, then any existing path in front of the template or in file is split off
                and pst_path is prepended.  If python is being run in a directory other than where the control
                file will reside, it is useful to pass `pst_path` as `.`. Default is None
            inschek (`bool`): flag to try to process the existing output file using the `pyemu.InstructionFile`
                class.  If successful, processed outputs are used as obsvals

        Returns:
            `pandas.DataFrame`: the data for the new observations that were added

        Note:
            populates the new observation information with default values

        Example::

            pst = pyemu.Pst(os.path.join("template", "my.pst"))
            pst.add_observations(os.path.join("template","new_obs.dat.ins"), pst_path=".")
            pst.write(os.path.join("template", "my_new.pst")

        """
        if not os.path.exists(ins_file):
            raise Exception(
                "ins file not found: {0}, {1}".format(os.getcwd(), ins_file)
            )
        if out_file is None:
            out_file = ins_file.replace(".ins", "")
        if ins_file == out_file:
            raise Exception("ins_file == out_file, doh!")

        # get the parameter names in the template file
        obsnme = pst_utils.parse_ins_file(ins_file)

        sobsnme = set(obsnme)
        sexist = set(self.obs_names)
        sint = sobsnme.intersection(sexist)
        if len(sint) > 0:
            raise Exception(
                "the following obs in instruction file {0} are already in the control file:{1}".format(
                    ins_file, ",".join(sint)
                )
            )

        # extend observation_data
        new_obs_data = pst_utils._populate_dataframe(
            obsnme,
            pst_utils.pst_config["obs_fieldnames"],
            pst_utils.pst_config["obs_defaults"],
            pst_utils.pst_config["obs_dtype"],
        )
        new_obs_data.loc[obsnme, "obsnme"] = obsnme
        new_obs_data.index = obsnme
        self.observation_data = pd.concat([self.observation_data, new_obs_data])
        cwd = "."
        if pst_path is not None:
            cwd = os.path.join(*os.path.split(ins_file)[:-1])
            ins_file = os.path.join(pst_path, os.path.split(ins_file)[-1])
            out_file = os.path.join(pst_path, os.path.split(out_file)[-1])
        # self.instruction_files.append(ins_file)
        # self.output_files.append(out_file)
        self.model_output_data.loc[ins_file, "pest_file"] = ins_file
        self.model_output_data.loc[ins_file, "model_file"] = out_file
        df = None
        if inschek:
            # df = pst_utils._try_run_inschek(ins_file,out_file,cwd=cwd)
            ins_file = os.path.join(cwd, ins_file)
            out_file = os.path.join(cwd, out_file)
            df = pst_utils.try_process_output_file(
                ins_file=ins_file, output_file=out_file
            )
        if df is not None:
            # print(self.observation_data.index,df.index)
            self.observation_data.loc[df.index, "obsval"] = df.obsval
            new_obs_data.loc[df.index, "obsval"] = df.obsval
        print("{0} obs added from instruction file {1}".format(len(obsnme), ins_file))
        return new_obs_data

    def write_input_files(self, pst_path="."):
        """writes model input files using template files and current `parval1` values.

        Args:
            pst_path (`str`): the path to where control file and template files reside.
                Default is '.'

        Note:
            adds "parval1_trans" column to Pst.parameter_data that includes the
            effect of scale and offset

        Example::

            pst = pyemu.Pst("my.pst")

            # load final parameter values
            pst.parrep("my.par")

            # write new model input files with final parameter values
            pst.write_input_files()

        """
        pst_utils.write_input_files(self, pst_path=pst_path)

    def process_output_files(self, pst_path="."):
        """processing the model output files using the instruction files
        and existing model output files.

        Args:
            pst_path (`str`): relative path from where python is running to
                where the control file, instruction files and model output files
                are located.  Default is "." (current python directory)

        Returns:
            `pandas.Series`: model output values

        Note:
            requires a complete set of model input files at relative path
            from where python is running to `pst_path`

        Example::

            pst = pyemu.Pst("pest.pst")
            obsvals = pst.process_output_files()
            print(obsvals)

        """
        return pst_utils.process_output_files(self, pst_path)

    def get_res_stats(self, nonzero=True):
        """get some common residual stats by observation group.

        Args:
            nonzero (`bool`): calculate stats using only nonzero-weighted observations.  This may seem
                obsvious to most users, but you never know....

        Returns:
            `pd.DataFrame`: a dataframe with columns for groups names and indices of statistic name.

        Note:
            Stats are derived from the current obsvals, weights and grouping in
            `Pst.observation_data` and the `modelled` values in `Pst.res`.  The
            key here is 'current' because if obsval, weights and/or groupings have
            changed in `Pst.observation_data` since the residuals file was generated
            then the current values for `obsval`, `weight` and `group` are used

            the normalized RMSE is normalized against the obsval range (max - min)

        Example::

            pst = pyemu.Pst("pest.pst")
            stats_df = pst.get_res_stats()
            print(stats_df.loc["mae",:])


        """
        res = self.res.copy()
        res.loc[:, "obsnme"] = res.pop("name")
        res.index = res.obsnme
        if nonzero:
            obs = self.observation_data.loc[self.nnz_obs_names, :]
        # print(obs.shape,res.shape)
        res = res.loc[obs.obsnme, :]
        # print(obs.shape, res.shape)

        # reset the res parts to current obs values and remove
        # duplicate attributes
        res.loc[:, "weight"] = obs.weight
        res.loc[:, "obsval"] = obs.obsval
        res.loc[:, "obgnme"] = obs.obgnme
        res.pop("group")
        res.pop("measured")

        # build these attribute lists for faster lookup later
        og_dict = {
            og: res.loc[res.obgnme == og, "obsnme"] for og in res.obgnme.unique()
        }
        og_names = list(og_dict.keys())

        # the list of functions and names
        sfuncs = [
            self._stats_rss,
            self._stats_mean,
            self._stats_mae,
            self._stats_rmse,
            self._stats_nrmse,
        ]
        snames = ["rss", "mean", "mae", "rmse", "nrmse"]

        data = []
        for sfunc in sfuncs:
            full = sfunc(res)
            groups = [full]
            for og in og_names:
                onames = og_dict[og]
                res_og = res.loc[onames, :]
                groups.append(sfunc(res_og))
            data.append(groups)

        og_names.insert(0, "all")
        stats = pd.DataFrame(data, columns=og_names, index=snames)
        return stats

    @staticmethod
    def _stats_rss(df):
        return (((df.modelled - df.obsval) * df.weight) ** 2).sum()

    @staticmethod
    def _stats_mean(df):
        return (df.modelled - df.obsval).mean()

    @staticmethod
    def _stats_mae(df):
        return ((df.modelled - df.obsval).apply(np.abs)).sum() / df.shape[0]

    @staticmethod
    def _stats_rmse(df):
        return np.sqrt(((df.modelled - df.obsval) ** 2).sum() / df.shape[0])

    @staticmethod
    def _stats_nrmse(df):
        return Pst._stats_rmse(df) / (df.obsval.max() - df.obsval.min())

    def plot(self, kind=None, **kwargs):
        """method to plot various parts of the control.  This is sweet as!

        Args:
            kind (`str`): options are 'prior' (prior parameter histograms, '1to1' (line of equality
                and sim vs res), 'obs_v_sim' (time series using datetime suffix), 'phi_pie'
                (pie chart of phi components)
            kwargs (`dict`): optional args for plots that are passed to pyemu plot helpers and ultimately
                to matplotlib

        Note:
            Depending on 'kind' argument, a multipage pdf is written

        Example::

            pst = pyemu.Pst("my.pst")
            pst.plot(kind="1to1") # requires Pst.res
            pst.plot(kind="prior")
            pst.plot(kind="phi_pie")


        """
        return plot_utils.pst_helper(self, kind, **kwargs)

    def write_par_summary_table(
        self,
        filename=None,
        group_names=None,
        sigma_range=4.0,
        report_in_linear_space=False,
    ):
        """write a stand alone parameter summary latex table or Excel sheet


        Args:
            filename (`str`): filename. If None, use <case>.par.tex to write as LaTeX. If filename extention is '.xls' or '.xlsx',
                tries to write as an Excel file. If `filename` is "none", no table is written
                Default is None
            group_names (`dict`): par group names : table names. For example {"w0":"well stress period 1"}.
                Default is None
            sigma_range (`float`): number of standard deviations represented by parameter bounds.  Default
                is 4.0, implying 95% confidence bounds
            report_in_linear_space (`bool`): flag, if True, that reports all logtransformed values in linear
                space. This renders standard deviation meaningless, so that column is skipped

        Returns:
            `pandas.DataFrame`: the summary parameter group dataframe

        Example::

            pst = pyemu.Pst("my.pst")
            pst.write_par_summary_table(filename="par.tex")

        """

        ffmt = lambda x: "{0:5G}".format(x)
        par = self.parameter_data.copy()
        pargp = par.groupby(par.pargp).groups
        # cols = ["parval1","parubnd","parlbnd","stdev","partrans","pargp"]
        if report_in_linear_space == True:
            cols = ["pargp", "partrans", "count", "parval1", "parlbnd", "parubnd"]

        else:
            cols = [
                "pargp",
                "partrans",
                "count",
                "parval1",
                "parlbnd",
                "parubnd",
                "stdev",
            ]

        labels = {
            "parval1": "initial value",
            "parubnd": "upper bound",
            "parlbnd": "lower bound",
            "partrans": "transform",
            "stdev": "standard deviation",
            "pargp": "type",
            "count": "count",
        }

        li = par.partrans == "log"
        if True in li.values and report_in_linear_space == True:
            print(
                "Warning: because log-transformed values being reported in linear space, stdev NOT reported"
            )

        if report_in_linear_space == False:
            par.loc[li, "parval1"] = par.parval1.loc[li].apply(np.log10)
            par.loc[li, "parubnd"] = par.parubnd.loc[li].apply(np.log10)
            par.loc[li, "parlbnd"] = par.parlbnd.loc[li].apply(np.log10)
            par.loc[:, "stdev"] = (par.parubnd - par.parlbnd) / sigma_range

        data = {c: [] for c in cols}
        for pg, pnames in pargp.items():
            par_pg = par.loc[pnames, :]
            data["pargp"].append(pg)
            for col in cols:
                if col in ["pargp", "partrans"]:
                    continue
                if col == "count":
                    data["count"].append(par_pg.shape[0])
                    continue
                # print(col)
                mn = par_pg.loc[:, col].min()
                mx = par_pg.loc[:, col].max()
                if mn == mx:
                    data[col].append(ffmt(mn))
                else:
                    data[col].append("{0} to {1}".format(ffmt(mn), ffmt(mx)))

            pts = par_pg.partrans.unique()
            if len(pts) == 1:
                data["partrans"].append(pts[0])
            else:
                data["partrans"].append("mixed")

        pargp_df = pd.DataFrame(data=data, index=list(pargp.keys()))
        pargp_df = pargp_df.loc[:, cols]
        if group_names is not None:
            pargp_df.loc[:, "pargp"] = pargp_df.pargp.apply(
                lambda x: group_names.pop(x, x)
            )
        pargp_df.columns = pargp_df.columns.map(lambda x: labels[x])

        preamble = (
            "\\documentclass{article}\n\\usepackage{booktabs}\n"
            + "\\usepackage{pdflscape}\n\\usepackage{longtable}\n"
            + "\\usepackage{booktabs}\n\\usepackage{nopageno}\n\\begin{document}\n"
        )

        if filename == "none":
            return pargp_df
        if filename is None:
            filename = self.filename.replace(".pst", ".par.tex")
        # if filename indicates an Excel format, try writing to Excel
        if filename.lower().endswith("xlsx") or filename.lower().endswith("xls"):
            try:
                pargp_df.to_excel(filename, index=None)
            except Exception as e:
                if filename.lower().endswith("xlsx"):
                    print(
                        "could not export {0} in Excel format. Try installing xlrd".format(
                            filename
                        )
                    )
                elif filename.lower().endswith("xls"):
                    print(
                        "could not export {0} in Excel format. Try installing xlwt".format(
                            filename
                        )
                    )
                else:
                    print("could not export {0} in Excel format.".format(filename))

        else:
            with open(filename, "w") as f:
                f.write(preamble)
                f.write("\\begin{center}\nParameter Summary\n\\end{center}\n")
                f.write("\\begin{center}\n\\begin{landscape}\n")
                try:
                    f.write(pargp_df.style.hide(axis='index').to_latex(
                        None, environment='longtable')
                    )
                except (TypeError, AttributeError) as e:
                    pargp_df.to_latex(index=False, longtable=True)
                f.write("\\end{landscape}\n")
                f.write("\\end{center}\n")
                f.write("\\end{document}\n")
        return pargp_df

    def write_obs_summary_table(self, filename=None, group_names=None):
        """write a stand alone observation summary latex table or Excel shet
            filename (`str`): filename. If None, use <case>.par.tex to write as LaTeX. If filename extention is '.xls' or '.xlsx',
                tries to write as an Excel file. If `filename` is "none", no table is written
                Default is None

        Args:
            filename (`str`): filename. If `filename` is "none", no table is written.
                If None, use <case>.obs.tex. If filename extention is '.xls' or '.xlsx',
                tries to write as an Excel file.
                Default is None
            group_names (`dict`): obs group names : table names. For example {"hds":"simulated groundwater level"}.
                Default is None

        Returns:
            `pandas.DataFrame`: the summary observation group dataframe

        Example::

            pst = pyemu.Pst("my.pst")
            pst.write_obs_summary_table(filename="obs.tex")
        """

        ffmt = lambda x: "{0:5G}".format(x)
        obs = self.observation_data.copy()
        obsgp = obs.groupby(obs.obgnme).groups
        cols = ["obgnme", "obsval", "nzcount", "zcount", "weight", "stdev", "pe"]

        labels = {
            "obgnme": "group",
            "obsval": "value",
            "nzcount": "non-zero weight",
            "zcount": "zero weight",
            "weight": "weight",
            "stdev": "standard deviation",
            "pe": "percent error",
        }

        obs.loc[:, "stdev"] = 1.0 / obs.weight
        obs.loc[:, "pe"] = 100.0 * (obs.stdev / obs.obsval.apply(np.abs))
        obs = obs.replace([np.inf, -np.inf], np.NaN)

        data = {c: [] for c in cols}
        for og, onames in obsgp.items():
            obs_g = obs.loc[onames, :]
            data["obgnme"].append(og)
            data["nzcount"].append(obs_g.loc[obs_g.weight > 0.0, :].shape[0])
            data["zcount"].append(obs_g.loc[obs_g.weight == 0.0, :].shape[0])
            for col in cols:
                if col in ["obgnme", "nzcount", "zcount"]:
                    continue

                # print(col)
                mn = obs_g.loc[:, col].min()
                mx = obs_g.loc[:, col].max()
                if np.isnan(mn) or np.isnan(mx):
                    data[col].append("NA")
                elif mn == mx:
                    data[col].append(ffmt(mn))
                else:
                    data[col].append("{0} to {1}".format(ffmt(mn), ffmt(mx)))

        obsg_df = pd.DataFrame(data=data, index=list(obsgp.keys()))
        obsg_df = obsg_df.loc[:, cols]
        if group_names is not None:
            obsg_df.loc[:, "obgnme"] = obsg_df.obgnme.apply(
                lambda x: group_names.pop(x, x)
            )
        obsg_df.sort_values(by="obgnme", inplace=True, ascending=True)
        obsg_df.columns = obsg_df.columns.map(lambda x: labels[x])

        preamble = (
            "\\documentclass{article}\n\\usepackage{booktabs}\n"
            + "\\usepackage{pdflscape}\n\\usepackage{longtable}\n"
            + "\\usepackage{booktabs}\n\\usepackage{nopageno}\n\\begin{document}\n"
        )

        if filename == "none":
            return obsg_df
        if filename is None:
            filename = self.filename.replace(".pst", ".obs.tex")
        # if filename indicates an Excel format, try writing to Excel
        if filename.lower().endswith("xlsx") or filename.lower().endswith("xls"):
            try:
                obsg_df.to_excel(filename, index=None)
            except Exception as e:
                if filename.lower().endswith("xlsx"):
                    print(
                        "could not export {0} in Excel format. Try installing xlrd".format(
                            filename
                        )
                    )
                elif filename.lower().endswith("xls"):
                    print(
                        "could not export {0} in Excel format. Try installing xlwt".format(
                            filename
                        )
                    )
                else:
                    print("could not export {0} in Excel format.".format(filename))

        else:
            with open(filename, "w") as f:

                f.write(preamble)

                f.write("\\begin{center}\nObservation Summary\n\\end{center}\n")
                f.write("\\begin{center}\n\\begin{landscape}\n")
                f.write("\\setlength{\\LTleft}{-4.0cm}\n")
                try:
                    f.write(obsg_df.style.hide(axis='index').to_latex(
                        None, environment='longtable')
                    )
                except (TypeError, AttributeError) as e:
                    obsg_df.to_latex(index=False, longtable=True)
                f.write("\\end{landscape}\n")
                f.write("\\end{center}\n")
                f.write("\\end{document}\n")

        return obsg_df

    # jwhite - 13 Aug 2019 - no one is using this write?
    # def run(self,exe_name="pestpp",cwd=None):
    #     """run a command related to the pst instance. If
    #     write() has been called, then the filename passed to write
    #     is in the command, otherwise the original constructor
    #     filename is used
    #
    #     exe_name : str
    #         the name of the executable to call.  Default is "pestpp"
    #     cwd : str
    #         the directory to execute the command in.  If None,
    #         os.path.split(self.filename) is used to find
    #         cwd.  Default is None
    #
    #
    #     """
    #     filename = self.filename
    #     if self.new_filename is not None:
    #         filename = self.new_filename
    #     cmd_line = "{0} {1}".format(exe_name,os.path.split(filename)[-1])
    #     if cwd is None:
    #         cwd = os.path.join(*os.path.split(filename)[:-1])
    #         if cwd == '':
    #             cwd = '.'
    #     print("executing {0} in dir {1}".format(cmd_line, cwd))
    #     pyemu.utils.os_utils.run(cmd_line,cwd=cwd)

    # @staticmethod
    # def _is_less_const(name):
    #     constraint_tags = ["l_", "less"]
    #     return True in [True for c in constraint_tags if name.startswith(c)]

    @property
    def less_than_obs_constraints(self):
        """get the names of the observations that
        are listed as active (non-zero weight) less than inequality constraints.

        Returns:
            `pandas.Series`: names of observations that are non-zero weighted less
            than constraints (`obgnme` starts with 'l_' or "less")

        Note:
             Zero-weighted obs are skipped

        """
        obs = self.observation_data
        lt_obs = obs.loc[
            obs.obgnme.str.startswith(self.get_constraint_tags('lt')) &
            (obs.weight != 0.0), "obsnme"
        ]
        return lt_obs

    @property
    def less_than_pi_constraints(self):
        """get the names of the prior information eqs that
        are listed as active (non-zero weight) less than inequality constraints.

        Returns:
            `pandas.Series`: names of prior information that are non-zero weighted
            less than constraints (`obgnme` starts with "l_" or "less")

        Note:
            Zero-weighted pi are skipped

        """

        pi = self.prior_information
        lt_pi = pi.loc[
            pi.obgnme.str.startswith(self.get_constraint_tags('lt')) &
            (pi.weight != 0.0), "pilbl"
        ]
        return lt_pi

    # @staticmethod
    # def _is_greater_const(name):
    #     constraint_tags = ["g_", "greater"]
    #     return True in [True for c in constraint_tags if name.startswith(c)]

    @property
    def greater_than_obs_constraints(self):
        """get the names of the observations that
        are listed as active (non-zero weight) greater than inequality constraints.

        Returns:
            `pandas.Series`: names obseravtions that are non-zero weighted
            greater than constraints (`obgnme` startsiwth "g_" or "greater")

        Note:
            Zero-weighted obs are skipped

        """

        obs = self.observation_data
        gt_obs = obs.loc[
            obs.obgnme.str.startswith(self.get_constraint_tags('gt')) &
            (obs.weight != 0.0), "obsnme"
        ]
        return gt_obs

    @property
    def greater_than_pi_constraints(self):
        """get the names of the prior information eqs that
        are listed as active (non-zero weight) greater than inequality constraints.

        Returns:
            `pandas.Series` names of prior information that are non-zero weighted
            greater than constraints (`obgnme` startsiwth "g_" or "greater")


        Note:
             Zero-weighted pi are skipped

        """

        pi = self.prior_information
        gt_pi = pi.loc[
            pi.obgnme.str.startswith(self.get_constraint_tags('gt')) &
            (pi.weight != 0.0),
            "pilbl"]
        return gt_pi

    def get_par_change_limits(self):
        """calculate the various parameter change limits used in pest.


        Returns:
            `pandas.DataFrame`: a copy of `Pst.parameter_data`
            with columns for relative and factor change limits
        Note:

            does not yet support absolute parameter change limits!

            Works in control file values space (not log transformed space).  Also
            adds columns for effective upper and lower which account for par bounds and the
            value of parchglim

        example::

            pst = pyemu.Pst("pest.pst")
            df = pst.get_par_change_limits()
            print(df.chg_lower)

        """
        par = self.parameter_data
        fpars = par.loc[par.parchglim == "factor", "parnme"]
        rpars = par.loc[par.parchglim == "relative", "parnme"]
        # apars = par.loc[par.parchglim == "absolute", "parnme"]

        change_df = par.copy()

        fpm = self.control_data.facparmax
        rpm = self.control_data.relparmax
        facorig = self.control_data.facorig
        base_vals = par.parval1.copy()

        # apply zero value correction
        base_vals[base_vals == 0] = par.loc[base_vals == 0, "parubnd"] / 4.0

        # apply facorig
        replace_pars = base_vals.index.map(
            lambda x: par.loc[x, "partrans"] != "log"
            and np.abs(base_vals.loc[x]) < facorig * np.abs(base_vals.loc[x])
        )
        # print(facorig,replace_pars)
        base_vals.loc[replace_pars] = base_vals.loc[replace_pars] * facorig

        # negative fac pars
        nfpars = par.loc[base_vals.apply(lambda x: x < 0)].index
        change_df.loc[nfpars, "fac_upper"] = base_vals / fpm
        change_df.loc[nfpars, "fac_lower"] = base_vals * fpm

        # postive fac pars
        pfpars = par.loc[base_vals.apply(lambda x: x > 0)].index
        change_df.loc[pfpars, "fac_upper"] = base_vals * fpm
        change_df.loc[pfpars, "fac_lower"] = base_vals / fpm

        # relative

        rdelta = base_vals.apply(np.abs) * rpm
        change_df.loc[:, "rel_upper"] = base_vals + rdelta
        change_df.loc[:, "rel_lower"] = base_vals - rdelta

        change_df.loc[:, "chg_upper"] = np.NaN
        change_df.loc[fpars, "chg_upper"] = change_df.fac_upper[fpars]
        change_df.loc[rpars, "chg_upper"] = change_df.rel_upper[rpars]
        change_df.loc[:, "chg_lower"] = np.NaN
        change_df.loc[fpars, "chg_lower"] = change_df.fac_lower[fpars]
        change_df.loc[rpars, "chg_lower"] = change_df.rel_lower[rpars]

        # effective limits
        change_df.loc[:, "eff_upper"] = change_df.loc[:, ["parubnd", "chg_upper"]].min(
            axis=1
        )
        change_df.loc[:, "eff_lower"] = change_df.loc[:, ["parlbnd", "chg_lower"]].max(
            axis=1
        )

        return change_df

    def get_adj_pars_at_bounds(self, frac_tol=0.01):
        """get list of adjustable parameter at/near bounds

        Args:
            frac_tol ('float`): fractional tolerance of distance to bound.  For upper bound,
                the value `parubnd * (1-frac_tol)` is used, lower bound uses `parlbnd * (1.0 + frac_tol)`

        Returns:
            tuple containing:

            - **[`str`]**: list of parameters at/near lower bound
            - **[`str`]**: list of parameters at/near upper bound

        Example::

            pst = pyemu.Pst("pest.pst")
            at_lb,at_ub = pst.get_adj_pars_at_bounds()
            print("pars at lower bound",at_lb)

        """

        par = self.parameter_data.loc[self.adj_par_names, :].copy()
        over_ub = par.loc[
            par.apply(lambda x: x.parval1 >= (1.0 - frac_tol) * x.parubnd, axis=1),
            "parnme",
        ].tolist()
        under_lb = par.loc[
            par.apply(lambda x: x.parval1 <= (1.0 + frac_tol) * x.parlbnd, axis=1),
            "parnme",
        ].tolist()

        return under_lb, over_ub

    def try_parse_name_metadata(self):
        """try to add meta data columns to parameter and observation data based on
        item names.  Used with the PstFrom process.

        Note:
            metadata is identified in key-value pairs that are separated by a colon.
            each key-value pair is separated from others by underscore

            This works with PstFrom style long names

            This method is called programmtically during `Pst.load()`

        """
        par = self.parameter_data
        obs = self.observation_data
        par_cols = pst_utils.pst_config["par_fieldnames"]
        obs_cols = pst_utils.pst_config["obs_fieldnames"]

        if "longname" in par.columns:
            partg = "longname"
        else:
            partg = "parnme"
        if "longname" in obs.columns:
            obstg = "longname"
        else:
            obstg = "obsnme"

        for df, name, fieldnames in zip(
            [par, obs], [partg, obstg], [par_cols, obs_cols]
        ):
            try:
                meta_dict = df.loc[:, name].apply(
                    lambda x: dict(
                        [item.split(":") for item in x.split("_") if ":" in item]
                    )
                )
                unique_keys = []
                for k, v in meta_dict.items():
                    for kk, vv in v.items():
                        if kk not in fieldnames and kk not in unique_keys:
                            unique_keys.append(kk)
                for uk in unique_keys:
                    if uk not in df.columns:
                        df.loc[:, uk] = np.NaN
                    df.loc[:, uk] = meta_dict.apply(lambda x: x.get(uk, np.NaN))
            except Exception as e:
                print("error parsing metadata from '{0}', continuing".format(name))

    def rename_parameters(self, name_dict, pst_path=".", tplmap=None):
        """rename parameters in the control and template files

        Args:
            name_dict (`dict`): mapping of current to new names.
            pst_path (str): the path to the control file from where python
                is running.  Default is "." (python is running in the
                same directory as the control file)

        Note:
            no attempt is made to maintain the length of the marker strings
            in the template files, so if your model is sensitive
            to changes in spacing in the template file(s), this
            is not a method for you

            This does a lot of string compare, so its gonna be slow as...

         Example::

            pst = pyemu.Pst(os.path.join("template","pest.pst"))
            name_dict = {"par1":"par1_better_name"}
            pst.rename_parameters(name_dict,pst_path="template")



        """

        missing = set(name_dict.keys()) - set(self.par_names)
        if len(missing) > 0:
            raise Exception(
                "Pst.rename_parameters(): the following parameters in 'name_dict'"
                + " are not in the control file:\n{0}".format(",".join(missing))
            )

        par = self.parameter_data
        par.loc[:, "parnme"] = par.parnme.apply(lambda x: name_dict.get(x, x))
        par.index = par.parnme.values

        for idx, eq in zip(
            self.prior_information.index, self.prior_information.equation
        ):
            for old, new in name_dict.items():
                eq = eq.replace(old, new)
            self.prior_information.loc[idx, "equation"] = eq

        # pad for putting to tpl
        name_dict = {k: v.center(12) for k, v in name_dict.items()}
        filelist = self.model_input_data.pest_file
        _replace_str_in_files(filelist, name_dict, file_obsparmap=tplmap,
                              pst_path=pst_path)


    def rename_observations(self, name_dict, pst_path=".", insmap=None):
        """rename observations in the control and instruction files

        Args:
            name_dict (`dict`): mapping of current to new names.
            pst_path (str): the path to the control file from where python
                is running.  Default is "." (python is running in the
                same directory as the control file)

        Note:
            This does a lot of string compare, so its gonna be slow as...

         Example::

            pst = pyemu.Pst(os.path.join("template","pest.pst"))
            name_dict = {"obs1":"obs1_better_name"}
            pst.rename_observations(name_dict,pst_path="template")



        """

        missing = set(name_dict.keys()) - set(self.obs_names)
        if len(missing) > 0:
            raise Exception(
                "Pst.rename_observations(): the following observations in 'name_dict'"
                + " are not in the control file:\n{0}".format(",".join(missing))
            )

        obs = self.observation_data
        obs.loc[:, "obsnme"] = obs.obsnme.apply(lambda x: name_dict.get(x, x))
        obs.index = obs.obsnme.values
        _replace_str_in_files(self.model_output_data.pest_file, name_dict,
                              file_obsparmap=insmap, pst_path=pst_path)


def _replace_str_in_files(filelist, name_dict, file_obsparmap=None, pst_path='.'):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(
            processes=min(os.cpu_count()-1, 60)) as pool:
        res = []
        for fname in filelist:
            sys_fname = os.path.join(
                pst_path,
                str(fname).replace("/", os.path.sep).replace("\\", os.path.sep)
            )
            if not os.path.exists(sys_fname):
                warnings.warn(
                    "template/instruction file '{0}' not found, continuing...",
                    PyemuWarning
                )
                continue
            if file_obsparmap is not None:
                if sys_fname not in file_obsparmap.keys():
                    continue
                sub_name_dict = {v: name_dict[v]
                                 for v in file_obsparmap[sys_fname]}
                rex = None
            else:
                sub_name_dict = name_dict
                trie = pyemu.helpers.Trie()
                [trie.add(onme) for onme in name_dict.keys()]
                rex = re.compile(trie.pattern())
            # _multiprocess_obspar_rename(sys_fname, sub_name_dict, rex)
            res.append(pool.apply_async(_multiprocess_obspar_rename,
                                        args=(sys_fname, sub_name_dict, rex)))
        [r.get for r in res]
        pool.close()
        pool.join()


def _multiprocess_obspar_rename(sys_file, map_dict, rex=None):
    print(f"    find/replace long->short in {sys_file}")
    t0 = time.time()
    _multiprocess_obspar_rename_v3(sys_file, map_dict, rex=rex)
    # with open(sys_file, "rt") as f:
    #     nl = len(f.readlines())
    # np = len(map_dict)
    # if rex is None:
    #     if np > 1e6:  # regex compile might be the major slowdown
    #         _multiprocess_obspar_rename_v0(sys_file, map_dict)
    #     elif nl > 100:  # favour line by line to conserve mem
    #         _multiprocess_obspar_rename_v2(sys_file, map_dict, rex)
    #     else: # read and replace whole file
    #         _multiprocess_obspar_rename_v1(sys_file, map_dict, rex)
    # else:
    #     if nl > 100:  # favour line by line to conserve mem
    #         _multiprocess_obspar_rename_v2(sys_file, map_dict, rex)
    #     else:  # read and replace whole file
    #         _multiprocess_obspar_rename_v1(sys_file, map_dict, rex)
    shutil.copy(sys_file+".tmp", sys_file)
    os.remove(sys_file+".tmp")
    print(f"    find/replace long->short in {sys_file}... "
          f"took {time.time()-t0: .2f} s")


# def _multiprocess_obspar_rename_v0(sys_file, map_dict):
#     # memory intensive when file is big
#     # slow when file is big & when map_dict is long
#     # although maybe less slow than v1 and v2 when map_dict is the same across
#     # files - unless rex is precompiled outside mp call
#     with open(sys_file, "rt") as f:
#         x = f.read()
#     with open(sys_file+".tmp", "wt") as f:
#         for old in sorted(map_dict.keys(), key=len, reverse=True):
#             x = x.replace(old, map_dict[old])
#         f.write(x)


# def _multiprocess_obspar_rename_v1(sys_file, map_dict, rex=None):
#     # memory intensive as whole file is read into memory
#     # maybe faster than v2 when file is big but map_dict is relativly small
#     # but look out for memory
#     if rex is None:
#         rex = re.compile("|".join(
#             map(re.escape, sorted(map_dict.keys(), key=len, reverse=True))))
#     with open(sys_file, "rt") as f:
#         x = f.read()
#     with open(sys_file+".tmp", "wt") as f:
#         f.write(rex.sub(lambda s: map_dict[s.group()], x))


# def _multiprocess_obspar_rename_v2(sys_file, map_dict, rex=None):
#     # line by line
#     if rex is None:
#         rex = re.compile("|".join(
#             map(re.escape, sorted(map_dict.keys(), key=len, reverse=True))))
#     with open(sys_file, "rt") as f, open(sys_file+'.tmp', 'w') as fo:
#         for line in f:
#             fo.write(rex.sub(lambda s: map_dict[s.group()], line))


def _multiprocess_obspar_rename_v3(sys_file, map_dict, rex=None):
    # build a trie for rapid regex interaction,
    if rex is None:
        trie = pyemu.helpers.Trie()
        _ = [trie.add(word) for word in map_dict.keys()]
        rex = re.compile(trie.pattern())
    with open(sys_file, "rt") as f:
        x = f.read()
    with open(sys_file + ".tmp", "wt") as f:
        f.write(rex.sub(lambda s: map_dict[s.group()], x))
