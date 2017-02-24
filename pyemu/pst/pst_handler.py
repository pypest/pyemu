from __future__ import print_function, division
import os
import re
import copy
import warnings
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
from pyemu.pst.pst_controldata import ControlData
from pyemu.pst import pst_utils

class Pst(object):
    """basic class for handling pest control files to support linear analysis
    as well as replicate some of the functionality of the pest utilities
    """
    def __init__(self, filename, load=True, resfile=None):
        """constructor of pst object
        Parameters:
        ----------
            filename : [str] pest control file name
            load : [bool] flag for loading
            resfile : [str] residual filename
        Returns:
        -------
            None
        """

        self.filename = filename
        self.resfile = resfile
        self.__res = None

        for key,value in pst_utils.pst_config.items():
            self.__setattr__(key,copy.copy(value))
        self.tied = None
        self.control_data = ControlData()

        if load:
            assert os.path.exists(filename),\
                "pst file not found:{0}".format(filename)
            self.load(filename)

    @property
    def phi(self):
        """get the weighted total objective function
        """
        sum = 0.0
        for grp, contrib in self.phi_components.items():
            sum += contrib
        return sum

    @property
    def phi_components(self):
        """ get the individual components of the total objective function
        Parameters:
        ----------
            None
        Returns:
        -------
            Dict{observation group : contribution}
        """

        # calculate phi components for each obs group
        components = {}
        ogroups = self.observation_data.groupby("obgnme").groups
        rgroups = self.res.groupby("group").groups
        for og in ogroups.keys():
            assert og in rgroups.keys(),"Pst.adjust_weights_res() obs group " +\
                "not found: " + str(og)
            og_res_df = self.res.ix[rgroups[og]]
            og_res_df.index = og_res_df.name
            og_df = self.observation_data.ix[ogroups[og]]
            og_df.index = og_df.obsnme
            og_res_df = og_res_df.loc[og_df.index,:]
            assert og_df.shape[0] == og_res_df.shape[0],\
            " Pst.phi_components error: group residual dataframe row length" +\
            "doesn't match observation data group dataframe row length" + \
                str(og_df.shape) + " vs. " + str(og_res_df.shape)
            components[og] = np.sum((og_res_df["residual"] *
                                     og_df["weight"]) ** 2)
        if not self.control_data.pestmode.startswith("reg") and \
            self.prior_information.shape[0] > 0:
            ogroups = self.prior_information.groupby("obgnme").groups
            for og in ogroups.keys():
                assert og in rgroups.keys(),"Pst.adjust_weights_res() obs group " +\
                    "not found: " + str(og)
                og_res_df = self.res.ix[rgroups[og]]
                og_res_df.index = og_res_df.name
                og_df = self.prior_information.ix[ogroups[og]]
                og_df.index = og_df.pilbl
                og_res_df = og_res_df.loc[og_df.index,:]
                assert og_df.shape[0] == og_res_df.shape[0],\
                " Pst.phi_components error: group residual dataframe row length" +\
                "doesn't match observation data group dataframe row length" + \
                    str(og_df.shape) + " vs. " + str(og_res_df.shape)
                components[og] = np.sum((og_res_df["residual"] *
                                         og_df["weight"]) ** 2)

        return components

    @property
    def phi_components_normalized(self):
        """ get the individual components of the total objective function
            normalized to the total PHI being 1.0
        Args:
            None
        Returns:
            Dict{observation group : normalized contribution}
        Raises:
            Assertion error if self.observation_data groups don't match
            self.res groups

        """
        # use a dictionary comprehension to go through and normalize each component of phi to the total
        phi_components_normalized = {i: self.phi_components[i]/self.phi for i in self.phi_components}
        return phi_components_normalized

    def set_res(self,res):
        self.__res = res

    @property
    def res(self):
        """get the residuals dataframe
        """
        if self.__res is not None:
            return self.__res
        else:
            if self.resfile is not None:
                assert os.path.exists(self.resfile),"Pst.res: self.resfile " +\
                    str(self.resfile) + " does not exist"
            else:
                self.resfile = self.filename.replace(".pst", ".res")
                if not os.path.exists(self.resfile):
                    self.resfile = self.resfile.replace(".res", ".rei")
                    if not os.path.exists(self.resfile):
                        raise Exception("Pst.res: " +
                                        "could not residual file case.res" +
                                        " or case.rei")

            res = pst_utils.read_resfile(self.resfile)
            missing_bool = self.observation_data.obsnme.apply\
                (lambda x: x not in res.name)
            missing = self.observation_data.obsnme[missing_bool]
            if missing.shape[0] > 0:
                raise Exception("Pst.res: the following observations " +
                                "were not found in " +
                                "{0}:{1}".format(self.resfile,','.join(missing)))
            self.__res = res
            return self.__res

    @property
    def nprior(self):
        """number of prior information equations
        """
        self.control_data.nprior = self.prior_information.shape[0]
        return self.control_data.nprior


    @property
    def par_data(self):
        """method to access parameter_data
        """
        warnings.warn("decorated property 'par_data' is deprecated because "+
                      "it doesn't propagate changes back 'parameter_data'")
        return self.parameter_data

    @property
    def obs_data(self):
        """method to access observation_data
        """
        warnings.warn("decorated property 'obs_data' is deprecated because "+
                      "it doesn't propagate changes back 'observation_data'")
        return self.observation_data

    @property
    def nnz_obs(self):
        nnz = 0
        for w in self.observation_data.weight:
            if w > 0.0:
                nnz += 1
        return nnz


    @property
    def nobs(self):
        """number of observations
        """
        self.control_data.nobs = self.observation_data.shape[0]
        return self.control_data.nobs


    @property
    def npar_adj(self):
        """number of adjustable parameters
        """
        pass
        np = 0
        for t in self.parameter_data.partrans:
            if t not in ["fixed", "tied"]:
                np += 1
        return np


    @property
    def npar(self):
        """number of parameters
        """
        self.control_data.npar = self.parameter_data.shape[0]
        return self.control_data.npar


    @property
    def obs_groups(self):
        """observation groups
        """
        og = list(self.observation_data.groupby("obgnme").groups.keys())
        #og = list(map(pst_utils.SFMT, og))
        return og


    @property
    def par_groups(self):
        """parameter groups
        """
        pass
        return list(self.parameter_data.groupby("pargp").groups.keys())


    @property
    def prior_groups(self):
        """prior info groups
        """
        og = list(self.prior_information.groupby("obgnme").groups.keys())
        #og = list(map(pst_utils.SFMT, og))
        return og

    @property
    def prior_names(self):
        return list(self.prior_information.groupby("pilbl").groups.keys())

    @property
    def par_names(self):
        """parameter names
        """
        return list(self.parameter_data.parnme.values)

    @property
    def adj_par_names(self):
        adj_names = []
        for t,n in zip(self.parameter_data.partrans,
                       self.parameter_data.parnme):
            if t.lower() not in ["tied","fixed"]:
                adj_names.append(n)
        return adj_names

    @property
    def obs_names(self):
        """observation names
        """
        pass
        return list(self.observation_data.obsnme.values)

    @property
    def nnz_obs_names(self):
        """non-zero weight obs names
        """
        nz_names = []
        for w,n in zip(self.observation_data.weight,
                       self.observation_data.obsnme):
            if w > 0.0:
                nz_names.append(n)
        return nz_names

    @property
    def zero_weight_obs_names(self):
        self.observation_data.index = self.observation_data.obsnme
        groups = self.observation_data.groupby(
                self.observation_data.weight.apply(lambda x: x==0.0)).groups
        if True in groups:
            return list(self.observation_data.loc[groups[True],"obsnme"])
        else:
            return []

    @property
    def regul_section(self):
        phimlim = float(self.nnz_obs)
        #sect = "* regularisation\n"
        sect = "{0:15.6E} {1:15.6E}\n".format(phimlim, phimlim*1.15)
        sect += "1.0 1.0e-10 1.0e10 linreg continue\n"
        sect += "1.3  1.0e-2  1\n"
        return sect

    @property
    def estimation(self):
        if self.control_data.pestmode == "estimation":
            return True
        return False

    @staticmethod
    def _read_df(f,nrows,names,converters,defaults=None):
        seek_point = f.tell()
        df = pd.read_csv(f, header=None,names=names,
                              nrows=nrows,delim_whitespace=True,
                              converters=converters, index_col=False)

        # in case there was some extra junk at the end of the lines
        if df.shape[1] > len(names):
            df = df.iloc[:,len(names)]
            df.columns = names

        if defaults is not None:
            for name in names:
                df.loc[:,name] = df.loc[:,name].fillna(defaults[name])
        elif np.any(pd.isnull(df)):
            raise Exception("NANs found")
        f.seek(seek_point)
        [f.readline() for _ in range(nrows)]
        return df

    def load(self, filename):
        """load the pest control file
        Parameters:
        ----------
            filename : str
                pst filename
        Returns:
        -------
            None
        """

        f = open(filename, 'r')
        f.readline()

        #control section
        line = f.readline()
        assert "* control data" in line,\
            "Pst.load() error: looking for control" +\
            " data section, found:" + line
        control_lines = []
        while True:
            line = f.readline()
            if line == '':
                raise Exception("Pst.load() EOF while " +\
                                "reading control data section")
            if line.startswith('*'):
                break
            control_lines.append(line)
        self.control_data.parse_values_from_lines(control_lines)

        #anything between control data and parameter groups
        while True:
            if line == '':
                raise Exception("EOF before parameter groups section found")
            if "* parameter groups" in line.lower():
                break
            self.other_lines.append(line)
            line = f.readline()
        try:
            self.parameter_groups = self._read_df(f,self.control_data.npargp,
                                                  self.pargp_fieldnames,
                                                  self.pargp_converters,
                                                  self.pargp_defaults)
            self.parameter_groups.index = self.parameter_groups.pargpnme
        except Exception as e:
            raise Exception("Pst.load() error reading parameter groups: {0}".format(str(e)))

        #parameter data
        line = f.readline()
        assert "* parameter data" in line.lower(),\
            "Pst.load() error: looking for parameter" +\
            " data section, found:" + line
        try:
            self.parameter_data = self._read_df(f,self.control_data.npar,
                                                self.par_fieldnames,
                                                self.par_converters,
                                                self.par_defaults)
            self.parameter_data.index = self.parameter_data.parnme
        except:
            raise Exception("Pst.load() error reading parameter data")

        # oh the tied parameter bullshit, how do I hate thee
        counts = self.parameter_data.partrans.value_counts()
        if "tied" in counts.index:
            # tied_lines = [f.readline().lower().strip().split() for _ in range(counts["tied"])]
            # self.tied = pd.DataFrame(tied_lines,columns=["parnme","partied"])
            # self.tied.index = self.tied.pop("parnme")
            self.tied = self._read_df(f,counts["tied"],self.tied_fieldnames,
                                      self.tied_converters)
            self.tied.index = self.tied.parnme
        # obs groups - just read past for now
        line = f.readline()
        assert "* observation groups" in line.lower(),\
            "Pst.load() error: looking for obs" +\
            " group section, found:" + line
        [f.readline() for _ in range(self.control_data.nobsgp)]

        # observation data
        line = f.readline()
        assert "* observation data" in line.lower(),\
            "Pst.load() error: looking for observation" +\
            " data section, found:" + line
        if self.control_data.nobs > 0:
            try:
                self.observation_data = self._read_df(f,self.control_data.nobs,
                                                      self.obs_fieldnames,
                                                      self.obs_converters)
                self.observation_data.index = self.observation_data.obsnme
            except:
                raise Exception("Pst.load() error reading observation data")
        else:
            raise Exception("nobs == 0")
        #model command line
        line = f.readline()
        assert "* model command line" in line.lower(),\
            "Pst.load() error: looking for model " +\
            "command section, found:" + line
        for i in range(self.control_data.numcom):
            self.model_command.append(f.readline().strip())

        #model io
        line = f.readline()
        assert "* model input/output" in line.lower(), \
            "Pst.load() error; looking for model " +\
            " i/o section, found:" + line
        for i in range(self.control_data.ntplfle):
            raw = f.readline().strip().split()
            self.template_files.append(raw[0])
            self.input_files.append(raw[1])
        for i in range(self.control_data.ninsfle):
            raw = f.readline().strip().split()
            self.instruction_files.append(raw[0])
            self.output_files.append(raw[1])

        #prior information - sort of hackish
        if self.control_data.nprior == 0:
            self.prior_information = self.null_prior
        else:
            pilbl, obgnme, weight, equation = [], [], [], []
            line = f.readline()
            assert "* prior information" in line.lower(), \
                "Pst.load() error; looking for prior " +\
                " info section, found:" + line
            for iprior in range(self.control_data.nprior):
                line = f.readline()
                if line == '':
                    raise Exception("EOF during prior information " +
                                    "section")
                raw = line.strip().split()
                pilbl.append(raw[0].lower())
                obgnme.append(raw[-1].lower())
                weight.append(float(raw[-2]))
                eq = ' '.join(raw[1:-2])
                equation.append(eq)
            self.prior_information = pd.DataFrame({"pilbl": pilbl,
                                                       "equation": equation,
                                                       "weight": weight,
                                                       "obgnme": obgnme})
            self.prior_information.index = self.prior_information.pilbl
        if "regul" in self.control_data.pestmode:
            line = f.readline()
            assert "* regul" in line.lower(), \
                "Pst.load() error; looking for regul " +\
                " section, found:" + line
            [self.regul_lines.append(f.readline()) for _ in range(3)]

        for line in f:
            if line.strip().startswith("++") and '#' not in line:
                args = line.replace('++','').strip().split()
                #args = ['++'+arg.strip() for arg in args]
                #self.pestpp_lines.extend(args)
                keys = [arg.split('(')[0] for arg in args]
                values = [arg.split('(')[1].replace(')','') for arg in args]
                for key,value in zip(keys,values):
                    if key in self.pestpp_options:
                        print("Pst.load() warning: duplicate pest++ option found:" + str(key))
                    self.pestpp_options[key] = value
        f.close()
        return


    def _update_control_section(self):

        self.control_data.npar = self.npar
        self.control_data.nobs = self.nobs
        self.control_data.npargp = self.parameter_groups.shape[0]
        self.control_data.nobsgp = self.observation_data.obgnme.\
            value_counts().shape[0] + self.prior_information.obgnme.\
            value_counts().shape[0]

        self.control_data.nprior = self.prior_information.shape[0]
        self.control_data.ntplfle = len(self.template_files)
        self.control_data.ninsfle = len(self.instruction_files)


    def _rectify_pgroups(self):
        # add any parameters groups
        pdata_groups = list(self.parameter_data.loc[:,"pargp"].\
            value_counts().keys())
        #print(pdata_groups)
        need_groups = []
        existing_groups = list(self.parameter_groups.pargpnme)
        for pg in pdata_groups:
            if pg not in existing_groups:
                need_groups.append(pg)
        if len(need_groups) > 0:
            print(need_groups)
            defaults = copy.copy(pst_utils.pst_config["pargp_defaults"])
            for grp in need_groups:
                defaults["pargpnme"] = grp
                self.parameter_groups = \
                    self.parameter_groups.append(defaults,ignore_index=True)

        # now drop any left over groups that aren't needed
        for gp in self.parameter_groups.loc[:,"pargpnme"]:
            if gp in pdata_groups and gp not in need_groups:
                need_groups.append(gp)
        self.parameter_groups.index = self.parameter_groups.pargpnme
        self.parameter_groups = self.parameter_groups.loc[need_groups,:]


    def _parse_pi_par_names(self):

        if self.prior_information.shape[0] == 0:
            return
        if "names" in self.prior_information.columns:
            self.prior_information.pop("names")
        if "rhs" in self.prior_information.columns:
            self.prior_information.pop("rhs")
        def parse(eqs):
            raw = eqs.split('=')
            rhs = float(raw[1])
            raw = re.split('[+,-]',raw[0].lower().strip())
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
            return [r.split('*')[1].replace("log(",'').replace(')','').strip() for r in raw if '*' in r]

        self.prior_information.loc[:,"names"] =\
            self.prior_information.equation.apply(lambda x: parse(x))



    def rectify_pi(self):
        if self.prior_information.shape[0] == 0:
            return
        self._parse_pi_par_names()
        adj_names = self.adj_par_names
        def is_good(names):
            for n in names:
                if n not in adj_names:
                    return False
            return True
        keep_idx = self.prior_information.names.\
            apply(lambda x: is_good(x))
        self.prior_information = self.prior_information.loc[keep_idx,:]

    def write(self,new_filename,update_regul=False):
        """write a pest control file
        Parameters:
        ----------
            new_filename (str) : name of the new pest control file
        Returns:
        -------
            None
        """


        self._rectify_pgroups()
        self.rectify_pi()
        self._update_control_section()

        f_out = open(new_filename, 'w')
        f_out.write("pcf\n* control data\n")
        self.control_data.write(f_out)

        for line in self.other_lines:
            f_out.write(line)

        f_out.write("* parameter groups\n")


        # to catch the byte code ugliness in python 3
        pargpnme = self.parameter_groups.loc[:,"pargpnme"].copy()
        self.parameter_groups.loc[:,"pargpnme"] = \
            self.parameter_groups.pargpnme.apply(self.pargp_format["pargpnme"])

        #self.parameter_groups.index = self.parameter_groups.pop("pargpnme")
        #gp_fieldnames = [name for name in self.pargp_fieldnames if name in self.parameter_groups.columns]
        f_out.write(self.parameter_groups.to_string(col_space=0,
                                                  formatters=self.pargp_format,
                                                  columns=self.pargp_fieldnames,
                                                  justify="right",
                                                  header=False,
                                                  index=False) + '\n')
        self.parameter_groups.loc[:,"pargpnme"] = pargpnme.values
        #self.parameter_groups.index = pargpnme

        f_out.write("* parameter data\n")
        #self.parameter_data.index = self.parameter_data.pop("parnme")
        f_out.write(self.parameter_data.to_string(col_space=0,
                                                  columns=self.par_fieldnames,
                                                  formatters=self.par_format,
                                                  justify="right",
                                                  header=False,
                                                  index=False) + '\n')
        #self.parameter_data.loc[:,"parnme"] = self.parameter_data.index

        if self.tied is not None:
            #self.tied.index = self.tied.pop("parnme")
            f_out.write(self.tied.to_string(col_space=0,
                                            columns=self.tied_fieldnames,
                                            formatters=self.tied_format,
                                            justify='right',
                                            header=False,
                                            index=False)+'\n')
            #self.tied.loc[:,"parnme"] = self.tied.index
        f_out.write("* observation groups\n")
        for group in self.obs_groups:
            try:
                group = group.decode()
            except:
                pass
            f_out.write(pst_utils.SFMT(str(group))+'\n')
        for group in self.prior_groups:
            try:
                group = group.decode()
            except:
                pass
            f_out.write(pst_utils.SFMT(str(group))+'\n')

        f_out.write("* observation data\n")
        #self.observation_data.index = self.observation_data.pop("obsnme")
        f_out.write(self.observation_data.to_string(col_space=0,
                                                  formatters=self.obs_format,
                                                  columns=self.obs_fieldnames,
                                                  justify="right",
                                                  header=False,
                                                  index=False) + '\n')
        #self.observation_data.loc[:,"obsnme"] = self.observation_data.index

        f_out.write("* model command line\n")
        for cline in self.model_command:
            f_out.write(cline+'\n')

        f_out.write("* model input/output\n")
        for tplfle,infle in zip(self.template_files,self.input_files):
            f_out.write(tplfle+' '+infle+'\n')
        for insfle,outfle in zip(self.instruction_files,self.output_files):
            f_out.write(insfle+' '+outfle+'\n')

        if self.nprior > 0:
            f_out.write("* prior information\n")
            #self.prior_information.index = self.prior_information.pop("pilbl")
            max_eq_len = self.prior_information.equation.apply(lambda x:len(x)).max()
            eq_fmt_str =  " {0:<" + str(max_eq_len) + "s} "
            eq_fmt_func = lambda x:eq_fmt_str.format(x)
            #  17/9/2016 - had to go with a custom writer loop b/c pandas doesn't want to
            # output strings longer than 100, even with display.max_colwidth
            #f_out.write(self.prior_information.to_string(col_space=0,
            #                                  columns=self.prior_fieldnames,
            #                                  formatters=pi_formatters,
            #                                  justify="right",
            #                                  header=False,
            #                                 index=False) + '\n')
            #self.prior_information["pilbl"] = self.prior_information.index
            for idx,row in self.prior_information.iterrows():
                f_out.write(pst_utils.SFMT(row["pilbl"]))
                f_out.write(eq_fmt_func(row["equation"]))
                f_out.write(pst_utils.FFMT(row["weight"]))
                f_out.write(pst_utils.SFMT(row["obgnme"]) + '\n')
        if self.control_data.pestmode.startswith("regul"):
            f_out.write("* regularisation\n")
            if update_regul or len(self.regul_lines) == 0:
                f_out.write(self.regul_section)
            else:
                [f_out.write(line) for line in self.regul_lines]


        for key,value in self.pestpp_options.items():
            f_out.write("++{0}({1})\n".format(str(key),str(value)))

        f_out.close()


    def get(self, par_names=None, obs_names=None):
        """get a new pst object with subset of parameters and observations
        Args:
            par_names (list of str) : parameter names
            obs_names (list of str) : observation names
        Returns:
            new pst instance
        Raises:
            None
        """
        pass
        #if par_names is None and obs_names is None:
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

        new_pargp = self.parameter_groups.copy()
        new_pargp.index = new_pargp.pargpnme
        new_pargp_names = new_par.pargp.value_counts().index
        new_pargp = new_pargp.loc[new_pargp_names,:]

        new_pst = Pst(self.filename, resfile=self.resfile, load=False)
        new_pst.parameter_data = new_par
        new_pst.observation_data = new_obs
        new_pst.parameter_groups = new_pargp
        new_pst.__res = new_res
        new_pst.prior_information = self.prior_information
        new_pst.rectify_pi()
        new_pst.control_data = self.control_data.copy()

        new_pst.model_command = self.model_command
        new_pst.template_files = self.template_files
        new_pst.input_files = self.input_files
        new_pst.instruction_files = self.instruction_files
        new_pst.output_files = self.output_files

        if self.tied is not None:
            print("Pst.get() warning: not checking for tied parameter " +
                  "compatibility in new Pst instance")
            new_pst.tied = self.tied.copy()
        new_pst.other_lines = self.other_lines
        new_pst.pestpp_options = self.pestpp_options
        new_pst.regul_lines = self.regul_lines

        return new_pst

    def zero_order_tikhonov(self,parbounds=True):
        raise Exception("Pst.zero_oder_tikhonov has moved to utils.helpers")


    def parrep(self, parfile=None):
        """replicates the pest parrep util. replaces the parval1 field in the
            parameter data section dataframe
        Parameters:
        ----------
            parfile (str) : parameter file to use.  If None, try to use
                            a parameter file that corresponds to the case name
        Returns:
        -------
            None
        """
        if parfile is None:
            parfile = self.filename.replace(".pst", ".par")
        par_df = pst_utils.read_parfile(parfile)
        self.parameter_data.index = self.parameter_data.parnme
        par_df.index = par_df.parnme
        self.parameter_data.parval1 = par_df.parval1
        self.parameter_data.scale = par_df.scale
        self.parameter_data.offset = par_df.offset



    def adjust_weights_recfile(self, recfile=None):
        """adjusts the weights of the observations based on the phi components
        in a recfile
        Parameters:
        ----------
            recfile (str) : record file name.  If None, try to use a record file
                            with the case name
        Returns:
        -------
            None
        """
        if recfile is None:
            recfile = self.filename.replace(".pst", ".rec")
        assert os.path.exists(recfile), \
            "Pst.adjust_weights_recfile(): recfile not found: " +\
            str(recfile)
        iter_components = pst_utils.get_phi_comps_from_recfile(recfile)
        iters = iter_components.keys()
        iters.sort()
        obs = self.observation_data
        ogroups = obs.groupby("obgnme").groups
        last_complete_iter = None
        for ogroup, idxs in ogroups.iteritems():
            for iiter in iters[::-1]:
                incomplete = False
                if ogroup not in iter_components[iiter]:
                    incomplete = True
                    break
                if not incomplete:
                    last_complete_iter = iiter
                    break
        if last_complete_iter is None:
            raise Exception("Pst.pwtadj2(): no complete phi component" +
                            " records found in recfile")
        self._adjust_weights_by_phi_components(
            iter_components[last_complete_iter])

    def adjust_weights_resfile(self, resfile=None):
        """adjust the weights by phi components in a residual file
        Parameters:
        ----------
            resfile (str) : residual filename.  If None, use self.resfile
        Returns:
        -------
            None
        """
        if resfile is not None:
            self.resfile = resfile
            self.__res = None
        phi_comps = self.phi_components
        self._adjust_weights_by_phi_components(phi_comps)

    def _adjust_weights_by_phi_components(self, components):
        """resets the weights of observations to account for
        residual phi components.
        Parameters:
        ----------
            components (dict{obs group:phi contribution}): group specific phi
                contributions
        Returns:
        -------
            None
        """
        obs = self.observation_data
        nz_groups = obs.groupby(obs["weight"].map(lambda x: x == 0)).groups
        ogroups = obs.groupby("obgnme").groups
        for ogroup, idxs in ogroups.items():
            if self.control_data.pestmode.startswith("regul") \
                    and "regul" in ogroup.lower():
                continue
            og_phi = components[ogroup]
            nz_groups = obs.loc[idxs,:].groupby(obs.loc[idxs,"weight"].\
                                                map(lambda x: x == 0)).groups
            og_nzobs = 0
            if False in nz_groups.keys():
                og_nzobs = len(nz_groups[False])
            if og_nzobs == 0 and og_phi > 0:
                raise Exception("Pst.adjust_weights_by_phi_components():"
                                " no obs with nonzero weight," +
                                " but phi > 0 for group:" + str(ogroup))
            if og_phi > 0:
                factor = np.sqrt(float(og_nzobs) / float(og_phi))
                obs.loc[idxs,"weight"] = obs.weight[idxs] * factor
        self.observation_data = obs

    def __reset_weights(self, target_phis, res_idxs, obs_idxs):
        """reset weights based on target phi vals for each group
        Parameters:
        ----------
            target_phis (dict) : target phi contribution for groups to reweight
            res_idxs (dict) : the index positions of each group of interest
                 in the res dataframe
            obs_idxs (dict) : the index positions of each group of interest
                in the observation data dataframe
        """

        for item in target_phis.keys():
            assert item in res_idxs.keys(),\
                "Pst.__reset_weights(): " + str(item) +\
                " not in residual group indices"
            assert item in obs_idxs.keys(), \
                "Pst.__reset_weights(): " + str(item) +\
                " not in observation group indices"
            actual_phi = ((self.res.loc[res_idxs[item], "residual"] *
                           self.observation_data.loc
                           [obs_idxs[item], "weight"])**2).sum()
            if actual_phi > 0.0:
                weight_mult = np.sqrt(target_phis[item] / actual_phi)
                self.observation_data.loc[obs_idxs[item], "weight"] *= weight_mult
            else:
                print("Pst.__reset_weights() warning: phi group {0} has zero phi, skipping...".format(item))

    def adjust_weights_by_list(self,obslist,weight):
        """apply a single weight to a list of obsevation names.  supports that
        data worth analyses in Schur
        Parameters:
        ----------
            obslist : list of obseravtion names
            weight : (flaot) new weight to assign
        """

        obs = self.observation_data
        obs = self.observation_data
        if not isinstance(obslist,list):
            obslist = [obslist]
        obslist = [str(i).lower() for i in obslist]
        groups = obs.groupby([lambda x:x in obslist,
                             obs.weight.apply(lambda x:x==0.0)]).groups
        if (True,True) in groups:
            obs.loc[groups[True,True],"weight"] = weight

    def adjust_weights(self,obs_dict=None,
                              obsgrp_dict=None):
        """reset the weights of observation groups to contribute a specified
        amount to the composite objective function
        Parameters:
        ----------
            obs_dict (dict{obs name:new contribution})
            obsgrp_dict (dict{obs group name:contribution})
        Returns:
        -------
            None
        Note:
        ----
            if all obs in a named obs group have zero weight, they will be
            assigned a non-zero weight so that the request phi contribution
            can be met.  Similarly, any obs listed in obs_dict with zero
            weight will also be reset
        """

        self.observation_data.index = self.observation_data.obsnme
        self.res.index = self.res.name

        if obsgrp_dict is not None:
            # reset groups with all zero weights
            obs = self.observation_data
            for grp in obsgrp_dict.keys():
                if obs.loc[obs.obgnme==grp,"weight"].sum() == 0.0:
                    obs.loc[obs.obgnme==grp,"weight"] = 1.0
            res_groups = self.res.groupby("group").groups
            obs_groups = self.observation_data.groupby("obgnme").groups
            self.__reset_weights(obsgrp_dict, res_groups, obs_groups)
        if obs_dict is not None:
            # reset obs with zero weight
            obs = self.observation_data
            for oname in obs_dict.keys():
                if obs.loc[oname,"weight"] == 0.0:
                    obs.loc[oname,"weight"] = 1.0

            res_groups = self.res.groupby("name").groups
            obs_groups = self.observation_data.groupby("obsnme").groups
            self.__reset_weights(obs_dict, res_groups, obs_groups)


    def proportional_weights(self, fraction_stdev=1.0, wmax=100.0,
                             leave_zero=True):
        """setup inversely proportional weights
        Parameters:
        ----------
            fraction_stdev (float) : the fraction portion of the observation
                val to treat as the standard deviation.  set to 1.0 for
                inversely proportional
            wmax (float) : maximum weight to allow
            leave_zero (bool) : flag to leave existing zero weights
        Returns:
        -------
            None
        """
        new_weights = []
        for oval, ow in zip(self.observation_data.obsval,
                            self.observation_data.weight):
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

        self.build_increments()
        self.parameter_data.loc[:,"pertubation"] = \
            self.parameter_data.parval1 + \
            self.parameter_data.increment

        self.parameter_data.loc[:,"out_forward"] = \
            self.parameter_data.loc[:,"pertubation"] > \
            self.parameter_data.loc[:,"parubnd"]

        out_forward = self.parameter_data.groupby("out_forward").groups
        if True in out_forward:
            self.parameter_data.loc[out_forward[True],"pertubation"] = \
                    self.parameter_data.loc[out_forward[True],"parval1"] - \
                    self.parameter_data.loc[out_forward[True],"increment"]

            self.parameter_data.loc[:,"out_back"] = \
            self.parameter_data.loc[:,"pertubation"] < \
            self.parameter_data.loc[:,"parlbnd"]
            out_back = self.parameter_data.groupby("out_back").groups
            if True in out_back:
                still_out = out_back[True]
                print(self.parameter_data.loc[still_out,:],flush=True)

                raise Exception("Pst.calculate_pertubations(): " +\
                                "can't calc pertubations for the following "+\
                                "parameters: {0}".format(','.join(still_out)))

    def build_increments(self):
        self.enforce_bounds()

        par_groups = self.parameter_data.groupby("pargp").groups
        inctype = self.parameter_groups.groupby("inctyp").groups
        for itype,inc_groups in inctype.items():
            pnames = []
            for group in inc_groups:
                pnames.extend(par_groups[group])
                derinc = self.parameter_groups.loc[group,"derinc"]
                self.parameter_data.loc[par_groups[group],"derinc"] = derinc
            if itype == "absolute":
                self.parameter_data.loc[pnames,"increment"] = \
                    self.parameter_data.loc[pnames,"derinc"]
            elif itype == "relative":
                self.parameter_data.loc[pnames,"increment"] = \
                    self.parameter_data.loc[pnames,"derinc"] * \
                    self.parameter_data.loc[pnames,"parval1"]
            elif itype == "rel_to_max":
                mx = self.parameter_data.loc[pnames,"parval1"].max()
                self.parameter_data.loc[pnames,"increment"] = \
                    self.parameter_data.loc[pnames,"derinc"] * mx
            else:
                raise Exception('Pst.get_derivative_increments(): '+\
                                'unrecognized increment type:{0}'.format(itype))

        #account for fixed pars
        isfixed = self.parameter_data.partrans=="fixed"
        self.parameter_data.loc[isfixed,"increment"] = \
            self.parameter_data.loc[isfixed,"parval1"]

    def add_transform_columns(self):
        for col in ["parval1","parlbnd","parubnd","increment"]:
            if col not in self.parameter_data.columns:
                continue
            self.parameter_data.loc[:,col+"_trans"] = (self.parameter_data.parval1 *
                                                          self.parameter_data.scale) +\
                                                         self.parameter_data.offset
            isnotfixed = self.parameter_data.partrans != "fixed"
            self.parameter_data.loc[isnotfixed,col+"_trans"] = \
                self.parameter_data.loc[isnotfixed,col+"_trans"].\
                    apply(lambda x:np.log10(x))

    def enforce_bounds(self):
        too_big = self.parameter_data.loc[:,"parval1"] > \
            self.parameter_data.loc[:,"parubnd"]
        self.parameter_data.loc[too_big,"parval1"] = \
            self.parameter_data.loc[too_big,"parubnd"]

        too_small = self.parameter_data.loc[:,"parval1"] < \
            self.parameter_data.loc[:,"parlbnd"]
        self.parameter_data.loc[too_small,"parval1"] = \
            self.parameter_data.loc[too_small,"parlbnd"]




