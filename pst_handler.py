import os
import copy
import numpy as np
import pandas
pandas.options.display.max_colwidth=100

class pst(object):
    """basic class for handling pest control files to support linear analysis
    as well as replicate some of the functionality of the pest utilities
    """
    def __init__(self,filename, load=True, resfile=None):
        """constructor of pst object
        Args:
            filename : [str] pest control file name
            load : [bool] flag for loading
            resfile : [str] residual filename
        Returns:
            None
        Raises:
            Assertion error if filename cannot be found
        """
        pass
        self.null_prior = pandas.DataFrame({"pilbl": None,
                                            "obgnme": None}, index=[])
        self.filename = filename

        self.resfile = resfile
        self.__res = None

        self.sfmt = lambda x: "{0:>20s}".format(str(x))
        self.sfmt_long = lambda x: "{0:>50s}".format(str(x))
        self.ifmt = lambda x: "{0:>10d}".format(int(x))
        self.ffmt = lambda x: "{0:>15.6E}".format(float(x))

        self.par_dtype = np.dtype([("parnme", "a20"),("parval1", np.float),
                                   ("scale", np.float),("offset", np.float)])
        self.par_fieldnames = "PARNME PARTRANS PARCHGLIM PARVAL1 PARLBND " +\
                              "PARUBND PARGP SCALE OFFSET DERCOM"
        self.par_fieldnames = self.par_fieldnames.lower().strip().split()
        self.par_format = {"parnme": self.sfmt, "partrans": self.sfmt,
                           "parchglim": self.sfmt, "parval1": self.ffmt,
                           "parlbnd": self.ffmt, "parubnd": self.ffmt,
                           "pargp": self.sfmt, "scale": self.ffmt,
                           "offset": self.ffmt, "dercom": self.ifmt}
        self.par_converters = {"parnme": str.lower, "pargp": str.lower}
        self.obs_fieldnames = "OBSNME OBSVAL WEIGHT OBGNME".lower().split()
        self.obs_format = {"obsnme": self.sfmt, "obsval": self.ffmt,
                           "weight": self.ffmt, "obgnme": self.sfmt}
        self.obs_converters = {"obsnme": str.lower, "obgnme": str.lower}

        self.prior_format = {"pilbl": self.sfmt, "equation": self.sfmt_long,
                             "obgnme": self.sfmt, "weight": self.ffmt}

        if load:
            assert os.path.exists(filename)
            self.load(filename)


    @property
    def phi(self):
        """get the weighted total objective function
        """
        sum = 0.0
        for grp,contrib in self.phi_components.iteritems():
            sum += contrib
        return sum

    @property
    def phi_components(self):
        """ get the individual components of the total objective function
        Args:
            None
        Returns:
            Dict{observation group : contribution}
        Raises:
            Assertion error if self.observation_data groups don't match
            self.res groups

        """

        # calculate phi components for each obs group
        components = {}
        ogroups = self.observation_data.groupby("obgnme").groups
        rgroups = self.res.groupby("group").groups
        for og in ogroups.keys():
            assert og in rgroups.keys(),"pst.adjust_weights_res() obs group " +\
                "not found: " + str(og)
            og_res_df = self.res.ix[rgroups[og]]
            og_res_df.index = og_res_df.name
            og_df = self.observation_data.ix[ogroups[og]]
            og_df.index = og_df.obsnme
            assert og_df.shape[0] == og_res_df.shape[0],\
            " pst.phi_components error: group residual dataframe row lenght" +\
            "doesn't match observation data group dataframe row length" + \
                str(og_df.shape) + " vs. " + str(og_res_df.shape)
            components[og] = np.sum((og_res_df["residual"] *
                                     og_df["weight"]) ** 2)
        return components


    @property
    def res(self):
        """get the residuals dataframe
        """
        pass
        if self.__res is not None:
            return self.__res
        else:
            if self.resfile is None:
                self.resfile = self.filename.replace(".pst", ".res")
                if not os.path.exists(self.resfile):
                    self.resfile = self.resfile.replace(".res", ".rei")
                    if not os.path.exists(self.resfile):
                        raise Exception("pst.get_residuals: " +
                                        "could not residual file case.res" +
                                        " or case.rei")
            self.__res = self.load_resfile(self.resfile)
            return self.__res


    @property
    def nprior(self):
        """number of prior information equations
        """
        pass
        return self.prior_information.shape[0]


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
        pass
        return self.observation_data.shape[0]


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
        pass
        return self.parameter_data.shape[0]


    @property
    def obs_groups(self):
        """observation groups
        """
        pass
        return self.observation_data.groupby("obgnme").groups.keys()


    @property
    def par_groups(self):
        """parameter groups
        """
        pass
        return self.parameter_data.groupby("pargp").groups.keys()


    @property
    def prior_groups(self):
        """prior info groups
        """
        pass
        return self.prior_information.groupby("obgnme").groups.keys()


    @property
    def par_names(self):
        """parameter names
        """
        pass
        return list(self.parameter_data.parnme.values)


    @property
    def obs_names(self):
        """observation names
        """
        pass
        return list(self.observation_data.obsnme.values)


    def load_resfile(self,resfile):
        """load the residual file
        """
        pass
        converters = {"name": str.lower, "group": str.lower}
        f = open(resfile, 'r')
        while True:
            line = f.readline()
            if line == '':
                raise Exception("pst.get_residuals: EOF before finding "+
                                "header in resfile: " + resfile)
            if "name" in line.lower():
                header = line.lower().strip().split()
                break
        res_df = pandas.read_csv(f, header=None, names=header, sep="\s+",
                                 converters=converters)
        f.close()
        return res_df


    def load(self, filename):
        """load the pest control file
        """
        pass
        f = open(filename, 'r')
        f.readline()
        f.readline()
        line = f.readline()
        raw = line.strip().split()
        self.mode = raw[1].lower()
        if self.mode == "estimation":
            self.estimation = True
        else:
            self.estimation = False
        raw = f.readline().strip().split()
        npar, nobs, nprior = int(raw[0]), int(raw[1]), int(raw[3])
        f.close()
        f = open(filename, 'r')
        while True:
            line = f.readline()
            if line == '':
                raise Exception("EOF before parameter data section found")
            if "* parameter data" in line.lower():
                break
        par = pandas.read_csv(f, header=None, names=self.par_fieldnames,
                              nrows=npar,delimiter="\s+",
                              converters=self.par_converters)
        self.parameter_data = par
        f.close()
        f = open(filename, 'r')
        while True:
            line = f.readline()
            if line == '':
                raise Exception("EOF before obs data section found")
            if "* observation data" in line.lower():
                break
        obs = pandas.read_csv(f, header=None, names=self.obs_fieldnames,
                              nrows=nobs, delimiter="\s+",
                              converters=self.obs_converters)
        self.observation_data = obs
        f.close()
        if nprior == 0:
            self.prior_information = self.null_prior
        else:
            pilbl, obgnme, weight, equation = [], [], [], []
            f = open(filename,'r')
            while True:
                line = f.readline()
                if line == '':
                    raise Exception("EOF before prior information " +
                                    "section found")
                if "* prior information" in line.lower():
                    for iprior in xrange(nprior):
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
                    break
            f.close()
            self.prior_information = pandas.DataFrame({"pilbl": pilbl,
                                                       "equation": equation,
                                                       "obgnme": obgnme,
                                                       "weight": weight})
            return


    def write(self,new_filename):
        """write a pest control file
        Args:
            new_filename (str) : name of the new pest control file
        Returns:
            None
        Raises:
            Assertion error if tied parameters are found - not supported
            Exception if self.filename pst is not the correct format
        """
        pass
        assert "tied" not in self.parameter_data.partrans,\
            "tied parameters not supported in pst.write()"
        f_in = open(self.filename, 'r')
        f_out = open(new_filename, 'w')
        for _ in xrange(3):
            f_out.write(f_in.readline())
        raw = f_in.readline().strip().split()
        npar_gp = len(self.par_groups)
        nobs_gp = len(self.obs_groups) + len(self.prior_groups)
        line = "{0:7d} {1:7d} {2:7d} {3:7d} {4:7d}\n"\
            .format(self.npar, self.nobs, npar_gp, self.nprior, nobs_gp)
        f_out.write(line)

        while True:
            line = f_in.readline()
            if line == '':
                raise Exception("pst.write(): EOF found while searching " +
                                "for * parameter groups")
            f_out.write(line)
            if "* parameter groups" in line.lower():
                break
        par_groups_found = [False] * len(self.par_groups)
        while True:
            line = f_in.readline()
            if line == '':
                raise Exception("pst.write(): EOF found while searching " +
                                "for * parameter data")

            if "* parameter data" in line.lower():

                break
            else:
                pgrp = line.strip().split()[0]
                if pgrp in self.par_groups:
                    f_out.write(line)
                    par_groups_found[self.par_groups.index(pgrp)] = True

        for found, group in zip(par_groups_found, self.par_groups):
            if not found:
                f_out.write(group +
                            " relative  0.01 0.0 switch 2.0 parabolic\n")

        f_out.write(line)
        self.parameter_data.index = self.parameter_data.parnme
        self.parameter_data.pop("parnme")
        f_out.write(self.parameter_data.to_string(colSpace=0,
                                                  formatters=self.par_format,
                                                  justify="right",
                                                  header=False,
                                                  index_names=False) + '\n')
        self.parameter_data["parnme"] = self.parameter_data.index
        #--read f_in past parameter data
        while True:
            line = f_in.readline()
            if line == '':
                raise Exception("pst.write(): EOF while searching " +
                                "for * observation groups")
            if "* observation" in line.lower():
                f_out.write(line)
                break

        for group in self.obs_groups:
            f_out.write(group+'\n')
        for group in self.prior_groups:
            f_out.write(group+'\n')

        while True:
            line = f_in.readline()
            if line == '':
                raise Exception("pst.write(): EOF while searching " +
                                "for * observation data")
            #f_out.write(line)
            if "* observation" in line.lower():
                f_out.write(line)
                break
        self.observation_data.index = self.observation_data.obsnme
        self.observation_data.pop("obsnme")
        f_out.write(self.observation_data.to_string(colSpace=0,
                                                  formatters=self.obs_format,
                                                  justify="right",
                                                  header=False,
                                                  index_names=False) + '\n')
        self.observation_data["obsnme"] = self.observation_data.index

        #--read f_in past observation data
        while True:
            line = f_in.readline()
            if line == '':
                raise Exception("pst.write(): EOF while searching " +
                                "for * model command line")
            if "* model" in line.lower():
                f_out.write(line)
                break
        while True:
            line = f_in.readline()
            if line == '' or "* prior" in line.lower():
                break
            f_out.write(line)
        if self.nprior > 0:
            f_out.write("* prior information\n")
            self.prior_information.index = self.prior_information.pilbl

            self.prior_information.pop("pilbl")
            f_out.write(self.prior_information.to_string(colSpace=0,
                                              formatters=self.prior_format,
                                              justify="right",
                                              header=False,
                                              index_names=False) + '\n')
            self.prior_information["pilbl"] = self.prior_information.index
        #--read past an option prior information section
        while True:
            line = f_in.readline()
            if line == '':
                break

            if line.strip().startswith('*') or line.strip().startswith("++"):
                f_out.write(line)
                break
        if line != '':
            while True:
                line = f_in.readline()
                if line == '':
                    break
                f_out.write(line)
        f_in.close()
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
        if par_names is None and obs_names is None:
            return copy.deepcopy(self)
        new_par = copy.deepcopy(self.parameter_data)
        if par_names is not None:
            new_par.index = new_par.parnme
            new_par = new_par.loc[par_names, :]
        new_obs = copy.deepcopy(self.observation_data)
        new_res = None



        if obs_names is not None:
            new_obs.index = new_obs.obsnme
            new_obs = new_obs.loc[obs_names]
            if self.res is not None:
                new_res = copy.deepcopy(self.res)
                new_res.index = new_res.name
                new_res = new_res.loc[obs_names,:]

        new_pst = pst(self.filename, resfile=self.resfile, load=False)
        new_pst.parameter_data = new_par
        new_pst.observation_data = new_obs
        new_pst.__res = new_res
        # this is too slow, just drop the prior info
        # new_prior = copy.deepcopy(self.prior_information)
        # if par_names is not None:
        #     # need to drop all prior information that mentions parameters that
        #     # are not in the new pst
        #     dropped = []
        #     for p in self.par_names:
        #         if p not in par_names:
        #             dropped.append(p)
        #     if len(dropped) > 0:
        #         # this is painful
        #         keep_idx = []
        #         for d in dropped:
        #             for row in new_prior.iterrows():
        #                 if d not in row[1].equation.lower():
        #                     keep_idx.append(row[1].index)
        #         new_prior = new_prior.loc[keep_idx, :]
        if par_names is not None:
            print "pst.get() warning: dropping all prior information in " + \
                  " new pst instance"
        new_pst.prior_information = self.null_prior
        new_pst.mode = self.mode
        new_pst.estimation = self.estimation
        return new_pst


    def zero_order_tikhonov(self,parbounds=True):
        """setup preferred-value regularization
        Args:
            parbounds (bool) : weight the prior information equations according
                to parameter bound width - approx the KL transform
        Returns:
            None
        Raises:
            None
        """
        pass
        obs_group = "regul"
        pilbl, obgnme, weight, equation = [], [], [], []
        for idx, row in self.parameter_data.iterrows():
            if row["partrans"].lower() not in ["tied", "fixed"]:
                pilbl.append(row["parnme"])
                weight.append(1.0)
                obgnme.append(obs_group)
                parnme = row["parnme"]
                parval1 = row["parval1"]
                if row["partrans"].lower() == "log":
                    parnme = "log(" + parnme + ")"
                    parval1 = np.log10(parval1)
                eq = "1.0 * " + parnme + " ={0:15.6E}".format(parval1)
                equation.append(eq)
        self.prior_information = pandas.DataFrame({"pilbl": pilbl,
                                                   "equation": equation,
                                                   "obgnme": obs_group,
                                                   "weight": weight})
        if parbounds:
            self.regweight_from_parbound()


    def regweight_from_parbound(self):
        """sets regularization weights from parameter bounds
            which approximates the KL expansion
        """
        self.parameter_data.index = self.parameter_data.parnme
        self.prior_information.index = self.prior_information.pilbl
        for idx, parnme in enumerate(self.prior_information.pilbl):
            if parnme in self.parameter_data.index:
                row =  self.parameter_data.loc[parnme, :]
                lbnd,ubnd = row["parlbnd"], row["parubnd"]
                if row["partrans"].lower() == "log":
                    weight = 1.0 / (np.log10(ubnd) - np.log10(lbnd))
                else:
                    weight = 1.0 / (ubnd - lbnd)
                self.prior_information.loc[parnme, "weight"] = weight
            else:
                print "prior information name does not correspond" +\
                      " to a parameter: " + str(parnme)


    def parrep(self,parfile=None):
        """replicates the pest parrep util. replaces the parval1 field in the
            parameter data section dataframe
        Args:
            parfile (str) : parameter file to use.  If None, try to use
                            a parameter file that corresponds to the case name
        Returns:
            None
        Raises:
            assertion error if parfile not found
        """
        if parfile is None:
            parfile = self.filename.replace(".pst", ".par")
        assert os.path.exists(parfile), "pst.parrep(): parfile not found: " +\
                                        str(parfile)
        f = open(parfile, 'r')
        header = f.readline()
        par_df = pandas.read_csv(f, header=None,
                                 names=["parnme", "parval1", "scale", "offset"],
                                 sep="\s+")
        self.parameter_data.index = self.parameter_data.parnme
        par_df.index = par_df.parnme
        self.parameter_data.parval1 = par_df.parval1


    def adjust_weights_recfile(self,recfile=None):
        """adjusts the weights of the observations based on the phi components
        in a recfile
        Args:
            recfile (str) : record file name.  If None, try to use a record file
                            with the case name
        Returns:
            None
        Raises:
            Assertion error if recfile not found
            Exception if no complete iteration output was found in recfile
        """
        if recfile is None:
            recfile = self.filename.replace(".pst", ".rec")
        assert os.path.exists(recfile), \
            "pst.adjust_weights_recfile(): recfile not found: " +\
            str(recfile)
        iter_components = self.get_phi_components_from_recfile(recfile)
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
            raise Exception("pst.pwtadj2(): no complete phi component" +
                            " records found in recfile")
        self.adjust_weights_by_phi_components(
            iter_components[last_complete_iter])


    def adjust_weights_resfile(self,resfile=None):
        """adjust the weights by phi components in a residual file
        Args:
            resfile (str) : residual filename.  If None, use self.resfile
        Returns:
            None
        Raises:
            None
        """
        if resfile is not None:
            self.resfile = resfile
            self.__res = None
        self.adjust_weights_by_phi_components(self.phi_components)


    def adjust_weights_by_phi_components(self, components):
        """resets the weights of observations to account for
        residual phi components.
        Args:
            components (dict{obs group:phi contribution}): group specific phi
                contributions
        Returns:
            None
        Raises:
            Exception if residual components don't agree with non-zero weighted
                observations
        """
        obs = self.observation_data
        nz_groups = obs.groupby(obs["weight"].map(lambda x: x == 0)).groups
        nzobs = 0
        if False in nz_groups.keys():
            nzobs = len(nz_groups[False])

        ogroups = obs.groupby("obgnme").groups
        for ogroup, idxs in ogroups.iteritems():
            if self.mode.startswith("regul") and "regul" in ogroup.lower():
                continue
            og_phi = components[ogroup]
            odf = obs.loc[idxs, :]
            nz_groups = odf.groupby(odf["weight"].map(lambda x: x == 0)).groups
            og_nzobs = 0
            if False in nz_groups.keys():
                og_nzobs = len(nz_groups[False])
            if og_nzobs == 0 and og_phi > 0:
                raise Exception("pst.adjust_weights_by_phi_components():"
                                " no obs with nonzero weight," +
                                " but phi > 0 for group:" + str(ogroup))
            if og_phi > 0:
                factor = np.sqrt(float(og_nzobs) / float(og_phi))
                obs.weight[idxs] *= factor
        self.observation_data = obs


    def get_phi_components_from_recfile(self, recfile):
        """read the phi components from a record file
        Args:
            recfile (str) : record file
        Returns:
            dict{iteration number:{group,contribution}}
        Raises:
            None
        """
        iiter = 1
        iters = {}
        f = open(recfile,'r')
        while True:
            line = f.readline()
            if line == '':
                break
            if "starting phi for this iteration" in line.lower():
                contributions = {}
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    if "contribution to phi" not in line.lower():
                        iters[iiter] = contributions
                        iiter += 1
                        break
                    raw = line.strip().split()
                    val = float(raw[-1])
                    group = raw[-3].lower().replace('\"', '')
                    contributions[group] = val
        return iters


    def __reset_weights(self, target_phis, res_idxs, obs_idxs):
        """reset weights based on target phi vals for each group
        Args:
            target_phis (dict) : target phi contribution for groups to reweight
            res_idxs (dict) : the index positions of each group of interest
                 in the res dataframe
            obs_idxs (dict) : the index positions of each group of interest
                in the observation data dataframe
        """
        pass
        for item in target_phis.keys():
            assert item in res_idxs.keys(),\
                "pst.__reset_weights(): "  + str(item) +\
                " not in residual group indices"
            assert item in obs_idxs.keys(), \
                "pst.__reset_weights(): " + str(item) +\
                " not in observation group indices"
            actual_phi = ((self.res.loc[res_idxs[item],"residual"] *
                           self.observation_data.loc
                           [obs_idxs[item],"weight"] )**2).sum()
            weight_mult = np.sqrt(target_phis[item] / actual_phi)
            self.observation_data.loc[obs_idxs[item],"weight"] *= weight_mult


    def adjust_weights_by_group(self,obs_dict=None,
                              obsgrp_dict=None,obsgrp_suffix_dict=None,
                              obsgrp_prefix_dict=None,obsgrp_phrase_dict=None):
        """reset the weights of observation groups to contribute a specified
        amount to the composite objective function
        Args:
            obs_dict (dict{obs name:new contribution})
            obsgrp_dict (dict{obs group name:contribution})
            obsgrp_suffic_dict (dict{obs group suffix:contribution})
            obsgrp_prefix_dict (dict{obs_group prefix:contribution})
            obsgrp_phrase_dict (dict{obs group phrase:contribution})
        Returns:
            None
        Raises:
            Exception if a key is not found in the obs or obs groups
        """
        if obsgrp_dict is not None:
            res_groups = self.res.groupby("group").groups
            obs_groups = self.observation_data.groupby("obgnme").groups
            self.__reset_weights(obsgrp_dict,res_groups,obs_groups)
        if obs_dict is not None:
            res_groups = self.res.groupby("name").groups
            obs_groups = self.observation_data.groupby("obsnme").groups
            self.__reset_weights(obs_dict,res_groups, obs_groups)
        if obsgrp_suffix_dict is not None:
            self.res.index = self.res.group
            self.observation_data.index = self.observation_data.obgnme
            res_idxs, obs_idxs = {}, {}
            for suffix,phi in obsgrp_suffix_dict.iteritems():
                res_groups = self.res.groupby(lambda x:
                                              x.endswith(suffix)).groups
                assert True in res_groups.keys(),\
                    "pst.adjust_weights_by_phi(): obs group suffix \'" +\
                    str(suffix)+"\' not found in res"
                obs_groups = self.observation_data.groupby(
                    lambda x: x.endswith(suffix)).groups
                assert True in obs_groups.keys(),\
                    "pst.adjust_weights_by_phi(): obs group suffix \'" +\
                    str(suffix) + "\' not found in observation_data"
                res_idxs[suffix] = res_groups[True]
                obs_idxs[suffix] = obs_groups[True]
            self.__reset_weights(obsgrp_suffix_dict, res_idxs, obs_idxs)
        if obsgrp_prefix_dict is not None:
            self.res.index = self.res.group
            self.observation_data.index = self.observation_data.obgnme
            res_idxs, obs_idxs = {}, {}
            for prefix, phi in obsgrp_prefix_dict.iteritems():
                res_groups = self.res.groupby(
                    lambda x: x.startswith(prefix)).groups
                assert True in res_groups.keys(),\
                    "pst.adjust_weights_by_phi(): obs group prefix \'" +\
                    str(prefix) + "\' not found in res"
                obs_groups = self.observation_data.groupby(
                    lambda x:x.startswith(prefix)).groups
                assert True in obs_groups.keys(),\
                    "pst.adjust_weights_by_phi(): obs group prefix \'" +\
                    str(prefix) + "\' not found in observation_data"
                res_idxs[prefix] = res_groups[True]
                obs_idxs[prefix] = obs_groups[True]
            self.__reset_weights(obsgrp_prefix_dict, res_idxs, obs_idxs)




if __name__ == "__main__":
    p = pst("pest.pst")
    pnew = p.get(p.par_names[:10],p.obs_names[-10:])
    print pnew.res
    pnew.write("test.pst")
    #p.adjust_phi_by_weights(obsgrp_dict={"head":10},obs_dict={"h_obs01_1":100})
    #p.adjust_phi_by_weights(obsgrp_prefix_dict={"he":10})
    #p.zero_order_tikhonov()
    #p.write("test.pst")
