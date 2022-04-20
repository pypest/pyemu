"""This module contains several class definitions for obseure parts of the
PEST control file: `ControlData` ('* control data'), `RegData` ('* regularization')
and `SvdData` ('* singular value decomposition').  These
classes are automatically created and appended to `Pst` instances;
users shouldn't need to deal with these classes explicitly
"""
from __future__ import print_function, division
import os
import copy
import warnings
import numpy as np
import pandas
from ..pyemu_warnings import PyemuWarning

pandas.options.display.max_colwidth = 100
# from pyemu.pst.pst_handler import SFMT,SFMT_LONG,FFMT,IFMT

# formatters
SFMT = lambda x: "{0:>20s}".format(str(x))
"""lambda function for string formatting `str` types into 20-char widths"""
SFMT_LONG = lambda x: "{0:>50s}".format(str(x))
"""lambda function for string formatting `str` types into 50-char widths"""
IFMT = lambda x: "{0:>10d}".format(int(x))
"""lambda function for string formatting `int` types into 10-char widths"""
FFMT = lambda x: "{0:>15.6E}".format(float(x))
"""lambda function for string formatting `float` types into 15-char widths"""

CONTROL_DEFAULT_LINES = """restart estimation
     0     0       0    0      0  0
0  0  single  point  1  0  0 noobsreref
2.000000e+001  -3.000000e+000  3.000000e-001  1.000000e-002 -7 999 lamforgive noderforgive
1.000000e+001  1.000000e+001  1.000000e-003  0  0
1.000000e-001 1 1.1 noaui nosenreuse noboundscale
30 1.000000e-002  3  3  1.000000e-002  3  0.0 1  -1.0
0  0  0  0 jcosave verboserec jcosaveitn reisaveitn parsaveitn noparsaverun""".lower().split(
    "\n"
)

CONTROL_VARIABLE_LINES = """RSTFLE PESTMODE
NPAR NOBS NPARGP NPRIOR NOBSGP [MAXCOMPDIM]
NTPLFLE NINSFLE PRECIS DPOINT [NUMCOM] [JACFILE] [MESSFILE] [OBSREREF]
RLAMBDA1 RLAMFAC PHIRATSUF PHIREDLAM NUMLAM [JACUPDATE] [LAMFORGIVE] [DERFORGIVE]
RELPARMAX FACPARMAX FACORIG [IBOUNDSTICK] [UPVECBEND]
PHIREDSWH [NOPTSWITCH] [SPLITSWH] [DOAUI] [DOSENREUSE] [BOUNDSCALE]
NOPTMAX PHIREDSTP NPHISTP NPHINORED RELPARSTP NRELPAR [PHISTOPTHRESH] [LASTRUN] [PHIABANDON]
ICOV ICOR IEIG [IRES] [JCOSAVE] [VERBOSEREC] [JCOSAVEITN] [REISAVEITN] [PARSAVEITN] [PARSAVERUN]""".lower().split(
    "\n"
)

CONTROL_VARIABLES_UNUSED = """RSTFLE NPAR NOBS NPARGP NPRIOR NOBSGP [MAXCOMPDIM] 
NTPLFLE NINSFLE PRECIS DPOINT [NUMCOM] [JACFILE] [MESSFILE] [OBSREREF]
RLAMBDA1 RLAMFAC PHIRATSUF PHIREDLAM NUMLAM [JACUPDATE] [LAMFORGIVE] FACORIG [IBOUNDSTICK] [UPVECBEND]
PHIREDSWH NRELPAR [NOPTSWITCH] [SPLITSWH] [DOAUI] [DOSENREUSE] [BOUNDSCALE] [PHISTOPTHRESH] [LASTRUN] [PHIABANDON]
ICOV ICOR IEIG [IRES] [JCOSAVE] [VERBOSEREC] [JCOSAVEITN] [REISAVEITN] [PARSAVEITN] [PARSAVERUN]""".lower().split(
    "\n"
)

REG_VARIABLE_LINES = """PHIMLIM PHIMACCEPT [FRACPHIM] [MEMSAVE]
WFINIT WFMIN WFMAX [LINREG] [REGCONTINUE]
WFFAC WFTOL IREGADJ [NOPTREGADJ REGWEIGHTRAT [REGSINGTHRESH]]""".lower().split(
    "\n"
)

REG_DEFAULT_LINES = """   1.0e-10    1.05e-10  0.1  nomemsave
1.0 1.0e-10 1.0e10 linreg continue
1.3  1.0e-2  1 1.5 1.5 0.5""".lower().split(
    "\n"
)


class RegData(object):
    """an object that encapsulates the regularization section
    of the PEST control file

    """

    def __init__(self):
        self.optional_dict = {}
        for vline, dline in zip(REG_VARIABLE_LINES, REG_DEFAULT_LINES):
            vraw = vline.split()
            draw = dline.split()
            for v, d in zip(vraw, draw):
                o = False
                if "[" in v:
                    o = True
                v = v.replace("[", "").replace("]", "")
                super(RegData, self).__setattr__(v, d)
                self.optional_dict[v] = o
        self.should_write = ["phimlim", "phimaccept", "fracphim", "wfinit"]

    def write(self, f):
        """write the regularization section to an open
        file handle

        Args:
            f (`file handle`): open file handle for writing

        """
        f.write("* regularization\n")
        for vline in REG_VARIABLE_LINES:
            vraw = vline.strip().split()
            for v in vraw:
                v = v.replace("[", "").replace("]", "")
                if v not in self.optional_dict.keys():
                    raise Exception("RegData missing attribute {0}".format(v))
                f.write("{0} ".format(self.__getattribute__(v)))
            f.write("\n")

    def write_keyword(self, f):
        """write the regularization section to an open
        file handle using the keyword-style format

        Args:
            f (`file handle`): open file handle for writing

        """
        for vline in REG_VARIABLE_LINES:
            vraw = vline.strip().split()
            for v in vraw:
                v = v.replace("[", "").replace("]", "")
                if v not in self.should_write:
                    continue
                if v not in self.optional_dict.keys():
                    raise Exception("RegData missing attribute {0}".format(v))
                f.write("{0:30} {1:>10}\n".format(v, self.__getattribute__(v)))


class SvdData(object):
    """encapsulates the singular value decomposition
    section of the PEST control file

    Args:
        kwargs (`dict`): optional keyword arguments


    """

    def __init__(self, **kwargs):
        self.svdmode = kwargs.pop("svdmode", 1)
        self.maxsing = kwargs.pop("maxsing", 10000000)
        self.eigthresh = kwargs.pop("eigthresh", 1.0e-6)
        self.eigwrite = kwargs.pop("eigwrite", 1)

    def write_keyword(self, f):
        """write an SVD section to a file handle using
        keyword-style format

                Args:
                    f (`file handle`): open file handle for writing

        """
        f.write("{0:30} {1:>10}\n".format("svdmode", self.svdmode))
        f.write("{0:30} {1:>10}\n".format("maxsing", self.maxsing))
        f.write("{0:30} {1:>10}\n".format("eigthresh", self.eigthresh))
        f.write("{0:30} {1:>10}\n".format("eigwrite", self.eigwrite))

    def write(self, f):
        """write an SVD section to a file handle

        Args:
            f (`file handle`): open file handle for writing

        """
        f.write("* singular value decomposition\n")
        f.write(IFMT(self.svdmode) + "\n")
        f.write(IFMT(self.maxsing) + " " + FFMT(self.eigthresh) + "\n")
        f.write("{0}\n".format(self.eigwrite))

    def parse_values_from_lines(self, lines):
        """parse values from lines of the SVD section of
        a PEST control file

        Args:
            lines ([`strs`]): the raw ASCII lines from the control file


        """
        assert (
            len(lines) == 3
        ), "SvdData.parse_values_from_lines: expected " + "3 lines, not {0}".format(
            len(lines)
        )
        try:
            self.svdmode = int(lines[0].strip().split()[0])
        except Exception as e:
            raise Exception(
                "SvdData.parse_values_from_lines: error parsing"
                + " svdmode from line {0}: {1} \n".format(lines[0], str(e))
            )
        try:
            raw = lines[1].strip().split()
            self.maxsing = int(raw[0])
            self.eigthresh = float(raw[1])
        except Exception as e:
            raise Exception(
                "SvdData.parse_values_from_lines: error parsing"
                + " maxsing and eigthresh from line {0}: {1} \n".format(
                    lines[1], str(e)
                )
            )
        # try:
        #     self.eigwrite = int(lines[2].strip())
        # except Exception as e:
        #     raise Exception("SvdData.parse_values_from_lines: error parsing" + \

        #                     " eigwrite from line {0}: {1} \n".format(lines[2],str(e)))
        self.eigwrite = lines[2].strip()


class ControlData(object):
    """an object that encapsulates the control data section
     of the PEST control file

    Notes:
        This class works hard to protect the variables in the control data section.
        It type checks attempts to change values to make sure the type being passed
        matches the expected type of the attribute.

    """

    def __init__(self):

        super(ControlData, self).__setattr__(
            "formatters", {np.int32: IFMT, np.float64: FFMT, str: SFMT}
        )
        super(ControlData, self).__setattr__("_df", self.get_dataframe())

        # acceptable values for most optional string inputs
        super(ControlData, self).__setattr__(
            "accept_values",
            {
                "doaui": ["aui", "noaui"],
                "dosenreuse": ["senreuse", "nosenreuse"],
                "boundscale": ["boundscale", "noboundscale"],
                "jcosave": ["jcosave", "nojcosave"],
                "verboserec": ["verboserec", "noverboserec"],
                "jcosaveitn": ["jcosaveitn", "nojcosvaeitn"],
                "reisaveitn": ["reisaveitn", "noreisaveitn"],
                "parsaveitn": ["parsaveitn", "noparsaveitn"],
                "parsaverun": ["parsaverun", "noparsaverun"],
            },
        )

        self._df.index = self._df.name.apply(lambda x: x.replace("[", "")).apply(
            lambda x: x.replace("]", "")
        )
        super(ControlData, self).__setattr__(
            "keyword_accessed", ["pestmode", "noptmax"]
        )
        counters = ["npar", "nobs", "npargp", "nobsgp", "nprior", "ntplfle", "ninsfle"]
        super(ControlData, self).__setattr__("counters", counters)
        # self.keyword_accessed = ["pestmode","noptmax"]
        super(ControlData, self).__setattr__("passed_options", {})

    def __setattr__(self, key, value):
        if key == "_df":
            super(ControlData, self).__setattr__("_df", value)
            return
        assert key in self._df.index, str(key) + " not found in attributes"
        self._df.loc[key, "value"] = self._df.loc[key, "type"](value)
        # super(ControlData, self).__getattr__("keyword_accessed").append(key)
        if key not in self.counters:
            self.keyword_accessed.append(key)

    def __getattr__(self, item):
        if item == "_df":
            return self._df.copy()
        assert item in self._df.index, str(item) + " not found in attributes"
        return self._df.loc[item, "value"]

    @staticmethod
    def get_dataframe():
        """get a generic (default) control section as a dataframe

        Returns:

            `pandas.DataFrame`: a dataframe of control data information

        """
        names = []
        [names.extend(line.split()) for line in CONTROL_VARIABLE_LINES]

        defaults = []
        [defaults.extend(line.split()) for line in CONTROL_DEFAULT_LINES]

        types, required, cast_defaults, formats = [], [], [], []
        for name, default in zip(names, defaults):
            if "[" in name or "]" in name:
                required.append(False)
            else:
                required.append(True)
            v, t, f = ControlData._parse_value(default)
            types.append(t)
            formats.append(f)
            cast_defaults.append(v)
        return pandas.DataFrame(
            {
                "name": names,
                "type": types,
                "value": cast_defaults,
                "required": required,
                "format": formats,
                "passed": False,
            }
        )

    @staticmethod
    def _parse_value(value):
        try:
            v = int(value)
            t = np.int32
            f = IFMT
        except Exception as e:
            try:
                v = float(value)
                t = np.float64
                f = FFMT
            except Exception as ee:
                v = value.lower()
                t = str
                f = SFMT
        return v, t, f

    def parse_values_from_lines(self, lines, iskeyword=False):
        """cast the string lines for a pest control file into actual inputs

        Args:
            lines ([`str`]): raw ASCII lines from pest control file

        """
        self._df.loc[:, "passed"] = False
        if iskeyword:
            # self._df.loc[:, "passed"] = True

            extra = {}
            for line in lines:
                raw = line.strip().split()
                if len(raw) == 0 or raw[0] == "#":
                    continue
                name = raw[0].strip().lower()

                value = raw[1].strip()
                v, t, f = self._parse_value(value)
                if name not in self._df.index:
                    extra[name] = v
                else:
                    # if the parsed values type isn't right
                    if t != self._df.loc[name, "type"]:

                        # if a float was expected and int return, not a problem
                        if t == np.int32 and self._df.loc[name, "type"] == np.float64:
                            self._df.loc[name, "value"] = np.float64(v)
                            self._df.loc[name, "passed"] = True

                        # if this is a required input, throw
                        elif self._df.loc[name, "required"]:
                            raise Exception(
                                "wrong type found for variable " + name + ":" + str(t)
                            )
                        else:

                            # else, since this problem is usually a string, check for acceptable values
                            found = False
                            for nname, avalues in self.accept_values.items():
                                if v in avalues:
                                    if t == self._df.loc[nname, "type"]:
                                        self._df.loc[nname, "value"] = v
                                        found = True
                                        self._df.loc[nname, "passed"] = True
                                        break
                            if not found:
                                warnings.warn(
                                    "non-conforming value found for "
                                    + name
                                    + ":"
                                    + str(v)
                                    + "...ignoring",
                                    PyemuWarning,
                                )

                    else:
                        self._df.loc[name, "value"] = v
                        self._df.loc[name, "passed"] = True
            return extra

        assert len(lines) == len(
            CONTROL_VARIABLE_LINES
        ), "ControlData error: len of lines not equal to " + str(
            len(CONTROL_VARIABLE_LINES)
        )

        for iline, line in enumerate(lines):
            vals = line.strip().split()
            names = CONTROL_VARIABLE_LINES[iline].strip().split()
            for name, val in zip(names, vals):
                v, t, f = self._parse_value(val)

                name = name.replace("[", "").replace("]", "")

                # if the parsed values type isn't right
                if t != self._df.loc[name, "type"]:

                    # if a float was expected and int return, not a problem
                    if t == np.int32 and self._df.loc[name, "type"] == np.float64:
                        self._df.loc[name, "value"] = np.float64(v)
                        # self._df.loc[name, "passed"] = True

                    # if this is a required input, throw
                    elif self._df.loc[name, "required"]:
                        raise Exception(
                            "wrong type found for variable " + name + ":" + str(t)
                        )
                    else:

                        # else, since this problem is usually a string, check for acceptable values
                        found = False
                        for nname, avalues in self.accept_values.items():
                            if v in avalues:
                                if t == self._df.loc[nname, "type"]:
                                    self._df.loc[nname, "value"] = v
                                    found = True
                                    # self._df.loc[nname, "passed"] = True
                                    break
                        if not found:
                            warnings.warn(
                                "non-conforming value found for "
                                + name
                                + ":"
                                + str(v)
                                + "...ignoring",
                                PyemuWarning,
                            )

                else:
                    self._df.loc[name, "value"] = v
                    self._df.loc[name, "passed"] = True

        return {}

    def copy(self):
        cd = ControlData()
        cd._df = self._df
        return cd

    @property
    def formatted_values(self):
        """list the entries and current values in the control data section

        Returns:
            pandas.Series:  formatted_values for the control data entries

        """
        # passed_df = self._df.copy()
        # blank = passed_df.apply(lambda x: x.type==str and x.passed==False,axis=1)
        # passed_df.loc[blank,"value"] = ""
        return self._df.apply(lambda x: self.formatters[x["type"]](x["value"]), axis=1)

    def write_keyword(self, f):
        """write the control data entries to an open file handle
        using keyword-style format.

        Args:
            f (file handle): open file handle to write to

        Notes:
            only writes values that have been accessed since instantiation

        """
        kw = super(ControlData, self).__getattribute__("keyword_accessed")
        kw = [kkw for kkw in kw if kkw != "numcom"]
        default_df = self.get_dataframe()
        default_df.index = default_df.name.values
        default_values = default_df.value.to_dict()
        dimen_vars = CONTROL_VARIABLE_LINES[1].split()
        dimen_vars.extend(CONTROL_VARIABLE_LINES[2].split())
        dimen_vars = set(dimen_vars)
        unused = CONTROL_VARIABLES_UNUSED[0].split()
        [unused.extend(line.split()) for line in CONTROL_VARIABLES_UNUSED[1:]]
        unused = [i.lower().replace("[","").replace("]","") for i in unused]
        f.write("* control data keyword\n")
        for n, v in zip(self._df.name, self.formatted_values):
            if n.replace("[","").replace("]","") not in kw:
                if n not in self._df.index:
                    continue
                if not self._df.loc[n, "passed"]:
                    continue
                if self._df.loc[n,"value"] == default_values.get(n,self._df.loc[n,"value"]):
                    continue
            if n.replace("[","").replace("]","") in dimen_vars:
                continue
            if n.replace("[", "").replace("]", "") in unused:
                continue
            f.write("{0:30} {1}\n".format(n.replace("[","").replace("]",""), v))

    def write(self, f):
        """write control data section to a file

        Args:
            f (file handle): open file handle to write to

        """
        if isinstance(f, str):
            f = open(f, "w")
            f.write("pcf\n")
            f.write("* control data\n")
        for line in CONTROL_VARIABLE_LINES:
            [
                f.write(self.formatted_values[name.replace("[", "").replace("]", "")])
                for name in line.split()
                if self._df.loc[name.replace("[", "").replace("]", ""), "passed"]
                == True
                or self._df.loc[name.replace("[", "").replace("]", ""), "required"]
                == True
                or name.replace("[", "").replace("]", "") in self.keyword_accessed
            ]
            f.write("\n")
