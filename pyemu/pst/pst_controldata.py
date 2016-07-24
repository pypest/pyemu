from __future__ import print_function, division
import os
import copy
import numpy as np
import pandas
pandas.options.display.max_colwidth = 100
#from pyemu.pst.pst_handler import SFMT,SFMT_LONG,FFMT,IFMT

#formatters
SFMT = lambda x: "{0:>20s}".format(str(x))
SFMT_LONG = lambda x: "{0:>50s}".format(str(x))
IFMT = lambda x: "{0:>10d}".format(int(x))
FFMT = lambda x: "{0:>15.6E}".format(float(x))

CONTROL_DEFAULT_LINES = """restart estimation
     0     0       0    0      0  0
0  0  single  point  1  0  0 noobsreref
2.000000e+001  -3.000000e+000  3.000000e-001  1.000000e-002 -7 999 lamforgive noderforgive
1.000000e+001  1.000000e+001  1.000000e-003  0  0
1.000000e-001 1 1.1 noaui nosenreuse noboundscale
30 1.000000e-002  3  3  1.000000e-002  3  0.0 1  -1.0
0  0  0  0 jcosave verboserec jcosaveitn reisaveitn parsaveitn noparsaverun"""\
    .lower().split('\n')

CONTROL_VARIABLE_LINES = """RSTFLE PESTMODE
NPAR NOBS NPARGP NPRIOR NOBSGP [MAXCOMPDIM]
NTPLFLE NINSFLE PRECIS DPOINT [NUMCOM] [JACFILE] [MESSFILE] [OBSREREF]
RLAMBDA1 RLAMFAC PHIRATSUF PHIREDLAM NUMLAM [JACUPDATE] [LAMFORGIVE] [DERFORGIVE]
RELPARMAX FACPARMAX FACORIG [IBOUNDSTICK] [UPVECBEND]
PHIREDSWH [NOPTSWITCH] [SPLITSWH] [DOAUI] [DOSENREUSE] [BOUNDSCALE]
NOPTMAX PHIREDSTP NPHISTP NPHINORED RELPARSTP NRELPAR [PHISTOPTHRESH] [LASTRUN] [PHIABANDON]
ICOV ICOR IEIG [IRES] [JCOSAVE] [VERBOSEREC] [JCOSAVEITN] [REISAVEITN] [PARSAVEITN] [PARSAVERUN]"""\
    .lower().split('\n')

class ControlData(object):
    def __init__(self):

        super(ControlData,self).__setattr__("formatters",{np.int32:IFMT,np.float64:FFMT,str:SFMT})
        super(ControlData,self).__setattr__("_df",self.get_dataframe())

        # acceptable values for most optional string inputs
        super(ControlData,self).__setattr__("accept_values",{'doaui':['aui','noaui'],
                                                              'dosenreuse':['senreuse','nosenreuse'],
                                                              'boundscale':['boundscale','noboundscale'],
                                                              'jcosave':['jcosave','nojcosave'],
                                                              'verboserec':['verboserec','noverboserec'],
                                                              'jcosaveitn':['jcosaveitn','nojcosvaeitn'],
                                                              'reisaveitn':['reisaveitn','noreisaveitn'],
                                                              'parsaveitn':['parsaveitn','noparsaveitn'],
                                                              'parsaverun':['parsaverun','noparsaverun']})

        self._df.index = self._df.name.apply(lambda x:x.replace('[',''))\
            .apply(lambda x: x.replace(']',''))

    def __setattr__(self, key, value):
        if key == "_df":
            super(ControlData,self).__setattr__("_df",value)
            return
        assert key in self._df.index, str(key)+" not found in attributes"
        self._df.loc[key,"value"] = self._df.loc[key,"type"](value)

    def __getattr__(self, item):
        if item == "_df":
            return self._df.copy()
        assert item in self._df.index, str(item)+" not found in attributes"
        return self._df.loc[item,"value"]

    @staticmethod
    def get_dataframe():
        """ get a generic (default) control section dataframe
        
        :return: dataframe
        """
        names = []
        [names.extend(line.split()) for line in CONTROL_VARIABLE_LINES]

        defaults = []
        [defaults.extend(line.split()) for line in CONTROL_DEFAULT_LINES]

        types, required,cast_defaults,formats = [],[],[],[]
        for name,default in zip(names,defaults):
            if '[' in name or ']' in name:
                required.append(False)
            else:
                required.append(True)
            v,t,f = ControlData._parse_value(default)
            types.append(t)
            formats.append(f)
            cast_defaults.append(v)
        return pandas.DataFrame({"name":names,"type":types,
                                     "value":cast_defaults,"required":required,
                                    "format":formats})


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
        return v,t,f


    def parse_values_from_lines(self,lines):
        """ cast the string lines for a pest control file into actual inputs
        Parameters:
        ----------
            lines: strings from pest control file
        Returns:
        -------
            None
        """
        assert len(lines) == len(CONTROL_VARIABLE_LINES),\
        "ControlData error: len of lines not equal to " +\
        str(len(CONTROL_VARIABLE_LINES))

        for iline,line in enumerate(lines):
            vals = line.strip().split()
            names = CONTROL_VARIABLE_LINES[iline].strip().split()
            for name,val in zip(names,vals):
                v,t,f = self._parse_value(val)
                name = name.replace('[','').replace(']','')
                
                #if the parsed values type isn't right
                if t != self._df.loc[name,"type"]:

                    # if a float was expected and int return, not a problem
                    if t == np.int32 and self._df.loc[name,"type"] == np.float64:
                        self._df.loc[name,"value"] = np.float64(v)


                    # if this is a required input, throw
                    elif self._df.loc[name,"required"]:
                        raise Exception("wrong type found for variable " + name + ":" + str(t))
                    else:
                        
                        #else, since this problem is usually a string, check for acceptable values
                        found = False
                        for nname,avalues in self.accept_values.items():
                            if v in avalues:
                                if t == self._df.loc[nname,"type"]:
                                    self._df.loc[nname,"value"] = v
                                    found = True
                                    break
                        if not found:
                            print("warning: non-conforming value found for " +\
                                  name + ":" + str(v))
                            print("ignoring...")

                else:
                    self._df.loc[name,"value"] = v


    def copy(self):
        cd = ControlData()
        cd._df = self._df
        return cd


    @property
    def formatted_values(self):
        return self._df.apply(lambda x: self.formatters[x["type"]](x["value"]),axis=1)

    def write(self,f):
        """ write control data section to a file
        
        Parameters:
        ----------
            f: file handle or string filename
        Returns:
        -------
            None
        """
        if isinstance(f,str):
            f = open(f,'w')
            f.write("pcf\n")
            f.write("* control data\n")
        for line in CONTROL_VARIABLE_LINES:
            [f.write(self.formatted_values[name.replace('[','').replace(']','')]) for name in line.split()]
            f.write('\n')


