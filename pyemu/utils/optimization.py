from __future__ import print_function, division
import os, sys
import numpy as np
import pandas as pd
from pyemu import Matrix,Pst

OPERATOR_WORDS = ["le","ge","gt","lt","eq"]
OPERATOR_SYMBOLS = ["<=",">=",">","<","="]

def to_mps(jco,obs_constraint_dict,pst=None,decision_var_names=None):
    if isinstance(jco,str):
        pst_name = jco.lower().replace('.jcb',".pst").replace(".jco",".pst")
        jco = Matrix.from_binary(jco)
    assert isinstance(jco,Matrix)
    assert isinstance(obs_constraint_dict,dict)
    if pst is None:
        if os.path.exists(pst_name):
            pst = Pst(pst_name)
        else:
            raise Exception("could not find pst file {0} and pst argument is None, a ".format(pst_name) +\
                            "pst instance is required for setting decision variable bound constraints")
    else:
        assert len(set(jco.row_names).difference(pst.observation_data.index)) == 0
        assert len(set(jco.col_names).difference(pst.parameter_data.index)) == 0
    if decision_var_names is None:
        decision_var_names = jco.col_names
    else:
        if not isinstance(decision_var_names,list):
            decision_var_names = [decision_var_names]
        for i,dv in enumerate(decision_var_names):
            dv = dv.lower()
            decision_var_names[i] = dv
            assert dv in jco.col_names,"decision var {0} not in jco column names".format(dv)
            assert dv in pst.parameter_data.index,"decision var {0} not in pst parameter names".format(dv)

    for obs_name,operator in obs_constraint_dict.items():
        assert obs_name.lower() in jco.row_names,"obs constraint {0} not in jco row names".format(obs_name)
        if operator.lower() not in OPERATOR_WORDS:
            if operator not in OPERATOR_SYMBOLS:
                raise Exception("operator {0} not in [{1}] or [{2}]".\
                    format(operator,','.join(OPERATOR_WORDS),','.join(OPERATOR_SYMBOLS)))
        obs_constraint_dict[obs_name.lower()] = obs_constraint_dict.pop(obs_name)