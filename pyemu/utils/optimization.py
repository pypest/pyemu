from __future__ import print_function, division
import os, sys
import numpy as np
import pandas as pd
from pyemu import Matrix,Pst

OPERATOR_WORDS = ["l","g","n","e"]
OPERATOR_SYMBOLS = ["<=",">=","=","="]


def to_mps(jco,obj_func=None,obs_constraint_sense=None,pst=None,decision_var_names=None,
           mps_filename=None,):

    if isinstance(jco,str):
        pst_name = jco.lower().replace('.jcb',".pst").replace(".jco",".pst")
        jco = Matrix.from_binary(jco)
    assert isinstance(jco,Matrix)

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

    if obs_constraint_sense is None:
        const_groups = [grp for grp in pst.obs_groups if grp.lower() in OPERATOR_WORDS]
        if len(const_groups) == 0:
            raise Exception("to_mps(): obs_constraint_sense is None and no "+\
                            "obseravtion groups in {0}".format(','.join(pst.obs_groups)))
        obs_constraint_sense = {}
        obs_groups = pst.observation_data.groupby(pst.observation_data.obgnme).groups
        for og,obs_names in obs_groups.items():
            if og == 'n':
                continue
            if og in const_groups:
                for oname in obs_names:
                    obs_constraint_sense[oname] = og

    assert isinstance(obs_constraint_sense,dict)

    operators = {}
    rhs = {}
    for obs_name,operator in obs_constraint_sense.items():
        obs_name = obs_name.lower()
        assert obs_name in pst.obs_names,"obs constraint {0} not in pst observation names"
        rhs[obs_name] = pst.observation_data.loc[obs_name,"obsval"]
        assert obs_name in jco.row_names,"obs constraint {0} not in jco row names".format(obs_name)
        if operator.lower() not in OPERATOR_WORDS:
            if operator not in OPERATOR_SYMBOLS:
                raise Exception("operator {0} not in [{1}] or [{2}]".\
                    format(operator,','.join(OPERATOR_WORDS),','\
                           .join(OPERATOR_SYMBOLS)))
            op = OPERATOR_WORDS[OPERATOR_SYMBOLS.index(operator)]
        else:
            op = operator.lower()
        operators[obs_name] = op
        obs_constraint_sense[obs_name.lower()] = obs_constraint_sense.\
                                                 pop(obs_name)

    #build a list of constaints in order WRT jco row order
    order_obs_constraints = [name for name in jco.row_names if name in
                             obs_constraint_sense]

    #build a list of decision var names in order WRT jco col order
    order_dec_var = [name for name in jco.col_names if name in
                     decision_var_names]

    #shorten constraint names if needed
    new_const_count = 0
    new_constraint_names = {}
    for name in order_obs_constraints:
        if len(name) > 8:
            new_name = name[:7]+"{0}".format(new_const_count)
            print("to_mps(): shortening constraint name {0} to {1}\n")
            new_constraint_names[name] = new_name
            new_const_count += 1
        else:
            new_constraint_names[name] = name

    #shorten decision var names if needed
    new_dec_count = 0
    new_decision_names = {}
    for name in order_dec_var:
        if len(name) > 8:
            new_name = name[:7]+"{0}".format(new_dec_count)
            print("to_mps(): shortening decision var name {0} to {1}\n")
            new_decision_names[name] = new_name
            new_dec_count += 1
        else:
            new_decision_names[name] = name


    if obj_func is None:
        # look for an obs group named 'n' with a single member
        og = pst.obs_groups
        if 'n' not in pst.obs_groups:
            raise Exception("to_mps(): obj_func is None but no "+\
                            "obs group named 'n'")
        grps = pst.observation_data.groupby(pst.observation_data.obgnme).groups
        assert len(grps["n"]) == 1,"to_mps(): 'n' obj_func group has more " +\
                                   " one member"
        obj_name = grps['n'][0]
        obj_iidx = jco.row_names.index(obj_name)
        obj = {}
        for name in order_dec_var:
            jco_jidx = jco.col_names.index(name)
            obj[name] = jco.x[obj_iidx,jco_jidx]

    elif isinstance(obj_func,str):
        obj_func = obj_func.lower()
        assert obj_func in jco.row_names,\
            "obj_func {0} not in jco.row_names".format(obj_func)
        assert obj_func in pst.observation_data.obsnme,\
            "obj_func {0} not in pst observations".format(obj_func)

        obj_iidx = jco.row_names.index(obj_func)
        obj = {}
        for name in order_dec_var:
            jco_jidx = jco.col_names.index(name)
            obj[name] = jco.x[obj_iidx,jco_jidx]
        obj_name = str(obj_func)

    elif isinstance(obj_func,dict):
        obj = {}
        for name,value in obj_func.items():
            assert name in jco.col_names,"to_mps(): obj_func key {0} not ".format(name) +\
                "in jco col names"
            obj[name] = float(value)
        obj_name = "obj_func"
    else:
        raise NotImplementedError

    if mps_filename is None:
        mps_filename = pst.filename.replace(".pst",".mps")

    with open(mps_filename,'w') as f:
        f.write("NAME {0}\n".format("pest_opt"))
        f.write("ROWS\n")
        for name in order_obs_constraints:
            f.write(" {0}  {1}\n".format(operators[name],
                                         new_constraint_names[name]))
        f.write(" {0}  {1}\n".format('n',obj_name))

        f.write("COLUMNS\n")
        for dname in order_dec_var:
            jco_jidx = jco.col_names.index(dname)
            for cname in order_obs_constraints:
                jco_iidx = jco.row_names.index(cname)
                f.write("    {0:8}  {1:8}   {2:10G}\n".\
                        format(new_decision_names[dname],
                               new_constraint_names[cname],
                               jco.x[jco_iidx,jco_jidx]))
            f.write("    {0:8}  {1:8}   {2:10G}\n".\
                    format(new_decision_names[dname],
                           obj_name,obj[dname]))

        f.write("RHS\n")
        for name in order_obs_constraints:
            f.write("    {0:8}  {1:8}   {2:10G}\n".
                    format("rhs",new_constraint_names[name],rhs[name]))
        f.write("BOUNDS\n")
        for name in order_dec_var:
            up,lw = pst.parameter_data.loc[name,"parubnd"],\
                    pst.parameter_data.loc[name,"parlbnd"]
            f.write(" {0:2} {1:8}  {2:8}  {3:10G}\n".\
                    format("UP","BOUND",name,up))
            f.write(" {0:2} {1:8}  {2:8}  {3:10G}\n".\
                    format("LO","BOUND",name,lw))
        f.write("ENDATA\n")



