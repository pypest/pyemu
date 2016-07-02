import os
import pandas as pd
import flopy
import pyemu

ml = flopy.modflow.Modflow.load("dewater.nam",check=False,verbose=False)


decvar_file = "dewater.decvar"
hedcon_file = "dewater.hedcon"

# get hedcon locations
with open(hedcon_file,'r') as f:
    [f.readline() for _ in range(4)]
    hed_df = pd.read_csv(f,usecols=[2,3],header=None,names=["row","col"],delim_whitespace=True)
hed_df.loc[:,'i'] = hed_df.pop("row") - 1
hed_df.loc[:,'j'] = hed_df.pop("col") - 1


ds3a_names = "FVNAME NC LAY ROW COL FTYPE FSTAT WSP".lower().split()
with open(decvar_file,"r") as f:
    line = '#'
    while line.startswith('#'):
        line = f.readline()
    print("iprn",line)
    nvars = [int(i) for i in f.readline().strip().split()[:3]]
    if nvars[1] > 0 or nvars[2] > 0:
        raise NotImplementedError()
    flow_lines = []
    for ivar in range(nvars[0]):
        line = f.readline().strip().split()[:8]
        if int(line[1]) != 1:
            raise NotImplementedError()
        if ':' in line[-1] or '-' in line[-1]:
            raise NotImplementedError
        flow_lines.append(line)
flow_df = pd.DataFrame(flow_lines,columns=ds3a_names)

flow_df.loc[:,"flux"] = 0.0
flow_df.loc[:,"k"] = flow_df.lay.apply(int) - 1
flow_df.loc[:,"i"] = flow_df.row.apply(int) - 1
flow_df.loc[:,"j"] = flow_df.col.apply(int) - 1
flow_df.loc[:,"wsp"] = flow_df.wsp.apply(int) - 1
print(flow_df.dtypes)
well_data = {}
for val in flow_df.wsp.unique():
    flow_df_sp = flow_df.loc[flow_df.groupby(flow_df.wsp==val).groups[True],:]
    well_data[val] = flow_df_sp.loc[:,["k","i","j","flux"]].values
print(flopy.modflow.ModflowWel.get_default_dtype())
wel = flopy.modflow.ModflowWel(ml,stress_period_data=well_data)
oc = flopy.modflow.ModflowOc(ml,chedfm="(30E15.6)")

ml.name = "dewater_pest"
ml.exe_name = "mf2005"
ml.write_input()
ml.run_model()



# write a template file for the well file
f_wel = open(ml.name+".wel",'r')
f_tpl = open(ml.name+".wel.tpl",'w')

f_tpl.write("ptf ~\n")
[f_tpl.write(f_wel.readline()) for _ in range(3)]
for i,line in enumerate(f_wel):
    pname = flow_df.fvname[i]
    tpl_line = line[:33] + "~    {0}    ~\n".format(pname)
    f_tpl.write(tpl_line)
f_tpl.close()



#write an instruction file for all obs locs
nnz_names = []
ijs = list(zip(hed_df.i.values,hed_df.j.values))

with open(ml.name+".hds.ins",'w') as f:
    f.write("pif ~\n")
    
    for i in range(ml.nrow):
        if i == 0:
            f.write("l2 ")
        else:
            f.write("l1 ")
        for j in range(ml.ncol):
            oname = "h{0:02d}_{1:02d}".format(i,j)
            if (i,j) in ijs:
                nnz_names.append(oname)
            f.write(" w !{0}!".format(oname))
        f.write("\n")

pst = pyemu.pst_utils.pst_from_io_files(ml.name+".wel.tpl",ml.name+".wel",ml.name+".hds.ins",ml.name+".hds")
pst.parameter_data.loc[:,"partrans"] = "none"
pst.parameter_data.loc[:,"parval1"] = 0.0
pst.parameter_data.loc[:,"parubnd"] = 20000.0
pst.parameter_data.loc[:,"parlbnd"] = 0.0
pst.parameter_data.loc[:,"scale"] = -1.0
pst.parameter_groups.loc[:,"inctyp"] = "absolute"
pst.parameter_groups.loc[:,"derinc"] = -10000.0

print(pst.observation_data)
pst.observation_data.loc[:,"obsval"] = 50.0
pst.observation_data.loc[:,"weight"] = 0.0
pst.observation_data.loc[nnz_names,"weight"] = 1.0
pst.observation_data.loc[nnz_names,"obgnme"] = "l"

pst.prior_information = pst.null_prior
pst.control_data.pestmode = "estimation"
pst.control_data.noptmax = -1
pst.model_command = [ml.exe_name + ' ' + ml.name+".nam"]

pst.write(ml.name+".pst")