import pyemu
pst = pyemu.Pst("dewater_pest.pst")

oc_names = pst.observation_data.groupby(pst.observation_data.weight.apply(lambda x:x!=0)).groups[True]
oc_dict = {oc:"le" for oc in oc_names}
pyemu.utils.to_mps("dewater_pest.jcb",oc_dict,pst=pst,decision_var_names="Q2")



