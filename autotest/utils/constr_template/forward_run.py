import os
import numpy as np
import pandas as pd
def constr(x):
    return (x[0],(1 + x[1]) / x[0]),[]

def helper(func=constr,pvals=None):
    if pvals is None:
        pvals = pd.read_csv("dv.dat", sep=r'\s+',index_col=0, header=None, names=["parnme","parval1"]).values
    #obj1,obj2 = func(pdf.values)
    objs,constrs = func(pvals)
    
    if os.path.exists("additive_par.dat"):
        obj1,obj2 = objs[0],objs[1]
        cdf = pd.read_csv("additive_par.dat", sep=r'\s+', index_col=0,header=None, names=["parnme","parval1"])
        obj1[0] += cdf.parval1.values[0]
        obj2[0] += cdf.parval1.values[1]
        for i in range(len(constrs)):
            constrs[i] += cdf.parval1.values[i+2]

    with open("obj.dat",'w') as f:
        for i,obj in enumerate(objs):
            f.write("obj_{0} {1}\n".format(i+1,float(obj.item())))
        #f.write("obj_2 {0}\n".format(float(obj2)))
        for i,constr in enumerate(constrs):
            f.write("constr_{0} {1}\n".format(i+1,float(constr)))
    return objs,constrs

if __name__ == '__main__':
    helper(constr)
