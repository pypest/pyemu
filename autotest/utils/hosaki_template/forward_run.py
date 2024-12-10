import os
import pandas as pd
import numpy as np

def hosaki(x):
    pval1 = x[0]
    pval2 = x[1]
    term1 = (pval2**2)*np.e**(-pval2)
    term2 = 1. - (8.*(pval1)) + (7.*(pval1**2))
    term3 = (-(7./3.)*(pval1**3)) + (0.25*(pval1**4))
    sim = (term2 + term3) * term1
    return sim

def helper(func=hosaki,pvals=None):
    if pvals is None:
        pvals = pd.read_csv('par.csv',index_col=0).values
    sim = hosaki(pvals)
    with open('sim.csv','w') as f:
        f.write('obsnme,obsval\n')
        f.write('sim,'+str(sim.item())+'\n')
    return sim

def hosaki_ppw_worker(pst_name,host,port):
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name,host,port,verbose=False)
    pvals = ppw.get_parameters()
    if pvals is None:
        return
    pvals.sort_index(inplace=True)

    while True:

        sim = helper(pvals=pvals.values)
        ppw.send_observations(np.array([sim]))
        pvals = ppw.get_parameters()
        if pvals is None:
            break
        pvals.sort_index(inplace=True)


if __name__ == "__main__":
    helper()
