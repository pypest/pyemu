import numpy as np

IDX1,IDX2 = 3,5

def get_grad(arr):
    return arr[IDX2] - arr[IDX1]

def setup():
    obs = np.loadtxt('simp_examp_syn.hds').flatten()
    oname = 'g_{0:02d}_{1:02d}'.format(IDX1+1,IDX2+1)
    oval = get_grad(obs)
    f_obs = open('grad.obs','w')
    f_obs.write(oname + ' {0:15.6E} 0.0\n'.format(oval))
    f_obs.close()
    
    f_ins = open('grad.ins','w')
    f_ins.write('pif ~\n')
    f_ins.write('l1 !'+oname+'!\n')
    f_ins.close()

def run():
    obs = np.loadtxt('simp_examp_cal.hds').flatten()
    oval = get_grad(obs)
    f = open('grad.out','w')
    f.write('{0:15.6E}\n'.format(oval))
    f.close()
    
 
if __name__ == '__main__':
    setup()
    run()       