import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.patches import Rectangle as rect
import matplotlib.cm as mplcm
import matplotlib.colors as colors

from matplotlib.font_manager import FontProperties

mpl.rcParams['font.sans-serif']          = 'Arial'
mpl.rcParams['font.serif']               = 'Arial'
mpl.rcParams['font.cursive']             = 'Arial'
mpl.rcParams['font.fantasy']             = 'Arial'
mpl.rcParams['font.monospace']           = 'Arial'
mpl.rcParams['pdf.compression']          = 0
mpl.rcParams['pdf.fonttype']             = 42

ticksize = 6
mpl.rcParams['legend.fontsize']  = 8
mpl.rcParams['axes.labelsize']   = 8
mpl.rcParams['xtick.labelsize']  = 8
mpl.rcParams['ytick.labelsize']  = 8
mpl.rcParams['legend.handlelength'] = 3

import pylab




def plot_bar(syn_h,cal_k,cal_h,cell_nums,obs_idxs,delx,plt_name,kmin,kmax):
    #fig = pylab.figure(figsize=(8,4))
    fig = pylab.figure(figsize=(4.72,4.72))
    axk = pylab.axes((0.1,0.77,0.7,0.21))
    ax1 = pylab.axes((0.1,0.418,0.7,0.25))
    ax2 = pylab.axes((0.1,0.1,0.7,0.25))    
    axk.text(-5.0,5.0,'a) Calibrated hydraulic conductivity distribution',fontsize=8)      
    ax1.text(-5.0,6.5,'b) Calibrated water level distribution',fontsize=8)   
    ax2.text(-5.0,6.5,'c ) Predictive water level distribution',fontsize=8)    
    arrowprops=dict(connectionstyle="angle,angleA=0,angleB=90,rad=10",arrowstyle='->')
    bbox_args = dict(fc="1.0")
    ax1.annotate('Q=0.5 $m^3/d$',fontsize=8,xy=(95,0),xytext=(70.0,1.0),
                 arrowprops=arrowprops,bbox=bbox_args)
    ax2.annotate('Q=1.0 $m^3/d$',fontsize=8,xy=(95,0),xytext=(70.0,1.0),
                 arrowprops=arrowprops,bbox=bbox_args)
    
    axk.text(0.0,-1.2,'Specified\nhead',ha='left',va='top',fontsize=8)
    axk.text(100,-1.2,'Specified\nflux',ha='right',va='top',fontsize=8)
    axk.text(50,-1.6,'Active model cells',ha='center',va='top',fontsize=8)
    arrowprops=dict(arrowstyle='<->')
    axk.annotate('',fontsize=8,xycoords='axes fraction',xy=(0.85,-0.075),xytext=(0.15,-0.075),
                arrowprops=arrowprops) 

    #cmap_name = 'gray'
    #cm = pylab.get_cmap(cmap_name)
    #cnorm = colors.Normalize(vmin=k_min,vmax=k_max)
    #smap = mplcm.ScalarMappable(norm=cnorm,cmap=cm)
    #color = []
    #for i,(k,col) in enumerate(zip(cal_k,cell_nums)):        
    #    c = smap.to_rgba(k)
    #    color.append(c)

    k_rects = axk.bar(cell_nums-(delx/2.0),cal_k,width=delx,color='#58ACFA',edgecolor='k',linewidth=0.5,alpha=0.5)
    axk.plot([0,cell_nums.max()+(0.5*delx)],[2.5,2.5],'k--',lw=1.5)
    axk.text(80,2.8,'True value',ha='left',va='bottom',fontsize=8)
    
        
    ax1.plot(cell_nums,syn_h[0,:],color='b',ls='-')
    ax1.scatter(cell_nums,syn_h[0,:],marker='.',s=25,edgecolor='b',facecolor='b',label='True')

    ax1.plot(cell_nums,cal_h[0,:],color='r',ls='-')
    ax1.scatter(cell_nums,cal_h[0,:],marker='.',s=25,edgecolor='r',facecolor='r',label='Calibrated')

    ax2.plot(cell_nums,syn_h[1,:],color='b',ls='--')
    ax2.scatter(cell_nums,syn_h[1,:],marker='.',s=25,edgecolor='b',facecolor='b',label='True')

    ax2.plot(cell_nums,cal_h[1,:],color='r',ls='--')
    ax2.scatter(cell_nums,cal_h[1,:],marker='.',s=25,edgecolor='r',facecolor='r',label='Calibrated')

    
    for iobs,obs_idx in enumerate(obs_idxs):
        if iobs == 0:
            ax1.scatter([cell_nums[obs_idx]],[syn_h[0,obs_idx]],marker='^',facecolor='k',edgecolor='k',s=50,label='Observation')
        else:
            ax1.scatter([cell_nums[obs_idx]],[syn_h[0,obs_idx]],marker='^',facecolor='k',edgecolor='k',s=50)
    
    for i,(col) in enumerate(cell_nums):
        xmn,xmx = col-(delx*0.5),col+(delx*0.5)
        ymn,ymx = -1.0,0.0   
        if i == 0:
            c = 'm'
        elif i == cell_nums.shape[0]-1:
            c = 'g'
        else:
            c = '#E5E4E2'        
        a = 0.75
        r1 = rect((xmn,ymn),xmx-xmn,ymx-ymn,color=c,ec='k',alpha=a)
        ax1.add_patch(r1)
        r2 = rect((xmn,ymn),xmx-xmn,ymx-ymn,color=c,ec='k',alpha=a)
        ax2.add_patch(r2)
        r3 = rect((xmn,ymn),xmx-xmn,ymx-ymn,color=c,ec='k',alpha=a)
        axk.add_patch(r3)
        x,y = (xmn+xmx)/2.0,(ymn+ymx)/2.0
        ax1.text(x,y,i+1,ha='center',va='center',fontsize=8)
        ax2.text(x,y,i+1,ha='center',va='center',fontsize=8)
        axk.text(x,y,i+1,ha='center',va='center',fontsize=8)
        


    axk.set_ylabel('Hydraulic conductivity\n ($m/d$)',multialignment='center')
    axk.set_xticklabels([])  
    axk.set_yticklabels(['','0','1','2','3','4'])  
    axk.set_ylim(-1,4.5)
    axk.set_xlim(0,cell_nums.max()+(0.5*delx))
    


    ax1.set_ylabel('Water level ($m$)')
    ax1.set_xticklabels([])    
    ax1.set_yticklabels(['','0','1','2','3','4','5','6'])  
    ax1.set_ylim(-1.0,6)
    ax1.set_xlim(0,cell_nums.max()+(0.5*delx))
    #ax1.grid()

    ax2.set_ylabel('Water level ($m$)')
    ax2.set_xlabel('Distance ($m$)')
    ax2.set_ylim(-1.0,6)
    ax2.set_yticklabels(['','0','1','2','3','4','5','6'])  
    ax2.set_xlim(0,cell_nums.max()+(0.5*delx))
    #ax2.grid()
    ax1.legend(scatterpoints=1,columnspacing=8,handletextpad=0.001,bbox_to_anchor=(.9,-1.35),ncol=2,frameon=False)
    ax2.xaxis.labelpad = 1.5
    pylab.savefig(plt_name,dpi=600,bbox_inches='tight')
    pylab.savefig(plt_name.replace('pdf','png'),dpi=600,bbox_inches='tight')



delx = 10.0 
obs_idxs,pred_idx = [3,5],8

syn_h = np.loadtxt('10par_xsec_truth.hds')

k_min,k_max = 0.85,2.65

os.system('pest.exe pest.pst')
cal_k = np.loadtxt('ref_cal\\hk_layer_1.ref')
cal_h = np.loadtxt('10par_xsec.hds')
cell_nums = np.arange(delx,(cal_k.shape[0]*delx)+delx,delx) - (0.5*delx)
plot_bar(syn_h,cal_k,cal_h,cell_nums,obs_idxs,delx,'results.png',k_min,k_max)
