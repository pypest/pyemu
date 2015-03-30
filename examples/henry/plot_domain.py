import os
import numpy as np
import pylab as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

from flopy.utils import binaryfile as bf

delx = 0.1
delz = 0.1
ncol,nlay = 120, 20

x = np.linspace(0, ncol*delx, num=ncol+1, endpoint=True)
z = np.linspace(0, -nlay*delz, num=nlay+1,endpoint=True)

ucn = bf.UcnFile("MT3D001.UCN")
times = ucn.get_times()
conc1,conc2 = ucn.get_data(totim=times[0])[:,0,:].copy(), ucn.get_data(totim=times[1])[:,0,:]
xc = np.arange(0,ncol*delx,delx)
zc = np.arange(-delz,-(nlay+1)*delz,-delz)
X,Z = np.meshgrid(xc,zc)

f = open(os.path.join("misc", "bore_coords_henry_coarse_conc.dat"), 'r')
cnames, cx, cz = [], [], []
for line in f:
	raw = line.strip().split()
	xx = float(raw[1])
	l = int(raw[3])
	zz = z[l - 1] - (delz / 2.0)
	cnames.append(raw[0])
	cx.append(xx)
	cz.append(zz)
f.close()

f = open(os.path.join("misc", "bore_coords_henry_coarse_head.dat"), 'r')
hnames, hx, hz = [], [], []
for line in f:
	raw = line.strip().split()
	xx = float(raw[1])
	l = int(raw[3])
	zz = z[l - 1] - (delz / 2.0)
	hnames.append(raw[0])
	hx.append(xx)
	hz.append(zz)
f.close()


pp_locs = np.loadtxt(os.path.join("misc", "pp_locs.dat"), usecols=[1, 2])


fig = plt.figure(figsize=(6.5, 3.35))
ax = plt.axes((0.085, 0.575, 0.875, 0.485),aspect="equal")
ax2 = plt.axes((0.085, 0.2, 0.875, 0.485),aspect="equal")
zlim = [z.min(), z.max()]
xlim = [x.min(), x.max()]
fp = FontProperties()
fp.set_size("small")
#for xx in x:
#	ax.plot([xx, xx], zlim, color="0.25", lw=0.1)

ax.contour(X,Z,conc1,levels=[0.01,0.1,0.5],colors='0.5',linewidths=[1,2,3])
cont = ax2.contour(X,Z,conc2,levels=[0.01,0.1,0.5],colors='0.5',linewidths=[1,2,3])

#ax.plot([x, x], zlim, color="0.25", lw=0.1)
#ax2.plot([x, x], zlim, color="0.25", lw=0.1)
#ax.plot(xlim, [z ,z], color="0.25", lw=0.1)
#ax2.plot(xlim, [z ,z], color="0.25", lw=0.1)

ax.scatter(hx, hz, marker='o', edgecolor='k', facecolor="none", s=25, label="head obs")
ax.scatter(cx, cz, marker='x', color='k', s=12, label="concentration obs")
#ax2.scatter(hx, hz, marker='o', edgecolor='k', facecolor="none", s=25, label="head obs")
#ax2.scatter(cx, cz, marker='x', color='k', s=12, label="conc obs")



for xx, zz, nn in zip(hx, hz, hnames):
	ax.text(xx,zz+0.125,nn[-2:],ha="center",va="bottom",fontsize="small",
		bbox={"fc":"1.0","boxstyle":"round,pad=0.01","ec":"none"})

ax.scatter(pp_locs[:, 0], pp_locs[:, 1], marker='.', s=1,color='b', label="pilot point")
ax2.scatter(pp_locs[:, 0], pp_locs[:, 1], marker='.', s=1,color='b', label="pilot point")



fresh = Rectangle((x.min(), z.min()), delx, nlay * delz, color='b')
ax.add_patch(fresh)
fresh2 = Rectangle((x.min(), z.min()), delx, nlay * delz, color='b')
ax2.add_patch(fresh2)

salt = Rectangle((x.max() - delx, z.min()), delx,nlay*delz, color='r')
ax.add_patch(salt)
salt2 = Rectangle((x.max() - delx, z.min()), delx,nlay*delz, color='r')
ax2.add_patch(salt2)

ax.plot([0,0],[0,0],lw=3,color='0.5',label="50% saltwater")
ax.plot([0,0],[0,0],lw=2,color='0.5',label="10% saltwater")
ax.plot([0,0],[0,0],lw=1,color='0.5',label="1% saltwater")

# ax2.plot([5,12],[-2,-0.75],lw=3,color='0.5',dashes=(2,2))
# ax2.plot([4,12],[-2,-0.65],lw=2,color='0.5',dashes=(2,2))
# ax2.plot([3,12],[-2,-0.5],lw=1,color='0.5',dashes=(2,2))

handles, labels = ax.get_legend_handles_labels()
handles.append(fresh)
handles.append(salt)
labels.append("freshwater boundary")
labels.append("saltwater boundary")

ax2.legend(handles,labels, loc="lower center", 
	ncol=3,frameon=False,prop=fp,
	bbox_to_anchor=(0.5,-1.2),scatterpoints=1)

# bbox_dict = {"fc":"1.0","boxstyle":"round"}#,pad=0.1"}

# ax.annotate('', xy=(1,-1),  xycoords='data',
#                 xytext=(0.25, -1), textcoords='data',color='b',fontsize=14,
#                 arrowprops=dict(facecolor='b', shrink=0.01,alpha=0.5),
#                 horizontalalignment='right', verticalalignment="center",
#                 )
# ax.annotate('4.5 $\\frac{m^3}{day}$',xy=(1.25,-1),xytext=(1.25,-1),
# 	horizontalalignment='left', verticalalignment="center",
# 	fontsize=14,color='k',bbox=bbox_dict)

# ax2.annotate('', xy=(1,-1),  xycoords='data',
#                 xytext=(0.25, -1), textcoords='data',color='b',fontsize=14,
#                 arrowprops=dict(facecolor='b', shrink=0.01,alpha=0.5),
#                 horizontalalignment='right', verticalalignment="center",
#                 )
# ax2.annotate('0.0 $\\frac{m^3}{day}$',xy=(1.25,-1),xytext=(1.25,-1),
# 	horizontalalignment='left', verticalalignment="center",
# 	fontsize=14,color='k',bbox=bbox_dict)


ax.set_xlim(xlim)
ax.set_ylim(zlim)
ax.set_xticklabels([])
ax2.set_xlim(xlim)
ax2.set_ylim(zlim)
ax.set_yticklabels(ax.get_yticks(),fontsize="small")
ax2.set_yticklabels(ax.get_yticks(),fontsize="small")
ax2.set_xticklabels(ax.get_xticks(),fontsize="small")
ax.set_ylabel("depth (m)",fontsize="small")
ax2.set_ylabel("depth (m)",fontsize="small")
ax2.set_xlabel("length (m)",fontsize="small",labelpad=0.5)
ax.text(0.0,0.0,"A.) Calibration stress period",ha="left",va="bottom",fontsize="small")
ax2.text(0.0,0.0,"B.) Forecast stress period",ha="left",va="bottom",fontsize="small")

plt.savefig("domain.png")
#plt.savefig("domain.eps")
