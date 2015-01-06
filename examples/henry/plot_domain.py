import os
import numpy as np
import pylab as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

delx = 0.1
delz = 0.1
ncol,nlay = 120, 20

x = np.linspace(0, ncol*delx, num=ncol+1, endpoint=True)
z = np.linspace(0, -nlay*delz, num=nlay+1,endpoint=True)

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


fig = plt.figure(figsize=(10, 4))
ax = plt.axes((0.1, 0.2, 0.8, 0.7))
zlim = [z.min(), z.max()]
xlim = [x.min(), x.max()]

#for xx in x:
#	ax.plot([xx, xx], zlim, color="0.25", lw=0.1)
ax.plot([x, x], zlim, color="0.25", lw=0.1)
#for zz in z:
#	ax.plot(xlim, [zz ,zz], color="0.25", lw=0.1)
ax.plot(xlim, [z ,z], color="0.25", lw=0.1)

ax.scatter(hx, hz, marker='o', edgecolor='k', facecolor="none", s=25, label="head obs")
ax.scatter(cx, cz, marker='x', color='k', s=12, label="conc obs")

for xx, zz, nn in zip(hx, hz, hnames):
	ax.text(xx,zz+0.025,nn[-2:],ha="center",va="bottom",fontsize="small")

ax.scatter(pp_locs[:, 0], pp_locs[:, 1], marker='.', s=1,color='b', label="pilot point")

fresh = Rectangle((x.min(), z.min()), delx, nlay * delz, color='g')
ax.add_patch(fresh)

salt = Rectangle((x.max() - delx, z.min()), delx,nlay*delz, color='r')
ax.add_patch(salt)

handles, labels = ax.get_legend_handles_labels()
handles.append(fresh)
handles.append(salt)
labels.append("freshwater")
labels.append("saltwater")
fp = FontProperties()
fp.set_size("small")
ax.legend(handles,labels, loc="lower center", ncol=5,fancybox=True,shadow=True,prop=fp,bbox_to_anchor=(0.5,-0.25))

ax.set_xlim(xlim)
ax.set_ylim(zlim)




plt.savefig("domain.png")
