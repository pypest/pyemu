import numpy as np
import shapefile


shp = shapefile.Reader("freyberg.shp")
fields = [item[0] for item in shp.fields[1:]]
r_idx = fields.index("row")
c_idx = fields.index("column")
z_idx = fields.index("k_zone")

rows,cols, zones = [],[],[]
for rec in shp.records():
	rows.append(int(rec[r_idx]))
	cols.append(int(rec[c_idx]))
	zones.append(int(rec[z_idx]))

arr = np.zeros((max(rows),max(cols)))
print(arr.shape)
for r,c,z in zip(rows,cols,zones):
	arr[r-1,c-1] = z

ib = np.loadtxt("ibound.ref")
arr += 1
arr[ib==0] = 1

np.savetxt("kzone.ref",arr,fmt="%3.0f",delimiter='')

with open("hk_zone.ref.tpl",'w') as f:
	f.write("ptf ~\n")
	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]):
			pname = "hkz_{0:1.0f}".format(arr[i,j])
			tpl_str = " ~    {0}   ~ ".format(pname)
			f.write(tpl_str)
		f.write('\n')