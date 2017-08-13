from __future__ import print_function
import flopy
from flopy import utils as fu
import platform
import numpy as np
if 'window' in platform.platform().lower():
    newln = '\n'
else:
    newln = '\r\n'
print ('Starting to read HYDMOD data')


obs = flopy.utils.HydmodObs('freyberg.hyd.bin')
times = obs.get_times()

read_obsnames = obs.get_obsnames()

with open('freyberg.heads', 'w') as ofp:
    ofp.write('obsname value{0}'.format(newln))
    for coutname in read_obsnames:
        if coutname.startswith('HDI001o'):
            cv = obs.get_data(obsname=coutname,totim=times[1])
            ofp.write('{0:20s} {1:15.6E} {2}'.format(coutname+'c', cv[0][1], newln))
    for coutname in read_obsnames:
        cv = obs.get_data(obsname=coutname,totim=times[2])
        ofp.write('{0:20s} {1:15.6E} {2}'.format(coutname+'f', cv[0][-1], newln))
    
print('Now read River flux from the LIST file')
lst = fu.MfListBudget('freyberg.list')
RIV_flux = lst.get_incremental()['RIVER_LEAKAGE_IN']-lst.get_incremental()['RIVER_LEAKAGE_OUT']
with open('freyberg.rivflux', 'w') as ofp:
    ofp.write('obsname value{0}'.format(newln))
    ofp.write('rivflux_cal  {1:15.6E}{0}rivflux_fore  {2:15.6E}{0}'.format(newln, RIV_flux[0], RIV_flux[1]))

print('Finally read endpoint file to get traveltime')

endpoint_file = 'freyberg.mpenpt'
lines = open(endpoint_file, 'r').readlines()
items = lines[-1].strip().split()
travel_time = float(items[4]) - float(items[3])

with open('freyberg.travel', 'w') as ofp:
    ofp.write('travetime {0:15.6e}{1}'.format(travel_time, newln))

print('Completed processing model output')

