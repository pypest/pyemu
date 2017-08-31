import os
import platform
if 'window' in platform.platform().lower():
    pref = ''
else:
    pref = './'
os.system('{0}mf2005 freyberg.truth.nam'.format(pref))
os.system('{0}mp6 <mpath.in'.format(pref))
os.system('python Process_output.py')
