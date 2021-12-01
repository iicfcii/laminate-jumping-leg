import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import statics

DEG_PER_COUNT = 0.088
GAP_COUNT = int(4/DEG_PER_COUNT)
ZERO_FORCE_TH = 1e-3
POS_MID = 2068
R = 25/1000

PI = np.pi
DEG_2_RAD = PI/180

fs = data.read('../data/15mil_40mm_20mm_1_2.csv')

ps0 = np.array(fs['p'])
tzs0 = np.array(fs['tz'])

ps = ps0 + (tzs0>0)*GAP_COUNT/2 - (tzs0<0)*GAP_COUNT/2 # accounts gap
non_zero_idx = np.logical_or(tzs0 > ZERO_FORCE_TH, tzs0 < -ZERO_FORCE_TH) # remove zero force
ps = ps[non_zero_idx]
tzs = tzs0[non_zero_idx]

# # Raw and adjusted data
# plt.figure()
# plt.plot(ps0,tzs0,label='raw')
# plt.plot(ps,tzs,'.',label='adjusted')
# plt.xlabel('Position Count (0.088deg/count)')
# plt.ylabel('Base Torque [Nm]')
# plt.legend()
# plt.show()

pv, tv = statics.base2virtual(ps,tzs)
coeff = np.polyfit(pv,tv,1)
tvfit = np.polyval(coeff,pv)

plt.figure()
plt.plot(pv,tv)
plt.plot(pv,tvfit,label='k={:.3f} b={:.3f}'.format(*coeff))
plt.xlabel('Virtual Angle [rad]')
plt.ylabel('Virtual Torque [Nm]')
plt.legend()

plt.show()
