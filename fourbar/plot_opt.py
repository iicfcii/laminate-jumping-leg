import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

pi = np.pi

# Example
ang = pi/2
l = [0.02,0.05,0.03,0.05,0.03]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

# m = 0.03
ang = -2.763701425319136
l = [0.1419177257747958, 0.05777685126888972, 0.13460936689096237, 0.08025894256522381, 0.0615142703191763]
kl = [0.08291220677239865, 0.22812260172365365, 0.16358487458002724, 0.052296712203683426]
c = 0.6056401948741517
dir = -0.8940270569874951
gnd = 0.4201807081543838

# m = 0.04
ang = -1.6676309995859535
l = [0.028880359240924316, 0.06458933718587634, 0.020136709406511355, 0.05449585918362976, 0.08177319725446737]
kl = [0.05000418474647815, 0.09129264955571645, 0.28278342524992484, 0.14780910456693147]
c = -0.20235785135937656
dir = -0.02647066819597288
gnd = 0.4620647164175584

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt.m,opt.cs,vis=True)
print('Error yb',opt.error_yb(data))
print('Error dyb',opt.error_dyb(data))
print('Max fxb',opt.fxb_max(data))

plt.figure()
plt.subplot(211)
plt.plot(data['t'],data['yb'])
plt.plot(opt.td,opt.ybd,'--')
plt.subplot(212)
plt.plot(data['t'],data['dyb'])
plt.plot(opt.td,opt.dybd,'--')

plt.figure()
plt.plot(data['t'],data['fy'],label='fy')
plt.plot(data['t'],data['fx'],label='fx')
plt.plot(data['t'],data['fxb'],label='fxb')
plt.legend()
plt.show()
