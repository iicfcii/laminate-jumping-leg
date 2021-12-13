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
ang = -2.792954810132125
l = [0.14235800521334388, 0.057775041091248265, 0.13501140377666365, 0.07853069964858439, 0.05834867111113904]
kl = [0.08736304442644566, 0.22348203262397276, 0.1639343103535335, 0.05093633150306032]
c = 0.45797288071140585
dir = -0.47173672750811313
gnd = 0.17204165254339832

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
