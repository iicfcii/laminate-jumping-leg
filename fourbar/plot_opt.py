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

ang = 2.244662709242469
l = [0.028579223741827087, 0.06749028524988998, 0.021679870634057408, 0.06830135199294628, 0.06202390513377365]
kl = [0.05120860604313203, 0.07882789231797914, 0.21892879500649076, 0.20697146390036542]
c = 0.5021682587598642
dir = 0.6416980516856645
gnd = 0.6730089723128119

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
