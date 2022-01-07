import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

# Example
ang = np.pi/2
l = [0.02,0.05,0.03,0.05,0.03]
w = [0.01,0.01]
c = 1

ang = 2.3216204786959906
l = [0.04015875608606949, 0.09998200125783568, 0.062000599936210904, 0.06963927061897716, 0.09995158444861704]
w = [0.005012474821183483, 0.010875312309250658]
c = 0.483487326384771

print('Total leg length',np.sum(l))
data = fourbar.solve(ang,l,w,c,opt.m,opt.cs,vis=True)
print('Error yb',opt.error_yb(data))
print('Error dyb',opt.error_dyb(data))
print('Max fxb',opt.fxb_max(data))
print('Rot',data['rot'][0],data['rot'][-1])

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
