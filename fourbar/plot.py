import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

# Example
ang = np.pi/2
l = [0.02,0.05,0.03,0.05,0.03]
w = [0.01,0.01]
c = 1

ang = 2.3604768825602203
l = [0.06116620767918308, 0.09754825655869397, 0.06867398447451244, 0.0830258411235155, 0.08263429360716978]
w = [0.035074930328427546, 0.02905278362101117]
c = 0.054193561679969804

data = fourbar.solve(ang,l,w,c,opt.m,opt.cs,vis=True)
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
