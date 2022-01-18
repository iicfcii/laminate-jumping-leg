import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

# Example
ang = np.pi/2
l = [0.02,0.05,0.03,0.05,0.03]
w = [0.01,0.01]
c = 1

ang = 2.2029910666303083
l = [0.07946564532277485, 0.058082251711145325, 0.054786734853385824, 0.07993246042476779, 0.07985428934824051]
w = [0.010018789519435294, 0.019917930967940693]
c = 0.6025797616490636

ang = 1.938885240600019
l = [0.059879679826619875, 0.056420406791583985, 0.041461379123213224, 0.059978847519599025, 0.0599586788428471]
w = [0.010016506240013451, 0.010471197810933574]
c = 0.7613383546335306

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
