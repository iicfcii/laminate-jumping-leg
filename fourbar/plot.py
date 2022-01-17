import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

# Example
ang = np.pi/2
l = [0.02,0.05,0.03,0.05,0.03]
w = [0.01,0.01]
c = 1

ang = 2.2913330204176514
l = [0.07807027914982999, 0.053799139129972355, 0.05563948920585282, 0.07440129960332645, 0.07682228491377874]
w = [0.006848308272490824, 0.009984457953306906]
c = 0.6044359945471032

ang = 2.210377033737202
l = [0.05999194486467203, 0.049616776624461754, 0.04482698752231912, 0.05998218508075233, 0.05999610414449435]
w = [0.005349973625895341, 0.007032767855536977]
c = 0.32291998292001267

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
