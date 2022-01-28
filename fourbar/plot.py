import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

# Example
ang = np.pi/2
l = [0.02,0.05,0.03,0.05,0.03]
w = [0.01,0.01]
c = 1

# a = 0.5
ang = 1.7755378150593206
l = [0.05992196857785843, 0.056550007709236655, 0.04327039453840696, 0.052401351839215154, 0.05866232514079765]
w = [0.01793250820013311, 0.0183556041989514]
c = 0.11201933995931124

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

plt.show(block=False)
while len(plt.get_fignums()) > 0:
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.start_event_loop(0.001)
