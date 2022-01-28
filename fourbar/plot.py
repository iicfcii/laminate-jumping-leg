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
ang = 1.5330165951890848
l = [0.07821086139798128, 0.05800230739903228, 0.05974491078404546, 0.039669563041889094, 0.07299110256333716]
w = [0.01992913580520928, 0.019969342936040753]
c = 0.6984033828096015

# a = 1
ang = 1.818981965059142
l = [0.04149849939200915, 0.07999850155041426, 0.04901079990098204, 0.05651422822074819, 0.07985902979555667]
w = [0.014534840571903303, 0.01986429251336412]
c = 0.80107402649113

# a = 1.5
ang = 2.392958814026799
l = [0.0252396715905875, 0.07999154860624597, 0.02596436026202186, 0.07992545909210672, 0.07008131735322609]
w = [0.010013014447389498, 0.01995920888969171]
c = 0.03913435083553174

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
