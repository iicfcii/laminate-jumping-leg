import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt_yb

pi = np.pi

ang = pi/2
l = [0.02,0.05,0.03,0.05,0.03]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

ang = 2.1979324373999862
l = [0.03239357,0.07413422,0.02155799,0.07805589,0.06864978]
kl = [0.1515252,0.0590249,0.21658157,0.23939019]
c = 1
dir = 1
gnd = 1

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt_yb.cs,vis=True)
print(opt_yb.error(data))

plt.figure()
plt.subplot(211)
plt.plot(data['t'],data['yb'])
plt.plot(opt_yb.td,opt_yb.ybd,'--')
plt.subplot(212)
plt.plot(data['t'],data['dyb'])

plt.figure()
plt.plot(data['t'],data['fy'])
plt.plot(data['t'],data['fx'])
plt.plot(data['t'],data['fxb'])
plt.show()
