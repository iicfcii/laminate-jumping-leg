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

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt_yb.cs,vis=True)
print(opt_yb.error(data))

plt.figure()
plt.subplot(211)
plt.plot(data['t'],data['yb'])
plt.plot(opt_yb.td,opt_yb.ybd,'--')
plt.subplot(212)
plt.plot(data['t'],data['dyb'])
plt.show()
