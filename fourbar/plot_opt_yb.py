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

ang = 3.0121799553868307
l = [0.06708632,0.05620438,0.05549923,0.06817905,0.05033273]
kl = [0.23731334,0.15245639,0.38972191,0.06602987]
c = 0.08978929783061318
dir = 0.8375915414165533
gnd = 0.5639741527612223

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt_yb.cs,vis=True)
print(opt_yb.error(data))

plt.figure()
plt.subplot(211)
plt.plot(data['t'],data['yb'])
plt.plot(opt_yb.td,opt_yb.ybd,'--')
plt.subplot(212)
plt.plot(data['t'],data['dyb'])

plt.figure()
plt.plot(data['t'],data['fy'],'r')
plt.plot(data['t'],data['fx'],'g')
plt.plot(data['t'],data['fxb'],'b')
plt.show()
