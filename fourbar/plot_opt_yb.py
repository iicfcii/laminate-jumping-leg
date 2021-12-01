import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt_yb

pi = np.pi

# Example
ang = pi/2
l = [0.02,0.05,0.03,0.05,0.03]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

# 500*yb + fxb, 0.25
ang = 2.02005212013153
l = [0.0219224,0.0653215,0.0205432,0.05751279,0.08330382]
kl  = [0.05513392,0.11265178,0.19884944,0.35118946]
c = 0.5818209278698765
dir = 0.34871113305293244
gnd = 0.8382979725911825

# 300*yb + fxb, 0.25
ang = 3.1302896716091846
l = [0.02035920711384507, 0.062040547756663095, 0.022038780972307014, 0.06062427204065637, 0.06168172348830375]
kl = [0.39295247734089656, 0.14532046277776883, 0.47365927066794095, 0.27146732920567757]
c = -0.27702697318808867
dir = -0.08762987871330707
gnd = 0.10976174612652079

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
