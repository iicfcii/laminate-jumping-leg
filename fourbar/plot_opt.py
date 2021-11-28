import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt_f

pi = np.pi

ang = pi/2
l = [0.02,0.05,0.03,0.05,0.03]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

# Only fy
ang = -2.4145167310824736
l = [0.026177558219390808, 0.0706788587361037, 0.020078055635557472, 0.07598183996392696, 0.04422031239799459]
kl = [0.050022823068156086, 0.05007658227588757, 0.050056068360975675, 0.07240317027515142]
c = -1
dir = -1
gnd = 1

# fy & fx
ang = -0.34680725297973986
l = [0.05066431,0.09284251,0.07083207,0.0411094,0.04399573]
kl = [0.0530217,0.07382111,0.06119576,0.13298325]
c = 1
dir = -1
gnd = -1

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt_f.cs,vis=True)
print(opt_f.error(data))

plt.figure()
plt.subplot(211)
plt.plot(data['t'],data['yb'])
plt.subplot(212)
plt.plot(data['t'],data['dyb'])

plt.figure()
plt.plot(data['t'],data['fy'],'r')
plt.plot(data['t'],data['fx'],'g')
plt.plot(data['t'],data['fxb'],'b')
plt.plot(opt_f.td,opt_f.fyd,'--r')
plt.show()
