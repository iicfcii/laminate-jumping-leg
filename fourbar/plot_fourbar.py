import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

pi = np.pi

ang = pi/2
l = [0.02,0.05,0.03,0.05,0.03]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

ang = -2.4145167310824736
l = [0.026177558219390808, 0.0706788587361037, 0.020078055635557472, 0.07598183996392696, 0.04422031239799459]
kl = [0.050022823068156086, 0.05007658227588757, 0.050056068360975675, 0.07240317027515142]
c = -1
dir = -1
gnd = 1

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=True)
print(opt.error(data))

plt.figure()
plt.plot(data['t'],data['fy'],'g')
plt.plot(opt.td,opt.fyd,'--g')
plt.show()
