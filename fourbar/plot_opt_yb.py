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

# yb, 0.3
ang = 2.1979324373999862
l = [0.03239357,0.07413422,0.02155799,0.07805589,0.06864978]
kl = [0.1515252,0.0590249,0.21658157,0.23939019]
c = 1
dir = 1
gnd = 1

# 100*yb + fxb, 0.3
ang = 2.941768779693176
l = [0.0718908670219532, 0.05088373436700236, 0.06593728979733518, 0.056406916594232186, 0.05442640516111975]
kl = [0.21759849235259887, 0.17577529466389866, 0.26239261752837345, 0.050362348358951536]
c = 1
dir = 1
gnd = 1

# 500*yb + fxb, 0.3
ang = -2.6012515184777114
l = [0.045291703949605064, 0.05376125929658659, 0.03490013595952238, 0.0659839156054198, 0.09999127940091848]
kl = [0.24873257043679, 0.050049453617168504, 0.21650317049997703, 0.05012297901536136]
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
plt.plot(data['t'],data['fy'],'r')
plt.plot(data['t'],data['fx'],'g')
plt.plot(data['t'],data['fxb'],'b')
plt.show()
