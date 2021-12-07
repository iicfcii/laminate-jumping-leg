import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

pi = np.pi

# Example
ang = pi/2
l = [0.02,0.05,0.03,0.05,0.03]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

# -dyb_max, no fxb limit, 0.25
# INTERESTING SOLUTION!!
ang = 2.760163375375682
l = [0.029730588730247382, 0.025131295373545158, 0.03935713145234707, 0.039507567564495916, 0.038731022572189244]
kl = [0.4591740394164685, 0.17641535492928823, 0.39224790663405096, 0.38980548287913863]
c = 0.5891687937944079
dir = -0.20008492607684736
gnd = 0.3648197007859846

ang = -2.126361718124675
l = [0.024816079209639616, 0.06904438816102355, 0.0203930516036767, 0.06604347779350797, 0.0696338741205973]
kl = [0.054547470914424745, 0.493613396362352, 0.16403917692818398, 0.49705297000614396]
c = -0.23250675631815454
dir = -0.27359780502206
gnd = 0.5040922079470005

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=True)
print('Error yb',opt.error_yb(data))
print('Error dyb',opt.error_dyb(data))
print('Max fxb',opt.fxb_max(data))

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
