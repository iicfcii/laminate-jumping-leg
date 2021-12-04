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

# -dyb_max, fxb < 1, 0.25
ang = 1.770752365436482
l = [0.0443112649450014, 0.05657637733700611, 0.0336042664304165, 0.04724909305360327, 0.05922732232156915]
kl = [0.050649966518107464, 0.43001386744651426, 0.23413659725325275, 0.41675635233050434]
c = 0.2423835183760188
dir = 0.8097738448823664
gnd = 0.7067053607156748

# e_fy, fxb < 1
ang = -2.9928473711254666
l = [0.04791774828015445, 0.052874922890627965, 0.05162823682773837, 0.04924049426164167, 0.04803513909845913]
kl = [0.11233497649620747, 0.18045603115482095, 0.13483364001677878, 0.06496510720758752]
c = -0.37663736457539765
dir = -0.38625148820798205
gnd = 0.3366572761454987

# e_fy, fxb < 1.5
ang = -1.9512716463450457
l = [0.024471082735499958, 0.06407981757666133, 0.021186007685710134, 0.05604502958973948, 0.06796539027239894]
kl = [0.057214857320147666, 0.08767206637739314, 0.08242810778640594, 0.3991327590770195]
c = -0.27045700459346167
dir = -0.4500156821208634
gnd = 0.16264069073742515

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

plt.figure()
plt.plot(data['t'],data['fy'],label='fy')
plt.plot(data['t'],data['fx'],label='fx')
plt.plot(data['t'],data['fxb'],label='fxb')
plt.legend()
plt.show()
