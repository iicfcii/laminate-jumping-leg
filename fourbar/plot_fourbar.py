import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

pi = np.pi

ang = pi/2
l = [0.02,0.05,0.03,0.05,0.05]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

# ang = -1.7002992941259667
# l = [0.04465517644032171, 0.08143877316527909, 0.061547424616484346, 0.07955661121068996, 0.032503209669644345]
# kl = [0.05193362578249802, 0.07823918247815259, 0.06415929824963923, 0.05280202940120371]
# c = 0.1909322494178718
# dir = -0.30557481405205733
# gnd = -0.1911573751747606
#
# ang = -2.078630075348734
# l = [0.0331561,0.07204999,0.03250145,.07525823,0.03039573]
# kl = [0.17093136,0.08716568,0.17115951,0.31366809]
# c = 1
# dir = -1
# gnd = -1

data = fourbar.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=True)
print(opt.error(data))

plt.figure()
plt.plot(data['t'],data['fx'],'r')
plt.plot(data['t'],data['fy'],'g')
plt.plot(data['t'],data['fbx'],'b')
plt.plot(opt.td,opt.fxd,'--r')
plt.plot(opt.td,opt.fyd,'--g')
plt.plot(opt.td,opt.fbxd,'--b')
plt.show()
