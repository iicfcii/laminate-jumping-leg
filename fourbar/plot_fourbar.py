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

ang = -1.7002992941259667
l = [0.04465517644032171, 0.08143877316527909, 0.061547424616484346, 0.07955661121068996, 0.032503209669644345]
kl = [0.05193362578249802, 0.07823918247815259, 0.06415929824963923, 0.05280202940120371]
c = 0.1909322494178718
dir = -0.30557481405205733
gnd = -0.1911573751747606

t,fx,fy,fxb = fourbar.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=True)
print(opt.error(t,fx,fy,fxb))

fxi = np.interp(opt.td,t,fx)
fyi = np.interp(opt.td,t,fy)
fxbi = np.interp(opt.td,t,fxb)

plt.figure()
plt.plot(t,fx,'r')
plt.plot(t,fy,'g')
plt.plot(t,fxb,'b')
# plt.plot(opt.td,fxi,'r')
# plt.plot(opt.td,fyi,'g')
# plt.plot(opt.td,fxbi,'b')
plt.plot(opt.td,opt.fxd,'--r')
plt.plot(opt.td,opt.fyd,'--g')
plt.plot(opt.td,opt.fxbd,'--b')
plt.show()
