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

t, fx, fy = fourbar.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=True)
print(opt.error(t, fx, fy))

fxi = np.interp(opt.td,t,fx)
fyi = np.interp(opt.td,t,fy)

plt.figure()
plt.plot(t,fx,'r')
plt.plot(t,fy,'g')
# plt.plot(opt.td,fxi,'r')
# plt.plot(opt.td,fyi,'g')
plt.plot(opt.td,opt.fxd,'--r')
plt.plot(opt.td,opt.fyd,'--g')
plt.show()
