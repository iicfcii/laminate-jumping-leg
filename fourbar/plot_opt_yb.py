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

# 500*yb + fxb, 0.25
ang = -1.9280456430847521
l = [0.022724071824681953, 0.06371768717854587, 0.020339503848124565, 0.05522763955020984, 0.08633304402073656]
kl = [0.051044677281452344, 0.11449003637773439, 0.295552321504255, 0.3331878977351622]
c = -0.04991036982346242
dir = -0.6620196824838007
gnd = 0.754993142986583

# 500*yb + fxb, 0.25
ang = 2.1249207404036206
l = [0.032706005186806775, 0.06521416927321094, 0.023393198906513396, 0.06589270725554587, 0.06252121570865696]
kl = [0.055140235946891825, 0.32777787440570244, 0.14178048774873203, 0.4999794090385143]
c = 0.01784176291680617
dir = 0.297637401877783
gnd = 0.74435030934043

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
