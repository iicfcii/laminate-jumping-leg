import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from anchor import geom
from anchor import stiffness
from anchor import motion
from stiffness import spring
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
fig,ax = plt.subplots(
    1,1,
    figsize=(3.25-plot.pad*2,2.5-plot.pad*2),dpi=150
)
lw = 0.8
ls = ['x-','+-','s-','*-']

s = design.springs[5]
k = s['k']
a = s['a']
x = s['x']

theta_d = np.linspace(0,cs['t']/k,100)
theta_dp = np.zeros((7,theta_d.shape[0]))
theta_dp[4,:] = theta_d
cs = jump.cs
cs['k'] = k
cs['a'] = a
tau_d = jump.f_ts(theta_dp,cs)

for i,s in enumerate([83,82,81,80]):
    theta,tau = spring.readn(s,k,a,type='spring')[:2]
    # kes_nl,aes_nl = spring.fit(theta_es_nl,tau_es_nl,k,a)
    # print('k={:.2f} a={:.2f}'.format(kes_nl,aes_nl))

    ax.plot(theta,tau,ls[i],color='C0',linewidth=lw,markersize=3,markevery=0.1)

ax.set_xlabel('Rotation (rad)',labelpad=1)
ax.set_ylabel('Torque (Nm)',labelpad=1)
ax.legend([
    'Non-laminate','16mm-wide','10mm-wide','5mm-wide',
],loc='lower right',handlelength=1,handletextpad=0.5)


plt.subplots_adjust(
    left=0.14,right=1,top=1,bottom=0.14,
    wspace=0,hspace=0
)
plot.savefig('soft.pdf',fig)
plt.show()
