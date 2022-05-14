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
    figsize=(3.4-plot.pad*2,2.7),dpi=150
)
lw = 0.8
c = ['C0','C1','C0','C1','C0','C1']
lines = []

es = []
kps = []
aps = []

for i,s in enumerate(design.springs):
    k = s['k']
    a = s['a']
    x = s['x']

    theta_d = np.linspace(0,cs['t']/k,100)
    theta_dp = np.zeros((6,theta_d.shape[0]))
    theta_dp[4,:] = theta_d
    cs = jump.cs
    cs['k'] = k
    cs['a'] = a
    tau_d = jump.f_ts(theta_dp,cs,bound=False)

    theta_es,tau_es = spring.readn(1,k,a,type='spring')[:2]
    theta_el,tau_el = spring.readn(1,k,a,type='leg')[:2]
    tau_el = tau_el*0.034

    es.append(np.sqrt(np.mean((tau_es-tau_el)**2)))
    kp,ap = spring.fit(theta_el,tau_el,k,a)
    print(k,a,kp,ap)
    kps.append(kp)
    aps.append(ap)

    # lines.append(ax.plot(theta_d,tau_d,'--',color=c[i],linewidth=lw)[0])
    lines.append(ax.plot(theta_es,tau_es,'.-',color=c[i],linewidth=lw,markersize=2)[0])
    lines.append(ax.plot(theta_el,tau_el,'-',color=c[i],linewidth=lw,markersize=2)[0])

print(np.mean(es))
kps_10 = [kps[i] for i in [0,2,4]]
kps_20 = [kps[i] for i in [1,3,5]]
print('k=0.1',np.mean(kps_10),np.std(kps_10),kps_10)
print('k=0.2',np.mean(kps_20),np.std(kps_20),kps_20)

# lg = plt.legend([lines[0],lines[3]],['0.15','0.30'],loc='upper left',handlelength=1,handletextpad=0.5)
# ax.add_artist(lg)
# ax.legend(lines[:3],['Desired','Simulation','Experiment'],loc='lower right',handlelength=1,handletextpad=0.5)
ax.set_xlabel('Rotation (rad)',labelpad=1)
ax.set_ylabel('Torque (Nm)',labelpad=1)
plt.subplots_adjust(
    left=0.125,right=1,top=1,bottom=0.12,
    wspace=0,hspace=0
)
plot.savefig('stiffness.pdf',fig)
plt.show()
