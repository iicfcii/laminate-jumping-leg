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
fig,axes = plt.subplots(
    1,2,sharey=True,
    figsize=(3.4-plot.pad*2,2.7),dpi=150
)
lw = 0.8
lines = []

es = []
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

    theta_o,tau_o = stiffness.sim(x,cs['t']/k)

    theta_es,tau_es = spring.readn(1,k,a,type='spring')[:2]
    theta_el,tau_el = spring.readn(1,k,a,type='leg')[:2]
    tau_el = tau_el*0.034

    es.append(np.sqrt(np.mean((tau_es-tau_el)**2)))

    ko,ao = spring.fit(theta_o,tau_o,k,a)
    kes,aes = spring.fit(theta_es,tau_es,k,a)
    kel,ael = spring.fit(theta_el,tau_el,k,a)
    print(
        'k: {:.2f}, {:.2f}, {:.2f}, {:.2f} a: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2e}'
        .format(k,ko,kes,kel,a,ao,aes,ael,s['b'])
    )

    idx_a = int(i/2)
    idx_k = i%2
    c = 'C{:d}'.format(2-idx_a)
    ax = axes[idx_k]
    lines.append(ax.plot(theta_d,tau_d,'--',color=c,linewidth=lw)[0])
    lines.append(ax.plot(theta_o,tau_o,'^-',color=c,linewidth=lw,markersize=3,markevery=0.1)[0])
    lines.append(ax.plot(theta_es,tau_es,'.-',color=c,linewidth=lw,markersize=3,markevery=0.1)[0])
    lines.append(ax.plot(theta_el,tau_el,'-',color=c,linewidth=lw)[0])

print(np.mean(es))

axes[0].annotate(
    '0.1',
    xy=(1, 0),xycoords='axes fraction',
    xytext=(-2,2),textcoords='offset points',ha='right',va='bottom',
)
axes[1].annotate(
    '0.2',
    xy=(1, 0),xycoords='axes fraction',
    xytext=(-2,2),textcoords='offset points',ha='right',va='bottom',
)
axes[0].legend(lines[20:],['Goal','Model','Spring','Leg'],loc='upper left',handlelength=1,handletextpad=0.5)
axes[1].legend([lines[i] for i in [23,15,7]],['0.5','1.0','2.0'],loc='upper left',handlelength=1,handletextpad=0.5)
axes[0].set_xlabel('Rotation (rad)',labelpad=1)
axes[1].set_xlabel('Rotation (rad)',labelpad=1)
axes[0].set_ylabel('Torque (Nm)',labelpad=1)
plt.subplots_adjust(
    left=0.125,right=1,top=1,bottom=0.12,
    wspace=0.1,hspace=0
)
plot.savefig('stiffness.pdf',fig)
plt.show()
