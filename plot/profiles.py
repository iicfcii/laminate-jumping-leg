import numpy as np
import matplotlib.pyplot as plt
from template import jump
from utils import plot

plot.set_default()
cs = jump.cs

k = 0.1
labels = ['(a)','(b)','(c)']
for i,a in enumerate([0.5,1,2]):
    fig,ax = plt.subplots(
        1,1,figsize=(3.4/3-plot.pad*2,3.4/3-plot.pad*2),dpi=150
    )

    theta_d = np.linspace(0,cs['t']/k,100)
    theta_dp = np.zeros((7,theta_d.shape[0]))
    theta_dp[4,:] = theta_d
    cs = jump.cs
    cs['k'] = k
    cs['a'] = a
    tau_d = jump.f_ts(theta_dp,cs)

    c = 'C{:d}'.format(i)
    ax.plot(theta_d,tau_d,'-',color=c,linewidth=1)

    ax.spines[["left", "bottom"]].set_linewidth(1)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1,0,'>k',markersize=3,transform=ax.get_yaxis_transform(),clip_on=False)
    ax.plot(0,1,'^k',markersize=3,transform=ax.get_xaxis_transform(),clip_on=False)
    ax.set_ylabel('$F$',labelpad=1)
    ax.set_xlabel('$x$',labelpad=1)
    ax.set_title('{} $a={:.1f}$'.format(labels[i],a),pad=4)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(
        left=0.08,right=1-0.04,top=1-0.14,bottom=0.05,
        wspace=0,hspace=0
    )
    plot.savefig('profile_{:d}.pdf'.format(int(a*10)),fig)
plt.show()
