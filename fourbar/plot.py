import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt
import jump

legs = [
    {
        'a': 0.3,
        'ang': 2.2075791077492672,
        'l': [0.04392701858320386, 0.05137145517551715, 0.02399309493688156, 0.06309574318150324, 0.03708307130126535],
        'w': [0.019999455957769828, 0.01032957786771581],
        'c': 0.4939096230603035
    },
    {
        'a': 1,
        'ang': 2.3804474791155825,
        'l': [0.06755081787168496, 0.07997256244616045, 0.07753063929427381, 0.07901632753699994, 0.07855616289287963],
        'w': [0.010109614087652333, 0.019987246920490132],
        'c': 0.21294326973473754
    },
]

for i,leg in enumerate(legs):
    c = 'C{:d}'.format(i)

    ps,pf,ls = fourbar.leg(leg['ang'],leg['l'],leg['c'])
    ax = plt.subplot(3,len(legs),i+1)
    for link in ls:
        plt.plot(link[:,0],link[:,1],color=c)
    plt.axis('scaled')
    plt.xlim([-0.1,0.1])
    plt.ylim([pf[0,1],pf[0,1]+0.15])
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.annotate(
        'a={:.1f}'.format(leg['a']),
        xy=(0, 1),xycoords='axes fraction',
        xytext=(10,-10),textcoords='offset points',ha='left',va='top',
    )

    cs = opt.cs
    cs['a'] = leg['a']
    x0 = [0,0,0,0]
    data_anchor = fourbar.solve(leg['ang'],leg['l'],leg['w'],leg['c'],opt.m,opt.cs,vis=True)
    sol_template = jump.solve(x0, cs)

    plt.subplot(3,1,2)
    plt.plot(sol_template.t,sol_template.y[2,:],color=c,label='slip')
    plt.plot(data_anchor['t'],data_anchor['dyb'],'--',color=c,label='sim')
    plt.ylabel('dyb [m/s]')
    if i == 0: plt.legend()

    plt.subplot(3,1,3)
    f_spring = -np.sign(sol_template.y[1,:])*cs['k']*cs['ds']*np.power(np.abs(sol_template.y[1,:]/cs['ds']),cs['a'])
    plt.plot(sol_template.t,f_spring,color=c)
    settle_idx = 50
    plt.plot(data_anchor['t'][settle_idx:],data_anchor['fy'][settle_idx:],'--',color=c)
    plt.ylabel('GRF [N]')
    plt.xlabel('Time [s]')
plt.show()
