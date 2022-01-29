import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt
import jump

legs = [
    {
        'a': 0.5,
        'ang': 1.5330165951890848,
        'l': [0.07821086139798128, 0.05800230739903228, 0.05974491078404546, 0.039669563041889094, 0.07299110256333716],
        'w': [0.01992913580520928, 0.019969342936040753],
        'c': 0.6984033828096015
    },
    {
        'a': 1,
        'ang': 1.818981965059142,
        'l': [0.04149849939200915, 0.07999850155041426, 0.04901079990098204, 0.05651422822074819, 0.07985902979555667],
        'w': [0.014534840571903303, 0.01986429251336412],
        'c': 0.80107402649113
    },
    {
        'a': 1.5,
        'ang': 2.392958814026799,
        'l': [0.0252396715905875, 0.07999154860624597, 0.02596436026202186, 0.07992545909210672, 0.07008131735322609],
        'w': [0.010013014447389498, 0.01995920888969171],
        'c': 0.03913435083553174
    }
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
    data_anchor = fourbar.solve(leg['ang'],leg['l'],leg['w'],leg['c'],opt.m,opt.cs,vis=False)
    sol_template = jump.solve(x0, cs)

    plt.subplot(3,1,2)
    plt.plot(sol_template.t,sol_template.y[2,:],color=c)
    plt.plot(data_anchor['t'],data_anchor['dyb'],'--',color=c)
    plt.ylabel('dyb [m/s]')

    plt.subplot(3,1,3)
    f_spring = -np.sign(sol_template.y[1,:])*cs['k']*cs['ds']*np.power(np.abs(sol_template.y[1,:]/cs['ds']),cs['a'])
    plt.plot(sol_template.t,f_spring,color=c)
    settle_idx = 100
    plt.plot(data_anchor['t'][settle_idx:],data_anchor['fy'][settle_idx:],'--',color=c)
    plt.ylabel('GRF [N]')
    plt.xlabel('Time [s]')



plt.show()
