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
        'ang': 2.570551068707523,
        'l': [0.05994585983921759, 0.08442453971863799, 0.06933555975856003, 0.07189836171466216, 0.039000680113049885],
        'w': [0.029998372043533394, 0.014069976672384188],
        'c': 0.46798736504096894
    },
    {
        'a': 1,
        'ang': 2.1089983192180397,
        'l': [0.05961906698960521, 0.09992781973555374, 0.08797699992125604, 0.06253605843706087, 0.08436185008542484],
        'w': [0.01326493292554198, 0.025911455268780813],
        'c': 0.7417733854253834
    },
    # {
    #     'a': 1.6,
    #     'ang': -2.319457248782482,
    #     'l': [0.07281165121513805, 0.05208670072140321, 0.06649918176112886, 0.06181435506879609, 0.09607799324941167],
    #     'w': [0.02999474116430211, 0.027490694762152865],
    #     'c': 0.1926050976657201,
    # }
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
    plt.plot(sol_template.t,sol_template.y[2,:],color=c,label='slip')
    plt.plot(data_anchor['t'],data_anchor['dyb'],'--',color=c,label='sim')
    plt.ylabel('dyb [m/s]')
    if i == 0: plt.legend()

    plt.subplot(3,1,3)
    f_spring = -np.sign(sol_template.y[1,:])*cs['k']*cs['ds']*np.power(np.abs(sol_template.y[1,:]/cs['ds']),cs['a'])
    plt.plot(sol_template.t,f_spring,color=c)
    plt.plot(data_anchor['t'][0:],data_anchor['fy'][0:],'--',color=c)
    plt.ylim([-0.1,1.3])
    plt.ylabel('GRF [N]')
    plt.xlabel('Time [s]')

    print(opt.total_mass(leg['l'],leg['w']))
    print(sol_template.y[2,-1])
plt.show()
