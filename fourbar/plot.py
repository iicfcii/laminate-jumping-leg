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
    # {
    #     'a': 0.7,
    #     'ang': 2.2200281553718217,
    #     'l': [0.05940487146667405, 0.09539904002402695, 0.07138260138964389, 0.07191363737539129, 0.05415807165432748],
    #     'w': [0.025582019648294124, 0.02969010388516696],
    #     'c': 0.2958124954343728
    # },
    {
        'a': 1,
        'ang': 2.1089983192180397,
        'l': [0.05961906698960521, 0.09992781973555374, 0.08797699992125604, 0.06253605843706087, 0.08436185008542484],
        'w': [0.01326493292554198, 0.025911455268780813],
        'c': 0.7417733854253834
    },
    # {
    #     'a': 1.3,
    #     'ang': 2.491367903035188,
    #     'l': [0.05556700892076306, 0.09999293730926566, 0.07217906391125171, 0.08882429444584387, 0.07509109676400946],
    #     'w': [0.01000144609345023, 0.029800295321308137],
    #     'c': 0.4237738994842939
    # },
    {
        'a': 1.6,
        'ang': 2.551825418110649,
        'l': [0.056659798104717246, 0.09997388680060479, 0.062482621009652024, 0.09973640592115117, 0.06971316781135563],
        'w': [0.010019847366950848, 0.029716186137395323],
        'c': 0.1258847876241307
    }
]

nrow = 4

for i,leg in enumerate(legs):
    c = 'C{:d}'.format(i)

    ps,pf,ls = fourbar.leg(leg['ang'],leg['l'],leg['c'])
    ax = plt.subplot(nrow,len(legs),i+1)
    for j, link in enumerate(ls):
        lw = 3
        if j == 1 or j == 2:
            lw = lw*leg['w'][0]/0.02
        elif j == 4 or j == 5:
            lw = lw*leg['w'][1]/0.02
        else:
            pass
        plt.plot(link[:,0],link[:,1],color=c,linewidth=lw)
    plt.axis('scaled')
    plt.xlim([-0.12,0.12])
    plt.ylim([pf[0,1]-0.005,pf[0,1]+0.16])
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

    f_spring = jump.f_spring(sol_template,cs)
    dy_spring = sol_template.y[3,:]
    dy_body = sol_template.y[2,:]

    plt.subplot(nrow,1,2)
    plt.plot(sol_template.t,sol_template.y[2,:],color=c,label='template')
    plt.plot(data_anchor['t'],data_anchor['dyb'],'--',color=c,label='anchor')
    plt.ylabel('dyb [m/s]')
    if i == 0: plt.legend()

    plt.subplot(nrow,1,3)
    f_spring = -np.sign(sol_template.y[1,:])*cs['k']*cs['ds']*np.power(np.abs(sol_template.y[1,:]/cs['ds']),cs['a'])
    plt.plot(sol_template.t,f_spring,color=c)
    plt.plot(data_anchor['t'],data_anchor['fy'],'--',color=c)
    plt.ylim([-0.1,1.3])
    plt.ylabel('GRF [N]')
    plt.xlabel('Time [s]')

    plt.subplot(nrow,1,4)
    plt.plot(sol_template.t,f_spring*dy_body,color=c)
    plt.plot(data_anchor['t'],np.array(data_anchor['fy'])*np.array(data_anchor['dyb']),'--',color=c)
    plt.ylim([-0.1,1.3])
    plt.ylabel('Power [W]')
    plt.xlabel('Time [s]')

    # print(opt.total_mass(leg['l'],leg['w']))
    # print(sol_template.y[2,-1])
plt.show()
