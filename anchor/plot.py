import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
import fourbar
import jump
import opt

xm = [2.6935314437637747, 0.030244462243688645, 0.04668319649977162, 0.02002235749858264, 0.05998841948291793, 0.059996931859852574, 0.14061111190360398]
opt.obj_motion(xm,plot=True)

springs = [
    {
        'k': 60,
        'a': 0.6,
        'x': [0.022719555787812505, 0.023043317483742193, 0.02764335338361519, 0.018813653588212513, 0.012424828688381656, -0.6971887222380353]
    },
    {
        'k': 60,
        'a': 1,
        'x': [0.010001950482507765, 0.04039852621473746, 0.027650636112590013, 0.057829100440527456, 0.010006477628465193, -0.17417191968543333]
    },
    {
        'k': 60,
        'a': 1.4,
        'x': [0.010003966955005869, 0.07850835338751745, 0.057705589820291635, 0.04314961391846071, 0.010002695668866359, 0.25747308323053986]
    }
]

for i,s in enumerate(springs):
    cs = opt.cs
    cs['k'] = s['k']
    cs['a'] = s['a']
    x = s['x']
    ls = x[:4]
    w = x[4]
    c = x[5]

    lk = fourbar.spring(ls,c)
    sol = jump.solve(cs)
    cs['dsm'] = -np.min(sol.y[1,:])

    sdatum = fourbar.stiffness(x,cs,plot=False)
    x_d = np.linspace(-cs['dsm'],0,50)
    f_d = -jump.f_spring(x_d,cs)
    f = np.interp(x_d,sdatum['x'],sdatum['f'])/cs['r']
    x_d = -x_d
    f_d = -f_d
    f = -f

    jdatum = fourbar.jump(xm,x,cs,plot=False)
    t_idx = np.nonzero(np.array(jdatum['t']) > sol.t[-1])[0][0]
    t_j = jdatum['t'][:t_idx]
    y_j = jdatum['y'][:t_idx]
    dy_j = jdatum['dy'][:t_idx]

    c = 'C{:d}'.format(i+1)

    plt.figure('spring')
    plt.subplot(2,3,i+1)
    plt.axis('scaled')
    r = 0.01
    plt.xlim([-0.01,0.08])
    plt.ylim([-0.03,0.06])
    lines = []
    line_styles = [':','-','--','-']
    line_widths = [2,4,2,2]
    for j,link in enumerate(lk):
        lines.append(plt.plot(link[:,0],link[:,1],line_styles[j],color=c,linewidth=line_widths[j])[0])
    if i == 0: plt.legend(lines[:3],['input','ground','flexure'])
    if i == 0: plt.ylabel('y [m]')
    if i == 1: plt.xlabel('x [m]')
    if i != 0: plt.tick_params(axis='y',which='both',labelleft=False)
    plt.title('a={:.1f}'.format(s['a']))

    plt.subplot(212)
    plt.plot(x_d,f_d,color=c)
    plt.plot(x_d,f,'o',color=c,markerfacecolor='none')
    r = np.amax(f_d)
    plt.ylim([-0.1*r,1.1*r])
    plt.ylabel('torque/r [N]')
    plt.xlabel('angle*r [m]')

    plt.figure('jump')
    plt.plot(sol.t,sol.y[2,:],color=c)
    plt.plot(t_j,dy_j,'o',color=c,markerfacecolor='none',markevery=100)
    plt.ylabel('dy [m/s]')
    plt.xlabel('t [s]')

plt.show()
