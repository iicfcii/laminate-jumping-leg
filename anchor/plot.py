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
        'k': 30,
        'a': 0.3,
        'x': [0.06537370705668276, 0.023284162574029352, 0.07996217581835618, 0.010672120755835596, 0.01000126926198963, -0.024262242372249054]
    },
    {
        'k': 30,
        'a': 1,
        'x': [0.010000735732930893, 0.06728554648775059, 0.02533001000361058, 0.0799974768780288, 0.010002220051112223, -0.69425169936661]
    },
    {
        'k': 30,
        'a': 1.5,
        'x': [0.010012720468809341, 0.14991587258031291, 0.13002315167020076, 0.06158692662450015, 0.010000461060269049, 0.44561305776894566]
    }
]

plt.figure()
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
    x_d = np.linspace(-cs['dsm'],0,50)
    f_d = -jump.f_spring(x_d,cs)

    datum = fourbar.stiffness(x,cs,plot=False)
    l = len(datum['x'])

    x1 = datum['x'][:int(l/2)]
    x1.reverse()
    x1 = np.array(x1)*cs['r']
    x2 = datum['x'][int(l/2):]
    x2 = np.array(x2)*cs['r']

    f1 = datum['f'][:int(l/2)]
    f1.reverse()
    f2 = datum['f'][int(l/2):]
    f1_i = np.interp(x_d,x1,f1)
    f2_i = np.interp(x_d,x2,f2)
    f = (f1_i+f2_i)/2/cs['r']

    x_d = -x_d
    f_d = -f_d
    f = -f

    c = 'C{:d}'.format(i+1)

    plt.subplot(2,3,i+1)
    plt.axis('scaled')
    r = 0.01
    plt.xlim([-0.01,0.11])
    plt.ylim([-0.06,0.14])
    lines = []
    line_styles = [':','-','--','-']
    line_widths = [2,4,2,2]
    for j,link in enumerate(lk):
        lines.append(plt.plot(link[:,0],link[:,1],line_styles[j],color=c,linewidth=line_widths[j])[0])
    if i == 0: plt.legend(lines[:3],['input','ground','flexure'])
    if i == 0: plt.ylabel('y [m]')
    if i == 1: plt.xlabel('x [m]')

    plt.subplot(212)
    plt.plot(x_d,f_d,color=c,label='a={:.1f}'.format(s['a']))
    plt.plot(x_d,f,'o',color=c,markerfacecolor='none')
    r = np.amax(f_d)
    plt.ylim([-0.1*r,1.1*r])
    plt.ylabel('torque/r [N]')
    plt.xlabel('angle*r [m]')
    plt.legend()

plt.show()
