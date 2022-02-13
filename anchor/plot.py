import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
import fourbar
import jump
import opt

springs = [
    {
        'k': 40,
        'a': 0.6,
        'x': [0.01873827104162616, 0.02806787363125099, 0.031059340682086066, 0.01678021004645687, 0.01216246716261322, -0.9110247494808081]
    },
    {
        'k': 60,
        'a': 0.6,
        'x': [0.022719555787812505, 0.023043317483742193, 0.02764335338361519, 0.018813653588212513, 0.012424828688381656, -0.6971887222380353]
    },
    {
        'k': 80,
        'a': 0.6,
        'x': [0.02645817609803363, 0.019255140323088185, 0.024537695187959396, 0.021801699065303846, 0.010504557378155996, -0.8922823063272325]
    },
    {
        'k': 40,
        'a': 1,
        'x': [0.010012131658306514, 0.05429677864028192, 0.035094721706915084, 0.0739106305463264, 0.01000406293402808, -0.3856909524602533]
    },
    {
        'k': 60,
        'a': 1,
        'x': [0.010001950482507765, 0.04039852621473746, 0.027650636112590013, 0.057829100440527456, 0.010006477628465193, -0.17417191968543333]
    },
    {
        'k': 80,
        'a': 1,
        'x': [0.010002857765228523, 0.03302683572682277, 0.02177805127743359, 0.048108483937276375, 0.010006617251138208, -0.4908704598225935]

    },
    {
        'k': 40,
        'a': 1.4,
        'x': [0.01000084823520056, 0.0987368832726395, 0.09424337856355897, 0.05295371989810913, 0.01002637466123165, 0.7551193325175176]
    },
    {
        'k': 60,
        'a': 1.4,
        'x': [0.010054699319005767, 0.07351665894667067, 0.06464472377895077, 0.042061331551113794, 0.01000897221556764, 0.07885124682829492]
    },
    {
        'k': 80,
        'a': 1.4,
        'x': [0.010097299615116129, 0.06042533137942364, 0.050572388949174416, 0.036647568879639957, 0.010011377415757075, 0.013018350790240607]
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
    color = 'C{:d}'.format(i+1)

    plt.figure('shape',figsize=(7.95,8))
    plt.subplot(3,3,i+1)

    lines = []
    line_styles = [':','-','--','-']
    line_widths = [2,3,2,2]
    for j,link in enumerate(lk):
        lines.append(plt.plot(link[:,0],link[:,1],line_styles[j],color=color,linewidth=line_widths[j])[0])
    # if i == 0: plt.legend(lines[:3],['input','ground','flexure'])

    plt.axis('scaled')
    ps = np.array(lk).reshape((-1,2))
    cx = (np.amax(ps[:,0])+np.amin(ps[:,0]))/2
    cy = (np.amax(ps[:,1])+np.amin(ps[:,1]))/2
    r = 0.05
    plt.xlim([cx-r,cx+r])
    plt.ylim([cy-r,cy+r])

    plt.xticks([])
    plt.yticks([])

    if i > 5: plt.xlabel('{:d}'.format(s['k']))
    if i % 3 == 0: plt.ylabel('{:.1f}'.format(s['a']),rotation='horizontal',ha='right')

plt.gcf().add_subplot(111, frameon=False)
plt.xticks([])
plt.yticks([])
plt.xlabel('k [N/m]',labelpad=16)
plt.ylabel('a',labelpad=20)
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()

# sol = jump.solve(cs)
# cs['dsm'] = -np.min(sol.y[1,:])
#
# sdatum = fourbar.stiffness(x,cs,plot=False)
# x_d = np.linspace(-cs['dsm'],0,50)
# f_d = -jump.f_spring(x_d,cs)
# f = np.interp(x_d,sdatum['x'],sdatum['f'])/cs['r']
# x_d = -x_d
# f_d = -f_d
# f = -f

# plt.figure('stiffness')
# plt.plot(x_d,f_d,'-',color=color)
# plt.plot(x_d,f,'o',color=color,markerfacecolor='none',markersize=4)
# plt.ylabel('torque/r [N]')
# plt.xlabel('angle*r [m]')

# jdatum = fourbar.jump(xm,x,cs,plot=True)
# t_idx = np.nonzero(np.array(jdatum['t']) > sol.t[-1])[0][0]
# t_j = jdatum['t'][:t_idx]
# y_j = jdatum['y'][:t_idx]
# dy_j = jdatum['dy'][:t_idx]
#
# plt.figure('jump')
# plt.plot(sol.t,sol.y[2,:],color=c)
# plt.plot(t_j,dy_j,'o',color=c,markerfacecolor='none',markevery=100)
# plt.ylabel('dy [m/s]')
# plt.xlabel('t [s]')

# xm = [2.6935314437637747, 0.030244462243688645, 0.04668319649977162, 0.02002235749858264, 0.05998841948291793, 0.059996931859852574, 0.14061111190360398]
# opt.obj_motion(xm,plot=True)
