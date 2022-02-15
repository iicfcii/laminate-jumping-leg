import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
import fourbar
import jump

def obj_motion(x,plot=False):
    rots = np.linspace(0,-cs['dl']/cs['r'],10)+x[0]
    try:
        xs,ys = fourbar.motion(rots,x,plot=plot)
    except AssertionError:
        return 1

    xs_d = np.zeros(xs.shape)
    ys_d = (rots-x[0])*cs['r']+ys[0]

    if plot:
        plt.figure()
        plt.plot(rots,ys_d)
        plt.plot(rots,ys,'.')
        plt.plot(rots,xs_d)
        plt.plot(rots,xs,'.')
        print('range',cs['dl']/cs['r'])

    ex = np.sqrt(np.sum((xs-xs_d)**2)/xs.shape[0])
    ey = np.sqrt(np.sum((ys-ys_d)**2)/ys.shape[0])
    return ex+ey

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def obj_stiffness(x,m,plot=False):
    try:
        datum = fourbar.stiffness(x,cs,plot=plot)
    except AssertionError:
        return 10

    x_d = np.linspace(-cs['dsm'],0,50)
    f_d = -jump.f_spring(x_d,cs)
    f = np.interp(x_d,datum['x'],datum['f'])/cs['r']
    e = np.sqrt(np.sum((f-f_d)**2)/f_d.shape[0])

    mass_spring = (
        (x[0]+x[2]+x[3])*fourbar.tr*fourbar.wr*fourbar.rho+
        x[1]*fourbar.tf*x[4]*fourbar.rho
    )
    em = np.abs(m-mass_spring)

    if plot:
        plt.figure()
        plt.plot(-x_d,-f_d)
        plt.plot(-x_d,-f,'--')
        # plt.plot(-np.array(datum['x'])*cs['r'],-np.array(datum['f'])/cs['r'],'--')
        print('mass error',em)

    return e+5*em

cs = {
    'g': 9.81,
    'mb': 0.025,
    'ml': 0.0001,
    'k': 80,
    'a': 1.4,
    'ds': 0.05,
    'tau': 0.15085776558260747,
    'v': 47.75363911922214,
    'dl': 0.06,
    'r': 0.05
}
sol = jump.solve(cs)
cs['dsm'] = -np.min(sol.y[1,:]) # spring travel range that needs to be matched
bounds_motion = [(-np.pi,np.pi)]+[(0.02,0.06)]*5+[(-1,1)]*1
bounds_stiffness = [(0.01,0.15)]*4+[(0.01,0.03)]+[(-1,1)]*1

xm = [2.6935314437637747, 0.030244462243688645, 0.04668319649977162, 0.02002235749858264, 0.05998841948291793, 0.059996931859852574, 0.14061111190360398]
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

if __name__ == '__main__':
    xm = None
    xs = None
    xm = [2.6935314437637747, 0.030244462243688645, 0.04668319649977162, 0.02002235749858264, 0.05998841948291793, 0.059996931859852574, 0.14061111190360398]
    xs = [0.010012131658306514, 0.05429677864028192, 0.035094721706915084, 0.0739106305463264, 0.01000406293402808, -0.3856909524602533]

    if xm is None:
        res = differential_evolution(
            obj_motion,
            bounds=bounds_motion,
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('xm',str(list(res.x)))
        print('Cost', res.fun)
        xm = res.x

    mass_spring = cs['mb']-0.02-np.sum(xm[1:6])*fourbar.tr*fourbar.wr*fourbar.rho
    if xs is None:
        res = differential_evolution(
            obj_stiffness,
            bounds=bounds_stiffness,
            args=(mass_spring,),
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('xs',str(list(res.x)))
        print('Cost', res.fun)
        xs = res.x

    print(obj_motion(xm,plot=True))
    print(obj_stiffness(xs,mass_spring,plot=True))
    plt.show()
