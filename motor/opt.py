from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy as np
from . import spin
from utils import data

N = 75
I_LOAD = 0.04*0.09**2 # Weights
I_HOLDER = (
    25399.083+ # beam
    62.63+ # shaft collar
    0.16*6**2*5+ # screws
    0.65*36**2+0.65*56**2 # markers
)/1e3/1e6
I_ROTOR = 3**2*np.pi*15*8/1e3*3**2/2*N**2/1e9 # 3mm radius 15mm height steel cylinder with gear ratio
K = 1/0.12*9.81/1000 if N == 75 else 1/0.1*9.81/1000 # From pololu
R = 6/1.5 # ohm
L = 580e-6 # h
VOLTS = [3.0,6.0,9.0] # V
t = np.linspace(0,spin.tfinal,100)

def read(has_load,plot=False):
    ws = []
    for v in VOLTS:
        dthetai = []
        for trial in [1,2]:
            # Exp motor data
            d = data.read('./data/motor/hpcb{:d}_{:d}V_{:d}_{:d}.csv'.format(N,int(np.ceil(v)),int(has_load),trial))
            t_raw = np.array(d['t'])
            t1_raw = np.array(d['t1'])

            assert np.sum(t_raw-t1_raw) == 0

            x = np.array(d['x'])
            x1 = np.array(d['x1'])
            y = np.array(d['y'])
            y1 = np.array(d['y1'])
            theta = np.arctan2(y1-y,x1-x)
            dtheta = np.concatenate(([0],(theta[1:]-theta[:-1])/(t_raw[1:]-t_raw[:-1])))

            # Remove dtheta jump
            idx_not_jump = dtheta < 50
            dtheta = dtheta[idx_not_jump]
            t_raw = t_raw[idx_not_jump]

            # Remove ddtehta jump
            ddtheta = np.concatenate(([0],(dtheta[1:]-dtheta[:-1])/(t_raw[1:]-t_raw[:-1])))
            idx_not_jump = np.abs(ddtheta) < 500
            dtheta = dtheta[idx_not_jump]
            t_raw = t_raw[idx_not_jump]

            # Find start index
            idx_ti = np.nonzero(t_raw > 0.5)[0][0]
            idx_tf = np.nonzero(t_raw > 0.5+spin.tfinal)[0][0]

            dtheta = -dtheta[idx_ti:idx_tf]
            t_raw = t_raw[idx_ti:idx_tf]
            t_raw -= t_raw[0]

            dthetai.append(np.interp(t,t_raw,dtheta))

        dthetai = np.sum(dthetai,axis=0)/len(dthetai)
        ws.append(dthetai)

    if plot:
        plt.figure()
        for i,v in enumerate(VOLTS):
            plt.plot(t,ws[i])

    return ws

# ws = read(True,plot=True)
# plt.show()
# exit()

def obj(x,ws,plot=False):
    e = 0
    sols = []
    for j in [0,1]:
        I = I_HOLDER+x[2]+I_LOAD*j

        for i in range(len(VOLTS)):
            cs = {
                'V': VOLTS[i],
                'b': x[0],
                'K': x[1],
                'J': I,
                'R': x[3],
                'L': L
            }
            sol = spin.solve(cs)
            w = np.interp(t,sol.t,sol.y[0,:])

            # steady state current cant be very hig
            iss = sol.y[1,-1]
            if iss > 0.2: return 100

            idx = t <= (0.5+j*0.5)
            ei = np.sqrt(np.sum((w[idx]-ws[j*3+i][idx])**2)/len(t[idx]))

            e += ei
            sols.append(sol)

    if plot:
        plt.figure('w')
        plt.figure('i')
        for i in range(len(ws)):
            c = 'C{:d}'.format(int(i%3))
            plt.figure('w')
            if i < 3:
                plt.subplot(121)
                plt.title('Without Load')
            else:
                plt.subplot(122)
                plt.title('With Load')
            plt.plot(t,ws[i],'--',color=c,label='{:.1f}V'.format(VOLTS[i%3]))
            plt.plot(sols[i].t,sols[i].y[0,:],color=c)
            plt.xlabel('Time (s)')
            plt.ylabel('Speed (rad/s)')
            plt.legend()

            plt.figure('i')
            if i < 3:
                plt.subplot(121)
            else:
                plt.subplot(122)
            plt.plot(sols[i].t,sols[i].y[1,:],color=c)

    return e

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

bounds=[(0,0.01),(K*0.5,K*2),(I_ROTOR*0.5,I_ROTOR*2),(0,R*5)]
x = None
# x = [0.00021493736182965403, 0.1834029440210516, 0.00012920296285840916, 12.030585215776586]
x = [0.00025370457449805806, 0.12076156825373549, 7.547067829477504e-05, 10.978193010075643]

if __name__ == '__main__':
    ws = read(False)+read(True)

    if x is None:
        res = differential_evolution(
            obj,
            bounds=bounds,
            args=(ws,),
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('x',str(list(res.x)))
        print('Cost', res.fun)
        x = res.x

    print(obj(x,ws,plot=True))
    plt.show()
