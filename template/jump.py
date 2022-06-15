from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import sympy.core

K_MB = 10
B_MB = 0.1
K_BB = 5000

def f_ts(x,cs):
    yb,dyb,phim,dphim,phis,phil,i = x
    t,k,a,r,bs = [cs['t'],cs['k'],cs['a'],cs['r'],cs['bs']]
    ts = np.sign(phis)*t*np.power(np.abs(phis*k/t),a)+bs*(dphim-dyb/r)

    return ts

def f_grf(x,cs):
    yb,dyb,phim,dphim,phis,phil,i = x
    r,g,mb,Il,mf = [cs['r'],cs['g'],cs['mb'],cs['Il'],cs['mf']]

    ts = f_ts(x,cs)
    fbb = -np.minimum(0,yb)/yb*(yb*K_BB)
    ddyb = (ts/r-mb*g+fbb)/(mb+Il/r/r)

    grf = (ts-Il*ddyb/r)/r+mf*g+fbb

    return grf

g, mb, r, k, a, t, d, bs, Il = symbols('g mb r k a t d bs Il')
bm, K, Im, R, L, V = symbols('bm K Im R L V') # motor
yb, dyb, phim, dphim, phis, phil, i = symbols('yb dyb phim dphim phis phil i')

rp = [-0.028783019958200613, 0.040053500503592146, 0.030340072489310328]
# r = rp[0]*phil**2+rp[1]*phil+rp[2]

ts = sign(phis)*t*Pow(abs(phis*k/t),a)+bs*(dphim-dyb/r) # spring force
tmb = (Max(0,phim-d)/(phim-d))*((phim-d)*K_MB+dphim*B_MB) # motor arm boundary
fbb = -Min(0,yb)/yb*(yb*K_BB) # Body lower boundary

ddyb = (ts/r-mb*g+fbb)/(mb+Il/r/r)
ddphim = (K*i-ts-bm*dphim-tmb)/Im
dphis = dphim-dyb/r
dphil = dyb/r
di = V/L-K*dphim/L-R*i/L

x = Matrix([yb,dyb,phim,dphim,phis,phil,i])
dx = Matrix([dyb,ddyb,dphim,ddphim,dphis,dphil,di])

# xm = [0.00015356505435637838, 0.19367850172392562, 0.00013013941021725156, 12.197322261486839]
xm = [0.00024817282734284404, 0.1217335758051555, 8.023874392878126e-05, 10.858458192019658]

cs = {
    'g': 9.81,
    'mb': 0.03,
    'mf': 0.00,
    'r': 0.04,
    'k': 0.1,
    'a': 1,
    't': 0.06, # max spring torque, Nm
    'd': 1.5, # max motor arm range, rad
    'bs': 0, # spring damping
    'Il': 0,
    'bm': xm[0],
    'K': xm[1],
    'Im': xm[2],
    'R': xm[3],
    'L': 580e-6,
    'V': 9
}

def solve(cs,plot=False):
    yi = -cs['mb']*cs['g']/K_BB
    x0 = [yi,0,0,0,0,0,0]

    dx_f = lambdify(x,dx.subs(cs))
    def f(t, x):
        return dx_f(*x).flatten()

    # spring decompress
    def lift_off(t,x):
        return f_grf(x,cs)
    lift_off.terminal = True
    lift_off.direction = -1

    sol = solve_ivp(f,[0,1],x0,events=[lift_off],max_step=1e-4)

    if plot:
        grf = f_grf(sol.y,cs)

        plt.figure('body')
        plt.subplot(211)
        plt.plot(sol.t,sol.y[0,:])
        plt.ylabel('yb')
        plt.subplot(212)
        plt.plot(sol.t,sol.y[1,:])
        plt.ylabel('dyb')

        plt.figure('rotor')
        plt.subplot(311)
        plt.plot(sol.t,sol.y[2,:])
        plt.ylabel('phim')
        plt.subplot(312)
        plt.plot(sol.t,sol.y[3,:])
        plt.ylabel('dphim')
        plt.subplot(313)
        plt.plot(sol.t,sol.y[6,:])
        plt.ylabel('i')

        plt.figure('spring')
        plt.subplot(311)
        plt.plot(sol.t,sol.y[4,:])
        plt.ylabel('phis')
        plt.subplot(312)
        plt.plot(sol.t,sol.y[5,:])
        plt.ylabel('phil')
        plt.subplot(313)
        plt.plot(sol.t,grf)
        plt.ylabel('grf')

    return sol

# solve(cs,plot=True)
# plt.show()
