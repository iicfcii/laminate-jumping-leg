from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import sympy.core

# Spring boundary constants
K_SB = 1
B_SB = 0.001
# Motor boundary constants
K_MB = 10
B_MB = 0.1
# Body boundary constants
K_BB = 100

def f_ts(x,cs,bound=True):
    y,dy,theta,dtheta,thetas,i = x
    t,k,a,r = [cs['t'],cs['k'],cs['a'],cs['r']]
    ts = np.sign(thetas)*t*np.power(np.abs(thetas*k/t),a)

    if not bound:
        return ts
    else:
        tsb = (np.maximum(0,thetas-t/k)/(thetas-t/k))*((thetas-t/k)*K_SB+(dtheta-dy/r)*B_SB)*0
        return ts+tsb

def f_grf(x,cs):
    y,dy,theta,dtheta,thetas,i = x
    t,k,a,r,g,ml = [cs['t'],cs['k'],cs['a'],cs['r'],cs['g'],cs['ml']]

    ts = f_ts(x,cs)
    tyb = -np.minimum(0,y)/y*(y/r*K_BB)
    grf = (ts+tyb)/r+ml*g

    return grf

g, m, r, k, a, t, d, bs = symbols('g m r k a t d bs')
b, K, I, R, L, V = symbols('b K I R L V')
y, dy, theta, dtheta, thetas, i = symbols('y dy theta dtheta thetas i')

tsb = (Max(0,thetas-t/k)/(thetas-t/k))*((thetas-t/k)*K_SB+(dtheta-dy/r)*B_SB)*0 # Spring boundary
tsd = bs*(dtheta-dy/r) # spring damping
tmb = (Max(0,theta-d)/(theta-d))*((theta-d)*K_MB+dtheta*B_MB) # motor arm boundary
tyb = -Min(0,y)/y*(y/r*K_BB) # Body lower boundary

ts = sign(thetas)*t*Pow(abs(thetas*k/t),a)+tsb+tsd # spring force

ddy = ((ts+tyb)/r-m*g-dy*0)/m
ddtheta = (K*i-ts-tmb-b*dtheta)/I
dthetas = dtheta-dy/r
di = V/L-K*dtheta/L-R*i/L

x = Matrix([y,dy,theta,dtheta,thetas,i])
dx = Matrix([dy,ddy,dtheta,ddtheta,dthetas,di])

# xm = [0.00015356505435637838, 0.19367850172392562, 0.00013013941021725156, 12.197322261486839]
xm = [0.00024817282734284404, 0.1217335758051555, 8.023874392878126e-05, 10.858458192019658]

cs = {
    'g': 9.81,
    'm': 0.03,
    'ml': 0.00,
    'r': 0.04,
    'k': 0.2,
    'a': 1,
    't': 0.06, # max spring torque, Nm
    'd': 1.5, # max motor arm range, rad
    'bs': 0, # spring damping
    'b': xm[0],
    'K': xm[1],
    'I': xm[2],
    'R': xm[3],
    'L': 580e-6,
    'V': 9
}

def solve(cs,plot=False):
    yi = -cs['m']*cs['g']*cs['r']/K_BB*cs['r']
    x0 = [yi,0,0,0,0,0]

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
        plt.ylabel('y')
        plt.subplot(212)
        plt.plot(sol.t,sol.y[1,:])
        plt.ylabel('dy')

        plt.figure('rotor')
        plt.subplot(311)
        plt.plot(sol.t,sol.y[2,:])
        plt.ylabel('theta')
        plt.subplot(312)
        plt.plot(sol.t,sol.y[3,:])
        plt.ylabel('dtheta')
        plt.subplot(313)
        plt.plot(sol.t,sol.y[5,:])
        plt.ylabel('i')

        plt.figure('spring')
        plt.subplot(211)
        plt.plot(sol.t,sol.y[4,:])
        plt.ylabel('thetas')
        plt.subplot(212)
        plt.plot(sol.t,grf)
        plt.ylabel('grf')

    return sol

# solve(cs,plot=True)
# plt.show()
