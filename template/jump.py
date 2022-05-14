from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import sympy.core

EPS = 1e-6
Kb = 10
Bb = 0.1

def f_ts(x,cs,bound=True):
    y,dy,theta,dtheta,thetas,i = x
    t,k,a,r = [cs['t'],cs['k'],cs['a'],cs['r']]
    ts = np.sign(thetas)*t*np.power(np.abs(thetas*k/t),a)

    if not bound:
        return ts
    else:
        tsb = (np.maximum(0,thetas-t/k)/(thetas-t/k))*((thetas-t/k)*Kb+(dtheta-dy/r)*Bb)
        return ts+tsb

def f_grf(x,cs):
    y,dy,theta,dtheta,thetas,i = x
    t,k,a,r = [cs['t'],cs['k'],cs['a'],cs['r']]

    ts = f_ts(x,cs)
    tyb = -np.minimum(EPS,y)/y*(y*Kb+dy*Bb)/r
    grf = (ts+tyb)/r

    return grf

g, m, Il, r, k, a, t, d = symbols('g m Il r k a t d')
b, K, I, R, L, V = symbols('b K I R L V')
y, dy, theta, dtheta, thetas, i = symbols('y dy theta dtheta thetas i')

tsb = (Max(0,thetas-t/k)/(thetas-t/k))*((thetas-t/k)*Kb+(dtheta-dy/r)*Bb) # Spring boundary
tmb = (Max(0,theta-d/r)/(theta-d/r))*((theta-d/r)*Kb+dtheta*Bb) # motor arm boundary
tyb = -Min(EPS,y)/y*(y*Kb+dy*Bb)/r # Body lower boundary

ts = sign(thetas)*t*Pow(abs(thetas*k/t),a)+tsb # spring force

ddy = ((ts+tyb)/r-m*g)/m
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
    'r': 0.04,
    'k': 0.2,
    'a': 1,
    't': 0.06, # max spring torque, Nm
    'd': 0.06, # max leg range, m
    'b': xm[0],
    'K': xm[1],
    'I': xm[2],
    'R': xm[3],
    'L': 580e-6,
    'V': 9
}

def solve(cs,plot=False):
    # Calculate initial condition
    cs['V'] = 0
    dx_f = lambdify(x,dx.subs(cs))
    def f(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(f,[0,0.2],[EPS,0,0,0,0,0],max_step=1e-4)

    # Simulate jump
    cs['V'] = 9
    dx_f = lambdify(x,dx.subs(cs))
    def f(t, x):
        return dx_f(*x).flatten()

    # spring decompress
    def lift_off(t,x):
        return x[4]
    lift_off.terminal = True
    lift_off.direction = -1

    sol = solve_ivp(f,[0,1],sol.y[:,-1],events=[lift_off],max_step=1e-4)

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
