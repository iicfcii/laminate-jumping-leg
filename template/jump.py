from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

g, m, r, k, a = symbols('g m r k a')
b, K, I, R, L, V = symbols('b K I R L V')
dyb, ys, dtheta, i = symbols('dyb ys dtheta i')

fs = -k*ys
ddyb = fs/m-g
ddtheta = -b*dtheta/I+K*i/I-fs*r/I
di = V/L-K*dtheta/L-R*i/L
dys = dyb-r*dtheta

x = Matrix([dyb,ys,dtheta,i])
dx = Matrix([ddyb,dys,ddtheta,di])

cs = {
    'g': 9.81,
    'm': 0.02,
    'r': 0.05,
    'k': 80,
    'a': 1,
    'b': 0.0006566656814173122,
    'K': 0.14705778874626846,
    'I': 9.345234544905957e-05,
    'R': 10,
    'L': 580e-6,
    'V': 8.7
}

# spring decompress
def lift_off(t,x):
    return x[1]+1e-6
lift_off.terminal = True
lift_off.direction = 1

def solve(cs):
    # Initial condition after settle
    ys_i = -cs['m']*cs['g']/cs['k']

    x0 = [0,ys_i,0,0]
    dx_f = lambdify(x,dx.subs(cs))

    def f(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(f,[0,1],x0,events=[lift_off],max_step=1e-4)
    return sol

# sol = solve(cs)
# for i in range(len(x)):
#     plt.subplot(4,1,i+1)
#     plt.plot(sol.t,sol.y[i,:])
# plt.show()
