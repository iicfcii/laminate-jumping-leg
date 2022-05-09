from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def t_spring(theta,k,a,t):
    import sympy.core
    if isinstance(theta,sympy.core.symbol.Symbol):
        return sign(theta)*t*Pow(abs(theta*k/t),a)
    else:
        return np.sign(theta)*t*np.power(np.abs(theta*k/t),a)

g, m, Il, r, k, a, t, d = symbols('g m Il r k a t d')
b, K, I, R, L, V = symbols('b K I R L V')
y, dy, theta, dtheta, thetas, i = symbols('y dy theta dtheta thetas i')

ts = t_spring(thetas,k,a,t)

# Spring boundary
tsb = (Max(0,thetas-t/k)/(thetas-t/k))*((thetas-t/k)*10+(dtheta-dy/r)*0.1)
ts += tsb

# Leg boundary by motor arm
tlb = (Max(0,theta-d/r)/(theta-d/r))*((theta-d/r)*10+dtheta*0.1)

ddy = (ts/r-m*g)/m
ddtheta = (K*i-ts-tlb-b*dtheta)/I
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
    # Initial condition after settle
    # thetas_i = np.power(cs['m']*cs['g']*cs['r']/cs['t'],1/cs['a'])*cs['t']/cs['k']
    thetas_i = 0
    y_i = (0-thetas_i)*cs['r']

    x0 = [y_i,0,0,0,thetas_i,0]
    dx_f = lambdify(x,dx.subs(cs))

    def f(t, x):
        return dx_f(*x).flatten()

    # spring decompress
    def lift_off(t,x):
        return x[4]
    lift_off.terminal = True
    lift_off.direction = -1

    sol = solve_ivp(f,[0,1],x0,events=[lift_off],max_step=1e-4)

    if plot:
        for i in range(len(x)):
            plt.subplot(len(x),1,i+1)
            plt.plot(sol.t,sol.y[i,:])

    return sol

# solve(cs,plot=True)
# plt.show()
