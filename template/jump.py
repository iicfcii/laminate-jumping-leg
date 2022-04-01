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

# r = 0.06
# t = 2*r
# for k in [30*r**2,70*r**2]:
#     for a in [0.5,1,2]:
#         theta = np.linspace(0,t/k,100)
#         ts = -t_spring(theta,k,a,t)
#
#         plt.plot(theta,ts)
# plt.show()

g, m, r, k, a, t, d = symbols('g m r k a t d')
b, K, I, R, L, V = symbols('b K I R L V')
y, dy, theta, dtheta, thetas, i = symbols('y dy theta dtheta thetas i')

ts = t_spring(thetas,k,a,t)
tb = (Max(0,theta-d)/(theta-d))*((theta-d)*10+dtheta*0.1)

ddy = (ts/r-m*g)/m
ddtheta = (K*i-ts-tb-b*dtheta)/I
dthetas = dtheta-dy/r
di = V/L-K*dtheta/L-R*i/L

x = Matrix([y,dy,theta,dtheta,thetas,i])
dx = Matrix([dy,ddy,dtheta,ddtheta,dthetas,di])

cs = {
    'g': 9.81,
    'm': 0.02,
    'r': 0.06,
    'k': 30*0.06**2,
    'a': 1,
    't': 1.5*0.06,
    'd': 1,
    'b': 0.0006566656814173122,
    'K': 0.14705778874626846,
    'I': 9.345234544905957e-05,
    'R': 10,
    'L': 580e-6,
    'V': 8.7
}

# spring decompress
def lift_off(t,x):
    return x[4]
lift_off.terminal = True
lift_off.direction = -1

def solve(cs,plot=False):
    # Initial condition after settle
    thetas_i = np.power(cs['m']*cs['g']*cs['r']/cs['t'],1/cs['a'])*cs['t']/cs['k']
    # print(t_spring(thetas_i,cs['k'],cs['a'],cs['t'])/cs['r'],cs['m']*cs['g'])

    x0 = [0,0,0,0,thetas_i,0]
    dx_f = lambdify(x,dx.subs(cs))

    def f(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(f,[0,1],x0,events=[lift_off],max_step=1e-4)

    if plot:
        for i in range(len(x)):
            plt.subplot(len(x),1,i+1)
            plt.plot(sol.t,sol.y[i,:])

    return sol

# solve(cs,plot=True)
# plt.show()
