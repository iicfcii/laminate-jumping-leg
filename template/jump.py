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

# p = [0.01290478170760566, -0.06128190277263893, 0.04575671701363875, 0.05348587213675358]
# r = p[0]*theta**3+p[1]*theta**2+p[2]*theta+p[3]

ts = t_spring(thetas,k,a,t)
tb = (Max(0,theta-d/r)/(theta-d/r))*((theta-d/r)*10+dtheta*0.1)

ddy = (ts/r-m*g)/m
ddtheta = (K*i-ts-tb-b*dtheta)/I
dthetas = dtheta-dy/r
di = V/L-K*dtheta/L-R*i/L

x = Matrix([y,dy,theta,dtheta,thetas,i])
dx = Matrix([dy,ddy,dtheta,ddtheta,dthetas,di])

cs = {
    'g': 9.81,
    'm': 0.025,
    'r': 0.06,
    'k': 0.2,
    'a': 1,
    't': 0.09,
    'd': 0.06,
    # 'b': 0.00021493736182965403,
    # 'K': 0.1834029440210516,
    # 'I': 0.00012920296285840916,
    # 'R': 12.030585215776586,
    'b': 0.0002537045744980580,
    'K': 0.12076156825373549,
    'I': 7.547067829477504e-05,
    'R': 10.978193010075643,
    'L': 580e-6,
    'V': 9
}

def solve(cs,plot=False):
    # Initial condition after settle
    thetas_i = np.power(cs['m']*cs['g']*cs['r']/cs['t'],1/cs['a'])*cs['t']/cs['k']
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
