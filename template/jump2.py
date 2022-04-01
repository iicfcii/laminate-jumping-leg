from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def f_spring(ys,k,a,d):
    import sympy.core
    if isinstance(ys,sympy.core.symbol.Symbol):
        return -sign(ys)*k*d*Pow(abs(ys/d),a)
    else:
        return -np.sign(ys)*k*d*np.power(np.abs(ys/d),a)

g, mb, ms, r, k, a, dl, ds = symbols('g mb ms r k a dl ds')
b, K, I, R, L, V = symbols('b K I R L V')
yb, dyb, ys, dys, theta, dtheta, i = symbols('yb dyb ys dys theta dtheta i')
ddyb, ddys = symbols('ddyb ddys')


fs = f_spring(ys,k,a,ds)-0.1*(dyb-r*dtheta) # spring force
fw = (Max(0,yb-ys-dl)/(yb-ys-dl))*((yb-ys-dl)*10000+dtheta*r*50)
f = fs-ms*g-ms*ddys+fw
ddtheta = (ddyb-ddys)/r
ans = solve(
    [
        K*i-b*dtheta-f*r-I*ddtheta,
        f-mb*g-mb*ddyb-fw
    ],
    [ddys,ddyb]
)
ddyb_e = ans[ddyb]
ddys_e = ans[ddys]
ddtheta_e = ddtheta.subs([(ddyb,ddyb_e),(ddys,ddys_e)])
di_e = V/L-K*dtheta/L-R*i/L

x = Matrix([yb,dyb,ys,dys,theta,dtheta,i])
dx = Matrix([dyb,ddyb_e,dys,ddys_e,dtheta,ddtheta_e,di_e])

cs = {
    'g': 9.81,
    'mb': 0.02,
    'ms': 0.005,
    'r': 0.06,
    'k': 50,
    'a': 1,
    'dl': 0.06,
    'ds': 0.06,
    'b': 0.0006566656814173122,
    'K': 0.14705778874626846,
    'I': 9.345234544905957e-05,
    'R': 10,
    'L': 580e-6,
    'V': 8.7
}

# spring decompress
def lift_off(t,x):
    return x[2]
lift_off.terminal = True
lift_off.direction = 1

def solve(cs,plot=False):
    # Initial condition after settle
    ys_i = -np.power((cs['mb']+cs['ms'])*cs['g']/cs['k']/cs['ds'],1/cs['a'])*cs['ds']
    yb_i = ys_i
    theta_i = 0

    x0 = [yb_i,0,ys_i,0,theta_i,0,0]
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
