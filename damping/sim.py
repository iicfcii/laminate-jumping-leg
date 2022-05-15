from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

step = 1e-3

k,b,I, = symbols('k b I')
theta,dtheta = symbols('thtea dtheta')

ddthtea = (-k*theta-b*dtheta)/I

x = Matrix([theta, dtheta])
dx = Matrix([dtheta, ddthtea])

cs = {
    'k': 0.1,
    'b': 0.0001,
    'I': 0.01*0.018**2+0.0005*0.0217**2+2.93e-7, # weight, marker, 3d printed part
    'theta0': 0.4
}

def solve(cs,tf=1,plot=False):
    dx_f = lambdify(x,dx.subs(cs))

    def spin(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(spin, [0,tf], [cs['theta0'],0], max_step=step)


    if plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(sol.t,sol.y[0,:])
        plt.ylabel('theta')
        plt.subplot(212)
        plt.plot(sol.t,sol.y[1,:])
        plt.ylabel('dtheta')

    return sol

# solve(cs,plot=True)
# plt.show()
