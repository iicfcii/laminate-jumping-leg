from sympy import *
from scipy.integrate import solve_ivp

step_sim = 0.0005

v, I, tau = symbols('v I tau')
theta, dtheta = symbols('theta dtheta')

ddtheta = (tau-dtheta*tau/v)/I

x = Matrix([theta, dtheta])
dx = Matrix([dtheta, ddtheta])

def solve(cs):
    dx_f = lambdify(x,dx.subs(cs))

    def spin(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(spin, [0,1], [0,0], max_step=step_sim)
    return sol
