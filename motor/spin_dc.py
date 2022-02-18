from sympy import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

step_sim = 0.0005

V, J, K, b, R, L = symbols('V J K b R L')
dtheta, i = symbols('dtheta i')

ddtheta = -b/J*dtheta+K/J*i
di = -K/L*dtheta-R/L*i+V/L

x = Matrix([dtheta, i])
dx = Matrix([ddtheta, di])

def solve(cs):
    dx_f = lambdify(x,dx.subs(cs))

    def spin(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(spin, [0,1], [0,0], max_step=step_sim)
    return sol
