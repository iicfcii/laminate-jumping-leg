from sympy import *
from scipy.integrate import solve_ivp
import numpy as np

# Vertical jumping
step_sim = 0.0005
pi = np.pi

g, mb, ml, k, a, el = symbols('g mb ml k a el')
tau, v, em, r  = symbols('tau v em r') # torque max, speed max, rotation range, moment arm,
yb, dyb, ddyb = symbols('yb dyb ddyb')
ys, dys, ddys = symbols('ys dys ddys')

# Lagrange's Method
Kb = mb*(dyb**2)/2 # Body kinetic
Kl = ml*(dys**2)/2 # Leg kinetic
Pb = mb*g*yb # Body gravitational potential
Pl = ml*g*ys # Leg potential

L = Kb+Kl-Pb-Pl

b = tau/r/(v*r)
fb = b*(dys-dyb) # damping
# input force, when not fully extended, constant output
# otherwise, increase quickly to model the motor travel limit
fm = (Max(0,yb-em)/(yb-em)-1)*tau/r

dL_d_yb = diff(L,yb)
dL_d_dyb = diff(L,dyb)
ddLddyb_d_t = diff(dL_d_dyb,dyb)*ddyb
ddyb_e = solve(ddLddyb_d_t-dL_d_yb+fm-fb, ddyb)[0]

dL_d_ys = diff(L,ys)-sign(ys)*k*el*Pow(abs(ys/el),a) # Add spring force
dL_d_dys = diff(L,dys)
ddLddys_d_t = diff(dL_d_dys,dys)*ddys
ddys_e = solve(ddLddys_d_t-dL_d_ys-fm+fb, ddys, simplify=False)[0]

# States model
x = Matrix([yb,ys,dyb,dys])
dx = Matrix([dyb,dys,ddyb_e,ddys_e])
# print(dx)

def lift_off(t,x): # spring decompress
    return x[1]+1e-6
lift_off.terminal = True
lift_off.direction = 1

def solve(x0, cs):
    dx_f = lambdify(x,dx.subs(cs))

    def jump(t, x):
        return dx_f(*x).flatten()

    sol = solve_ivp(jump, [0,10], x0, events=[lift_off], max_step=step_sim)
    return sol
