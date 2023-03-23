import numpy as np
import scipy as sp
import sympy

outlist = np.loadtxt("out", delimiter=" ")
p0 = outlist[0:2]
x0 = outlist[2:4]
omega0 = outlist[5]

# fmt: off
# Van der Pol Equation
xdim = 2
pdim = 2
def func(x, p):
    return sympy.Matrix([
        x[1] - x[0] * x[0] - x[0] * x[0] * x[0] / 3,
        p[0] * (p[1] - x[0])
    ])
# fmt: on

sym_x = sympy.MatrixSymbol("x", xdim, 1)
sym_p = sympy.MatrixSymbol("p", pdim, 1)

f_expanded = func(sym_x, sym_p).expand()

# 線形項の係数行列
M = sympy.zeros(2, 2)
M[0, 0] = f_expanded[0].coeff(sym_x[0], 1)
M[0, 1] = f_expanded[0].coeff(sym_x[1], 1)
M[1, 0] = f_expanded[1].coeff(sym_x[0], 1)
M[1, 1] = f_expanded[1].coeff(sym_x[1], 1)

# 非線形項
F = f_expanded - M * sym_x
f = F[0]
g = F[1]

f_x = sympy.diff(f, sym_x[0])
f_y = sympy.diff(f, sym_x[1])
g_x = sympy.diff(g, sym_x[0])
g_y = sympy.diff(g, sym_x[1])

f_xx = sympy.diff(f_x, sym_x[0])
f_xy = sympy.diff(f_x, sym_x[1])
f_yy = sympy.diff(f_y, sym_x[1])
g_xx = sympy.diff(g_x, sym_x[0])
g_xy = sympy.diff(g_x, sym_x[1])
g_yy = sympy.diff(g_y, sym_x[1])

f_xxx = sympy.diff(f_xx, sym_x[0])
f_xxy = sympy.diff(f_xx, sym_x[1])
f_xyy = sympy.diff(f_xy, sym_x[1])
g_xxy = sympy.diff(g_xx, sym_x[1])
g_xyy = sympy.diff(g_xy, sym_x[1])
g_yyy = sympy.diff(g_yy, sym_x[1])


l = (M[0, 1] / (16 * omega0**4)) * (
    omega0**2
    * ((f_xxx + g_xxy) + 2 * M[1, 1] * (f_xxy + g_xyy) - M[1, 0] * (f_xyy + g_yyy))
    - M[0, 1]
    * M[1, 1]
    * (f_xx**2 - f_xx * g_xy - f_xy * g_xx - g_xx * g_yy - 2 * g_xy**2)
    - M[1, 0]
    * M[1, 1]
    * (g_yy**2 - g_yy * f_xy - g_xy * f_yy - f_xx * f_yy - 2 * f_xy**2)
    + M[0, 1] ** 2 * (f_xx * g_xx + g_xx * g_xy)
    - M[1, 0] ** 2 * (f_yy * g_yy + f_xy * f_yy)
    - (omega0**2 + 3 * M[1, 1] ** 2) * (f_xx * f_xy - g_xy * g_yy)
).subs({sym_x[0]: x0[0], sym_x[1]: x0[1], sym_p[0]: p0[0], sym_p[1]: p0[1]}).evalf()


print(l)

lambda_c = p0[1] - l * p0[0]
print(lambda_c)
