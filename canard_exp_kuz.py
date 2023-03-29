import numpy as np
import scipy
import sympy
import tqdm

input_file = "data/bifdata"
output_file = "data/tempout"
eps_index = 0
param_index = 1

# fmt: off
# non organized multivib
xdim = 2
pdim = 2

def func(x, p, x0):
    return -sympy.Matrix([
        x[1] - x[0] * x[0] - x[0] * x[0] * x[0] / 3,
        p[0] * (p[1] - x[0])
    ])
# fmt: on

sym_x = sympy.MatrixSymbol("x", xdim, 1)
sym_p = sympy.MatrixSymbol("p", pdim, 1)
sym_x0 = sympy.MatrixSymbol("x0", xdim, 1)

F = func(sym_x, sym_p, sym_x0).expand()

# linear term
sym_A = sympy.derive_by_array(
    [F[i] for i in range(xdim)], [sym_x[i] for i in range(xdim)]
).transpose()
sym_A = sympy.Matrix(sym_A)

sym_B_diff = [sympy.zeros(xdim, xdim) for i in range(xdim)]
for i in range(xdim):
    for j in range(xdim):
        for k in range(xdim):
            sym_B_diff[i][j, k] = sympy.diff(sympy.diff(F[i], sym_x[j]), sym_x[k])


def B(u, v, x0, p0):
    ret = np.zeros((xdim, 1), dtype=complex)
    for i in range(xdim):
        for j in range(xdim):
            for k in range(xdim):
                temp = float(
                    sym_B_diff[i][j, k]
                    .subs(
                        {
                            sym_x[0]: 0,
                            sym_x[1]: 0,
                            sym_p[0]: p0[0],
                            sym_p[1]: p0[1],
                            sym_x0[0]: x0[0],
                            sym_x0[1]: x0[1],
                        }
                    )
                    .evalf()
                )
                ret[i] += temp * u[j] * v[k]
    return np.array(ret)


sym_C_diff = [[sympy.zeros(xdim, xdim) for i in range(xdim)] for j in range(xdim)]
for i in range(xdim):
    for j in range(xdim):
        for k in range(xdim):
            for l in range(xdim):
                sym_C_diff[i][j][k, l] = sympy.diff(
                    sympy.diff(sympy.diff(F[i], sym_x[j]), sym_x[k]),
                    sym_x[l],
                )


def C(u, v, w, x0, p0):
    ret = np.zeros((xdim, 1), dtype=complex)
    for i in range(xdim):
        for j in range(xdim):
            for k in range(xdim):
                for l in range(xdim):
                    temp = float(
                        sym_C_diff[i][j][k, l]
                        .subs(
                            {
                                sym_x[0]: 0,
                                sym_x[1]: 0,
                                sym_p[0]: p0[0],
                                sym_p[1]: p0[1],
                                sym_x0[0]: x0[0],
                                sym_x0[1]: x0[1],
                            }
                        )
                        .evalf()
                    )
                    ret[i] += temp * u[j] * v[k] * w[l]
    return np.array(ret)


outlist = np.loadtxt(input_file, delimiter=" ")

f = open(output_file, "w")

# for iter in range(1):
for iter in tqdm.tqdm(range(len(outlist))):
    p0 = outlist[iter][0:2]
    x0 = outlist[iter][2:4]
    omega0 = outlist[iter][5]

    A = sym_A.subs(
        {
            sym_x[0]: 0,
            sym_x[1]: 0,
            sym_p[0]: p0[0],
            sym_p[1]: p0[1],
            sym_x0[0]: x0[0],
            sym_x0[1]: x0[1],
        }
    )
    A = np.array(A).astype(np.float64)

    # print("Eigenvalues of A:", end=" ")
    # print(scipy.linalg.eigvals(A))

    a, temp = scipy.linalg.eig(A)
    q = temp[:, 0]
    q_bar = q.conjugate()

    b, temp = scipy.linalg.eig(A.T)
    # aとbの固有値リストの並びが一緒なので[:, 1]にしないとだめ
    p = temp[:, 1]
    p_bar = p.conjugate()

    # debug q and p
    # print(a)
    # print(q)
    # print(q_bar)
    # print(b)
    # print(p)
    # print(p_bar)
    # print(A @ q - 1j * omega0 * q)
    # print(A @ q_bar + 1j * omega0 * q_bar)
    # print(A.T @ p + 1j * omega0 * p)
    # print(A.T @ p_bar - 1j * omega0 * p_bar)

    # scaling p to satisfy <p, q> = 1
    p_bar = p_bar / (p_bar @ q)
    p = p_bar.conjugate()

    # check the restriction <p, q> = 1
    # print(p_bar @ q)

    # # Kuznetsov's 2-dim definition
    # l1 = (
    #     1
    #     / (2 * omega0**2)
    #     * (
    #         (1j * p_bar @ B(q, q)) * (p_bar @ B(q, q_bar)) + omega0 * p_bar @ C(q, q, q_bar)
    #     ).real
    # )

    # Kuznetsov's n-dim definition
    l1 = (
        1
        / (2 * omega0)
        * (
            p_bar @ C(q, q, q_bar, x0, p0)
            - 2 * p_bar @ B(q, np.linalg.inv(A) @ B(q, q_bar, x0, p0), x0, p0)
            + p_bar
            @ B(
                q_bar,
                np.linalg.inv(2j * omega0 * np.eye(xdim) - A) @ B(q, q, x0, p0),
                x0,
                p0,
            )
        ).real
    )

    # check the first lyapunov exponent
    # print(l1)

    canard_point = p0[param_index] - (l1 * p0[eps_index] ** 1.5 / 4)

    # print(1 / p0[0], end=" ")
    # print(canard_point[0], end=" ")
    # print(canard_point[0] - p0[0] ** 1.5, end=" ")
    # print(canard_point[0] + p0[0] ** 1.5)

    f.write("{:.16f}".format(p0[eps_index]))
    f.write(" ")
    f.write("{:.16f}".format(canard_point[eps_index]))
    f.write(" ")
    f.write("{:.16f}".format(canard_point[0] - p0[eps_index] ** 1.5))
    f.write(" ")
    f.write("{:.16f}".format(canard_point[0] + p0[eps_index] ** 1.5))
    f.write("\n")

f.close()
