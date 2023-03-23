import numpy as np
import scipy
import sympy

outlist = np.loadtxt("data/out", delimiter=" ")

for iter in range(len(outlist)):
    p0 = outlist[iter][0:2]
    x0 = outlist[iter][2:4]
    omega0 = outlist[iter][5]

    # fmt: off
    # Van der Pol Equation
    xdim = 2
    pdim = 2
    def func(x, p):
        return -sympy.Matrix([
            x[1] - x[0] * x[0] - x[0] * x[0] * x[0] / 3,
            p[0] * (p[1] - x[0])
        ])
    # fmt: on

    sym_x = sympy.MatrixSymbol("x", xdim, 1)
    sym_p = sympy.MatrixSymbol("p", pdim, 1)

    f_expanded = func(sym_x, sym_p).expand()

    A = (
        sympy.derive_by_array(
            [f_expanded[i] for i in range(xdim)], [sym_x[i] for i in range(xdim)]
        )
        .transpose()
        .subs(
            {
                sym_x[0]: 0,
                sym_x[1]: 0,
                sym_p[0]: p0[0],
                sym_p[1]: p0[1],
            }
        )
    )
    A = np.array(A).astype(np.float64)

    # 非線形項
    F = f_expanded - A @ sym_x

    def B(u, v):
        ret = np.zeros((xdim, 1), dtype=complex)
        for i in range(xdim):
            for j in range(xdim):
                for k in range(xdim):
                    temp = float(
                        sympy.diff(sympy.diff(F[i], sym_x[j]), sym_x[k])
                        .subs(
                            {
                                sym_x[0]: 0,
                                sym_x[1]: 0,
                                sym_p[0]: p0[0],
                                sym_p[1]: p0[1],
                            }
                        )
                        .evalf()
                    )
                    ret[i] += temp * u[j] * v[k]
        return np.array(ret)

    def C(u, v, w):
        ret = np.zeros((xdim, 1), dtype=complex)
        for i in range(xdim):
            for j in range(xdim):
                for k in range(xdim):
                    for l in range(xdim):
                        temp = float(
                            sympy.diff(
                                sympy.diff(sympy.diff(F[i], sym_x[j]), sym_x[k]),
                                sym_x[l],
                            )
                            .subs(
                                {
                                    sym_x[0]: 0,
                                    sym_x[1]: 0,
                                    sym_p[0]: p0[0],
                                    sym_p[1]: p0[1],
                                }
                            )
                            .evalf()
                        )
                        ret[i] += temp * u[j] * v[k] * w[l]
        return np.array(ret)

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
            p_bar @ C(q, q, q_bar)
            - 2 * p_bar @ B(q, np.linalg.inv(A) @ B(q, q_bar))
            + p_bar @ B(q_bar, np.linalg.inv(2j * omega0 * np.eye(xdim) - A) @ B(q, q))
        ).real
    )

    # check the first lyapunov exponent
    # print(l1)

    canard_point = p0[1] - (l1 * p0[0] ** 1.5 / 4)

    print(p0[0], end=" ")
    print(canard_point[0], end=" ")
    print(canard_point[0] - p0[0] ** 1.5, end=" ")
    print(canard_point[0] + p0[0] ** 1.5)
