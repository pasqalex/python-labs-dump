import numpy as np

def f(x):
    return np.log(np.abs(x - 4))

A, B = 0.0, np.pi # отрезок [A; B]
N_CONTROL = 100 # количество контрольных точек (не менее 20)
N_LIST = [2, 4, 8, 16, 32] # значения n (количество узлов = n + 1) для исследования
N_PLOT = 8  # n, для которого выводится таблица контрольных точек

def sinc_approx(x, n, a=A, b=B, func=f):
    x = np.asarray(x, dtype=float)
    h = (b - a) / n
    nodes = a + h * np.arange(n + 1)
    fk = func(nodes)
    t = (x[:, None] - nodes[None, :]) / h
    return np.sinc(t) @ fk


def main():
    if N_CONTROL < 20:
        raise ValueError("Контрольных точек должно быть не менее 20")

    x_ctrl = np.linspace(A, B, N_CONTROL)
    f_ctrl = f(x_ctrl)

    print(f"Контрольных точек: {N_CONTROL}, отрезок [{A:.4f}; {B:.4f}]\n")
    print(f"{'n':>5} {'узлов':>7} {'max |f - L_n|':>16} {'средняя погр.':>16}")
    for n in N_LIST:
        err = np.abs(f_ctrl - sinc_approx(x_ctrl, n))
        print(f"{n:>5} {n + 1:>7} {err.max():>16.3e} {err.mean():>16.3e}")

    L = sinc_approx(x_ctrl, N_PLOT)
    err = np.abs(f_ctrl - L)
    print(f"\nКонтрольные точки для n = {N_PLOT}:")
    print(f"{'x':>10} {'f(x)':>12} {'L_n(x)':>12} {'погрешность':>14}")
    for xi, fi, Li, ei in zip(x_ctrl, f_ctrl, L, err):
        print(f"{xi:>10.5f} {fi:>12.6f} {Li:>12.6f} {ei:>14.3e}")


if name == "main":
    main()