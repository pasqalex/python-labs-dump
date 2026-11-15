"""
Метод Ритца для краевой задачи (вариант 1):

    d/dx( x^2 dφ/dx ) - 2φ = -1 - ln(x^2),   φ(1) = 0,  φ(2) = 0,   m = 2..6.

Эквивалентная форма:  L φ = -(x² φ')' + 2φ = f,  f(x) = 1 + 2 ln x.
Функционал: J(φ) = 1/2 ∫ (x² φ'² + 2 φ²) dx - ∫ f φ dx.

Базис:  w_k(x) = (x-1)(2-x)(x-1)^(k-1),  k = 1..m.
Система Ритца:  A c = b,
    A_ij = ∫ (x² w_i' w_j' + 2 w_i w_j) dx,   b_i = ∫ f w_i dx.

Точное решение (уравнение Эйлера):
    φ(x) = C1 x + C2 / x² + 1 + ln x,
    C1 = -(3 + 4 ln2)/7,  C2 = 4(ln2 - 1)/7.

Результат: три таблицы и коэффициенты выводятся в консоль (файлы не создаются).
"""
import numpy as np
from scipy.integrate import quad, trapezoid

A_, B_ = 1.0, 2.0
M_LIST = list(range(2, 7))      # m = 2, 3, 4, 5, 6
N_STEPS = 10                    # шаг таблицы h = 0.1


def f(x):
    return 1.0 + 2.0 * np.log(x)


def exact(x):
    c1 = -(3 + 4 * np.log(2)) / 7
    c2 = 4 * (np.log(2) - 1) / 7
    return c1 * x + c2 / x**2 + 1 + np.log(x)


def w(k, x):
    t = x - 1
    return t**k * (2 - x)


def dw(k, x):
    t = x - 1
    return k * t**(k - 1) * (2 - x) - t**k


def ritz(m):
    A = np.zeros((m, m))
    b = np.zeros(m)
    for i in range(1, m + 1):
        b[i - 1] = quad(lambda x: f(x) * w(i, x), A_, B_)[0]
        for j in range(i, m + 1):
            A[i - 1, j - 1] = A[j - 1, i - 1] = quad(
                lambda x: x**2 * dw(i, x) * dw(j, x) + 2 * w(i, x) * w(j, x),
                A_, B_)[0]
    return np.linalg.solve(A, b), np.linalg.cond(A)


def phi_m(c, x):
    return sum(ck * w(k, x) for k, ck in enumerate(c, start=1))


def print_table(title, header, rows, fmt):
    print(f"\n{title}")
    line = " | ".join(f"{h:>12}" for h in header)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(f"{v:>12{fmt[i]}}" for i, v in enumerate(r)))


def main():
    # коэффициенты и решения
    coefs, conds = {}, {}
    for m in M_LIST:
        coefs[m], conds[m] = ritz(m)

    # Таблица 1: значения в узлах
    xs = np.linspace(A_, B_, N_STEPS + 1)
    ex = exact(xs) + 0.0
    ex[np.abs(ex) < 1e-15] = 0.0
    hdr1 = ["x", "точное"] + [f"m={m}" for m in M_LIST]
    rows1 = [[x, e] + [phi_m(coefs[m], x) for m in M_LIST]
             for x, e in zip(xs, ex)]
    print_table("Таблица 1. Значения φ(x): точное и метод Ритца",
                hdr1, rows1, [".1f", ".8f"] + [".8f"] * len(M_LIST))

    # Таблица 2: абсолютные погрешности в узлах
    hdr2 = ["x"] + [f"m={m}" for m in M_LIST]
    rows2 = [[x] + [abs(phi_m(coefs[m], x) - e) for m in M_LIST]
             for x, e in zip(xs, ex)]
    print_table("Таблица 2. Абсолютная погрешность |φ_m(x) - φ(x)|",
                hdr2, rows2, [".1f"] + [".3e"] * len(M_LIST))

    # Таблица 3: сходимость по m (на мелкой сетке)
    xf = np.linspace(A_, B_, 2001)
    exf = exact(xf)
    hdr3 = ["m", "max-ошибка", "L2-ошибка", "отношение", "cond(A)"]
    rows3, prev = [], None
    for m in M_LIST:
        d = phi_m(coefs[m], xf) - exf
        e_max = np.max(np.abs(d))
        e_l2 = np.sqrt(trapezoid(d**2, xf))
        ratio = prev / e_max if prev else float("nan")
        rows3.append([m, e_max, e_l2, ratio, conds[m]])
        prev = e_max
    print_table("Таблица 3. Сходимость (отношение = ошибка(m-1)/ошибка(m))",
                hdr3, rows3, ["d", ".3e", ".3e", ".2f", ".3e"])

    print("\nКоэффициенты c_k:")
    for m in M_LIST:
        print(f"  m={m}: ", np.array2string(coefs[m], precision=6))


if __name__ == "__main__":
    main()