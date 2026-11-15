"""
Задачи линейного программирования (вариант 1). Нужен только numpy.

Запуск:  python3 lp_tasks.py          - оба задания
         python3 lp_tasks.py 1        - задание 1 (графический метод)
         python3 lp_tasks.py 1 --plot - то же + рисунок (если установлен matplotlib)
         python3 lp_tasks.py 2        - задание 2 (симплекс-метод)

Данные варианта меняются в блоках DATA1 и DATA2 ниже.
"""
import sys
import itertools
from fractions import Fraction as Fr
import numpy as np

DATA1 = dict(
    names=("A", "B"),
    usage=[[6, 3, 2],        # расход сырья I, II, III на одно изделие A
           [3, 4, 5]],       # ... на одно изделие B
    stock=[102, 91, 105],    # b1, b2, b3
    profit=[5, 9],           # c1, c2
)

# Задание 2: max П·x,  R x <= З
DATA2 = dict(
    products=("N1", "N2", "N3"),
    resources=("P1", "P2", "P3"),
    R=[[1, 8, 5],            # расход P1 на N1, N2, N3
       [4, 1, 6],            # P2
       [5, 2, 2]],           # P3
    stock=[43, 74, 35],
    profit=[5, 7, 6],
)


def hr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def fmt(v, nd=4):
    """Fraction/float -> короткая десятичная запись."""
    s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s

def task1(plot=False):
    d = DATA1
    nm = d["names"]
    c = np.array(d["profit"], float)
    m = len(d["stock"])
    A = np.array([[d["usage"][j][i] for j in range(2)] for i in range(m)], float)  # строки = сырьё
    b = np.array(d["stock"], float)
    hr("ЗАДАНИЕ 1. Графический метод")
    print(f"Переменные: x1 = число изделий {nm[0]}, x2 = число изделий {nm[1]}.")
    print(f"Целевая функция:  F = {c[0]:g}·x1 + {c[1]:g}·x2  -> max\nОграничения:")
    for i in range(m):
        print(f"  сырьё {'I'*(i+1) if i < 3 else i+1:<3}: {A[i,0]:g}·x1 + {A[i,1]:g}·x2 <= {b[i]:g}")
    print("  x1 >= 0, x2 >= 0")

    print("\nПрямые для построения (точки пересечения с осями):")
    for i in range(m):
        print(f"  ({i+1}) {A[i,0]:g}x1 + {A[i,1]:g}x2 = {b[i]:g}:  "
              f"(x1={b[i]/A[i,0]:.4g}, x2=0)  и  (x1=0, x2={b[i]/A[i,1]:.4g})")

    # все прямые границы: ограничения + оси
    lines = [(A[i, 0], A[i, 1], b[i], f"({i+1})") for i in range(m)]
    lines += [(1.0, 0.0, 0.0, "x1=0"), (0.0, 1.0, 0.0, "x2=0")]
    feas = lambda p: (p[0] >= -1e-9 and p[1] >= -1e-9 and np.all(A @ p <= b + 1e-9))
    verts = []
    for (l1, l2) in itertools.combinations(lines, 2):
        M = np.array([[l1[0], l1[1]], [l2[0], l2[1]]])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, [l1[2], l2[2]])
        if feas(p) and not any(np.allclose(p, q[0]) for q in verts):
            verts.append((p, f"{l1[3]} ∩ {l2[3]}"))
    cen = np.mean([v[0] for v in verts], axis=0)
    verts.sort(key=lambda v: np.arctan2(v[0][1] - cen[1], v[0][0] - cen[0]))

    print("\nВершины многоугольника допустимых решений (обход по контуру) и значения F:")
    print(f"{'вершина':>14} | {'x1':>8} | {'x2':>8} | {'F':>9} | пересечение")
    best = max(verts, key=lambda v: c @ v[0])
    for p, lab in verts:
        mark = "  <-- max" if np.allclose(p, best[0]) else ""
        print(f"{'('+fmt(p[0])+'; '+fmt(p[1])+')':>14} | {p[0]:8.4f} | {p[1]:8.4f} | {c@p:9.4f} | {lab}{mark}")

    p = best[0]
    print(f"\nВектор градиента  grad F = ({c[0]:g}; {c[1]:g}).  Линия уровня F = C:  {c[0]:g}x1 + {c[1]:g}x2 = C")
    print("Её нужно двигать параллельно самой себе по направлению градиента до последней точки")
    print("касания с многоугольником.")
    for C in (0.5 * (c @ p), c @ p):
        print(f"  C = {C:8.3f}:  точки ({C/c[0]:.3f}; 0) и (0; {C/c[1]:.3f})")
    print(f"\nОПТИМУМ: x1 = {fmt(p[0])}, x2 = {fmt(p[1])},  F_max = {fmt(c@p)}")
    print(f"Это точка пересечения прямых {best[1]}.")
    print(f"ОТВЕТ: выпускать {fmt(p[0])} изд. {nm[0]} и {fmt(p[1])} изд. {nm[1]}; максимальная прибыль {fmt(c@p)} ден.ед.")
    print("Остатки сырья:")
    for i in range(m):
        used = A[i] @ p
        st = "израсходовано полностью (дефицитный ресурс)" if abs(b[i] - used) < 1e-9 else "остаток"
        print(f"  сырьё {i+1}: использовано {used:g} из {b[i]:g}, остаток {b[i]-used:g}  - {st}")
    print(f"Решение целочисленное: {'да' if np.allclose(p, np.round(p)) else 'нет'}")

    # ---- ASCII-график
    print("\nГрафик (# - область допустимых решений, o - вершины, * - оптимум):")
    xmax = max(v[0][0] for v in verts) * 1.15 + 1
    ymax = max(v[0][1] for v in verts) * 1.15 + 1
    W, H = 64, 26
    grid = [[" "] * (W + 1) for _ in range(H + 1)]
    for r in range(H + 1):
        for k in range(W + 1):
            q = np.array([k / W * xmax, r / H * ymax])
            if feas(q):
                grid[r][k] = "#"
    for q, _ in verts:
        grid[round(q[1] / ymax * H)][round(q[0] / xmax * W)] = "o"
    grid[round(p[1] / ymax * H)][round(p[0] / xmax * W)] = "*"
    for r in range(H, -1, -1):
        lab = f"{r/H*ymax:6.1f} |" if r % 5 == 0 else "       |"
        print(lab + "".join(grid[r]))
    print("       +" + "-" * (W + 1))
    ticks = [" "] * (W + 8)
    for k in range(0, W + 1, 8):
        s = f"{k/W*xmax:.0f}"
        for j, ch in enumerate(s):
            ticks[k + j] = ch
    print("        " + "".join(ticks) + "   x1")

    if plot:
        try:
            import matplotlib.pyplot as plt
            xs = np.linspace(0, xmax, 200)
            plt.figure(figsize=(7, 6))
            for i in range(m):
                plt.plot(xs, (b[i] - A[i, 0] * xs) / A[i, 1], label=f"({i+1}) {A[i,0]:g}x1+{A[i,1]:g}x2={b[i]:g}")
            poly = np.array([v[0] for v in verts])
            plt.fill(poly[:, 0], poly[:, 1], alpha=0.25, label="ОДР")
            plt.plot(xs, (c @ p - c[0] * xs) / c[1], "k--", label=f"F={c@p:g}")
            plt.arrow(0, 0, c[0], c[1], head_width=0.5, color="r")
            plt.plot(*p, "r*", ms=14)
            plt.xlim(0, xmax); plt.ylim(0, ymax); plt.grid(alpha=.3); plt.legend()
            plt.xlabel("x1"); plt.ylabel("x2"); plt.savefig("task1_graph.png", dpi=150)
            print("\nРисунок сохранён: task1_graph.png")
        except ImportError:
            print("\nmatplotlib не установлен - рисунок не построен.")

def simplex(c, A, b, names, slack_names):
    m, n = len(A), len(c)
    cols = list(names) + list(slack_names)
    T = [[Fr(v) for v in A[i]] + [Fr(int(i == j)) for j in range(m)] + [Fr(b[i])] for i in range(m)]
    cost = [Fr(v) for v in c] + [Fr(0)] * m
    basis = [n + i for i in range(m)]

    def zrow():
        z = [sum(cost[basis[i]] * T[i][j] for i in range(m)) - (cost[j] if j < n + m else 0)
             for j in range(n + m + 1)]
        return z                       # z[-1] = значение целевой функции

    def show(it, enter=None, leave=None, ratios=None):
        print(f"\nСимплексная таблица №{it}")
        head = f"{'Базис':>6} {'Cб':>5} |" + "".join(f"{cn:>9}" for cn in cols) + f" |{'b':>9} |{'b/a':>9}"
        print(head)
        print("-" * len(head))
        for i in range(m):
            r = ratios[i] if ratios else None
            print(f"{cols[basis[i]]:>6} {fmt(cost[basis[i]]):>5} |" +
                  "".join(f"{fmt(T[i][j]):>9}" for j in range(n + m)) +
                  f" |{fmt(T[i][-1]):>9} |{('' if r is None else fmt(r)):>9}")
        z = zrow()
        print(f"{'Δj=zj-cj':>12} |" + "".join(f"{fmt(z[j]):>9}" for j in range(n + m)) + f" |{fmt(z[-1]):>9} |")
        if enter is not None:
            print(f"-> ведущий столбец: {cols[enter]} (Δ = {fmt(z[enter])}); "
                  f"ведущая строка: {cols[basis[leave]]}; ведущий элемент = {fmt(T[leave][enter])}")

    it = 1
    while True:
        z = zrow()
        enter = min(range(n + m), key=lambda j: (z[j], j))
        if z[enter] >= 0:
            show(it)
            print("\nВсе Δj >= 0 -> план оптимален.")
            break
        ratios = [T[i][-1] / T[i][enter] if T[i][enter] > 0 else None for i in range(m)]
        cand = [(r, i) for i, r in enumerate(ratios) if r is not None]
        if not cand:
            show(it, ratios=ratios)
            raise RuntimeError("Целевая функция не ограничена")
        leave = min(cand)[1]
        show(it, enter, leave, ratios)
        pv = T[leave][enter]
        T[leave] = [v / pv for v in T[leave]]
        for i in range(m):
            if i != leave and T[i][enter] != 0:
                f = T[i][enter]
                T[i] = [a - f * bb for a, bb in zip(T[i], T[leave])]
        basis[leave] = enter
        it += 1
    return T, basis, zrow(), cols, it


def task2():
    d = DATA2
    prod, res = d["products"], d["resources"]
    R, b, c = d["R"], d["stock"], d["profit"]
    n, m = len(c), len(b)
    hr("ЗАДАНИЕ 2. Симплекс-метод")
    print("Математическая модель:  П = " + " + ".join(f"{c[j]}·{prod[j]}" for j in range(n)) + " -> max")
    for i in range(m):
        print("  " + " + ".join(f"{R[i][j]}·{prod[j]}" for j in range(n)) + f" <= {b[i]}   ({res[i]})")
    print("  " + ", ".join(prod) + " >= 0")
    print("Каноническая форма: добавлены остаточные переменные s1..s3 (остатки ресурсов):")
    for i in range(m):
        print("  " + " + ".join(f"{R[i][j]}·{prod[j]}" for j in range(n)) + f" + s{i+1} = {b[i]}")

    sl = [f"s{i+1}" for i in range(m)]
    T, basis, z, cols, iters = simplex(c, R, b, prod, sl)
    x = [Fr(0)] * (n + m)
    for i in range(m):
        x[basis[i]] = T[i][-1]
    Z = z[-1]

    print("\n" + "-" * 78 + "\nРЕЗУЛЬТАТЫ И АНАЛИЗ\n" + "-" * 78)
    print("Оптимальный ассортимент и объёмы выпуска:")
    for j in range(n):
        print(f"  {prod[j]} = {x[j]} ≈ {fmt(x[j])}" + ("   (не выпускается)" if x[j] == 0 else ""))
    print(f"Максимальная прибыль: П = {Z} ≈ {fmt(Z)}")
    print("Остатки ресурсов:")
    for i in range(m):
        used = sum(Fr(R[i][j]) * x[j] for j in range(n))
        st = "дефицитный (использован полностью)" if x[n + i] == 0 else "недефицитный"
        print(f"  {res[i]}: запас {b[i]}, использовано {fmt(used)}, остаток s{i+1} = {x[n+i]} ≈ {fmt(x[n+i])}  - {st}")

    print("\nДвойственные оценки ресурсов (теневые цены, Δj остаточных переменных):")
    y = [z[n + i] for i in range(m)]
    for i in range(m):
        print(f"  y_{res[i]} = {y[i]} ≈ {fmt(y[i])}   -> +1 ед. ресурса {res[i]} даёт {fmt(y[i])} ед. прибыли"
              if y[i] != 0 else f"  y_{res[i]} = 0   -> ресурс избыточен, дополнительная единица прибыли не даёт")
    dual = sum(Fr(b[i]) * y[i] for i in range(m))
    print(f"Проверка (теорема двойственности): Σ b_i·y_i = {fmt(dual)} {'=' if dual == Z else '≠'} П = {fmt(Z)}")

    print("\nОценки невыпускаемых изделий (Δj, на сколько снизится прибыль при выпуске 1 ед.):")
    any_ = False
    for j in range(n):
        if j not in basis:
            any_ = True
            print(f"  {prod[j]}: Δ = {fmt(z[j])}")
    if not any_:
        print("  все изделия входят в план")

    print("\nУстойчивость плана: диапазон изменения запаса ресурса, в котором базис (и теневая цена) сохраняется:")
    for i in range(m):
        col = [T[k][n + i] for k in range(m)]            # столбец B^-1 для s_i
        xb = [T[k][-1] for k in range(m)]
        lo = max([-xb[k] / col[k] for k in range(m) if col[k] > 0], default=None)
        hi = min([-xb[k] / col[k] for k in range(m) if col[k] < 0], default=None)
        lo_s = "-∞" if lo is None else fmt(b[i] + lo)
        hi_s = "+∞" if hi is None else fmt(b[i] + hi)
        print(f"  {res[i]}: запас {b[i]} ∈ [{lo_s}; {hi_s}]")

    # ---- независимая проверка: перебор всех базисных решений через numpy
    Afull = np.hstack([np.array(R, float), np.eye(m)])
    cfull = np.array(list(c) + [0] * m, float)
    best, bestx = -1e18, None
    for B in itertools.combinations(range(n + m), m):
        Bm = Afull[:, B]
        if abs(np.linalg.det(Bm)) < 1e-12:
            continue
        xb = np.linalg.solve(Bm, b)
        if np.all(xb >= -1e-9) and cfull[list(B)] @ xb > best:
            best = cfull[list(B)] @ xb
            bestx = (B, xb)
    print(f"\nПроверка перебором всех базисных решений (numpy): max П = {best:.6f}; "
          f"симплекс-метод: {float(Z):.6f}  {'(совпадает)' if abs(best - float(Z)) < 1e-8 else '(ОШИБКА!)'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args or ["1", "2"]
    if "1" in which:
        task1(plot="--plot" in sys.argv)
    if "2" in which:
        task2()