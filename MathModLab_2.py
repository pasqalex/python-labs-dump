"""
Задачи по уравнениям математической физики (чистый Python, без внешних пакетов).

Запуск:   python3 pde_tasks.py - все пять задач
          python3 pde_tasks.py 3 - только задача 3  (номера 1..5)

Задача 1. (2x+y+1)U_x + (x-y)U_y = 0 - общее решение
Задача 2. 2U_xx + 2U_xy + 5U_yy = 0 - тип и канонический вид
Задача 3. U_tt = a²U_xx, a=3, U(0,x)=x², U_t(0,x)=1/x, U(t,0)=U(t,l)=0 - метод Фурье
Задача 4. U_tt = a²U_xx - 50(l-x)sin4t, a=1.5, нулевые н.у. и г.у. - метод Фурье
Задача 5. U_t = U_xx + sin(πx/12), U(x,0)=x sin πx, U(0,t)=0.5, U(l,t)=e^-t,
          x∈[0,1], t∈[0,2] - разностные схемы

ВНИМАНИЕ: в задачах 3 и 4 длина отрезка l в условии не задана. Здесь l = L = 1
(как в задаче 5); поменяйте константу L ниже, если у вас другое значение.
"""
import math
import cmath
import sys

PI = math.pi
L = 1.0


def lin(a, b, c):
    """Красивая запись a*x + b*y + c."""
    out = ""
    for coef, name in ((a, "x"), (b, "y"), (c, "")):
        if abs(coef) < 1e-12:
            continue
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        term = (f"{mag:g}" if (name == "" or mag != 1) else "") + name
        out += (f" {sign} " if out else ("-" if coef < 0 else "")) + term
    return out or "0"


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


#1 task
def task1(P=(2.0, 1.0, 1.0), Q=(1.0, -1.0, 0.0)):
    a1, b1, c1 = P
    a2, b2, c2 = Q
    hr("ЗАДАЧА 1. Общее решение линейного однородного уравнения 1-го порядка")
    print(f"({lin(a1, b1, c1)}) U_x + ({lin(a2, b2, c2)}) U_y = 0")
    print("Характеристическая система: dx/P = dy/Q = ds,  т.е.  dy/dx = Q/P.")
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        print("det = 0: особая точка не единственна - случай не реализован.")
        return
    x0 = (b1 * c2 - c1 * b2) / det
    y0 = (a2 * c1 - a1 * c2) / det
    print(f"Точка покоя (P=Q=0): x0 = {x0:.6f}, y0 = {y0:.6f}.  Сдвиг X = x-x0, Y = y-y0")
    print("Тогда  dX/ds = a1*X + b1*Y,  dY/ds = a2*X + b2*Y  (линейная система с матрицей M).")
    tr = a1 + b2
    disc = tr * tr - 4 * det
    print(f"Характеристическое уравнение M: λ² - {tr:g}·λ + ({det:g}) = 0,  D = {disc:g}")
    if abs(disc) < 1e-12:
        print("Кратный корень - случай не реализован.")
        return
    sq = cmath.sqrt(disc)
    lams = [(tr + sq) / 2, (tr - sq) / 2]

    def left(lam):  # левый собственный вектор: l*M = lam*l, тогда u=l·(X,Y), du/ds = lam*u
        if abs(a2) > 1e-12:
            return (a2, lam - a1)
        return (lam - b2, b1)

    if disc > 0:
        l1, l2 = lams[0].real, lams[1].real
        v1, v2 = left(l1), left(l2)
        print(f"λ1 = {l1:.6f}, λ2 = {l2:.6f} (вещественные, разные).")
        print(f"u1 = ({v1[0]:.6f})·X + ({v1[1]:.6f})·Y   (du1/ds = λ1·u1)")
        print(f"u2 = ({v2[0]:.6f})·X + ({v2[1]:.6f})·Y   (du2/ds = λ2·u2)")
        print("Первый интеграл:  Φ = λ2·ln|u1| - λ1·ln|u2| = const,  т.е.")
        print(f"   |u1|^({l2:.6f}) · |u2|^({-l1:.6f}) = const")

        def Phi(x, y):
            X, Y = x - x0, y - y0
            u1 = v1[0] * X + v1[1] * Y
            u2 = v2[0] * X + v2[1] * Y
            return l2 * math.log(abs(u1)) - l1 * math.log(abs(u2))
    else:
        lam = lams[0] if lams[0].imag > 0 else lams[1]
        al, be = lam.real, lam.imag
        v = left(lam)
        print(f"λ = {al:.6f} ± {be:.6f}i (комплексные).")
        print(f"u = ({v[0]:.6f})·X + ({v[1]:.6f})·Y  (комплексная),  du/ds = λ·u")
        print(f"Первый интеграл:  Φ = ln|u| - ({al/be:.6f})·arg(u) = const")

        def Phi(x, y):
            X, Y = x - x0, y - y0
            u = v[0] * X + v[1] * Y
            return math.log(abs(u)) - al / be * cmath.phase(u)

    print("\nОБЩЕЕ РЕШЕНИЕ:  U(x,y) = F(Φ(x,y)),  F - произвольная дифференцируемая функция.")

    #проверка 1: Φ постоянна вдоль характеристики (RK4)
    def rk4(x, y, s_end, n=4000):
        h = s_end / n
        fx = lambda x, y: a1 * x + b1 * y + c1
        fy = lambda x, y: a2 * x + b2 * y + c2
        for _ in range(n):
            k1x, k1y = fx(x, y), fy(x, y)
            k2x, k2y = fx(x + h / 2 * k1x, y + h / 2 * k1y), fy(x + h / 2 * k1x, y + h / 2 * k1y)
            k3x, k3y = fx(x + h / 2 * k2x, y + h / 2 * k2y), fy(x + h / 2 * k2x, y + h / 2 * k2y)
            k4x, k4y = fx(x + h * k3x, y + h * k3y), fy(x + h * k3x, y + h * k3y)
            x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        return x, y

    print("\nПроверка 1: Φ вдоль численно проинтегрированной характеристики (RK4)")
    for (xs, ys) in [(1.0, 0.5), (2.0, -1.0), (0.0, 1.0)]:
        try:
            ph0 = Phi(xs, ys)
            dev = max(abs(Phi(*rk4(xs, ys, s)) - ph0) for s in (0.1, 0.2, 0.3))
            print(f"  старт ({xs:g},{ys:g}): Φ = {ph0:+.8f},  max|ΔΦ| = {dev:.2e}")
        except (ValueError, ZeroDivisionError):
            print(f"  старт ({xs:g},{ys:g}): характеристика проходит через особую линию, пропущено")

    #проверка 2: невязка уравнения P·Φx + Q·Φy
    print("Проверка 2: невязка P·Φ_x + Q·Φ_y (центральные разности)")
    h = 1e-6
    for (xs, ys) in [(1.0, 0.5), (2.0, -1.0), (0.7, 0.2)]:
        try:
            px = (Phi(xs + h, ys) - Phi(xs - h, ys)) / (2 * h)
            py = (Phi(xs, ys + h) - Phi(xs, ys - h)) / (2 * h)
            Pv = a1 * xs + b1 * ys + c1
            Qv = a2 * xs + b2 * ys + c2
            print(f"  ({xs:g},{ys:g}): {Pv*px + Qv*py:+.2e}")
        except (ValueError, ZeroDivisionError):
            print(f"  ({xs:g},{ys:g}): пропущено")


#2 task
def task2(A=2.0, B=2.0, C=5.0):
    hr("ЗАДАЧА 2. Тип уравнения 2-го порядка и канонический вид")
    print(f"{A:g}·U_xx + ({B:g})·U_xy + ({C:g})·U_yy = 0")
    a11, a12, a22 = A, B / 2, C
    D = a12 ** 2 - a11 * a22
    print(f"a11 = {a11:g}, a12 = {a12:g}, a22 = {a22:g};  Δ = a12² - a11·a22 = {D:g}")
    typ = "эллиптический" if D < -1e-12 else ("гиперболический" if D > 1e-12 else "параболический")
    print(f"Δ {'<' if D < -1e-12 else ('>' if D > 1e-12 else '=')} 0  =>  уравнение {typ.upper()}")
    print("Характеристическое уравнение: a11·dy² - 2·a12·dx·dy + a22·dx² = 0")

    sw = abs(a11) < 1e-12 and abs(a22) > 1e-12
    m11, m22 = (a22, a11) if sw else (a11, a22)
    m12 = a12
    if abs(m11) < 1e-12:
        rows = [(1.0, 0.0), (0.0, 1.0)]
        print("a11 = a22 = 0: уже вид U_xy, ξ = x, η = y.")
    elif D < -1e-12:
        r = math.sqrt(-D)
        print(f"dy/dx = ({m12:g} ± {r:g}·i)/{m11:g}  =>  y - ({m12/m11:g})x ∓ i({r/m11:g})x = const")
        rows = [(-m12 / m11, 1.0), (r / m11, 0.0)]
    elif D > 1e-12:
        r = math.sqrt(D)
        l1, l2 = (m12 + r) / m11, (m12 - r) / m11
        print(f"dy/dx = {l1:g} и {l2:g}")
        rows = [(-l1, 1.0), (-l2, 1.0)]
    else:
        lam = m12 / m11
        print(f"dy/dx = {lam:g} (двойной корень)")
        rows = [(-lam, 1.0), (1.0, 0.0)]
    if sw:
        rows = [(q, p) for (p, q) in rows]
    (p1, q1), (p2, q2) = rows
    print(f"\nЗамена переменных:  ξ = {lin(p1, q1, 0)},   η = {lin(p2, q2, 0)}")

    M = [[a11, a12], [a12, a22]]
    J = [[p1, q1], [p2, q2]]
    N = [[sum(J[i][k] * M[k][l] * J[j][l] for k in range(2) for l in range(2))
          for j in range(2)] for i in range(2)]
    cxx, cxy, cyy = N[0][0], 2 * N[0][1], N[1][1]
    cxx, cxy, cyy = [0.0 if abs(v) < 1e-12 else v for v in (cxx, cxy, cyy)]
    print(f"Канонический вид:  {cxx:g}·U_ξξ + {cxy:g}·U_ξη + {cyy:g}·U_ηη = 0")
    nz = [v for v in (cxx, cxy, cyy) if v != 0][0]
    print(f"После деления на {nz:g}:  {cxx/nz:g}·U_ξξ + {cxy/nz:g}·U_ξη + {cyy/nz:g}·U_ηη = 0")
    if typ == "эллиптический":
        print("(уравнение Лапласа: U_ξξ + U_ηη = 0)")

    #оператор исходного уравнения на U(x,y)=V(e,т) и каноническое выражение для V
    V = lambda s, e: math.sin(0.7 * s) * math.exp(0.3 * e) + s * s * e + 0.5 * s * e * e
    U = lambda x, y: V(p1 * x + q1 * y, p2 * x + q2 * y)
    h = 1e-3

    def d2(F, u, v):
        Fxx = (F(u + h, v) - 2 * F(u, v) + F(u - h, v)) / h ** 2
        Fyy = (F(u, v + h) - 2 * F(u, v) + F(u, v - h)) / h ** 2
        Fxy = (F(u + h, v + h) - F(u + h, v - h) - F(u - h, v + h) + F(u - h, v - h)) / (4 * h ** 2)
        return Fxx, Fxy, Fyy

    print("\nПроверка на пробной функции U = V(ξ(x,y), η(x,y)):")
    for (x, y) in [(0.3, 0.4), (-0.5, 0.9)]:
        uxx, uxy, uyy = d2(U, x, y)
        vss, vse, vee = d2(V, p1 * x + q1 * y, p2 * x + q2 * y)
        left_ = A * uxx + B * uxy + C * uyy
        right = cxx * vss + cxy * vse + cyy * vee
        print(f"  ({x:+.1f},{y:+.1f}): исходный оператор = {left_:+.6f},  канонический = {right:+.6f}")

#3 task
def wave_fd(a, l, nx, t_list, u0, v0, f=None):
    """Разностная схема 'крест' при числе Куранта = 1 (шаг по времени tau = h/a)."""
    h = l / nx
    tau = h / a
    x = [i * h for i in range(nx + 1)]
    prev = [u0(xi) for xi in x]
    prev[0] = prev[-1] = 0.0
    vel = [v0(xi) if 0 < i < nx else 0.0 for i, xi in enumerate(x)]
    src = (lambda t, xi: f(t, xi)) if f else (lambda t, xi: 0.0)
    cur = [0.0] * (nx + 1)
    for i in range(1, nx):
        cur[i] = 0.5 * (prev[i - 1] + prev[i + 1]) + tau * vel[i] + 0.5 * tau ** 2 * src(0.0, x[i])
    want = {max(1, round(t / tau)): t for t in t_list}
    snaps = {}
    nmax = max(want)
    for n in range(1, nmax):
        nxt = [0.0] * (nx + 1)
        for i in range(1, nx):
            nxt[i] = cur[i - 1] + cur[i + 1] - prev[i] + tau ** 2 * src(n * tau, x[i])
        prev, cur = cur, nxt
        if n + 1 in want:
            snaps[n + 1] = cur[:]
    if 1 in want:
        snaps[1] = cur[:]
    return h, tau, snaps


def task3(a=3.0, l=L, N=2000):
    hr(f"ЗАДАЧА 3. Однородное волновое уравнение, метод Фурье  (a={a:g}, l={l:g}, N={N} членов ряда)")
    print("U_tt = a²U_xx,  U(0,x)=x²,  U_t(0,x)=1/x,  U(t,0)=U(t,l)=0")
    print("U = Σ [A_n cos(ω_n t) + B_n sin(ω_n t)] sin(nπx/l),  ω_n = a·nπ/l")
    print("A_n = (2/l)·( -l²(-1)^n/k_n + 2((-1)^n - 1)/k_n³ ),  k_n = nπ/l")
    print("B_n = 2·Si(nπ)/(a·nπ),   Si(z) = ∫_0^z sin(s)/s ds")

    def seg(n):  # ∫_{(n-1)π}^{nπ} sin s / s ds  (Симпсон)
        lo, hi, m = (n - 1) * PI, n * PI, 40
        hh = (hi - lo) / m
        g = lambda s: math.sin(s) / s if abs(s) > 1e-14 else 1.0
        S = g(lo) + g(hi)
        for j in range(1, m):
            S += (4 if j % 2 else 2) * g(lo + j * hh)
        return S * hh / 3

    An, Bn, k, w = [], [], [], []
    si = 0.0
    for n in range(1, N + 1):
        kn = n * PI / l
        s = (-1) ** n
        An.append(2 / l * (-l * l * s / kn + 2 * (s - 1) / kn ** 3))
        si += seg(n)
        Bn.append(2 * si / (a * n * PI))
        k.append(kn)
        w.append(a * kn)

    def U(t, x):
        return sum((An[i] * math.cos(w[i] * t) + Bn[i] * math.sin(w[i] * t)) * math.sin(k[i] * x)
                   for i in range(N))

    print(f"\nПервые коэффициенты:  A_1..3 = {An[0]:.6f}, {An[1]:.6f}, {An[2]:.6f};"
          f"  B_1..3 = {Bn[0]:.6f}, {Bn[1]:.6f}, {Bn[2]:.6f}")
    xs = [0.1 * l, 0.25 * l, 0.5 * l, 0.75 * l, 0.9 * l]
    print("\nПроверка начального условия U(0,x) = x²:")
    print(f"{'x':>8} | {'U(0,x)':>10} | {'x²':>10}")
    for x in xs:
        print(f"{x:8.3f} | {U(0, x):10.5f} | {x*x:10.5f}")
    print("(для U_t(0,x)=1/x ряд поточечно не сходится: 1/x не принадлежит L², B_n·ω_n = 2Si(nπ)/l -> π/l)")
    ts = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    print("\nТаблица U(t,x):")
    print(f"{'t \\ x':>8} | " + " | ".join(f"{x:9.3f}" for x in xs))
    print("-" * (11 + 12 * len(xs)))
    for t in ts:
        print(f"{t:8.3f} | " + " | ".join(f"{U(t, x):9.5f}" for x in xs))

    # независимая проверка разностной схемой
    nx = 500
    tt = [0.1, 0.2, 0.5]
    h, tau, snaps = wave_fd(a, l, nx, tt, lambda x: x * x, lambda x: 1 / x)
    print(f"\nСравнение с разностной схемой (h={h:g}, tau={tau:.2e}, число Куранта 1):")
    def fold(z):                      # отражение от концов отрезка [0,l]
        z %= 2 * l
        return 2 * l - z if z > l else z

    print(f"{'t':>6} {'x':>7} | {'ряд Фурье':>10} | {'схема':>10} | {'разность':>10}")
    for t in tt:
        n = max(1, round(t / tau))
        tn = n * tau
        lines = (fold(a * tn), fold(l - a * tn))   # особые линии: x = at и фронт от x = l
        for x in (0.2 * l, 0.5 * l, 0.8 * l):
            i = round(x / h)
            u_fd = snaps[n][i]
            u_f = U(tn, i * h)
            near = min(abs(i * h - p) for p in lines) < 0.03 * l
            print(f"{tn:6.3f} {i*h:7.3f} | {u_f:10.5f} | {u_fd:10.5f} | {abs(u_f-u_fd):10.2e}"
                  + ("   <- на особой линии" if near else ""))
    print("\nЗамечания:")
    print(" * x² не равно 0 при x = l, а 1/x не ограничена при x -> 0: условия не согласованы с")
    print("   краевыми, поэтому ряд сходится медленно, около x = l видны колебания (явление Гиббса),")
    print("   а решение имеет особенности на линиях x = a·t (log-сингулярность) и отражённых от неё.")
    print("   Вблизи этих линий разность 'ряд - схема' может быть заметной.")


#4 task
def task4(a=1.5, l=L, N=200):
    hr(f"ЗАДАЧА 4. Неоднородное волновое уравнение, метод Фурье  (a={a:g}, l={l:g}, N={N})")
    print("U_tt = a²U_xx - 50(l-x)·sin 4t,   U(0,x) = U_t(0,x) = U(t,0) = U(t,l) = 0")
    print("U = Σ T_n(t) sin(nπx/l),  F_n = -100/k_n,  T_n'' + ω_n²T_n = F_n sin 4t")
    print("T_n = F_n/(ω_n² - 16)·[ sin 4t - (4/ω_n) sin(ω_n t) ]   (ω_n = a·nπ/l)")
    res = [n for n in range(1, N + 1) if abs(a * n * PI / l - 4) < 1e-9]
    if res:
        print(f"Резонанс при n = {res}: используется T_n = F_n(sin 4t/32 - t cos 4t/8)")

    def T(n, t):
        kn = n * PI / l
        wn = a * kn
        Fn = -100.0 / kn
        if abs(wn - 4) < 1e-9:
            return Fn * (math.sin(4 * t) / 32 - t * math.cos(4 * t) / 8)
        return Fn / (wn * wn - 16) * (math.sin(4 * t) - 4 / wn * math.sin(wn * t))

    def U(t, x):
        return sum(T(n, t) * math.sin(n * PI * x / l) for n in range(1, N + 1))

    xs = [0.1 * l, 0.25 * l, 0.5 * l, 0.75 * l, 0.9 * l]
    ts = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    print("\nТаблица U(t,x):")
    print(f"{'t \\ x':>8} | " + " | ".join(f"{x:10.3f}" for x in xs))
    print("-" * (11 + 13 * len(xs)))
    for t in ts:
        print(f"{t:8.3f} | " + " | ".join(f"{U(t, x):10.5f}" for x in xs))

    nx = 400
    tt = [0.5, 1.0, 2.0, 3.0]
    f = lambda t, x: -50 * (l - x) * math.sin(4 * t)
    h, tau, snaps = wave_fd(a, l, nx, tt, lambda x: 0.0, lambda x: 0.0, f)
    print(f"\nСравнение с разностной схемой (h={h:g}, tau={tau:.2e}):")
    print(f"{'t':>6} {'x':>7} | {'ряд Фурье':>10} | {'схема':>10} | {'разность':>10}")
    worst = 0.0
    for t in tt:
        n = max(1, round(t / tau))
        tn = n * tau
        for x in (0.25 * l, 0.5 * l, 0.75 * l):
            i = round(x / h)
            u_fd, u_f = snaps[n][i], U(tn, i * h)
            worst = max(worst, abs(u_f - u_fd))
            print(f"{tn:6.3f} {i*h:7.3f} | {u_f:10.6f} | {u_fd:10.6f} | {abs(u_f-u_fd):10.2e}")
    print(f"max |разность| = {worst:.2e}")
    print("Сходимость ряда по числу членов (x = l/2, t = 2):")
    prev = None
    for n_ in (5, 10, 20, 40, 80, 160):
        v = sum(T(n, 2.0) * math.sin(n * PI * 0.5) for n in range(1, n_ + 1))
        print(f"  N = {n_:4d}: U = {v:.8f}" + ("" if prev is None else f"   |Δ| = {abs(v-prev):.2e}"))
        prev = v


#5 task
def thomas(a, b, c, d):
    n = len(d)
    cp, dp = [0.0] * n, [0.0] * n
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def heat(N, tau, theta, T=2.0, l=1.0, times=(0.5, 1.0, 2.0), rannacher=False):
    h = l / N
    x = [i * h for i in range(N + 1)]
    u = [xi * math.sin(PI * xi) for xi in x]
    u[0], u[N] = 0.5, math.exp(0.0)
    f = [math.sin(PI * xi / 12) for xi in x]

    def step(u, t, dt, th):
        r = dt / h ** 2
        g0, g1 = 0.5, math.exp(-(t + dt))
        rhs = [u[i] + (1 - th) * r * (u[i - 1] - 2 * u[i] + u[i + 1]) + dt * f[i] for i in range(1, N)]
        if th == 0:
            sol = rhs
        else:
            rhs[0] += th * r * g0
            rhs[-1] += th * r * g1
            n = N - 1
            sol = thomas([-th * r] * n, [1 + 2 * th * r] * n, [-th * r] * n, rhs)
        return [g0] + sol + [g1]

    steps = round(T / tau)
    want = {round(t / tau): t for t in times}
    snaps = {}
    for n in range(steps):
        t = n * tau
        if n == 0 and rannacher:       # старт Раннахера: два шага Эйлера по tau/2 (гасит колебания КН)
            u = step(u, t, tau / 2, 1.0)
            u = step(u, t + tau / 2, tau / 2, 1.0)
        else:
            u = step(u, t, tau, theta)
        if n + 1 in want:
            snaps[want[n + 1]] = u[:]
    return x, snaps, u


def task5(l=1.0):
    hr("ЗАДАЧА 5. Неоднородное уравнение теплопроводности, разностные схемы")
    print("U_t = U_xx + sin(πx/12),  U(x,0) = x·sin(πx),  U(0,t) = 0.5,  U(1,t) = e^{-t},")
    print("x ∈ [0,1], t ∈ [0,2].   Схемы: явная, неявная (Эйлера), Кранка-Николсон (КН).")
    print("Замечание: на границе при t=0 условия не согласованы (U(0,0)=0.5 против 0 из н.у.,")
    print("U(1,0)=1 против 0): в угловых узлах берутся значения краевых условий.")
    times = (0.5, 1.0, 2.0)

    N = 20
    h = l / N
    tau_e = 0.4 * h * h
    tau_i = 0.01
    print(f"\nСетка: N = {N}, h = {h:g}.  Явная: tau = {tau_e:g} (r = tau/h² = 0.4 ≤ 0.5);"
          f"  неявная и КН: tau = {tau_i:g}")
    xe, se, _ = heat(N, tau_e, 0.0, times=times)
    xb, sb, _ = heat(N, tau_i, 1.0, times=times)
    xc, sc, _ = heat(N, tau_i, 0.5, times=times, rannacher=True)
    NR = 320
    tau_r = 0.5 * l / NR
    _, sr, _ = heat(NR, tau_r, 0.5, times=times, rannacher=True)
    ratio = NR // N
    print(f"Эталон: КН (старт Раннахера) на сетке N = {NR}, tau = {tau_r:g}.")

    for t in times:
        print(f"\nt = {t:g}:")
        print(f"{'x':>6} | {'явная':>10} | {'неявная':>10} | {'КН':>10} | {'эталон':>10}")
        for i in range(0, N + 1, 2):
            print(f"{xe[i]:6.2f} | {se[t][i]:10.6f} | {sb[t][i]:10.6f} | {sc[t][i]:10.6f} | {sr[t][i*ratio]:10.6f}")

    print("\nМаксимальная погрешность относительно эталона (по узлам сетки N = 20):")
    print(f"{'t':>6} | {'явная':>10} | {'неявная':>10} | {'КН':>10}")
    for t in times:
        e = lambda s: max(abs(s[t][i] - sr[t][i * ratio]) for i in range(N + 1))
        print(f"{t:6.2f} | {e(se):10.2e} | {e(sb):10.2e} | {e(sc):10.2e}")

    print("\nСходимость КН (tau = h/2): max|U_N - U_2N| по общим узлам")
    print(f"{'N':>5} {'h':>9} | " + " | ".join(f"t={t:g}: разность  отнош." for t in times))
    Ns = [10, 20, 40, 80, 160]
    sols = {n: heat(n, 0.5 * l / n, 0.5, times=times, rannacher=True)[1] for n in Ns}
    prevd = {}
    for n in Ns[:-1]:
        row = []
        for t in times:
            d = max(abs(sols[n][t][i] - sols[2 * n][t][2 * i]) for i in range(n + 1))
            rr = "   -  " if t not in prevd else f"{prevd[t]/d:6.2f}"
            prevd[t] = d
            row.append(f"{d:12.3e} {rr}")
        print(f"{n:5d} {l/n:9.5f} | " + " | ".join(row))
    print("(отношение ≈ 4 означает второй порядок точности по h)")

    print("\nУстойчивость явной схемы (N = 20):")
    for r in (0.4, 0.5, 0.6):
        tau = r * h * h
        _, _, u = heat(N, tau, 0.0)
        print(f"  r = {r:.1f}:  max|U| при t≈2 = {max(abs(v) for v in u):.3e}" +
              ("   <- схема неустойчива" if r > 0.5 else ""))


TASKS = {"1": task1, "2": task2, "3": task3, "4": task4, "5": task5}

if __name__ == "__main__":
    which = sys.argv[1:] or sorted(TASKS)
    for k in which:
        TASKS[k]()