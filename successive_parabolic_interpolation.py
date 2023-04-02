import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('dark_background')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def f(x):
    return math.log(math.pow(x, 2)) + 1 - math.sin(x)


def painter_decorator(func):
    def draw(*args):
        p = func(*args)
        xx = np.arange(max(-25, p - 25), p + 25, 0.05)
        yy = np.array([f(x) for x in xx])
        plt.plot(xx, yy)
        plt.scatter(p, f(p), c='r')
        plt.show()

    return draw

prev_step = 0
debug_list = []


def debug_start(l, r, *args):
    global prev_step, debug_list
    debug_list = [[l, r] + list(args) + [None]]
    prev_step = r - l


def debug_tick(l, r, step, *args):
    global prev_step, debug_list
    attitude = step / prev_step
    prev_step = step
    debug_list.append([l, r] + list(args) + [attitude])


def debug_result(*args):
    global debug_list
    args = list(args) + ['attitude']
    frame = pd.DataFrame(debug_list, columns=args)
    ord_args = args
    r_id = 1
    while r_id < len(args) - 1 and not ord_args[r_id + 1].startswith('f'):
        ord_args[r_id], ord_args[r_id + 1] = ord_args[r_id + 1], ord_args[r_id]
        r_id += 1

    return frame[ord_args]

from scipy.interpolate import lagrange

@painter_decorator
def successive_parabolic_interpolation(l, r, eps, s=1):
    m = (l + r) / 2

    # for lagrange
    xx = np.arange(l, r, 0.05)
    yy = np.array([f(x) for x in xx])
    parables = []

    f1, f2, f3 = s * f(l), s * f(m), s * f(r)
    debug_start(l, r, *[m, f1, f2, f3])
    while r - l > eps:
        p = ((m - l) ** 2) * (f2 - f3) - ((m - r) ** 2) * (f2 - f1)
        q = 2 * ((m - l) * (f2 - f3) - (m - r) * (f2 - f1))
        u = m - p / q
        fu = s * f(u)
        lagr = lagrange([l, m, r], [f1, f2, f3])
        parables.append(s * lagr(xx))
        if m > u:
            if f2 < fu:
                l, f1 = u, fu
            else:
                r, f3 = m, f2
                m, f2 = u, fu
        else:
            if f2 > fu:
                l, f1 = m, f2
                m, f2 = u, fu
            else:
                r, f3 = u, fu
        debug_tick(l, r, r - l, *[m, f1, f2, f3])

    p = (l + r) / 2
    plt.scatter(p, f(p), c='r')
    for i in parables:
        plt.plot(xx, i, c='w', linestyle='--', alpha=.3)
    plt.plot(xx, yy)
    plt.show()
    return p
successive_parabolic_interpolation(-1.5, 2, 0.001)
print(debug_result(*['l', 'r', 'm', 'f(l)', 'f(m)', 'f(r)']))