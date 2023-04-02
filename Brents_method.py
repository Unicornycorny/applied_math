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

@painter_decorator
def brents_method(l, r, eps, s=1):
    gr = (math.sqrt(5) - 1) / 2
    m = w = v = l + gr * (r - l)
    fm = fw = fv = s * f(m)
    d = e = 0
    u = float('+inf')
    algo_type = None
    debug_start(l, r, *[m, w, v, fm, fw, fv, algo_type])
    while r - l > eps:
        g, e = e, d
        if len({m, w, v}) == len({fm, fw, fv}) == 3:
            p = ((m - w) ** 2) * (fm - fv) - ((m - v) ** 2) * (fm - fw)
            q = 2 * ((m - w) * (fm - fv) - (m - v) * (fm - fw))
            u = m - p / q

        if l + eps <= u <= r - eps and 2 * abs(u - m) < g:
            algo_type = 'spi'
            d = abs(u - m)
        else:
            algo_type = 'gss'
            if m < (r + l) * .5:
                d = r - m
                u = m + gr * d
            else:
                d = m - l
                u = m - gr * d

        if abs(u - m) < eps:
            return u

        fu = s * f(u)

        if fu <= fm:
            if u >= m:
                l = m
            else:
                r = m
            v, w, m = w, m, u
            fv, fw, fm = fw, fm, fu

        else:
            if u >= m:
                r = u
            else:
                l = u

            if fu <= fw or w == m:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == m or v == w:
                v = u
                fv = fu

        debug_tick(l, r, d, *[m, w, v, fm, fw, fv, algo_type])

    return (l + r) / 2
brents_method(-1.5, 2, 0.001)
print(debug_result(*['l', 'r', 'm', 'w', 'v', 'f(m)', 'f(w)', 'f(v)', 'algo_type']))