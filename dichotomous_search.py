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
def dichotomous_search(l, r, eps, s=1):
    debug_start(l, r, *[0, 0, 0, 0])
    delta = eps / 3

    while r - l > eps:
        m = (l + r) * .5
        x1 = m - delta
        x2 = m + delta
        if s * f(x1) > s * f(x2):
            l = x1
        else:
            r = x2
        debug_tick(l, r, r - l, *[x1, x2, f(x1), f(x2)])
    return (l + r) * .5
dichotomous_search(-1.5, 2, 0.001)
print(debug_result(*['l', 'r', 'x1', 'x2', 'f(x1)', 'f(x2)']))