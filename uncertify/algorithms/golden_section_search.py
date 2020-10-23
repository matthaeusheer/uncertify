import math

import numpy as np

from typing import Tuple, Callable, Union

INV_PHI = (math.sqrt(5) - 1) / 2
INV_PHI_SQUARED = (3 - math.sqrt(5)) / 2


def golden_section_search(func: Callable, low: float, up: float, tolerance: float = 1e-5,
                          return_mean: bool = True) -> Union[Tuple[float, float], float]:
    """Golden Section Search Algorithm - Taken directly from Wikipedia.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    This implementation
    reuses function evaluations, saving 1/2 of the evaluations per
    iteration, and returns a bounding interval if return_mean=False, else it takes the mean of the
    upper and lower bound of the interval.

    Example:
        f = lambda x: (x-2)**2
        a = 1
        b = 5
        tol = 1e-5
        (c,d) = golden_section_search(f, a, b, tol)
        print(c, d)
        1.9999959837979107 2.0000050911830893
    """
    (low, up) = (min(low, up), max(low, up))
    h = up - low
    if h <= tolerance:
        if return_mean:
            return float(np.mean([low, up]))
        else:
            return low, up

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tolerance / h) / math.log(INV_PHI)))

    c = low + INV_PHI_SQUARED * h
    d = low + INV_PHI * h
    yc = func(c)
    yd = func(d)

    for k in range(n - 1):
        if yc < yd:
            up = d
            d = c
            yd = yc
            h = INV_PHI * h
            c = low + INV_PHI_SQUARED * h
            yc = func(c)
        else:
            low = c
            c = d
            yc = yd
            h = INV_PHI * h
            d = low + INV_PHI * h
            yd = func(d)

    if yc < yd:
        if return_mean:
            return float(np.mean([low, d]))
        else:
            return low, d
    else:
        if return_mean:
            return float(np.mean([c, up]))
        else:
            return c, up
