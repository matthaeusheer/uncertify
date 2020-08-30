import math

from typing import Tuple, Callable

INV_PHI = (math.sqrt(5) - 1) / 2
INV_PHI_SQUARED = (3 - math.sqrt(5)) / 2


def golden_section_search(func: Callable, a: float, b: float, tol: float = 1e-5) -> Tuple[float, float]:
    """Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    This implementation
    reuses function evaluations, saving 1/2 of the evaluations per
    iteration, and returns a bounding interval.

    Example:
        f = lambda x: (x-2)**2
        a = 1
        b = 5
        tol = 1e-5
        (c,d) = golden_section_search(f, a, b, tol)
        print(c, d)
        1.9999959837979107 2.0000050911830893
    """
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return a, b

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(INV_PHI)))

    c = a + INV_PHI_SQUARED * h
    d = a + INV_PHI * h
    yc = func(c)
    yd = func(d)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = INV_PHI * h
            c = a + INV_PHI_SQUARED * h
            yc = func(c)
        else:
            a = c
            c = d
            yc = yd
            h = INV_PHI * h
            d = a + INV_PHI * h
            yd = func(d)

    if yc < yd:
        return a, d
    else:
        return c, b
