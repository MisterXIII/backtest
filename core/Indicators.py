from numba import njit, float64, int64, f8, i8
from numba.core.types import UniTuple
import numpy as np


@njit("float64[:](float64[:], int64)")
def ema(x, period):
    alpha = 2 / (period + 1)
    y = np.empty_like(x)
    n = len(x)
    o = 0
    while np.isnan(x[o]):
        o += 1
    y[:o + period - 1] = np.nan
    y[o + period - 1] = np.mean(
        x[o: o + period])
    for k in range(o + period, n):
        y[k] = alpha * x[k] + (1 - alpha) * y[k - 1]
    return y


@njit(float64[:](float64[:], int64))
def sma(x, period):
    y = np.full_like(x, np.nan)
    n = len(x)
    y[period] = np.mean(x[:period])
    cum = np.sum(x[:period])
    for k in range(period, n):
        cum += x[k] - x[k - period]
        y[k] = cum / period

    return y


@njit("float64[:](float64[:], int64)")
def slope(x, a):
    if a < 1:
        raise ValueError("a must be greater than or equal to 1")
    y = np.full_like(x, np.nan)
    n = len(x)
    y[0] = x[0]
    for k in range(1, n):
        y[k] = (x[k] - x[k - a]) / a

    return y


@njit(float64[:](float64[:], int64, int64))
def macd(x, a=12, b=26):
    return ema(x, b) - ema(x, a)


@njit(UniTuple(f8[:], 3)(f8[:], i8))
def bollinger_bands(x, period=20):
    middle = sma(x, period)
    upper = np.full_like(x, np.nan)
    lower = np.full_like(x, np.nan)

    o = 0
    while x[o] == np.nan:
        o += 1

    for i in range(o + period - 1, len(x)):
        std = np.std(x[i - period + 1:i + 1])
        upper[i] = middle[i] + 2 * std
        lower[i] = middle[i] - 2 * std

    return lower, middle, upper

