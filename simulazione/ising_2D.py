import math
import random

import numpy as np
from numba import njit


@njit(fastmath=True)
def set_seed(seed):
    random.seed(seed)


@njit(fastmath=True, inline="always")
def initialize_lattice(L, ordered=False):
    S = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            if ordered:
                S[i, j] = 1
            else:
                S[i, j] = 1 if random.random() < 0.5 else -1
    return S


@njit(fastmath=True, inline="always")
def magnetization(S):
    L = S.shape[0]
    total = 0
    for i in range(L):
        for j in range(L):
            total += S[i, j]
    return total


@njit(fastmath=True, inline="always")
def absolute_magnetization(S):
    m = magnetization(S)
    return -m if m < 0 else m


@njit(fastmath=True, inline="always")
def energy(S, bc="periodic"):
    L = S.shape[0]
    E = 0
    for i in range(L):
        for j in range(L):
            if i < L - 1 or bc == "periodic":
                E -= S[i, j] * S[(i + 1) % L, j]
            if j < L - 1 or bc == "periodic":
                E -= S[i, j] * S[i, (j + 1) % L]
    return E


@njit(fastmath=True, inline="always")
def autocorrelation(E, t_max):
    n_mis = E.shape[0]
    C_c = np.zeros(t_max, dtype=np.float64)

    sum_e = 0.0
    for i in range(n_mis):
        sum_e += E[i]
    E_mean = sum_e / n_mis

    for tau in range(t_max):
        limit = n_mis - tau
        sum_c = 0.0
        for t in range(limit):
            sum_c += (E[t] - E_mean) * (E[t + tau] - E_mean)

        C_c[tau] = sum_c / limit

    return C_c / C_c[0]


@njit(fastmath=True, inline="always")
def metropolis_step(S, T, bc="periodic", flip_all=False, sequential=False):
    L = S.shape[0]
    dM = 0
    dE = 0

    is_periodic = bc == "periodic"
    n_flips = L * L if flip_all else 1
    use_sequential = sequential and flip_all

    if flip_all:
        w2 = math.exp(-2.0 / T)
        w4 = w2 * w2
        w6 = w4 * w2
        w8 = w4 * w4
    else:
        w2 = w4 = w6 = w8 = 0.0

    i = 0
    j = 0
    for _ in range(n_flips):
        if not use_sequential:
            i = random.randrange(L)
            j = random.randrange(L)

        s = S[i, j]

        if is_periodic:
            up = S[i - 1, j] if i > 0 else S[L - 1, j]
            down = S[i + 1, j] if i + 1 < L else S[0, j]
            left = S[i, j - 1] if j > 0 else S[i, L - 1]
            right = S[i, j + 1] if j + 1 < L else S[i, 0]
            nn = up + down + left + right
        else:
            nn = 0
            if i > 0:
                nn += S[i - 1, j]
            if i + 1 < L:
                nn += S[i + 1, j]
            if j > 0:
                nn += S[i, j - 1]
            if j + 1 < L:
                nn += S[i, j + 1]

        d = 2 * s * nn
        if d <= 0:
            accept = True
        else:
            if flip_all:
                if d == 4:
                    w = w4
                elif d == 8:
                    w = w8
                elif d == 2:
                    w = w2
                else:
                    w = w6
            else:
                w = math.exp(-d / T)
            accept = random.random() < w

        if accept:
            S[i, j] = -s
            dM -= 2 * s
            dE += d

        if use_sequential:
            j += 1
            if j == L:
                j = 0
                i += 1

    return dM, dE


@njit(fastmath=True, inline="always")
def evolve(S, T, nmcs, nequil, bc="periodic", flip_all=False, sequential=True):
    n = int(nmcs - nequil)
    M = np.empty(n, dtype=np.float64)
    E = np.empty(n, dtype=np.float64)

    M_cur = magnetization(S)
    E_cur = energy(S, bc)
    sample = 0

    sum_m = 0.0
    sum_m_abs = 0.0
    sum_e = 0.0

    for step in range(nmcs):
        dM, dE = metropolis_step(S, T, bc, flip_all, sequential)
        M_cur += dM
        E_cur += dE
        if step >= nequil:
            m_val = float(M_cur)
            e_val = float(E_cur)
            M[sample] = m_val
            E[sample] = e_val
            sum_m += m_val
            sum_m_abs += -m_val if m_val < 0.0 else m_val
            sum_e += e_val
            sample += 1

    L = S.shape[0]
    area = float(L * L)

    mean_m = sum_m / n
    mean_m_abs = sum_m_abs / n
    mean_e = sum_e / n

    var_e = 0.0
    sum_m2 = 0.0
    for k in range(n):
        de = E[k] - mean_e
        var_e += de * de
        mk = M[k]
        sum_m2 += mk * mk
    var_e /= n
    mean_m2 = sum_m2 / n

    m_avg = mean_m / area
    m_abs_avg = mean_m_abs / area
    e_avg = mean_e / area
    c = var_e / (area * T * T)
    chi = (mean_m2 - mean_m_abs * mean_m_abs) / (area * T)

    return (
        M,
        E,
        m_avg,
        m_abs_avg,
        e_avg,
        c,
        chi,
    )


@njit(fastmath=True, inline="always")
def mean_err(a):
    n_s, n_t = a.shape
    mean = np.empty(n_t, dtype=np.float64)
    err = np.empty(n_t, dtype=np.float64)
    denom = float(n_s * (n_s - 1))

    for j in range(n_t):
        s = 0.0
        for r in range(n_s):
            s += a[r, j]
        m = s / n_s
        v = 0.0
        for r in range(n_s):
            diff = a[r, j] - m
            v += diff * diff
        mean[j] = m
        err[j] = math.sqrt(v / denom)

    return mean, err
