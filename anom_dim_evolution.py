import sys
import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
# import scipy.special as spec
from scipy.special import zeta

N = 3
Ca, Cf, Tf = N, (N**2 - 1)/(2*N), 0.5
dada = (N**2)*(N**2 + 36)/24
dfda = N*(N**2 + 6)/48
dfdf = (N**4 - 6*N**2 + 18)/(96*N**2)
nf = 3


def Hn(n, k=1):
    sign, K = np.sign(k), abs(k)
    vals = np.array([(sign/j)**K for j in range(1, n+1)])
    return vals.sum()


def Hnm(n, klist):
    if len(klist) == 1:
        return Hn(n, klist[0])
    else:
        sign = sign_list[0]
        K = abs(klist[0])
        vals = np.array([
            ((sign/j)**K)*Hnm(j, klist[1:])
            for j in range(1, n+1)
        ])
        return vals.sum()


def NpHn(i, n, k=1):
    return Hn(n+i, k)


def NpHnm(i, n, klist):
    return Hnm(n+i, klist)


def gamma_0l(n):
    val = NpHn(-1, n) + NpHn(1, n) - 1.5
    return 2*Cf*val


def gamma_1l_plus(n):
    a1 = 2*NpHn(1, n, 3) - 17/24 - 2*Hn(n, 3, -1) - (28/3)*Hn(n)
    a2 = (151/18)*NpHn(-1, n) + 2*NpHnm(-1, n, [1, -2]) - (11/6)*NpHn(-1, n, 2)
    a3 = (151/18)*NpHn(1, n) + 2*NpHnm(1, n, [1, -2]) - (11/6)*NpHn(1, n, 2)
    a = 4*Ca*Cf*(a1 + a2 + a3)

    b1 = (1/12)+(4/3)*Hn(n)
    b2 = -((11/9)*NpHn(-1,n)-(1/3)*NpHn(-1,n,2)+(11/9)*NpHn(1,n)-(1/3)*NpHn(1,n,2))
    b = 4*Cf*nf*(b1 + b2)

    c1 = 4*Hn(n, -3) + 2*Hn(n) + 2*Hn(n, 2) - 3/8
    c2 = NpHn(-1, n, 2) + 2*NpHn(-1, n, 3)
    c3 = NpHn(-1, n)+4*NpHnm(-1, n, [1, -2])+2*NpHnm(-1, n, [1, 2])+NpHn(-1, n, 3)
    c4 = NpHn(1, n)+4*NpHnm(1, n, [1, -2])+2*NpHnm(1, n, [1, 2])+NpHn(1, n, 3)
    c = 4*(Cf**2)*(c1 + c2 - (c3 + c4))

    return a + b + c


def gamma_1l(n):
    a = (NpHn(-1, n, 2) - NpHn(-1, n, 3)) - (NpHn(1, n, 2) - NpHn(1, n, 3))
    b = 2*(NpHn(-1, n) + NpHn(1, n) - 2*Hn(n))
    c = 16*Cf*(Cf - 0.5*Ca)*(a - b)
    return gamma_1l_plus(n) + c


def gamma_2l_plus(n):
    a1 = 1.5*zeta(3) - 1.25 + (10/9)*Hn(n, 3) + (4/3)*Hnm(n, [1, -2]) - (2/3)*Hn(n, -4) + 2*Hnm(n, [1, 1]) - (25/9)*Hn(n, 2)
    a2 = (257/27)*Hn(n) - (2/3)*Hnm(n, [-3, 1])
    a3 = 
