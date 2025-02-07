import sys
import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import scipy.special as spec


# This will set up everything necessary for alpha_s evolution to specified order
# order up to alpha_s^6
# number of flavors will be automatically set

# arxiv:1701.01404


N = 3
Ca, Cf, Tf = N, (N**2 - 1)/(2*N), 0.5
dada = (N**2)*(N**2 + 36)/24
dfda = N*(N**2 + 6)/48
dfdf = (N**4 - 6*N**2 + 18)/(96*N**2)


def b0(nf):
    a1, a2 = (11./3)*Ca, -(4./3)*Tf*nf
    return a1 + a2


def b1(nf):
    a1, a2, a3 = (34./3)*Ca**2, -(20./3)*Ca*Tf*nf, -4*Cf*Tf*nf
    return a1 + a2 + a3


def b2(nf):
    a1, a2 = (2857./54)*Ca**3, -(1415./27)*(Ca**2)*Tf*nf
    a3, a4 = -(205./9)*Cf*Ca*Tf*nf, 2*(Cf**2)*Tf*nf
    a5, a6 = (44./9)*Cf*(Tf**2)*nf**2, (158./27)*Ca*(Tf**2)*nf**2
    return a1 + a2 + a3 + a4 + a5 + a6


def b3(nf):
    a1 = (Ca**4)*(150653./486 - (44./9)*spec.zeta(3))
    a2 = dada*(-80./9 + (704./3)*spec.zeta(3))
    a3 = (Ca**3)*Tf*nf*(-39143./81 + (136./3)*spec.zeta(3))
    a4 = (Ca**2)*Cf*Tf*nf*(7073./243 - (653./9)*spec.zeta(3))
    a5 = Ca*(Cf**2)*Tf*nf*(-4204./27 + (352./9)*spec.zeta(3))
    a6 = dfda*nf*(512./9 - (1664./3)*spec.zeta(3))
    a7 = 46*(Cf**3)*Tf*nf
    a8 = ((Ca*Tf*nf)**2)*(7930./81 + (224./9)*spec.zeta(3))
    a9 = ((Ca*Tf*nf)**2)*(1352./27 - (704./9)*spec.zeta(3))
    a10 = Ca*Cf*((Tf*nf)**2)*(17152./243 + (448./9)*spec.zeta(3))
    a11 = dfdf*(nf**2)*(-704./9 + (512./3)*spec.zeta(3))
    a12, a13 = (424./243)*Ca*(Tf*nf)**3, (1232./243)*Cf*(Tf*nf)**3
    return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13


def b4(nf):
    a1 = (8157455/16) + (621885/2)*spec.zeta(3) - (88209/2)*spec.zeta(4) - 288090*spec.zeta(5)
    a2 = nf*(-(336460813/1944) - (4811164/81)*spec.zeta(3) + (33935/6)*spec.zeta(4) + (1358995/27)*spec.zeta(5))
    a3 = (nf**2)*((25960913/1944) + (698531/81)*spec.zeta(3) - (10526/9)*spec.zeta(4) - (381760/81)*spec.zeta(5))
    a4 = (nf**3)*(-(630559/5832) - (48722/243)*spec.zeta(3) + (1618/27)*spec.zeta(4) + (460/9)*spec.zeta(5))
    a5 = (nf**4)*((1205/2916) - (152/81)*spec.zeta(3))
    return a1 + a2 + a3 + a4 + a5


def betaFunc(a, mu, n):
    if (0.1 < mu and mu < 1.27):
        nf = 3
    elif (1.27 <= mu and mu < 4.18):
        nf = 4
    else:
        nf = 5
    fpi = 4*np.pi
    term_1 = b0(nf)*(a/fpi)**2
    term_2 = b1(nf)*(a/fpi)**3
    term_3 = b2(nf)*(a/fpi)**4
    term_4 = b3(nf)*(a/fpi)**5
    term_5 = b4(nf)*(a/fpi)**6
    terms = np.array([term_1, term_2, term_3, term_4, term_5])
    dadmu = -2*fpi*terms[:n+1].sum()/mu
    return dadmu


def get_new_alpha(alpha, mu, new_mu, order=0, niter=1000):
    mu_array = np.linspace(mu, new_mu, niter)
    new_alphas = scint.odeint(betaFunc, alpha, mu_array, args=(order,))
    return new_alphas[-1]


# a0 = 0.1179 # /(4*np.pi)
# a0_err = 0.0010
# Mz = 91.1876
# Mz_err = 0.0021
# # mu = np.linspace(Mz, 2.0, 1000)
# mu = np.linspace(Mz, 1.0, 1000)
# mu_up = np.linspace(Mz, 5000, 1000)
# mu_tot = np.append(np.flip(mu_up), mu)
# # mu = np.linspace(Mz, 0.75, 1000)
# # mu = np.linspace(3.2, 91.1876, 100)
# a_nl1 = scint.odeint(betaFunc, a0, mu, args=(0,))
# a_nl1_up = scint.odeint(betaFunc, a0, mu_up, args=(0,))
# a_nl1 = np.append(np.flip(a_nl1_up), a_nl1)
# print(10*"*")
# a_nl2 = scint.odeint(betaFunc, a0, mu, args=(1,))
# a_nl2_up = scint.odeint(betaFunc, a0, mu_up, args=(0,))
# a_nl2 = np.append(np.flip(a_nl1_up), a_nl2)
# print(10*"*")
# a_nl3 = scint.odeint(betaFunc, a0, mu, args=(2,))
# a_nl3_up = scint.odeint(betaFunc, a0, mu_up, args=(0,))
# a_nl3 = np.append(np.flip(a_nl1_up), a_nl3)
# print(10*"*")
# a_nl4 = scint.odeint(betaFunc, a0, mu, args=(3,))
# a_nl4_up = scint.odeint(betaFunc, a0, mu_up, args=(0,))
# a_nl4 = np.append(np.flip(a_nl1_up), a_nl4)
# print(10*"*")
# a_nl5 = scint.odeint(betaFunc, a0, mu, args=(4,))
# a_nl5_up = scint.odeint(betaFunc, a0, mu_up, args=(0,))
# a_nl5 = np.append(np.flip(a_nl1_up), a_nl5)
# print(10*"*")
# # plt.plot(mu_tot, a_nl1, label="# loops: 1")
# # plt.plot(mu_tot, a_nl2, label="# loops: 2")
# # plt.plot(mu_tot, a_nl3, label="# loops: 3")
# # plt.plot(mu_tot, a_nl4, label="# loops: 4")
# plt.plot(mu_tot, a_nl5,
#          label=r'$\beta^{\overline{MS}}(\alpha_s) \sim \mathcal{O}(\alpha_s^6)$')
# plt.xlabel(r'$Q$ GeV', fontsize=12)
# plt.title(r'$\alpha_s^{\overline{MS}}(Q)$', fontsize=15)
# plt.errorbar([3.2], [0.2464534483542745], yerr=[0.004708273178506332],
#              fmt='o', capsize=5, color='m', label=r'$Q$=3.2 GeV')
# plt.errorbar([2.0], [0.3002707248978929], yerr=[0.00719450987351185],
#              fmt='o', capsize=5, color='b', label=r'$Q$=2.0 GeV')
# plt.errorbar([Mz], [a0], xerr=[Mz_err], yerr=[a0_err],
#              fmt='s', capsize=5, color='k', label=r'$Q$=M$_Z$')
# plt.legend()
# plt.xscale('log')
# plt.xlim(1.0, 2000)
# plt.ylim(0.05, 0.35)
# plt.grid(True)
# plt.show()
