import itertools as it
# import functools as ft
import numpy as np

Lt = 64
lt = 32


# ------------------------- Fix Function Parameters -------------------------#
def fix_func(f, fix_pars):
    # takes in function f(x, *pars)
    # fix_pars = ((i_1, A_1), (i_2, A_2), ...)
    # this function returns a wrapped function with
    # the i_n'th parameter fixed to A_n
    def new_func(x, *pars):
        new_pars = [None]*(len(pars) + len(fix_pars))
        for j, fp in fix_pars:
            new_pars[j] = fp
        for par in pars:
            for j, npar in enumerate(new_pars):
                if npar is None:
                    new_pars[j] = par
                    break
        return f(x, *new_pars)
    return new_func


def fix_jac(jac, fix_pars):
    # takes in function f(x, *pars)
    # fix_pars = ((i_1, A_1), (i_2, A_2), ...)
    # this function returns a wrapped function with
    # the i_n'th parameter fixed to A_n
    def new_jac(x, *pars):
        new_pars = [None]*(len(pars) + len(fix_pars))
        for j, fp in fix_pars:
            new_pars[j] = fp
        for par in pars:
            for j, npar in enumerate(new_pars):
                if npar is None:
                    new_pars[j] = par
                    break
                # breakpoint()
        del_index_list = [fix[0] for fix in fix_pars]
        return np.delete(jac(x, *new_pars), del_index_list, axis=1)
    return new_jac


def fix_hess(h, fix_pars):
    # takes in function f(x, *pars)
    # fix_pars = ((i_1, A_1), (i_2, A_2), ...)
    # this function returns a wrapped function with
    # the i_n'th parameter fixed to A_n
    def new_hess(x, *pars):
        new_pars = [None]*(len(pars) + len(fix_pars))
        for j, fp in fix_pars:
            new_pars[j] = fp
        for par in pars:
            for j, npar in enumerate(new_pars):
                if npar is None:
                    new_pars[j] = par
                    break
        del_index_list = [fix[0] for fix in fix_pars]
        return np.delete(
            np.delete(h(x, *new_pars), del_index_list, axis=2),
            del_index_list, axis=1)
    return new_hess


# ------------------------- Set Up for Priors -------------------------#
def prior_vals(vals, prior_list):
    what = np.zeros(len(prior_list), dtype=vals.dtype)
    return np.append(vals, what)


def prior_func(func, prior_list):
    # prior_list = ((i_1, A_1), ..., (i_n, A_n))
    def new_func(x, *args):
        prior_diff_array = np.array([
            prior - args[i] for i, prior in prior_list
        ])
        return np.append(func(x, *args), prior_diff_array)
    return new_func


def prior_jac(jac, prior_list):
    def new_jac(x, *args):
        add_to_jac = np.ones((len(prior_list), len(prior_list)))
        return np.vstack((jac(x, *args), add_to_jac))
    return new_jac


def prior_covar(covar, prior_error, eta=1):
    add_size = len(prior_error)
    N = covar.shape[0]
    new_covar = np.zeros((N+add_size, N+add_size), dtype=covar.dtype)
    new_covar[:N, :N] = covar
    for j, prerr in prior_error:
        new_covar[N+j, N+j] = eta/(prerr[j]**2)
    return new_covar


# ------------------------- Two-Point Function -------------------------#
def c2pt_smeson_ratio(E, t):
    # For computing effective mass plots
    return np.cosh(E*(0.5*Lt-t))/np.cosh(E*(0.5*Lt-t+1))


def c2pt_meson_func_half(t, *p):
    val = 0
    for j in range(0, len(p), 2):
        val += p[j]*np.exp(-p[j+1]*t)
    return val


def c2pt_meson_func(t, *p):
    val = 0
    for j in range(0, len(p), 2):
        val += p[j]*(np.exp(-p[j+1]*t) + np.exp(-p[j+1]*(Lt - t)))
    return val


def c2pt_meson_jac(t, *p):
    tau, jac = Lt - t, np.zeros((len(p), len(t)))
    for j in range(0, len(p), 2):
        jac[j] = np.exp(-p[j+1]*t) + np.exp(-p[j+1]*tau)
        jac[j+1] = -p[j]*(tau*np.exp(-p[j+1]*tau) + t*np.exp(-p[j+1]*t))
    return jac.T


def c2pt_meson_hess(t, *p):
    tau, hess = Lt - t, np.zeros((len(p), len(p), len(t)))
    for j in range(0, len(p), 2):
        hess[j+1, j] = -(tau*np.exp(-p[j+1]*(Lt-t)) + t*np.exp(-p[j+1]*t))
        hess[j, j+1] = -(tau*np.exp(-p[j+1]*(Lt-t)) + t*np.exp(-p[j+1]*t))
        hess[j+1, j+1] = p[j]*(
            (tau**2)*np.exp(-p[j+1]*tau) + (t**2)*np.exp(-p[j+1]*t)
        )
    return np.swapaxes(hess, 0, -1)


def remove_wrap(data, popts):
    T = data.shape[-1]
    tL = np.arange(T)
    wrap_around = c2pt_meson_func_half(T-tL, *popts)
    return data - wrap_around


# ------------------------- Three-Point Function -------------------------#
def sumFit_X(dts):
    # Design Matrix for Summation Method in Three-Point Function
    return np.vstack([dts, np.ones(len(dts))]).T


def flatten_taus(dts, tau_o):
    # Returns np.array([[dt_j, tau_k]])
    # for j in dts and k in (tau_o, dt-(tau_o+1))
    time_slices = []
    for dt in dts:
        time_slices += [[dt, dtau] for dtau in range(tau_o, dt-(tau_o+1))]
    return np.asarray(time_slices)


def flatten_c3pt(data, dts, tau_o):
    # data[dts, tau]
    # Flattens c3pt to a 1-D array matching flatten_taus
    ndat = []
    for j, dt in enumerate(dts):
        ndat += [data[j, k] for k in range(tau_o, dt-(tau_o+1))]
    return np.asarray(ndat)


def FF_nstateFit_X(Esnk, Esrc, t):
    # t = [(delt, tau), ...]
    # Presumably I am dealing with c3pt*FF_norm
    nterms = len(Esnk)
    dEsnk = Esnk - Esnk[0]
    dEsrc = Esrc - Esrc[0]

    def factor(n, m, delt, tau):
        if n == m:
            return np.exp(-dEsnk[n]*(delt-tau))*np.exp(-dEsrc[m]*tau)
        else:
            val_a = np.exp(-dEsnk[n]*(delt-tau))*np.exp(-dEsrc[m]*tau)
            val_b = np.exp(-dEsrc[n]*(delt-tau))*np.exp(-dEsnk[m]*tau)
            return val_a + val_b

    X = np.array([
        [factor(n, m, delt, tau)
         for n, m in it.product(range(nterms), repeat=2) if n >= m]
        for delt, tau in t
    ], dtype=np.complex128)
    return X


def c3pt_nstateFit_X(popt_snk, popt_src, t):
    # t = [(delt, tau), ...]
    # Lt = 64
    nvals = popt_snk.shape[0]
    popt_snk = popt_snk.reshape(int(nvals/2), 2)
    popt_src = popt_src.reshape(int(nvals/2), 2)

    def factor(n, m, delt, tau):
        Asnk, Esnk = popt_snk[n]
        Asrc, Esrc = popt_src[m]
        amps = np.sqrt(Asnk*Asrc)
        exps = np.exp(-Esnk*(delt-tau))*np.exp(-Esrc*tau)
        return amps*exps

    X = np.array([
        [factor(n, m, delt, tau)
         for n, m in it.product(range(nvals), repeat=2)]
        for delt, tau in t
    ], dtype=np.complex128)
    return X
