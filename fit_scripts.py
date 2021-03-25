import sys
import functools as ft

import numpy as np
import scipy.linalg as scilin
import scipy.optimize as opt
import scipy.sparse as sps
from scipy.sparse import linalg as splinalg

from . import fitfuncs as ffunc


def sparse_cholesky(A):
    # Sparse Matrix Implementation of Cholesky Decomposition
    #
    # taken from:
    # https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
    #
    # matrix A must be a sparse symmetric positive-definite.
    #
    n = A.shape[0]
    LU = splinalg.splu(A, diag_pivot_thresh=0)  # sparse LU decomposition
    # check the matrix A is positive definite below.
    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
        return LU.L.dot(sps.diags(LU.U.diagonal()**0.5))
    else:
        sys.exit('The matrix is not positive definite')


# ----------------------- For Scipy.Optimize.Minimize -----------------------#
def chisquare(xo, xdata, ydata, covar, func, priors=None):
    # returns $\chi^2_{\nu}$
    dof = len(xdata) - len(xo)
    res = func(xdata, *xo) - ydata
    L = scilin.cholesky(covar, lower=True)
    xres = scilin.solve_triangular(L, res, lower=True)
    chisq = xres.dot(xres.conj())/dof
    if priors is None:
        return chisq
    else:
        popt, perr = priors
        prior = 0
        for j in range(len(popt)):
            prior += ((xo[j+1] - popt[j])**2)/(perr[j]**2)
        return chisq + 0.5*prior


def chiJac(jac, xo, xdata, ydata, covar, func, priors=None):
    dof = len(xdata) - len(xo)
    res = func(xdata, *xo) - ydata
    L = scilin.cholesky(covar, lower=True)
    # (J * Linv) * (Linv.H * res.conj)
    lhs = scilin.solve_triangular(L, jac(xdata, *xo), lower=True)
    rhs = scilin.solve_triangular(L, res)
    chiJ = lhs.conj().T.dot(rhs)/dof
    if priors is None:
        return 2*chiJ.real
    else:
        popt, perr = priors
        prior_jac = np.zeros(len(popt))
        for j in range(len(popt)):
            prior_jac[j] = (xo[j+1] - popt[j])/(perr[j]**2)
        return 2*chiJ.real + prior_jac


def chiHess(jac, hess, xo, xdata, ydata, covar, func, priors=None):
    dof = len(xdata) - len(xo)
    res = func(xdata, *xo) - ydata
    L = scilin.cholesky(covar, lower=True)
    # Part (a) with Hessian: (Hess * Linv).H * (Linv * res)
    lhs_a = scilin.solve_triangular(L, hess(xdata, *xo), lower=True)
    rhs_a = scilin.solve_triangular(L, res, lower=True)
    a_term = lhs_a.conj().dot(rhs_a)/dof
    # Part (b) Jacobians: (Jac * Linv).H * (Linv * Jac)
    xJac = scilin.solve_triangular(L, jac(xdata, *xo), lower=True)
    b_term = xJac.conj().dot(xJac)
    return 2*(a_term + b_term)


def minsolve(xdata, ydata, covar, func, xo,
             cJac=None, cHess=None, *args, **kwargs):
    # scipy.minimize implementation of solving my problems
    # cJac is the parameter jacobian of func
    # cHess is the parameter hessian of func
    # *args and **kwargs are inputs for scipy.optimize.minimize
    my_args = (xdata, ydata, covar, func)
    if cJac is None:
        res = opt.minimize(chisquare, xo, args=my_args, *args, **kwargs)
    else:
        wrap_jac = ft.partial(chiJac, cJac)
        if cHess is None:
            res = opt.minimize(chisquare, xo, args=my_args,
                               jac=wrap_jac, *args, **kwargs)
        else:
            wrap_hess = ft.partial(chiHess, cJac, cHess)
            res = opt.minimize(chisquare, xo, args=my_args,
                               jac=wrap_jac, hess=wrap_hess,
                               *args, **kwargs)
    if not res.success:
        rmes = "Status: {0};\nMessage: {1}".format(res.status, res.message)
        raise Exception(rmes)
    return res.x


# ------------------------- Effective Mass Plot -------------------------#
def eff_mass(xdata, ydata, miter=1000, verbose=False):
    dratio = ydata[1:]/ydata[:-1]
    meff, xvals = [], []

    def res(E, xdat, ydat):
        return ffunc.c2pt_smeson_ratio(E, xdat) - ydat

    for j, xdat in enumerate(xdata[1:]):
        try:
            mval = opt.brentq(
                res, 0, 1, args=(xdat, dratio[j]), maxiter=miter
            )
            meff.append(mval)
            xvals.append(xdat)
        except Exception as e:
            if verbose:
                print("Error: {0}".format(e))
    if len(meff) == 0:
        raise Exception("Empty Effective Mass Plot")
    meff, xvals = np.asarray(meff), np.asarray(xvals)
    return xvals, meff


# ------------------------- C3pt Fits ------------------------- #
def gen_lstsq_FF_sumFit(dts, dat, cov, chisquare=True):
    # Solves generalized least-squares problem
    # Summation Method for FormFactor = c3pt * FF_norm
    def linfit(x, a, b):
        return a*x + b

    X = ffunc.sumFit_X(dts)
    L = scilin.cholesky(cov, lower=True)
    Xl = scilin.solve_triangular(L, X, lower=True)
    yl = scilin.solve_triangular(L, dat, lower=True)
    val, res, rnk, d = scilin.lstsq(Xl, yl)
    if not chisquare:
        return val
    else:
        xres = dat - linfit(dts, *val)
        dof = 2*(len(dts) - 2)
        chisq = xres.conj().dot(scilin.inv(cov).dot(xres))
        return val, chisq.real/dof


def gen_lstsq_FF_nstateFit(Esnk, Esrc, dtaus, dat, cov, chisquare=True):
    # Solves generalized least-squares problem
    # For n-state fit for FormFactor = c3pt*FF_norm
    # breakpoint()
    X = ffunc.FF_nstateFit_X(Esnk, Esrc, dtaus)
    L = scilin.cholesky(cov, lower=True)
    Xl = scilin.solve_triangular(L, X, lower=True)
    yl = scilin.solve_triangular(L, dat, lower=True)
    val, res, rnk, d = scilin.lstsq(Xl, yl)
    if not chisquare:
        return val
    else:
        res = dat - X.dot(val)
        xres = scilin.solve_triangular(L, res, lower=True)
        dof = 2*(len(dtaus) - len(val))
        chisq = xres.conj().dot(xres)/dof
        # chisq = res.conj().dot(scilin.inv(cov).dot(res))/dof
        if dof < 0:
            print(xres)
            print('what?')
        return val, chisq.real


def gen_lstsq_c3pt_nstateFit(popt_snk, popt_src, dtaus,
                             dat, cov, chisquare=True):
    # Solves generalized least-squares problem
    # For n-state fit for c3pt
    X = ffunc.c3pt_nstateFit_X(popt_snk, popt_src, dtaus)
    L = scilin.cholesky(cov, lower=True)
    Xl = scilin.solve_triangular(L, X, lower=True)
    yl = scilin.solve_triangular(L, dat, lower=True)
    val, res, rnk, d = scilin.lstsq(Xl, yl)
    if not chisquare:
        return val
    else:
        res = dat - X.dot(val)
        xres = scilin.solve_triangular(L, res, lower=True)
        dof = 2*(len(dtaus) - len(val))
        chisq = xres.conj().dot(xres)/dof
        return val, chisq


def FF_sumFit(dts, tau_o, data, dblocks, chisquare=False):
    # data[wlines, dt, tau]
    wshape, dt_shape, tau_shape = data.shape
    nbins = dblocks.shape[0]
    coeff = (nbins-1.0)/nbins

    dsums = np.array([
        [data[k, j, tau_o:dt-(tau_o+1)].sum() for j, dt in enumerate(dts)]
        for k in range(wshape)
    ])
    dsums_blocks = (
        np.array([[dblk[k, j, tau_o:dt-(tau_o+1)].sum()
                   for j, dt in enumerate(dts)]
                  for dblk in dblocks]) for k in range(wshape)
    )  # shape (wlines, blocks, dts)  # (blocks, wlines, dts)
    dsums_cov = (
        coeff*np.sum(np.array([
            np.outer((dsms - dsums[k]).conj(), dsms-dsums[k])
            for dsms in dsums_blks]), axis=0)
        for k, dsums_blks in enumerate(dsums_blocks)
    )
    if chisquare:
        chisqs = np.zeros(wshape)
    vals = np.zeros((wshape, 2), dtype=data.dtype)
    for k, dcov in enumerate(dsums_cov):
        if chisquare:
            val, chisq = gen_lstsq_FF_sumFit(
                dts, dsums[k], dcov, chisquare=chisquare
            )
            vals[k] = val
            chisqs[k] = chisq
        else:
            val = gen_lstsq_FF_sumFit(dts, dsums[k], dcov,
                                      chisquare=chisquare)
            vals[k] = val
    if chisquare:
        return vals, chisqs
    else:
        return vals


def FF_forward_nstateFit(dts, tau_o, E, data, dblocks, chisquare=False):
    wshape, dt_shape, tau_shape = data.shape
    nbins = dblocks.shape[0]
    coeff = (nbins-1.0)/nbins

    dtaus = ffunc.flatten_taus(dts, tau_o)
    dflats = np.array([ffunc.flatten_c3pt(data[k], dts, tau_o)
                       for k in range(wshape)])
    dflats_blocks = (
        [ffunc.flatten_c3pt(dblk[k], dts, tau_o) for dblk in dblocks]
        for k in range(wshape)
    )
    dflats_cov = (
        coeff*np.sum(np.array([
            np.outer((dflt - dflats[k]).conj(), dflt - dflats[k])
            for dflt in dflats_blks]), axis=0)
        for k, dflats_blks in enumerate(dflats_blocks)
    )
    if chisquare:
        chisqs = np.zeros(wshape)
    vals = None
    for k, dcov in enumerate(dflats_cov):
        if chisquare:
            print('Incomplete')
        break
    raise Exception('Incomplete Function')


def FF_nstateFit(dts, tau_o, Esnk, Esrc, data, dblocks, chisquare=False):
    wshape, dt_shape, tau_shape = data.shape
    nbins = dblocks.shape[0]
    coeff = (nbins-1.0)/nbins

    dtaus = ffunc.flatten_taus(dts, tau_o)
    dflats = np.array([ffunc.flatten_c3pt(data[k], dts, tau_o)
                       for k in range(wshape)])
    dflats_blocks = (
        [ffunc.flatten_c3pt(dblk[k], dts, tau_o) for dblk in dblocks]
        for k in range(wshape)
    )
    dflats_cov = (
        coeff*np.sum(np.array([
            np.outer((dflt - dflats[k]).conj(), dflt - dflats[k])
            for dflt in dflats_blks]), axis=0)
        for k, dflats_blks in enumerate(dflats_blocks)
    )
    if chisquare:
        chisqs = np.zeros(wshape)
    vals = None
    for k, dcov in enumerate(dflats_cov):
        if chisquare:
            val, chisq = gen_lstsq_FF_nstateFit(
                Esnk, Esrc, dtaus, dflats[k], dcov, chisquare=chisquare
            )
            if vals is None:
                vals = np.array([val])
            else:
                vals = np.vstack((vals, np.array([val])))
            chisqs[k] = chisq
        else:
            val = gen_lstsq_FF_nstateFit(
                Esnk, Esrc, dtaus, dflats[k], dcov, chisquare=chisquare
            )
            if vals is None:
                vals = np.array([val])
            else:
                vals = np.vstack((vals, np.array([val])))
    if chisquare:
        return vals, chisqs
    else:
        return vals


def c3pt_nstateFit(dts, tau_o, popt_snk, popt_src,
                   data, dblocks, chisquare=False):
    wshape, dt_shape, tau_shape = data.shape
    nbins = dblocks.shape[0]
    coeff = (nbins-1.0)/nbins

    dtaus = ffunc.flatten_taus(dts, tau_o)
    dflats = np.array([ffunc.flatten_c3pt(data[k], dts, tau_o)
                       for k in range(wshape)])
    dflats_blocks = (
        [ffunc.flatten_c3pt(dblk[k], dts, tau_o) for dblk in dblocks]
        for k in range(wshape)
    )
    dflats_cov = (
        coeff*np.sum(np.array([
            np.outer((dflt - dflats[k]).conj(), dflt - dflats[k])
            for dflt in dflats_blks]), axis=0)
        for k, dflats_blks in enumerate(dflats_blocks)
    )
    if chisquare:
        chisqs = np.zeros(wshape)
    vals = None
    for k, dcov in enumerate(dflats_cov):
        if chisquare:
            val, chisq = gen_lstsq_c3pt_nstateFit(
                popt_snk, popt_src, dtaus, dflats[k], dcov, chisquare=chisquare
            )
            if vals is None:
                vals = np.array([val])
            else:
                vals = np.vstack((vals, np.array([val])))
            chisqs[k] = chisq
        else:
            val = gen_lstsq_c3pt_nstateFit(
                popt_snk, popt_src, dtaus, dflats[k], dcov, chisquare=chisquare
            )
            if vals is None:
                vals = np.array([val])
            else:
                vals = np.vstack((vals, np.array([val])))
    if chisquare:
        return vals, chisqs
    else:
        return vals
