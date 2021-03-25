# import itertools as it
import functools as ft
import types
import numpy as np
import scipy.linalg as scilin
import scipy.optimize as opt
import scipy.stats as sts

from . import fitfuncs as ffunc
from . import fit_scripts as fitter


def sort_c2pt(popt):
    cool = popt.reshape((int(popt.shape[0]/2), 2))
    how = np.asarray(sorted(cool, key=lambda x: x[1]))
    return how.flatten()


def get_mean_cov(data):
    mean = data.mean(0)
    cov = np.array([np.outer(d-mean, (d-mean).conj()) for d in data])
    cov = cov.mean(0)
    return mean, cov


def get_mean_var(data):
    mean, cov = get_mean_cov(data)
    return mean, cov.diagonal()


def jk_blocks(data, nbins):
    Ncfg, cut = data.shape[0], data.shape[0]/nbins
    inds = np.arange(0, cut)
    while inds[-1] < Ncfg:
        blocks = np.delete(data, inds, axis=0)
        inds += cut
        yield blocks


sdef jk_blocks_arr(data, nbins):
    blocks_gen = jk_blocks(data, nbins)
    blocks = np.array([block for block in blocks_gen])
    return blocks


def calc_jk_mean_cov(data, nbin):
    mean, coeff, blocks = data.mean(0), (nbin-1.0)/nbin, jk_blocks(data, nbin)
    dcov_blocks = np.array([
        np.outer(db.mean(0)-mean, (db.mean(0)-mean).conj()) for db in blocks
    ])
    jk_cov = coeff*np.sum(dcov_blocks, 0)
    return mean, jk_cov


def calc_jk_mean_var(data, nbin):
    mean, cov = calc_jk_mean_cov(data, nbin)
    return mean, cov.diagonal()


# ------------------------- C2pt Fit -------------------------#
def match_lists(my_lists, match=True):
    """ returns domain and range of list """
    func_table = {True: np.intersect1d, False: np.union1d}
    domain_list = [my_list[0] for my_list in my_lists]
    domain = ft.reduce(func_table[match], domain_list)
    im_shape, im_dtype = (len(my_lists), len(domain)), my_lists[0][1].dtype
    image = np.empty(im_shape, dtype=im_dtype)
    for j, my_list in enumerate(my_lists):
        for k, dm in enumerate(domain):
            for tval, ival in zip(my_list[0], my_list[1]):
                if dm == tval:
                    image[j, k] = ival
                    break
    return domain, image


def eff_mass_jk_mean_var(xdata, data, nbin, miter=1000, verbose=False):
    mean, coeff, blocks = data.mean(0), (nbin-1)/nbin, jk_blocks(data, nbin)
    xvals, meff = fitter.eff_mass(xdata, mean, miter=miter, verbose=verbose)
    val_list = [(xvals, meff)]
    for dblk in blocks:
        val_list.append(
            fitter.eff_mass(xdata, dblk.mean(0), miter=miter, verbose=False)
        )
    trange, image_list = match_lists(val_list)
    cv = image_list[0]
    err = np.sqrt(coeff*np.sum((image_list[1:]-cv)**2, 0))
    return trange, np.array([cv, err])


def check_fit_params(pars, smear_param):
    if smear_param == 'SS' and any(pars < 0):
        raise Exception('SS: Negative amplitude and(or) energy')
    elif smear_param == 'SP' and any(pars[1::2] < 0):
        raise Exception('SP: Negative energy')
    else:
        pass
    N = len(pars)
    pars = np.asarray(sorted(pars.reshape(N//2, 2), key=lambda b: b[1]))
    return pars.reshape(N)


def cv_fit(f, xdat, dat, xo, nbins, smear_param, *args, **kwargs):
    mu, cov = calc_jk_mean_cov(dat, nbins)
    pars, pcov = opt.curve_fit(f, xdat, mu, p0=xo, sigma=cov, *args, **kwargs)
    pars = check_fit_params(pars, smear_param)


def ret_chisq(xdat, dat, nbins, f, popt):
    mu, cov = calc_jk_mean_cov(dat, nbins)
    res, dof = f(xdat, *popt) - mu, len(xdat) - len(popt)
    L = scilin.cholesky(cov, lower=True)
    xres = scilin.solve_triangular(L, res, lower=True)
    chisq = xres.dot(xres.conj())/dof
    pval = 1 - sts.chi2.cdf(chisq*dof, dof)
    return chisq, pval


def cv_fit_jk_mean_var(xdata, data, nbins, f, po,
                       smear_param=None, return_chisq=True,
                       return_blocks=False, *args, **kwargs):
    coeff, blks = (nbins-1)/nbins, jk_blocks(data, nbins)
    popt = cv_fit(f, xdata, data, xo=po, nbins, smear_param, *args, **kwargs)
    popt_blocks = np.array([
        cv_fit(f, xdata, dblk, xo=popt, nbins, smear_param, *args, **kwargs)
        for dblk in blks
    ])
    popt_err = np.sqrt(coeff*np.sum((popt_blocks - popt)**2, 0))
    if return_chisq:
        chisq, pval = ret_chisq(xdata, data, nbins, f, popt)
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks, chisq, pval
        else:
            return np.array([popt, popt_err]), chisq, pval
    else:
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks
        else:
            return np.array([popt, popt_err])
