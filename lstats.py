# import itertools as it
import functools as ft
import types
import numpy as np
import scipy.linalg as scilin
import scipy.optimize as opt
import scipy.stats as sts

from . import fitfuncs as ffunc
from . import fit_scripts as fitter
# from . import excited_states as exc
# import fit_scripts as fitter


def sort_c2pt(popt):
    cool = popt.reshape((int(popt.shape[0]/2), 2))
    how = np.asarray(sorted(cool, key=lambda x: x[1]))
    return how.flatten()


def get_mean_cov(data):
    Ncfg, Nt = data.shape
    mean = data.mean(0)
    cov = np.array([
        np.outer(data[j]-mean, (data[j]-mean).conj()) for j in range(Ncfg)
    ])
    cov = cov.mean(0)
    return mean, cov


def get_mean_var(data):
    mean, cov = get_mean_cov(data)
    return mean, cov.diagonal()


def old_jk_blocks(data, nbins):
    if isinstance(data, types.GeneratorType):
        data = np.array(list(data))
    Ncfg = data.shape[0]
    inds = np.arange(0, Ncfg, nbins) - 1
    # inds = np.arange(0, nbins-1)
    while inds[-1] < Ncfg:
        inds += 1  # nbins  # 1
        yield np.delete(data, inds, axis=0)


def jk_blocks(data, nbins):
    Ncfg = data.shape[0]
    cut = int(Ncfg/nbins)
    inds = np.arange(0, cut, dtype=int)
    while inds[-1] < Ncfg:  # inds[-1] + 1 < Ncfg:
        blocks = np.delete(data, inds, axis=0)
        inds += cut
        yield blocks  # np.delete(data, inds, axis=0)


def jk_blocks_arr(data, nbins):
    Ncfg = data.shape[0]
    cut = int(Ncfg/nbins)
    inds = np.arange(0, cut, dtype=int)
    blocks = None
    while inds[-1] < Ncfg:
        # inds += cut
        block = np.delete(data, inds, axis=0)
        if blocks is None:
            blocks = np.array([block])
        else:
            blocks = np.vstack((blocks, np.array([block])))
        inds += cut
    return blocks


def calc_jk_mean_cov(data, nbin):
    Ncfg, Nt = data.shape
    mean, coeff, blocks = data.mean(0), (nbin-1.0)/nbin, jk_blocks(data, nbin)
    dcov_blocks = np.array([
        np.outer(dblk.mean(0)-mean, (dblk.mean(0)-mean).conj())
        for dblk in blocks
    ])
    jk_cov = coeff*np.sum(dcov_blocks, 0)
    return mean, jk_cov


def calc_jk_mean_var(data, nbins):
    mean, cov = calc_jk_mean_cov(data, nbins)
    return mean, cov.diagonal()


# ------------------------- C2pt Fit -------------------------#
def match_lists(my_lists, match=True):
    """ returns domain and range of list """
    func_table = {True: np.intersect1d, False: np.union1d}
    domain_list = [my_list[0] for my_list in my_lists]
    domain = ft.reduce(func_table[match], domain_list)
    image = np.empty(shape=(len(my_lists), len(domain)),
                     dtype=my_lists[0][1].dtype)
    for j, my_list in enumerate(my_lists):
        for k, dm in enumerate(domain):
            for tval, ival in zip(my_list[0], my_list[1]):
                if dm == tval:
                    image[j, k] = ival
                    break
    return domain, image


def eff_mass_jk_mean_var(xdata, data, nbins, miter=1000, verbose=False):
    coeff = (nbins-1)/nbins
    dmean = data.mean(0)
    dblocks = jk_blocks(data, nbins)

    xvals, meff = fitter.eff_mass(xdata, dmean, miter=miter, verbose=verbose)
    val_list = [(xvals, meff)]
    for dblk in dblocks:
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
    pars = np.asarray(
        sorted(pars.reshape(N//2, 2), key=lambda b: b[1])
    ).reshape(N)
    return pars


def cv_fit_jk_mean_var(xdata, data, nbins, f, po,
                       smear_param=None, return_chisq=True,
                       return_blocks=False, *args, **kwargs):
    def cFilter(func, xdata, data, xo, *args, **kwargs):
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        a, b = opt.curve_fit(
            func, xdata, dmean, p0=xo, sigma=dcov, *args, **kwargs
        )
        a = check_fit_params(a, smear_param)
        return a

    Ncfg, Nt = data.shape
    dof = Nt - len(po)
    coeff, blks = (nbins-1)/nbins, jk_blocks(data, nbins)
    # breakpoint()
    popt = cFilter(f, xdata, data, xo=po, *args, **kwargs)
    popt_blocks = np.array([
        cFilter(f, xdata, dblk, xo=popt, *args, **kwargs)
        for dblk in blks
    ])
    popt_err = np.sqrt(coeff*np.sum((popt_blocks - popt)**2, 0))
    # compute chi-square
    if return_chisq:
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        res = f(xdata, *popt) - dmean
        L = scilin.cholesky(dcov, lower=True)
        xres = scilin.solve_triangular(L, res, lower=True)
        chisq = xres.dot(xres.conj())/dof
        pval = 1 - sts.chi2.cdf(chisq*dof, dof)
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks, chisq, pval
        else:
            return np.array([popt, popt_err]), chisq, pval
    else:
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks
        else:
            return np.array([popt, popt_err])


def fill_fix_params(pars, fixpars):
    new_pars = [None]*(len(pars) + len(fixpars))
    for j, fp in fixpars:
        new_pars[j] = fp
    for par in pars:
        for j, npar in enumerate(new_pars):
            if npar is None:
                new_pars[j] = par
                break
    new_pars = np.asarray(new_pars)
    return new_pars


def cv_fixfit_jk_mean_var(xdata, data, nbins, f, po,
                          fixpars, smear_param, return_chisq=True,
                          return_blocks=False, *args, **kwargs):
    def cFilter(func, xdata, data, xo, *args, **kwargs):
        # breakpoint()
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        my_func = ffunc.fix_func(f, fixpars)
        if 'jac' in kwargs.keys():
            mjac = kwargs['jac']
            my_jac = ffunc.fix_jac(mjac, fixpars)
            kwargs['jac'] = my_jac
        a, b = opt.curve_fit(my_func, xdata, dmean, p0=xo,
                             sigma=dcov, *args, **kwargs)
        # refill a with fixed value
        a = fill_fix_params(a, fixpars)
        a = check_fit_params(a, smear_param)
        return a

    Ncfg, Nt = data.shape
    dof = Nt - (len(po) - len(fixpars))
    coeff, blks = (nbins-1)/nbins, jk_blocks(data, nbins)
    # breakpoint()
    popt = cFilter(f, xdata, data, xo=po, *args, **kwargs)
    del_inds = [fix[0] for fix in fixpars]
    new_po = np.delete(popt, del_inds)
    popt_blocks = np.array([
        cFilter(f, xdata, dblk, xo=new_po, *args, **kwargs)
        for dblk in blks
    ])
    popt_err = np.sqrt(coeff*np.sum((popt_blocks - popt)**2, 0))
    # compute chi-square
    if return_chisq:
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        res = f(xdata, *popt) - dmean
        L = scilin.cholesky(dcov, lower=True)
        xres = scilin.solve_triangular(L, res, lower=True)
        chisq = xres.dot(xres.conj())/dof
        pval = 1 - sts.chi2.cdf(chisq*dof, dof)
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks, chisq, pval
        else:
            return np.array([popt, popt_err]), chisq, pval
    else:
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks
        else:
            return np.array([popt, popt_err])


def cv_fitprior_jk_mean_var(xdata, data, nbins, f, po, prior_list,
                            prior_err, eta=1, smear_param='SS',
                            return_chisq=True, return_blocks=False,
                            *args, **kwargs):
    def cFilter(func, xdata, data, xo, *args, **kwargs):
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        dmean = ffunc.prior_vals(dmean, prior_list)
        dcov = ffunc.prior_covar(dcov, prior_err, eta=eta)
        my_func = ffunc.prior_func(func, prior_list)
        if 'jac' in kwargs.keys():
            mjac = kwargs['jac']
            my_jac = ffunc.prior_jac(mjac, prior_list)
            kwargs['jac'] = my_jac
        a, b = opt.curve_fit(my_func, xdata, dmean, p0=xo,
                             sigma=dcov, *args, **kwargs)
        a = check_fit_params(a, smear_param)
        return a

    Ncfg, Nt = data.shape
    dof = Nt - len(po)   # - len(prior_list)
    coeff, blks = (nbins-1)/nbins, jk_blocks(data, nbins)
    # breakpoint()
    popt = cFilter(f, xdata, data, xo=po, *args, **kwargs)
    popt_blocks = np.array([
        cFilter(f, xdata, dblk, xo=popt, *args, **kwargs)
        for dblk in blks
    ])
    popt_err = np.sqrt(coeff*np.sum((popt_blocks - popt)**2, 0))
    # compute chi-square
    if return_chisq:
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        res = f(xdata, *popt) - dmean
        L = scilin.cholesky(dcov, lower=True)
        xres = scilin.solve_triangular(L, res, lower=True)
        chisq = xres.dot(xres.conj())/dof
        pval = 1 - sts.chi2.cdf(chisq*dof, dof)
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks, chisq, pval
        else:
            return np.array([popt, popt_err]), chisq, pval
    else:
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks
        else:
            return np.array([popt, popt_err])


def cv_fitfixprior_jk_mean_var(xdata, data, nbins, f, po, fixpars,
                               prior_list, prior_err, eta=1,
                               smear_param='SS', return_chisq=True,
                               return_blocks=False, *args, **kwargs):
    def cFilter(func, xdata, data, xo, *args, **kwargs):
        dmean, dcov = calc_jk_mean_cov(func, nbins)
        my_func = ffunc.fix_func(func, fixpars)
        dmean = ffunc.prior_vals(dmean, prior_list)
        dcov = ffunc.prior_covar(dcov, prior_err, eta=eta)
        my_func = ffunc.prior_func(my_func, prior_list)
        if 'jac' in kwargs.keys():
            mjac = kwargs['jac']
            my_jac = ffunc.fix_jac(mjac, fixpars)
            my_jac = ffunc.prior_jac(my_jac, prior_list)
            kwargs['jac'] = my_jac
        a, b = opt.curve_fit(my_func, xdata, dmean, p0=xo,
                             sigma=dcov, *args, **kwargs)
        a = check_fit_params(a, smear_param)
        return a
    Ncfg, Nt = data.shape
    dof = Nt - (len(po) - len(fixpars))
    coeff, blks = (nbins-1)/nbins, jk_blocks(data, nbins)
    # breakpoint()
    popt = cFilter(f, xdata, data, xo=po, *args, **kwargs)
    del_inds = [fix[0] for fix in fixpars]
    new_po = np.delete(popt, del_inds)
    popt_blocks = np.array([
        cFilter(f, xdata, dblk, xo=new_po, *args, **kwargs)
        for dblk in blks
    ])
    popt_err = np.sqrt(coeff*np.sum((popt_blocks - popt)**2, 0))
    # compute chi-square
    if return_chisq:
        dmean, dcov = calc_jk_mean_cov(data, nbins)
        res = f(xdata, *popt) - dmean
        L = scilin.cholesky(dcov, lower=True)
        xres = scilin.solve_triangular(L, res, lower=True)
        chisq = xres.dot(xres.conj())/dof
        pval = 1 - sts.chi2.cdf(chisq*dof, dof)
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks, chisq, pval
        else:
            return np.array([popt, popt_err]), chisq, pval
    else:
        if return_blocks:
            return np.array([popt, popt_err]), popt_blocks
        else:
            return np.array([popt, popt_err])


# ------------------------- Form Factor Set Up -------------------------#
def do_sym(data):
    dreal = 0.5*(data.real + np.flip(data.real, 0))
    dimag = 0.5*(data.imag - np.flip(data.imag, 0))
    return dreal + 1j*dimag


def remove_wrap(data, popts):
    T = data.shape[-1]
    tL = np.arange(T)
    wrap_around = ffunc.c2pt_meson_func_half(T-tL, *popts)
    return data - wrap_around


def c2pt_norm_dat(c2pt_snk, c2pt_src, dt, taus, popt_snk=None, popt_src=None):
    # FF_norm from two-point function data
    Lt = c2pt_snk.shape[-1]
    dt_taus = (dt - taus) % Lt
    c2pt_snk, c2pt_src = c2pt_snk.mean(0), c2pt_src.mean(0)
    if popt_snk is not None:
        c2pt_snk = remove_wrap(c2pt_snk, popt_snk)
    if popt_src is not None:
        c2pt_src = remove_wrap(c2pt_src, popt_src)

    num = c2pt_snk[dt]*c2pt_snk[taus]*c2pt_src[dt_taus]
    den = c2pt_src[dt]*c2pt_src[taus]*c2pt_snk[dt_taus]

    rsqrt = np.sqrt(num/den)
    # rsqrt = np.append(np.flip(rsqrt[:dt]), rsqrt[dt:])
    return rsqrt/c2pt_snk[dt]  # rsqrt/c2pt_src[dt]


def c2pt_norm(p_snk, p_src, dt, taus):
    Lt = 64
    dt_taus = (dt - taus) % Lt

    c2func = ffunc.c2pt_meson_func
    num = c2func(dt, *p_snk)*c2func(taus, *p_snk)*c2func(dt_taus, *p_src)
    den = c2func(dt, *p_src)*c2func(taus, *p_src)*c2func(dt_taus, *p_snk)
    rsqrt = np.sqrt(num/den)
    return rsqrt/c2func(dt, *p_snk)


def make_FormFactor_dat(c3pt, c2pt_snk, c2pt_src, dts, symm=False,
                        popt_snk=None, popt_src=None):
    taus = np.arange(c3pt.shape[-1])
    FF = np.swapaxes(
        np.array([c3pt.mean(0)[:, j]*c2pt_norm_dat(c2pt_snk, c2pt_src,
                                                   dt, taus,
                                                   popt_snk=popt_snk,
                                                   popt_src=popt_src)
                  for j, dt in enumerate(dts)]),
        0, 1)
    if symm:
        FF = do_sym(FF)
    return FF


def make_nFormFactor_dat(c3pt, c2pt_snk, c2pt_src, dts, symm=False,
                         popt_snk=None, popt_src=None):
    taus = np.arange(c3pt.shape[-1])
    FF = np.swapaxes(
        np.array([c3pt.mean(0)[:, :, j]*c2pt_norm_dat(c2pt_snk, c2pt_src,
                                                      dt, taus,
                                                      popt_snk=popt_snk,
                                                      popt_src=popt_src)
                  for j, dt in enumerate(dts)]),
        0, 2)
    if symm:
        FF = do_sym(FF)
    return FF


def make_FormFactor(c3pt, popt_snk, popt_src, dts, symm=False):
    taus = np.arange(c3pt.shape[-1])
    FF = np.swapaxes(
        np.array([c3pt.mean(0)[:, j]*c2pt_norm(popt_snk, popt_src, dt, taus)
                  for j, dt in enumerate(dts)]),
        0, 1)
    if symm:
        FF = do_sym(FF)
    return FF


def jk_blocks_FormFactor_dat(c3pt, c2pt_snk, c2pt_src, nbin, dts, symm=False,
                             popt_snk_blks=None, popt_src_blks=None):
    if isinstance(c3pt, types.GeneratorType):
        c3pt = np.array(list(c3pt))
    if isinstance(c2pt_snk, types.GeneratorType):
        c2pt_snk = np.array(list(c2pt_snk))
    if isinstance(c2pt_src, types.GeneratorType):
        c2pt_src = np.array(list(c2pt_src))

    bA = jk_blocks(c3pt, nbin)
    bB, bC = jk_blocks(c2pt_snk, nbin), jk_blocks(c2pt_src, nbin)
    if popt_src_blks is None:
        popt_src_blks = [None]*nbin
    if popt_snk_blks is None:
        popt_snk_blks = [None]*nbin

    FFblocks = (
        make_FormFactor_dat(
            ba, bb, bc, dts, symm=symm, popt_snk=pb, popt_src=pc
        )
        for ba, bb, bc, pb, pc in zip(bA, bB, bC, popt_snk_blks, popt_src_blks)
    )
    return FFblocks


def jk_blocks_nFormFactor_dat(c3pt, c2pt_snk, c2pt_src, nbin, dts,
                              symm=False, popt_snk_blks=None,
                              popt_src_blks=None, to_arr=False):
    if isinstance(c3pt, types.GeneratorType):
        c3pt = np.array(list(c3pt))
    if isinstance(c2pt_snk, types.GeneratorType):
        c2pt_snk = np.array(list(c2pt_snk))
    if isinstance(c2pt_src, types.GeneratorType):
        c2pt_src = np.array(list(c2pt_src))

    bA = jk_blocks(c3pt, nbin)
    bB, bC = jk_blocks(c2pt_snk, nbin), jk_blocks(c2pt_src, nbin)
    if popt_src_blks is None:
        popt_src_blks = [None]*nbin
    if popt_snk_blks is None:
        popt_snk_blks = [None]*nbin

    if not to_arr:
        FFblocks = (
            make_nFormFactor_dat(
                ba, bb, bc, dts, symm=symm, popt_snk=pb, popt_src=pc
            )
            for ba, bb, bc, pb, pc in zip(bA, bB, bC,
                                          popt_snk_blks,
                                          popt_src_blks)
        )
    else:
        FFblocks = np.array([
         make_nFormFactor_dat(
                ba, bb, bc, dts, symm=symm, popt_snk=pb, popt_src=pc
            )
            for ba, bb, bc, pb, pc in zip(bA, bB, bC,
                                          popt_snk_blks,
                                          popt_src_blks)
        ])
    return FFblocks


def jk_blocks_FormFactor(c3pt, psnk_blocks, psrc_blocks,
                         nbin, dts, symm=False):
    c3pt_blocks = jk_blocks(c3pt, nbin)
    FFblocks = (
        make_FormFactor(blk, psnk_b, psrc_b, dts, symm=symm)
        for blk, psnk_b, psrc_b in zip(c3pt_blocks, psnk_blocks, psrc_blocks)
    )
    return FFblocks


def jk_FormFactor_dat_mean_err(c3pt, c2pt_snk, c2pt_src, nbin, dts, symm=False,
                               popt_src=None, popt_snk=None,
                               popt_src_blks=None, popt_snk_blks=None):
    coeff = (nbin-1)/nbin
    FF = make_FormFactor_dat(c3pt, c2pt_snk, c2pt_src, dts, symm=symm,
                             popt_snk=popt_snk, popt_src=popt_src)
    FFblocks = jk_blocks_FormFactor_dat(c3pt, c2pt_snk, c2pt_src,
                                        nbin, dts, symm=symm,
                                        popt_snk_blks=popt_snk_blks,
                                        popt_src_blks=popt_src_blks)
    FFblocks = np.array(list(FFblocks))
    FF_err_real = np.sqrt(coeff*np.sum((FFblocks.real - FF.real)**2, 0))
    FF_err_imag = np.sqrt(coeff*np.sum((FFblocks.imag - FF.imag)**2, 0))
    FF_err = FF_err_real + 1j*FF_err_imag
    return FF, FF_err


def jk_nFormFactor_dat_mean_err(c3pt, c2pt_snk, c2pt_src, nbin, dts,
                                symm=False, popt_src=None, popt_snk=None,
                                popt_src_blks=None, popt_snk_blks=None):
    coeff = (nbin-1)/nbin
    FF = make_nFormFactor_dat(c3pt, c2pt_snk, c2pt_src, dts, symm=symm,
                              popt_snk=popt_snk, popt_src=popt_src)
    FFblocks = jk_blocks_nFormFactor_dat(c3pt, c2pt_snk, c2pt_src,
                                         nbin, dts, symm=symm,
                                         popt_snk_blks=popt_snk_blks,
                                         popt_src_blks=popt_src_blks)
    FFblocks = np.array(list(FFblocks))
    FF_err_real = np.sqrt(coeff*np.sum((FFblocks.real - FF.real)**2, 0))
    FF_err_imag = np.sqrt(coeff*np.sum((FFblocks.imag - FF.imag)**2, 0))
    FF_err = FF_err_real + 1j*FF_err_imag
    return FF, FF_err


def jk_FormFactor_mean_err(c3pt, psnk, psrc, psnk_blocks, psrc_blocks,
                           nbin, dts, symm=False):
    coeff = (nbin-1)/nbin
    FF = make_FormFactor(c3pt, psnk, psrc, dts, symm=symm)
    FFblocks = jk_blocks_FormFactor(
        c3pt, psnk_blocks, psrc_blocks, nbin, dts, symm=symm
    )
    FFblocks = np.array(list(FFblocks))
    FF_err_real = np.sqrt(coeff*np.sum((FFblocks.real - FF.real)**2, 0))
    FF_err_imag = np.sqrt(coeff*np.sum((FFblocks.imag - FF.imag)**2, 0))
    FF_err = FF_err_real + 1j*FF_err_imag
    return FF, FF_err
