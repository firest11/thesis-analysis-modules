import itertools as it
import functools as ft
# import types
import numpy as np
import scipy.optimize as opt

# import fitfuncs as ffunc
from . import fit_scripts as fitter
from . import lstats


# ------------------------- C2pt Fit Wrappers -------------------------#
# Effective Mass Plots
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


def wrap_eff_mass_jk_mean_var(xdata, Dat, miter=1000, verbose=False):
    jk_smp = Dat.shape[0]
    dmean = Dat.mean(0)
    blocks = lstats.jk_blocks(Dat, jk_smp)
    mean = fitter.eff_mass(xdata, dmean, verbose=verbose)
    val_list = [mean]

    for blk in blocks:
        val_list.append(fitter.eff_mass(xdata, blk.mean(0), verbose=verbose))
    trange, image_list = match_lists(val_list)
    cv = image_list[0]
    coeff = (jk_smp-1.0)/jk_smp
    err = np.sqrt(coeff*np.sum((image_list[1:]-cv)**2, 0))
    return trange, cv, err


# --- C2pt Fit: Nstate
# minsolve
def wrap_minsolve_jk_mean_cov(xdata, data, nbins, f, xo,
                              jac=None, hess=None, chisquare=False,
                              *margs, **mkwargs):
    Ncfg, Nt = data.shape
    coeff, blks = (nbins-1)/nbins, lstats.jk_blocks(data, nbins)
    blks, blks2 = it.tee(blks)
    dat_mean, dat_jcov = lstats.calc_jk_mean_cov(data, nbins)

    try:
        val = fitter.minsolve(xdata, dat_mean, dat_jcov, f, xo,
                              cJac=jac, cHess=hess, *margs, **mkwargs)
        if chisquare:
            chisq = fitter.chisquare(val, xdata, dat_mean, dat_jcov, f)
        vcov_blocks = np.zeros((nbins, len(val), len(val)))
        for n, dblk in enumerate(blks2):
            bmean, bcov = lstats.calc_jk_mean_cov(dblk, nbins-1)
            y = fitter.minsolve(xdata, bmean, bcov, f, val,
                                cJac=jac, cHess=hess, *margs, **mkwargs)
            res = y-val
            vcov_blocks[n] = np.outer(res, res)
        vcov = coeff*np.sum(vcov_blocks, 0)
    except Exception as e:
        print("Error: {0}".format(e))
        raise Exception(str(e))
    if chisquare:
        return val, vcov, chisq
    else:
        return val, vcov


def wrap_minsolve_jk_mean_var(xdata, data, nbins, f, xo,
                              jac=None, hess=None, chisquare=False,
                              *margs, **mkwargs):
    if chisquare:
        mean, cov, chisq = wrap_minsolve_jk_mean_cov(xdata, data, nbins, f, xo,
                                                     jac=jac, hess=hess,
                                                     chisquare=chisquare,
                                                     *margs, **mkwargs)
        return mean, np.sqrt(cov.diagonal()), chisq
    else:
        mean, cov = wrap_minsolve_jk_mean_cov(xdata, data, nbins, f, xo,
                                              jac=jac, hess=hess,
                                              chisquare=chisquare,
                                              *margs, **mkwargs)
        return mean, np.sqrt(cov.diagonal())


def wrap_stuff():
    return opt.curve_fit
# def wrap_cv_jk_mean_cov(xdata, data, nbins, f, po, sp_val,
#                         verb=0, *args, **kwargs):
#     def cFilter(func, xdata, data, xo, sigma, *args, **kwargs):
#         a, b = opt.curve_fit(
#             func, xdata, data, p0=xo, sigma=sigma, *args, **kwargs
#         )
#         return a

#     Ncfg, Nt = data.shape
#     coeff, blks = (nbins-1)/nbins, jk_blocks(data, nbins)
#     dat_mean, dat_jcov = calc_jk_mean_cov(data, nbins)
#     try:
#         popt, pcov = opt.curve_fit(f, xdata, dat_mean, p0=po,
#                                    sigma=dat_jcov, *args, **kwargs)


def wrap_sumFF_dat(c3pt, c2pt_snk, c2pt_src, dts, tau_o,
                   nbins=None, symm=False, chisquare=False,
                   popt_snk=None, popt_snk_blks=None,
                   popt_src=None, popt_src_blks=None):
    FF = lstats.make_FormFactor_dat(c3pt, c2pt_snk, c2pt_src, dts, symm=symm,
                                    popt_snk=popt_snk, popt_src=popt_src)
    FFblocks = lstats.jk_blocks_FormFactor_dat(c3pt, c2pt_snk, c2pt_src,
                                               nbins, dts, symm=symm,
                                               popt_snk_blks=popt_snk_blks,
                                               popt_src_blks=popt_src_blks)
    FFblocks = np.array(list(FFblocks))

    c3pt_blocks = lstats.jk_blocks(c3pt, nbins)
    c2snk_blocks = lstats.jk_blocks(c2pt_snk, nbins)
    c2src_blocks = lstats.jk_blocks(c2pt_src, nbins)
    coeff = (nbins-1.0)/nbins
    if chisquare:
        vals, chisq = fitter.FF_sumFit(dts, tau_o, FF, FFblocks,
                                       chisquare=chisquare)
        vblocks = np.zeros((nbins, *vals.shape), dtype=vals.dtype)
        for j, (c3blk, c2snk_blk, c2src_blk) in enumerate(
                zip(c3pt_blocks, c2snk_blocks, c2src_blocks)):
            FFb = lstats.make_FormFactor_dat(c3blk, c2snk_blk, c2src_blk, dts,
                                             symm=symm, popt_snk=popt_snk,
                                             popt_src=popt_src)
            FFblocks2 = lstats.jk_blocks_FormFactor_dat(
                c3blk, c2snk_blk, c2src_blk, nbins-1, dts, symm=symm,
                popt_snk_blks=popt_snk_blks[1:], popt_src_blks=popt_src_blks[1:]
            )
            FFblocks2 = np.array(list(FFblocks))
            vblocks[j] = fitter.FF_sumFit(dts, tau_o, FFb, FFb, FFblocks2,
                                          chisquare=False)
        verr_r = np.sqrt(coeff*np.sum((vblocks.real - vals.real)**2, 0))
        verr_i = np.sqrt(coeff*np.sum((vblocks.imag - vals.imag)**2, 0))
        verr = verr_r + 1j*verr_i
        return vals, vblocks, verr, chisq
    else:
        vals = fitter.FF_sumFit(dts, tau_o, FF, FFblocks,
                                chisquare=chisquare)
        vblocks = np.zeros((nbins, *vals.shape), dtype=vals.dtype)
        for j, (c3blk, c2snk_blk, c2src_blk) in enumerate(
                zip(c3pt_blocks, c2snk_blocks, c2src_blocks)):
            FFb = lstats.make_FormFactor_dat(c3blk, c2snk_blk, c2src_blk, dts,
                                             symm=symm,
                                             popt_snk=popt_snk_blks[j],
                                             popt_src=popt_src_blks[j])
            FFblocks2 = lstats.jk_blocks_FormFactor_dat(
                c3blk, c2snk_blk, c2src_blk, nbins-1, dts, symm=symm,
                popt_snk_blks=popt_snk_blks[1:], popt_src_blks=popt_src_blks[1:])
            FFblocks2 = np.array(list(FFblocks))
            vblocks[j] = fitter.FF_sumFit(dts, tau_o, FFb, FFb, FFblocks2,
                                          chisquare=False)
        verr_r = np.sqrt(coeff*np.sum((vblocks.real - vals.real)**2, 0))
        verr_i = np.sqrt(coeff*np.sum((vblocks.imag - vals.imag)**2, 0))
        verr = verr_r + 1j*verr_i
        return vals, vblocks, verr


def wrap_sumFF(c3pt, psnk, psnk_blocks, psrc, psrc_blocks, dts, tau_o,
               nbins=None, symm=False, chisquare=False):
    FF = lstats.make_FormFactor(c3pt, psnk, psrc, dts, symm=symm)
    FFblocks = lstats.jk_blocks_FormFactor(c3pt, psnk_blocks, psrc_blocks,
                                           nbins, dts, symm=symm)
    FFblocks = np.array(list(FFblocks))

    c3pt_blocks = lstats.jk_blocks(c3pt, nbins)
    coeff = (nbins-1.0)/nbins
    if chisquare:
        vals, chisq = fitter.FF_sumFit(dts, tau_o, FF, FFblocks,
                                       chisquare=chisquare)
        vblocks = np.zeros((nbins, *vals.shape), dtype=vals.dtype)
        for j, c3blk in enumerate(c3pt_blocks):
            FFb = lstats.make_FormFactor(c3blk, psnk_blocks[j], psrc_blocks[j],
                                         dts, symm=symm)
            FFblocks2 = lstats.jk_blocks_FormFactor(c3blk, psnk_blocks[1:],
                                                    psrc_blocks[1:],
                                                    nbins-1, dts, symm=symm)
            FFblocks2 = np.array(list(FFblocks2))
            vblocks[j] = fitter.FF_sumFit(dts, tau_o, FFb, FFblocks2,
                                          chisquare=False)
        verr_r = np.sqrt(coeff*np.sum((vblocks.real - vals.real)**2, 0))
        verr_i = np.sqrt(coeff*np.sum((vblocks.imag - vals.imag)**2, 0))
        verr = verr_r + 1j*verr_i
        return vals, vblocks, verr, chisq
    else:
        vals = fitter.FF_sumFit(dts, tau_o, FF, FFblocks,
                                chisquare=chisquare)
        vblocks = np.zeros((nbins, *vals.shape), dtype=vals.dtype)
        for j, c3blk in enumerate(c3pt_blocks):
            FFb = lstats.make_FormFactor(c3blk, psnk_blocks[j], psrc_blocks[j],
                                         dts, symm=symm)
            FFblocks2 = lstats.jk_blocks_FormFactor(c3blk, psnk_blocks[1:],
                                                    psrc_blocks[1:],
                                                    nbins-1, dts, symm=symm)
            FFblocks2 = np.array(list(FFblocks2))
            vblocks[j] = fitter.FF_sumFit(dts, tau_o, FFb, FFblocks2,
                                          chisquare=False)
        verr_r = np.sqrt(coeff*np.sum((vblocks.real - vals.real)**2, 0))
        verr_i = np.sqrt(coeff*np.sum((vblocks.imag - vals.imag)**2, 0))
        verr = verr_r + 1j*verr_i
        return vals, vblocks, verr


def wrap_FF_nstateFit(c3pt, psnk, psnk_blocks, psrc, psrc_blocks, dts, tau_o,
                      nbins=None, symm=False, chisquare=False):
    FF = lstats.make_FormFactor(c3pt, psnk[:2], psrc[:2], dts, symm=symm)
    FFblocks = lstats.jk_blocks_FormFactor(c3pt, psnk_blocks[:, :2],
                                           psrc_blocks[:, :2],
                                           nbins, dts, symm=symm)
    FFblocks = np.array(list(FFblocks))

    c3pt_blocks = lstats.jk_blocks(c3pt, nbins)
    coeff = (nbins-1.0)/nbins

    if chisquare:
        vals, chisq = fitter.FF_nstateFit(dts, tau_o, psnk[1::2], psrc[1::2],
                                          FF, FFblocks, chisquare=chisquare)
        vblocks = np.zeros((nbins, *vals.shape), dtype=vals.dtype)
        for j, c3blk in enumerate(c3pt_blocks):
            FFb = lstats.make_FormFactor(c3blk, psnk_blocks[j, :2],
                                         psrc_blocks[j, :2], dts, symm=symm)
            FFblocks2 = lstats.jk_blocks_FormFactor(c3blk, psnk_blocks[1:, :2],
                                                    psrc_blocks[1:, :2],
                                                    nbins-1, dts, symm=symm)
            FFblocks2 = np.array(list(FFblocks2))
            vblocks[j] = fitter.FF_nstateFit(dts, tau_o, psnk_blocks[j, 1::2],
                                             psrc_blocks[j, 1::2],
                                             FFb, FFblocks2, chisquare=False)
        verr_r = np.sqrt(coeff*np.sum((vblocks.real - vals.real)**2, 0))
        verr_i = np.sqrt(coeff*np.sum((vblocks.imag - vals.imag)**2, 0))
        verr = verr_r + 1j*verr_i
        return vals, vblocks, verr, chisq
    else:
        vals = fitter.FF_nstateFit(dts, tau_o, psnk[1::2], psrc[1::2],
                                   FF, FFblocks, chisquare=chisquare)
        vblocks = np.zeros((nbins, *vals.shape), dtype=vals.dtype)
        for j, c3blk in enumerate(c3pt_blocks):
            FFb = lstats.make_FormFactor(c3blk, psnk_blocks[j, :2],
                                         psrc_blocks[j, :2], dts, symm=symm)
            FFblocks2 = lstats.jk_blocks_FormFactor(c3blk, psnk_blocks[1:, :2],
                                                    psrc_blocks[1:, :2],
                                                    nbins-1, dts, symm=symm)
            FFblocks2 = np.array(list(FFblocks2))
            vblocks[j] = fitter.FF_nstateFit(dts, tau_o, psnk_blocks[j, 1::2],
                                             psrc_blocks[j, 1::2],
                                             FFb, FFblocks2, chisquare=False)
        verr_r = np.sqrt(coeff*np.sum((vblocks.real - vals.real)**2, 0))
        verr_i = np.sqrt(coeff*np.sum((vblocks.imag - vals.imag)**2, 0))
        verr = verr_r + 1j*verr_i
        return vals, vblocks, verr
