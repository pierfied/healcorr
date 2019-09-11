import numpy as np
import ctypes
import os
import healpy as hp
import glob


def compute_corr(maps, mask, bins, premasked=False, cross_correlate=False, verbose=False):
    npix = len(mask)
    nside = hp.npix2nside(npix)

    nbins = len(bins) - 1

    pix_inds = np.arange(npix)[mask]
    theta, phi = hp.pix2ang(nside, pix_inds)

    if not premasked:
        masked_maps = maps[:, mask]
    else:
        masked_maps = maps

    nmaps = masked_maps.shape[0]
    mask_npix = masked_maps.shape[1]

    lib_path = glob.glob(os.path.join(os.path.dirname(__file__), '../healcorr*.so'))[0]
    healcorr_lib = ctypes.cdll.LoadLibrary(lib_path)

    dptr = ctypes.POINTER(ctypes.c_double)

    healcorr_run = healcorr_lib.healcorr
    healcorr_run.argtypes = [ctypes.c_long, dptr, dptr, ctypes.c_long, dptr, ctypes.c_long, dptr, ctypes.c_long,
                             ctypes.c_long]
    healcorr_run.restype = dptr

    masked_maps = np.ascontiguousarray(masked_maps)
    theta = np.ascontiguousarray(theta)
    phi = np.ascontiguousarray(phi)
    bins = np.ascontiguousarray(bins)

    masked_maps_ptr = masked_maps.ctypes.data_as(dptr)
    theta_ptr = theta.ctypes.data_as(dptr)
    phi_ptr = phi.ctypes.data_as(dptr)
    bins_ptr = bins.ctypes.data_as(dptr)

    if verbose:
        verbose_flag = 1
    else:
        verbose_flag = 0

    if cross_correlate:
        crosscorr_flag = 1
        nxis = (nmaps * (nmaps + 1)) / 2
    else:
        crosscorr_flag = 0
        nxis = nmaps

    xis_ptr = healcorr_run(mask_npix, theta_ptr, phi_ptr, nmaps, masked_maps_ptr, nbins, bins_ptr, verbose_flag,
                           crosscorr_flag)
    xis = np.ctypeslib.as_array(xis_ptr, shape=(nxis, nbins))

    if cross_correlate:
        xi_mat = np.zeros([nmaps, nmaps, nbins])
        xi_mask = np.tri(nmaps, nmaps, dtype=True)

        for i in range(nbins):
            xi_mat[:,:,i][xi_mask] =  xis[:,i]
            xi_mat[:,:,i][xi_mask.T] =  xis[:,i]

        xis = xi_mat

    return xis
