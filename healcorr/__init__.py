import numpy as np
import ctypes
import os
import healpy as hp
import glob


def compute_corr(maps, mask, bins, premasked=False, cross_correlate=False, verbose=False):
    print('starting')
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

    lib_path = glob.glob(os.path.join(os.path.dirname(__file__), '../libhealcorr.so'))[0]
    healcorr_lib = ctypes.cdll.LoadLibrary(lib_path)

    fptr = ctypes.POINTER(ctypes.c_float)

    healcorr_run = healcorr_lib.healcorr
    healcorr_run.argtypes = [ctypes.c_int, fptr, fptr, ctypes.c_int, fptr, ctypes.c_int, fptr, ctypes.c_int,
                             ctypes.c_int]
    healcorr_run.restype = fptr

    masked_maps = np.ascontiguousarray(masked_maps.astype(np.float32))
    theta = np.ascontiguousarray(theta.astype(np.float32))
    phi = np.ascontiguousarray(phi.astype(np.float32))
    bins = np.ascontiguousarray(bins.astype(np.float32))

    masked_maps_ptr = masked_maps.ctypes.data_as(fptr)
    theta_ptr = theta.ctypes.data_as(fptr)
    phi_ptr = phi.ctypes.data_as(fptr)
    bins_ptr = bins.ctypes.data_as(fptr)

    if verbose:
        verbose_flag = 1
    else:
        verbose_flag = 0

    if cross_correlate:
        crosscorr_flag = 1
        nxis = int((nmaps * (nmaps + 1)) / 2)
    else:
        crosscorr_flag = 0
        nxis = nmaps

    print('about to run')

    xis_ptr = healcorr_run(mask_npix, theta_ptr, phi_ptr, nmaps, masked_maps_ptr, nbins, bins_ptr, verbose_flag,
                           crosscorr_flag)
    xis = np.ctypeslib.as_array(xis_ptr, shape=(nxis, nbins))

    print('done running')

    if cross_correlate:
        xi_mat = np.zeros([nmaps, nmaps, nbins])
        xi_mask = np.tri(nmaps, nmaps, dtype=bool)

        for i in range(nbins):
            xi_mat[:,:,i][xi_mask] =  xis[:,i]
            xi_mat[:,:,i][xi_mask.T] =  xi_mat[:,:,i].T[xi_mask.T]

        xis = xi_mat

    return xis
