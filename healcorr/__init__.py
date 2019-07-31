import numpy as np
import ctypes
import os
import healpy as hp
import glob


def compute_corr(maps, mask, bins):
    npix = len(mask)
    nside = hp.npix2nside(npix)

    nbins = len(bins) - 1

    pix_inds = np.arange(npix)[mask]
    theta, phi = hp.pix2ang(nside, pix_inds)

    masked_maps = maps[:, mask]

    nmaps = masked_maps.shape[0]
    mask_npix = masked_maps.shape[1]

    lib_path = glob.glob(os.path.join(os.path.dirname(__file__), '../healcorr*.so'))[0]
    healcorr_lib = ctypes.cdll.LoadLibrary(lib_path)

    dptr = ctypes.POINTER(ctypes.c_double)

    healcorr_run = healcorr_lib.healcorr
    healcorr_run.argtypes = [ctypes.c_long, dptr, dptr, ctypes.c_long, dptr, ctypes.c_long, dptr]
    healcorr_run.restype = dptr

    masked_maps = np.ascontiguousarray(masked_maps)
    theta = np.ascontiguousarray(theta)
    phi = np.ascontiguousarray(phi)
    bins = np.ascontiguousarray(bins)

    masked_maps_ptr = masked_maps.ctypes.data_as(dptr)
    theta_ptr = theta.ctypes.data_as(dptr)
    phi_ptr = phi.ctypes.data_as(dptr)
    bins_ptr = bins.ctypes.data_as(dptr)

    xis_ptr = healcorr_run(mask_npix, theta_ptr, phi_ptr, nmaps, masked_maps_ptr, nbins, bins_ptr)
    xis = np.ctypeslib.as_array(xis_ptr, shape=(nmaps, nbins))

    return xis
