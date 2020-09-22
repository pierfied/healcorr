//
// Created by pierfied on 7/30/19.
//

#include "healcorr.h"
#include <stdlib.h>
#include <iostream>

extern "C" {
    __global__
    void calc_map_means(long nmaps, long npix, double *maps, double *map_means){
        long index = blockIdx.x * blockDim.x + threadIdx.x;
        long stride = blockDim.x * gridDim.x;

        for (long i = index; i < nmaps; i += stride){
            map_means[i] = 0;

            for (long j = i * npix; j < (i + 1) * npix; j++){
                map_means[i] += maps[j];
            }

            map_means[i] /= npix;
        }
    }

    __global__
    void calc_vec(long npix, double *theta, double *phi, double *x, double *y, double *z){
        long index = blockIdx.x * blockDim.x + threadIdx.x;
        long stride = blockDim.x * gridDim.x;

        for (long i = index; i < npix; i += stride){
            x[i] = cos(phi[i]) * sin(theta[i]);
            y[i] = sin(phi[i]) * sin(theta[i]);
            z[i] = cos(theta[i]);
        }
    }

    __global__
    void calc_xis(long npix, long nxis, long nbins, double *x, double *y, double *z, double *bins, double *maps,
                  long *map1, long *map2, double *xis, double *counts){
        long index = blockIdx.x * blockDim.x + threadIdx.x;
        long stride = blockDim.x * gridDim.x;

        for (long ind = index; ind < npix * npix; ind += stride) {
            long i = ind / npix;
            long j = ind % npix;

            if (j > i) continue;

            double dot_prod = x[i] * x[j] + y[i] * y[j] + z[i] * z[j];
            dot_prod = fmin(fmax(dot_prod, -1.), 1.);
            double ang_sep = acos(dot_prod);

            long bin_num;
            if (ang_sep < bins[0] || ang_sep > bins[nbins]) continue;
            for (long k = 0; k < nbins; k++) {
                if (ang_sep < bins[k + 1]) {
                    bin_num = k;
                    break;
                }
            }

            double *addr = &(counts[bin_num]);
            atomicAdd(addr, 1);

            for (long k = 0; k < nxis; k++) {
                addr = &(xis[k * nbins + bin_num]);
                atomicAdd(addr, maps[map1[k] * npix + i] * maps[map2[k] * npix + j]);
            }
        }
    }

    __global__
    void avg_xis(long nxis, long nbins, double *xis, double *counts, double *map_means,
                 long *map1, long *map2){
        long index = blockIdx.x * blockDim.x + threadIdx.x;
        long stride = blockDim.x * gridDim.x;

        for (long ind = index; ind < nxis * nbins; ind += stride){
            long i = ind / nbins;
            long j = ind % nbins;

            xis[ind] = xis[ind] / counts[j] - map_means[map1[i]] * map_means[map2[i]];
        }
    }

    double *healcorr(long npix, double *theta, double *phi, long nmaps, double *maps, long nbins, double *bins,
                     long verbose, long cross_correlate){
        double *cu_maps, *cu_theta, *cu_phi, *cu_bins;
        cudaMallocManaged(&cu_maps, sizeof(double) * nmaps * npix);
        cudaMallocManaged(&cu_theta, sizeof(double) * npix);
        cudaMallocManaged(&cu_phi, sizeof(double) * npix);
        cudaMallocManaged(&cu_bins, sizeof(double) * (nbins + 1));
        cudaMemcpy(cu_maps, maps, sizeof(double) * nmaps * npix, cudaMemcpyDefault);
        cudaMemcpy(cu_theta, theta, sizeof(double) * npix, cudaMemcpyDefault);
        cudaMemcpy(cu_phi, phi, sizeof(double) * npix, cudaMemcpyDefault);
        cudaMemcpy(cu_bins, bins, sizeof(double) * (nbins + 1), cudaMemcpyDefault);

        double *map_means;
        cudaMallocManaged(&map_means, sizeof(double) * nmaps);

        int blockSize, numBlocks;

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, calc_map_means, 0, 0);
        calc_map_means<<<numBlocks, blockSize>>>(nmaps, npix, cu_maps, map_means);

        double *x, *y, *z;
        cudaMallocManaged(&x, sizeof(double) * npix);
        cudaMallocManaged(&y, sizeof(double) * npix);
        cudaMallocManaged(&z, sizeof(double) * npix);

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, calc_vec, 0, 0);
        calc_vec<<<numBlocks, blockSize>>>(npix, cu_theta, cu_phi, x, y, z);

        long nxis;
        if (cross_correlate) {
            nxis = (nmaps * (nmaps + 1)) / 2;
        } else {
            nxis = nmaps;
        }

        long *map1, *map2;
        cudaMallocManaged(&map1, sizeof(long) * nxis);
        cudaMallocManaged(&map2, sizeof(long) * nxis);

        long xi_num = 0;
        if (cross_correlate) {
            for (long i = 0; i < nmaps; ++i) {
                for (long j = 0; j <= i; ++j) {
                    map1[xi_num] = i;
                    map2[xi_num] = j;

                    xi_num++;
                }
            }
        } else {
            for (long i = 0; i < nmaps; ++i) {
                map1[i] = i;
                map2[i] = i;
            }
        }

        double *cu_xis, *counts;
        cudaMallocManaged(&cu_xis, sizeof(double) * nbins * nxis);
        cudaMallocManaged(&counts, sizeof(double) * nbins * nxis);
#pragma omp parallel for
        for (long i = 0; i < nbins * nxis; ++i) {
            cu_xis[i] = 0;
        }
#pragma omp parallel for
        for (long i = 0; i < nbins; ++i) {
            counts[i] = 0;
        }

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, calc_xis, 0, 0);
        calc_xis<<<numBlocks, blockSize>>>(npix, nxis, nbins, x, y, z, cu_bins, cu_maps,
                                           map1, map2, cu_xis, counts);

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, avg_xis, 0, 0);
        avg_xis<<<numBlocks, blockSize>>>(nxis, nbins, cu_xis, counts, map_means, map1, map2);

        cudaDeviceSynchronize();

        double *xis;
        xis = (double *) malloc(sizeof(double) * nxis * nbins);
        cudaMemcpy(xis, cu_xis, sizeof(double) * nxis * nbins, cudaMemcpyDefault);

        if (verbose) {
            std::cout << "Done: " << cudaGetLastError() << std::endl;
        }

        cudaFree(map_means);
        cudaFree(x);
        cudaFree(y);
        cudaFree(z);
        cudaFree(counts);
        cudaFree(map1);
        cudaFree(map2);

        cudaFree(cu_maps);
        cudaFree(cu_theta);
        cudaFree(cu_phi);
        cudaFree(cu_bins);

        return xis;
    }
}