//
// Created by pierfied on 7/30/19.
//

#include "healcorr.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>

extern "C" {

    __global__
    void calc_ang_sep(int ind, float *x, float *y, float *z, float *ang_sep){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        float xi = x[ind];
        float yi = y[ind];
        float zi = z[ind];

        if(index == 0){
            ang_sep[ind] = 0;
        }

        for (int j = index; j < ind; j += stride){
            ang_sep[j] = acosf(xi * x[j] + yi * y[j] + zi * z[j]);
        }
    }

    __global__
    void calc_map_means(int nmaps, int npix, float *maps, float *map_means){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < nmaps; i += stride){
            map_means[i] = 0;

            for (int j = i * npix; j < (i + 1) * npix; j++){
                map_means[i] += maps[j];
            }

            map_means[i] /= npix;
        }
    }

    __global__
    void calc_vec(int npix, float *theta, float *phi, float *x, float *y, float *z){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < npix; i += stride){
            x[i] = cosf(phi[i]) * sinf(theta[i]);
            y[i] = sinf(phi[i]) * sinf(theta[i]);
            z[i] = cosf(theta[i]);
        }
    }

    __global__
    void calc_xis(long npix, int nxis, int nbins, float *x, float *y, float *z, float *bins, float *maps,
                  int *map1, int *map2, float *xis, float *counts){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (long ind = index; ind < npix * npix; ind += stride) {
            int i = ind / npix;
            int j = ind % npix;

            if (j > i) continue;

            float ang_sep = acosf(x[i] * x[j] + y[i] * y[j] + z[i] * z[j]);
            ang_sep = fminf(fmaxf(ang_sep, -1), 1);

            int bin_num;
            if (ang_sep < bins[0] || ang_sep > bins[nbins]) continue;
            for (int k = 0; k < nbins; k++) {
                if (ang_sep < bins[k + 1]) {
                    bin_num = k;
                    break;
                }
            }

            float *addr = &(counts[bin_num]);
            atomicAdd(addr, 1);

            for (int k = 0; k < nxis; k++) {
                addr = &(xis[k * nbins + bin_num]);
                atomicAdd(addr, maps[map1[k] * npix + i] * maps[map2[k] * npix + j]);
            }
        }
    }

    __global__
    void avg_xis(int nxis, int nbins, float *xis, float *counts, float *map_means,
                 int *map1, int *map2){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int ind = index; ind < nxis * nbins; ind += stride){
            int i = ind / nbins;
            int j = ind % nbins;

            xis[ind] = xis[ind] / counts[j] - map_means[map1[i]] * map_means[map2[i]];
        }
    }

    float *healcorr(int npix, float *theta, float *phi, int nmaps, float *maps, int nbins, float *bins,
                     int verbose, int cross_correlate){
        float *cu_maps, *cu_theta, *cu_phi, *cu_bins;
        cudaMallocManaged(&cu_maps, sizeof(float) * nmaps * npix);
        cudaMallocManaged(&cu_theta, sizeof(float) * npix);
        cudaMallocManaged(&cu_phi, sizeof(float) * npix);
        cudaMallocManaged(&cu_bins, sizeof(float) * (nbins + 1));
        cudaMemcpy(cu_maps, maps, sizeof(float) * nmaps * npix, cudaMemcpyDefault);
        cudaMemcpy(cu_theta, theta, sizeof(float) * npix, cudaMemcpyDefault);
        cudaMemcpy(cu_phi, phi, sizeof(float) * npix, cudaMemcpyDefault);
        cudaMemcpy(cu_bins, bins, sizeof(float) * (nbins + 1), cudaMemcpyDefault);

        float *map_means;
        cudaMallocManaged(&map_means, sizeof(float) * nmaps);

        int blockSize, numBlocks;

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, calc_map_means, 0, 0);
        calc_map_means<<<numBlocks, blockSize>>>(nmaps, npix, cu_maps, map_means);

        float *x, *y, *z;
        cudaMallocManaged(&x, sizeof(float) * npix);
        cudaMallocManaged(&y, sizeof(float) * npix);
        cudaMallocManaged(&z, sizeof(float) * npix);

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, calc_vec, 0, 0);
        calc_vec<<<numBlocks, blockSize>>>(npix, cu_theta, cu_phi, x, y, z);

        int nxis;
        if (cross_correlate) {
            nxis = (nmaps * (nmaps + 1)) / 2;
        } else {
            nxis = nmaps;
        }

        int *map1, *map2;
        cudaMallocManaged(&map1, sizeof(int) * nxis);
        cudaMallocManaged(&map2, sizeof(int) * nxis);

        int xi_num = 0;
        if (cross_correlate) {
            for (int i = 0; i < nmaps; ++i) {
                for (int j = 0; j <= i; ++j) {
                    map1[xi_num] = i;
                    map2[xi_num] = j;

                    xi_num++;
                }
            }
        } else {
            for (int i = 0; i < nmaps; ++i) {
                map1[i] = i;
                map2[i] = i;
            }
        }

        float *cu_xis, *counts;
        cudaMallocManaged(&cu_xis, sizeof(float) * nbins * nxis);
        cudaMallocManaged(&counts, sizeof(float) * nbins * nxis);
#pragma omp parallel for
        for (long i = 0; i < nbins * nxis; ++i) {
            cu_xis[i] = 0;
        }
#pragma omp parallel for
        for (long i = 0; i < nbins; ++i) {
            counts[i] = 0;
        }

        float *ang_sep;
        int *bin_nums;
        cudaMallocManaged(&ang_sep, sizeof(float) * npix);
        cudaMallocManaged(&bin_nums, sizeof(int) * npix);

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, calc_xis, 0, 0);
        calc_xis<<<numBlocks, blockSize>>>(npix, nxis, nbins, x, y, z, cu_bins, cu_maps,
                                           map1, map2, cu_xis, counts);

        cudaOccupancyMaxPotentialBlockSize( &numBlocks, &blockSize, avg_xis, 0, 0);
        avg_xis<<<numBlocks, blockSize>>>(nxis, nbins, cu_xis, counts, map_means, map1, map2);

        cudaDeviceSynchronize();

        float *xis;
        xis = (float *) malloc(sizeof(float) * nxis * nbins);
        cudaMemcpy(xis, cu_xis, sizeof(float) * nxis * nbins, cudaMemcpyDefault);

        std::cout << "Done: " << cudaGetLastError() << std::endl;

        cudaFree(map_means);
        cudaFree(x);
        cudaFree(y);
        cudaFree(z);
        cudaFree(counts);
        cudaFree(ang_sep);
        cudaFree(bin_nums);
        cudaFree(map1);
        cudaFree(map2);

        cudaFree(cu_maps);
        cudaFree(cu_theta);
        cudaFree(cu_phi);
        cudaFree(cu_bins);

        return xis;
    }
}