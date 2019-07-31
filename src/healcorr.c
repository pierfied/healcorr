//
// Created by pierfied on 7/30/19.
//

#include "healcorr.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void calc_ang_sep(long ind, double *x, double *y, double *z, double *ang_sep) {
    double xi = x[ind];
    double yi = y[ind];
    double zi = z[ind];

#pragma omp parallel for
    for (long j = 0; j < ind; ++j) {
        ang_sep[j] = acos(xi * x[j] + yi * y[j] + zi * z[j]);
    }
    ang_sep[ind] = 0;
}

double *healcorr(long npix, double *theta, double *phi, long nmaps, double *maps, long nbins, double *bins) {
    double *map_means = malloc(sizeof(double) * nmaps);
#pragma omp parallel for
    for (long i = 0; i < nmaps; ++i) {
        map_means[i] = 0;

        for (long j = 0; j < npix; ++j) {
            map_means[i] += maps[i * npix + j];
        }

        map_means[i] /= npix;
    }

    double *x = malloc(sizeof(double) * npix);
    double *y = malloc(sizeof(double) * npix);
    double *z = malloc(sizeof(double) * npix);
#pragma omp parallel for
    for (long i = 0; i < npix; ++i) {
        x[i] = cos(phi[i]) * sin(theta[i]);
        y[i] = sin(phi[i]) * sin(theta[i]);
        z[i] = cos(theta[i]);
    }

    double *xis = malloc(sizeof(double) * nbins * nmaps);
    double *counts = malloc(sizeof(double) * nbins);
#pragma omp parallel for
    for (long i = 0; i < nbins * nmaps; ++i) {
        xis[i] = 0;
        counts[i] = 0;
    }

    double *ang_sep = malloc(sizeof(double) * npix);
    long *bin_nums = malloc(sizeof(long) * npix);
    for (long i = 0; i < npix; ++i) {
        calc_ang_sep(i, x, y, z, ang_sep);

#pragma omp parallel for
        for (long j = 0; j <= i; ++j) {
            if (ang_sep[j] < bins[0] || ang_sep[j] > bins[nbins]) {
                bin_nums[j] = -1;
            } else {
                for (long k = 0; k < nbins; ++k) {
                    if (ang_sep[j] < bins[k + 1]) {
                        bin_nums[j] = k;
                        break;
                    }
                }
            }
        }

        for (long j = 0; j <= i; ++j) {
            if (bin_nums[j] >= 0){
                counts[j]++;
            }
        }

        for (long j = 0; j <= i; ++j) {
            if (bin_nums[j] >= 0){
#pragma omp parallel for
                for (long k = 0; k < nmaps; ++k) {
                    xis[k * nbins + bin_nums[j]] += maps[k * npix + i] * maps[k * npix + j];
                }
            }
        }

//#pragma omp parallel for
//        for (long j = 0; j < nmaps; ++j) {
//            double mi = maps[j * npix + i];
//
//            for (long k = 0; k <= i; ++k) {
//                if (bin_nums[k] >= 0) {
//                    xis[j * nbins + bin_nums[k]] += mi * maps[j * npix + k];
//                }
//            }
//        }
    }

#pragma omp parallel for
    for (long i = 0; i < nmaps; ++i) {
        for (long j = 0; j < nbins; ++j) {
            long ind = i * nbins + j;
            xis[ind] = (xis[ind] - (map_means[i] * map_means[i] * counts[j])) / (counts[j] - 1);
        }
    }

    free(map_means);
    free(x);
    free(y);
    free(z);
    free(counts);
    free(ang_sep);
    free(bin_nums);

    return xis;
}