//
// Created by pierfied on 7/30/19.
//

#include "healcorr.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

double *healcorr(long npix, double *theta, double *phi, long nmaps, double *maps, long nbins, double *bins,
                 long verbose, long cross_correlate) {
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

    long nxis;
    if (cross_correlate) {
        nxis = (nmaps * (nmaps + 1)) / 2;
    }else{
        nxis = nmaps;
    }

    long *map1 = malloc(sizeof(long) * nxis);
    long *map2 = malloc(sizeof(long) * nxis);

    long xi_num = 0;
    if(cross_correlate){
        for (int i = 0; i < nmaps; ++i) {
            for (int j = 0; j <= i; ++j) {
                map1[xi_num] = i;
                map2[xi_num] = j;

                xi_num++;
            }
        }
    }else{
        for (int i = 0; i < nmaps; ++i) {
            map1[i] = i;
            map2[i] = i;
        }
    }

    double *xis = malloc(sizeof(double) * nbins * nxis);
    double *counts = malloc(sizeof(double) * nbins);
#pragma omp parallel for
    for (long i = 0; i < nbins * nmaps; ++i) {
        xis[i] = 0;
    }
#pragma omp parallel for
    for (long i = 0; i < nbins; ++i) {
        counts[i] = 0;
    }

    time_t prev_time = 0;
    time_t cur_time = 0;

    if (verbose) {
        printf("\n");
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
            if (bin_nums[j] >= 0) {
                counts[bin_nums[j]]++;
            }
        }

        for (long j = 0; j <= i; ++j) {
            if (bin_nums[j] >= 0) {
#pragma omp parallel for
                for (long k = 0; k < nxis; ++k) {
                    xis[k * nbins + bin_nums[j]] += maps[map1[k] * npix + i] * maps[map2[k] * npix + j];
                }
            }
        }

        cur_time = time(NULL);
        if (verbose && (cur_time > prev_time || i == npix - 1)) {
            printf("\033[A\33[2K\r%ld/%ld: %ld%% Complete\n", i + 1, npix, 100 * (i + 1) / npix);
            prev_time = cur_time;
        }
    }

#pragma omp parallel for
    for (long i = 0; i < nxis; ++i) {
        for (long j = 0; j < nbins; ++j) {
            long ind = i * nbins + j;
            xis[ind] = (xis[ind] - (map_means[map1[i]] * map_means[map2[i]] * counts[j])) / (counts[j] - 1);
        }
    }

    free(map_means);
    free(x);
    free(y);
    free(z);
    free(counts);
    free(ang_sep);
    free(bin_nums);
    free(map1);
    free(map2);

    return xis;
}